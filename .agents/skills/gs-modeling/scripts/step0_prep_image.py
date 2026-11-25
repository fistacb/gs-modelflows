"""Step 0a: Crop front view from multi-view design sheet and create clean RGBA PNG (background-color agnostic)."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from config_loader import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--panels", type=int, default=4, help="Number of horizontal panels in design sheet (1 if single image)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])
    src = Path(args.src) if args.src else (work_dir / cfg["input_design_sheet"])
    dst = Path(args.dst) if args.dst else (work_dir / "input" / "front_refined.png")
    preview_dst = dst.parent / (dst.stem + "_preview.png")

    im = Image.open(src).convert("RGB")
    arr = np.array(im)
    h, w, _ = arr.shape

    if args.panels > 1:
        col_mean = arr[: int(h * 0.85), :, :].mean(axis=(0, 2))
        s0, s1 = int(w * (1.0 / args.panels - 0.03)), int(w * (1.0 / args.panels + 0.03))
        divider_x = s0 + int(np.argmax(col_mean[s0:s1]))
        crop = arr[: int(h * 0.87), : divider_x - 3, :]
    else:
        crop = arr
    ch, cw, _ = crop.shape

    # Estimate background color from borders unless explicitly given in config
    bg_cfg = cfg["background"]
    if bg_cfg.get("bg_rgb") is not None:
        bg_rgb = np.array(bg_cfg["bg_rgb"], dtype=np.float32)
    else:
        border = np.concatenate([
            crop[0:8, :, :].reshape(-1, 3),
            crop[:, 0:6, :].reshape(-1, 3),
            crop[:, -6:, :].reshape(-1, 3),
        ], axis=0)
        bg_rgb = np.median(border, axis=0).astype(np.float32)

    # Save detected bg_rgb alongside output so downstream fringe removal can reuse it automatically
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(np.uint8([[bg_rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(crop_lab - bg_lab[None, None, :], axis=2)

    outer_th = float(bg_cfg.get("lab_outer_thresh", 22.0))
    inner_th = float(bg_cfg.get("lab_interior_thresh", 26.0))

    # Flood-fill outer background + remove enclosed interior pockets matching background Lab color
    bg_cand = (dist < outer_th).astype(np.uint8)
    ff = bg_cand.copy()
    mask_ff = np.zeros((ch + 2, cw + 2), np.uint8)
    for x in range(cw):
        if ff[0, x] == 1: cv2.floodFill(ff, mask_ff, (x, 0), 2)
        if ff[ch - 1, x] == 1: cv2.floodFill(ff, mask_ff, (x, ch - 1), 2)
    for y in range(ch):
        if ff[y, 0] == 1: cv2.floodFill(ff, mask_ff, (0, y), 2)
        if ff[y, cw - 1] == 1: cv2.floodFill(ff, mask_ff, (cw - 1, y), 2)

    interior_bg = dist < inner_th
    fg_mask = ((ff != 2) & (~interior_bg)).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg_mask = (labels == largest).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    alpha = (fg_mask * 255).astype(np.uint8)
    eroded = cv2.erode(fg_mask, kernel, iterations=1)
    edge_band = (fg_mask > 0) & (eroded == 0)
    edge_alpha = np.clip((dist - (outer_th * 0.65)) / max(outer_th * 0.65, 1.0), 0.0, 1.0)
    alpha_f = alpha.astype(np.float32) / 255.0
    alpha_f[edge_band] = np.minimum(alpha_f[edge_band], edge_alpha[edge_band])
    alpha = np.clip(alpha_f * 255.0, 0, 255).astype(np.uint8)

    ys, xs = np.where(alpha > 16)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    rgba_tight = np.dstack([crop[y0:y1, x0:x1], alpha[y0:y1, x0:x1]])

    canvas_size = 1024
    pad = int(canvas_size * 0.08)
    sh, sw = rgba_tight.shape[:2]
    scale = (canvas_size - 2 * pad) / max(sh, sw)
    nw, nh = int(round(sw * scale)), int(round(sh * scale))
    resized = Image.fromarray(rgba_tight, mode="RGBA").resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    canvas.paste(resized, ((canvas_size - nw) // 2, (canvas_size - nh) // 2))

    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    Image.alpha_composite(Image.new("RGBA", (canvas_size, canvas_size), (40, 44, 52, 255)), canvas).convert("RGB").save(preview_dst)
    (dst.parent / "detected_bg.json").write_text(json.dumps({"bg_rgb": bg_rgb.tolist()}, indent=2), encoding="utf-8")
    print(f"Saved: {dst}, bg_rgb={bg_rgb.round(1).tolist()}, preview={preview_dst}")


if __name__ == "__main__":
    main()
