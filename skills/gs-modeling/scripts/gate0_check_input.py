"""Gate 0: Design-agnostic input image resolution & pose readiness checker (Hard Stop before TripoSplat)."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from config_loader import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-crop", required=True, help="Uncanvas-resized tight front-view crop PNG/JPG before 1024x1024 padding")
    ap.add_argument("--rgba", default=None, help="Optional prepared RGBA PNG to check foreground bbox")
    ap.add_argument("--head-ratio", type=float, default=None, help="Estimated head-to-body ratio (e.g. 3.0 for chibi, 7.5 for tall anime)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    th = cfg.get("gate0_input_thresholds", {})
    min_body_h = int(th.get("min_body_height_px", 600))
    min_face_w = int(th.get("min_face_width_px", 120))
    min_hand_w = int(th.get("min_hand_width_px", 40))

    raw_path = Path(args.raw_crop)
    im = Image.open(raw_path)
    w, h = im.size
    arr = np.array(im.convert("RGBA"))
    alpha = arr[:, :, 3]
    if alpha.min() < 250:
        ys, xs = np.where(alpha > 16)
        body_h = int(ys.max() - ys.min() + 1) if len(ys) else h
        body_w = int(xs.max() - xs.min() + 1) if len(xs) else w
    else:
        # Estimate non-background bbox from border median
        rgb = arr[:, :, :3].astype(np.float32)
        border = np.concatenate([rgb[0:4].reshape(-1, 3), rgb[:, 0:4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)], axis=0)
        bg = np.median(border, axis=0)
        fg = np.linalg.norm(rgb - bg[None, None, :], axis=2) > 25.0
        ys, xs = np.where(fg)
        body_h = int(ys.max() - ys.min() + 1) if len(ys) else h
        body_w = int(xs.max() - xs.min() + 1) if len(xs) else w

    # Estimate head-to-body ratio from config face_z_range if not explicitly provided
    if args.head_ratio is not None:
        head_ratio = float(args.head_ratio)
    else:
        fz0, fz1 = cfg["proportions"]["face_z_range"]
        head_frac = max(1.0 - float(fz0), 0.10)
        head_ratio = 1.0 / head_frac

    est_face_px = int(round(body_h / max(head_ratio, 1.5) * 0.72))
    est_hand_px = int(round(body_h * 0.085))
    aspect_w_over_h = body_w / max(body_h, 1)
    pose_type = "T-pose (high shoulder-drop load -> enable 2-stage clavicle rig)" if aspect_w_over_h > 0.68 else "A-pose / relaxed (optimal)"

    failed = []
    if body_h < min_body_h:
        failed.append(f"body_height_px={body_h} < {min_body_h}")
    if est_face_px < min_face_w:
        failed.append(f"est_face_width_px={est_face_px} < {min_face_w}")
    if est_hand_px < min_hand_w:
        failed.append(f"est_hand_width_px={est_hand_px} < {min_hand_w}")

    report = {
        "gate": "Gate 0 (Input Quality)",
        "source": str(raw_path),
        "raw_image_wh": [w, h],
        "foreground_body_wh": [body_w, body_h],
        "estimated_head_ratio": round(head_ratio, 2),
        "est_face_width_px": est_face_px,
        "est_hand_width_px": est_hand_px,
        "aspect_w_over_h": round(aspect_w_over_h, 3),
        "pose_assessment": pose_type,
        "numeric_gate": "OK (awaiting_mask_visual_review)" if not failed else "HARD_STOP_FAIL",
        "failed_checks": failed,
        "recommendation": (
            "Proceed to mask visual review and TripoSplat."
            if not failed
            else "HARD STOP: Input resolution is too low for clean 3DGS face/hands. Provide or generate a higher-resolution standalone front view (body height >= 600px, face >= 120px, preferably A-pose) before running TripoSplat."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
