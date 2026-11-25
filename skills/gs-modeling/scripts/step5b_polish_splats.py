"""Step 5b: Edge-preserving bilateral smoothing & background-color-agnostic fringe cleanup."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from ply_io import read_ply, write_ply
from config_loader import load_config

SH_C0 = 0.28209479177387814


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])
    src = Path(args.src) if args.src else (work_dir / "outputs" / "step3_carved_head.ply")
    dst = Path(args.dst) if args.dst else (work_dir / "outputs" / "step3_carved_head_polished.ply")

    data, _ = read_ply(src)
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    fdc = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    sc = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float32)

    # 1. Background-color-agnostic fringe removal using Lab distance from detected/configured bg_rgb
    bg_rgb = cfg["background"].get("bg_rgb")
    det_path = work_dir / "input" / "detected_bg.json"
    if bg_rgb is None and det_path.exists():
        bg_rgb = json.loads(det_path.read_text(encoding="utf-8")).get("bg_rgb")

    keep = np.ones(len(data), dtype=bool)
    if bg_rgb is not None:
        lab_th = float(cfg["background"].get("remove_fringe_lab_thresh", 28.0))
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        pts_lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        bg_u8 = np.uint8([[bg_rgb]])
        bg_lab = cv2.cvtColor(bg_u8, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        lab_dist = np.linalg.norm(pts_lab - bg_lab[None, :], axis=1)
        keep &= (lab_dist >= lab_th)

    n_removed = int((~keep).sum())
    data = data[keep].copy()
    xyz, fdc, rgb, sc = xyz[keep], fdc[keep], rgb[keep], sc[keep]

    # 2. Edge-preserving bilateral smoothing
    tree = cKDTree(xyz)
    dists, idx = tree.query(xyz, k=12)
    d_col2 = np.sum((rgb[idx] - rgb[:, None, :]) ** 2, axis=2)
    w = np.exp(-0.5 * (dists / 0.010) ** 2 - 0.5 * d_col2 / (0.095 ** 2))
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-8)

    xyz_new = 0.72 * xyz + 0.28 * np.sum(w[:, :, None] * xyz[idx], axis=1)
    fdc_new = 0.65 * fdc + 0.35 * np.sum(w[:, :, None] * fdc[idx], axis=1)
    sc_cap = float(np.percentile(sc, 98.0))
    sc_new = np.minimum(sc, sc_cap)

    data["x"], data["y"], data["z"] = xyz_new[:, 0], xyz_new[:, 1], xyz_new[:, 2]
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = fdc_new[:, 0], fdc_new[:, 1], fdc_new[:, 2]
    data["scale_0"], data["scale_1"], data["scale_2"] = sc_new[:, 0], sc_new[:, 1], sc_new[:, 2]

    write_ply(dst, data)
    print(f"Polished -> {dst}: kept={len(data)}, removed_bg_fringe={n_removed}, scale_cap={sc_cap:.3f}")


if __name__ == "__main__":
    main()
