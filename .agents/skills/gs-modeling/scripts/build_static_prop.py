#!/usr/bin/env python3
"""Package a cleaned static 3DGS prop/terrain asset into a standardized `v1_prop/` bundle.

Designed for `gs-modeling` `mode: static_prop` (trees, rocks, chests, lamps, buildings, furniture)
that skip Stage 4 (face/eyes) and Stage 5 (rigging) and freeze right after Stage 3 cleanup.

Outputs:
  <out_dir>/
    prop.ply           # Z-up, -Y-forward, bottom-center pivot (Z=0), scaled to --target-height-m
    meta.json          # Dimensions, bbox, footprint_radius_m, collision_cylinder for gs-stage / downstream
    preview_8view.png  # Optional 8-view turnaround sheet (when --render-qa is passed)

Usage:
  python skills/gs-modeling/scripts/build_static_prop.py \
      --ply work/chest-01/03_body/prop.ply \
      --out-dir work/chest-01/v1_prop \
      --name treasure_chest --category chest \
      --target-height-m 0.75 --clean --render-qa
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from ply_io import read_ply, write_ply  # noqa: E402
from remove_detached import remove_detached  # noqa: E402


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 in [w, x, y, z] format. q1: (4,), q2: (N, 4)."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    out = np.empty_like(q2)
    out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standardized v1_prop/ 3DGS static prop package.")
    parser.add_argument("--ply", type=Path, required=True, help="Input cleaned Stage-3 3DGS .ply (Z-up).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory (e.g. work/<id>/v1_prop).")
    parser.add_argument("--name", type=str, required=True, help="Asset name/id (e.g. treasure_chest, pine_tree).")
    parser.add_argument("--category", type=str, default="prop", help="Semantic category (tree, rock, chest, lamp, building, furniture, prop).")
    parser.add_argument("--target-height-m", type=float, default=1.0, help="Real-world target Z height in meters (e.g. 0.75 for chest, 2.8 for lamp).")
    parser.add_argument("--pivot", choices=("bottom_center", "center"), default="bottom_center", help="Origin placement convention.")
    parser.add_argument("--yaw-deg", type=float, default=0.0, help="Optional Z-axis yaw rotation in degrees to align front to -Y.")
    parser.add_argument("--from-triposplat", action="store_true", help="Convert raw TripoSplat axes (front=+X, up=-Y) to Blender Z-up (front=-Y, up=+Z) before cleanup/scaling.")
    parser.add_argument("--clean", action="store_true", help="Run remove_detached micro-floater cleanup before packaging.")
    parser.add_argument("--min-opacity", type=float, default=0.15, help="Visibility threshold for bounds & footprint calculation.")
    parser.add_argument("--render-qa", action="store_true", help="Render an 8-view preview sheet via Spark (render_views.py).")
    args = parser.parse_args()

    data, _ = read_ply(args.ply)
    n_in = len(data)
    if n_in == 0:
        raise RuntimeError(f"Input PLY has 0 splats: {args.ply}")

    # Drop any NaN/Inf splats defensively
    xyz_check = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    finite = np.isfinite(xyz_check).all(axis=1)
    if not finite.all():
        data = data[finite].copy()

    # Optional raw TripoSplat (front=+X, up=-Y) -> Blender Z-up (front=-Y, up=+Z) conversion
    if args.from_triposplat:
        R_ts = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
        q_ts_xyzw = Rotation.from_matrix(R_ts).as_quat()
        q_ts = np.array([q_ts_xyzw[3], q_ts_xyzw[0], q_ts_xyzw[1], q_ts_xyzw[2]], dtype=np.float32)
        xyz_ts = (np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32) @ R_ts.T).astype(np.float32)
        rot_ts = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float32)
        rn_ts = np.linalg.norm(rot_ts, axis=1, keepdims=True)
        rot_ts = np.where(rn_ts > 1e-8, rot_ts / np.maximum(rn_ts, 1e-8), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        rot_ts = quat_mul_wxyz(q_ts, rot_ts)
        rot_ts /= np.maximum(np.linalg.norm(rot_ts, axis=1, keepdims=True), 1e-8)
        data = data.copy()
        data["x"], data["y"], data["z"] = xyz_ts[:, 0], xyz_ts[:, 1], xyz_ts[:, 2]
        data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = rot_ts[:, 0], rot_ts[:, 1], rot_ts[:, 2], rot_ts[:, 3]

    cleanup_report = None
    if args.clean:
        xyz_tmp = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
        op_tmp = sigmoid(data["opacity"].astype(np.float32))
        vis_tmp = op_tmp >= args.min_opacity
        ext_tmp = float(np.ptp(xyz_tmp[vis_tmp if vis_tmp.sum() >= 20 else slice(None)], axis=0).max())
        vox_s = max(ext_tmp * 0.025, 1e-4)
        cleaned, rep = remove_detached(data, voxel=vox_s, fine_voxel=ext_tmp * 0.010, min_opacity=args.min_opacity)
        if rep.get("removed_fraction", 0.0) > 0.25:
            # Guard against over-pruning sparse procedural/thin props: fall back to coarse component filter only
            cleaned, rep = remove_detached(data, voxel=vox_s, fine_voxel=0.0, min_fine_pts=0, min_opacity=args.min_opacity)
        data = cleaned
        cleanup_report = {
            "removed_points": int(rep.get("removed_points", 0)),
            "removed_fraction": round(float(rep.get("removed_fraction", 0.0)), 4),
            "kept_components": int(rep.get("kept_components", 1)),
        }

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    rot = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float32)
    rn = np.linalg.norm(rot, axis=1, keepdims=True)
    rot = np.where(rn > 1e-8, rot / np.maximum(rn, 1e-8), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    sc = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float32)
    op_prob = sigmoid(data["opacity"].astype(np.float32))

    # Optional Z-axis yaw rotation
    if abs(args.yaw_deg) > 1e-4:
        r_yaw = Rotation.from_euler("z", args.yaw_deg, degrees=True)
        R = r_yaw.as_matrix().astype(np.float32)
        q_xyzw = r_yaw.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
        xyz = xyz @ R.T
        rot = quat_mul_wxyz(q_wxyz, rot)
        rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8)

    # Robust visible bounds (1st..99th percentile on Z for height, 0.5..99.5 for XY so stray wisps don't skew pivot)
    vis = op_prob >= args.min_opacity
    if vis.sum() < 20:
        vis = np.ones(len(xyz), dtype=bool)
    v_xyz = xyz[vis]
    z_lo = float(np.percentile(v_xyz[:, 2], 0.5))
    z_hi = float(np.percentile(v_xyz[:, 2], 99.5))
    cur_h = max(z_hi - z_lo, 1e-5)

    # Scale uniformly to --target-height-m
    scale_factor = float(args.target_height_m) / cur_h
    xyz = xyz * scale_factor
    sc = sc + math.log(scale_factor)

    # Recompute visible bounds after scaling and apply pivot
    v_xyz = xyz[vis]
    x_lo, x_hi = float(np.percentile(v_xyz[:, 0], 0.5)), float(np.percentile(v_xyz[:, 0], 99.5))
    y_lo, y_hi = float(np.percentile(v_xyz[:, 1], 0.5)), float(np.percentile(v_xyz[:, 1], 99.5))
    z_lo, z_hi = float(np.percentile(v_xyz[:, 2], 0.5)), float(np.percentile(v_xyz[:, 2], 99.5))

    cx = 0.5 * (x_lo + x_hi)
    cy = 0.5 * (y_lo + y_hi)
    cz = z_lo if args.pivot == "bottom_center" else 0.5 * (z_lo + z_hi)
    xyz[:, 0] -= cx
    xyz[:, 1] -= cy
    xyz[:, 2] -= cz

    # Write updated attributes back to PLY
    out_data = data.copy()
    out_data["x"], out_data["y"], out_data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out_data["rot_0"], out_data["rot_1"], out_data["rot_2"], out_data["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    out_data["scale_0"], out_data["scale_1"], out_data["scale_2"] = sc[:, 0], sc[:, 1], sc[:, 2]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = args.out_dir / "prop.ply"
    write_ply(out_ply, out_data)

    # Compute final footprint & collision cylinder in meters
    v_xyz = xyz[vis]
    bmin = np.percentile(v_xyz, 0.5, axis=0)
    bmax = np.percentile(v_xyz, 99.5, axis=0)
    dims = bmax - bmin
    xy_r = np.hypot(v_xyz[:, 0], v_xyz[:, 1])
    footprint_r = float(np.percentile(xy_r, 98.0))
    coll_r = float(np.percentile(xy_r, 90.0))

    preview_rel = None
    if args.render_qa:
        render_script = SCRIPTS / "render_views.py"
        if render_script.exists():
            qa_dir = args.out_dir / "_qa_views"
            qa_dir.mkdir(parents=True, exist_ok=True)
            out_prefix = qa_dir / "prop"
            angles = [0, 45, 90, 135, 180, -135, -90, -45]
            angle_arg = ",".join(str(a) for a in angles)
            cmd = [
                sys.executable, str(render_script),
                "--ply", str(out_ply),
                "--out", str(out_prefix),
                "--only", f"full:{angle_arg}",
            ]
            from config_loader import resolve_spark_dir
            vendor_cand = resolve_spark_dir() / "vendor"
            if vendor_cand.exists():
                cmd += ["--vendor", str(vendor_cand)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[build_static_prop] render_views.py failed ({res.returncode}): {res.stderr.strip()}", file=sys.stderr)
            else:
                from PIL import Image, ImageDraw
                tiles = []
                for a in angles:
                    tag = "000" if a == 0 else (f"p{a:03d}" if a > 0 else f"m{abs(a):03d}")
                    found = qa_dir / f"prop_full_{tag}.png"
                    if found.exists():
                        tiles.append((a, Image.open(found).convert("RGB").resize((360, 360), Image.Resampling.LANCZOS)))
                if tiles:
                    cols = 4
                    rows = (len(tiles) + cols - 1) // cols
                    sheet = Image.new("RGB", (360 * cols, 360 * rows), (24, 24, 30))
                    draw = ImageDraw.Draw(sheet)
                    for idx, (a, im) in enumerate(tiles):
                        x, y = (idx % cols) * 360, (idx // cols) * 360
                        sheet.paste(im, (x, y))
                        draw.rectangle([x + 8, y + 8, x + 92, y + 32], fill=(12, 16, 24))
                        draw.text((x + 14, y + 13), f"{a:+d} deg" if a != 0 else "0 deg", fill=(235, 240, 250))
                    preview_path = args.out_dir / "preview_8view.png"
                    sheet.save(preview_path)
                    preview_rel = preview_path.name

    meta = {
        "schema_version": "1.0",
        "mode": "static_prop",
        "name": args.name,
        "category": args.category,
        "convention": f"Z-up, -Y-forward, pivot={args.pivot}, units=meters",
        "pivot": args.pivot,
        "splat_count": int(len(out_data)),
        "input_splat_count": int(n_in),
        "target_height_m": round(float(args.target_height_m), 4),
        "dimensions_m": {
            "x": round(float(dims[0]), 4),
            "y": round(float(dims[1]), 4),
            "z": round(float(dims[2]), 4),
        },
        "bbox_min_m": [round(float(v), 4) for v in bmin],
        "bbox_max_m": [round(float(v), 4) for v in bmax],
        "footprint_radius_m": round(footprint_r, 4),
        "collision_cylinder": {
            "radius_m": round(coll_r, 4),
            "height_m": round(float(dims[2]), 4),
        },
        "cleanup": cleanup_report,
        "files": {
            "ply": "prop.ply",
            "preview": preview_rel,
        },
    }
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
