"""Real-Time Position+Normal Map 3D Relighting + Slab Occlusion + Character Self-Shadow + Emissive Bloom.

Takes an already-rendered sequence of stage walkthrough PNGs (from render_stage_walk.py) plus the
static stage PLY, frame JSON, and 3D walk trajectory JSON:
  1. Renders ONE solid Position Map of the static stage (excluding translucent mist/sparkles).
  2. Reconstructs grain-free 3D world coordinates & smooth surface normals per pixel.
  3. Relights every frame in real time with:
       - Moving handheld lantern 3D point light with multi-layer slab occlusion (prevents vertical
         light bleed through bridges/terraces!) and camera Z-depth halo occlusion.
       - Character pixel separation (median static background diff) + back-facing self-shadowing
         so a front-held lantern never shines through the character's back/cape.
       - Automatic emissive bloom & subtle flicker for all light anchors declared in frame.json.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from ply_io import read_ply, write_ply
from assemble_stage import blender_to_spark_data

SH_C0 = 0.28209479177387814

DEFAULT_LIGHT_COLORS = {
    "soft_ivory": [1.00, 0.74, 0.28],
    "warm_lamp":  [1.00, 0.72, 0.26],
    "fire":       [1.00, 0.56, 0.18],
    "soft_aqua":  [0.32, 0.80, 1.00],
    "crystal":    [0.32, 0.80, 1.00],
    "soft_mint":  [0.64, 1.00, 0.62],
    "magic":      [0.64, 1.00, 0.62],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply Position+Normal 3D real-time relighting & emissive bloom to a stage walk sequence.")
    ap.add_argument("--map-ply", type=Path, required=True, help="Assembled static stage PLY (Blender Z-up).")
    ap.add_argument("--frames-dir", type=Path, required=True, help="Directory of rendered walkthrough PNGs (f_000.png ...).")
    ap.add_argument("--frame-json", type=Path, required=True, help="Volumetric 3D frame JSON (volumes + lights).")
    ap.add_argument("--trajectory", type=Path, required=True, help="walk_3d_trajectory.json.")
    ap.add_argument("--out-gif", type=Path, required=True, help="Output relit animated GIF path.")
    ap.add_argument("--spark-dir", type=Path, required=True, help="spark_demo directory for 1-frame Position Map render.")
    ap.add_argument("--cam-front", type=float, default=-20.0)
    ap.add_argument("--cam-pitch", type=float, default=19.0)
    ap.add_argument("--cam-dist", type=float, default=12.2)
    ap.add_argument("--cam-fov", type=float, default=35.0)
    ap.add_argument("--cam-target", default="0.0,-1.15,0.25", help="Spark target x,y,z.")
    ap.add_argument("--char-height-m", type=float, default=1.40)
    ap.add_argument("--lantern-strength", type=float, default=1.95)
    ap.add_argument("--slab-z", type=float, default=1.82, help="Vertical slab occlusion split height (m) between Layer 0 and Layer 1.")
    ap.add_argument("--duration-ms", type=int, default=90)
    args = ap.parse_args()

    t0 = time.time()
    frame_cfg = json.loads(args.frame_json.read_text(encoding="utf-8"))
    traj = json.loads(args.trajectory.read_text(encoding="utf-8"))
    pngs = sorted(args.frames_dir.glob("f_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No f_*.png frames found in {args.frames_dir}")
    n_frs = min(len(pngs), len(traj))
    W, H = Image.open(pngs[0]).size
    target_s = [float(v) for v in args.cam_target.split(",")]

    # 1. Render ONE solid Position Map of the static stage (exclude translucent mist/sparkles)
    map_d, _ = read_ply(args.map_ply)
    op_m = 1.0 / (1.0 + np.exp(-np.clip(map_d["opacity"], -20.0, 20.0)))
    d_sol = map_d[op_m > 0.28]
    xyz_b = np.column_stack([d_sol["x"], d_sol["y"], d_sol["z"]]).astype(np.float32)
    lo, hi = xyz_b.min(axis=0) - 0.1, xyz_b.max(axis=0) + 0.1
    span = np.maximum(hi - lo, 1e-4)

    pos_s = blender_to_spark_data(d_sol)
    nrgb = np.clip((xyz_b - lo[None, :]) / span[None, :], 0.0, 1.0) * 0.90 + 0.05
    fdc_p = (nrgb - 0.5) / SH_C0
    pos_s["f_dc_0"], pos_s["f_dc_1"], pos_s["f_dc_2"] = fdc_p[:, 0], fdc_p[:, 1], fdc_p[:, 2]

    tag = f"relight_{args.out_gif.stem}"
    tmp_dir = args.spark_dir / "frames" / tag
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pos_rel = f"frames/{tag}/stage_pos.ply"
    write_ply(args.spark_dir / pos_rel, pos_s)

    cam = {
        "width": W, "height": H, "background": "#000000",
        "target": target_s, "distance": args.cam_dist, "pitch": args.cam_pitch,
        "fov": args.cam_fov, "front": args.cam_front, "side": args.cam_front + 90.0,
    }
    crel, mrel = f"frames/{tag}/cam_pos.json", f"frames/{tag}/man_pos.json"
    (args.spark_dir / crel).write_text(json.dumps(cam), encoding="utf-8")
    (args.spark_dir / mrel).write_text(
        json.dumps({"static": [pos_rel], "review": [], "frames": [{"blenderFrame": 0, "ply": [], "png": "stage_pos.png"}]}),
        encoding="utf-8",
    )
    subprocess.run(["node", "capture.mjs", tag, f"{tag}_out", crel, mrel], cwd=str(args.spark_dir), check=True, capture_output=True)

    # 2. Reconstruct smooth 3D world-space xyz_s & smooth surface normals nrm
    pos_im = np.asarray(Image.open(args.spark_dir / f"{tag}_out" / "stage_pos.png").convert("RGB"), dtype=np.float32) / 255.0
    fg = pos_im.max(axis=-1) > 0.032
    fg_f = fg.astype(np.float32)
    xyz_px = lo[None, None, :] + np.clip((pos_im - 0.05) / 0.90, 0.0, 1.0) * span[None, None, :]
    wb = cv2.GaussianBlur(fg_f, (0, 0), 3.0)
    xyz_s = np.zeros_like(xyz_px)
    for ch in range(3):
        xyz_s[:, :, ch] = np.where(fg, cv2.GaussianBlur(xyz_px[:, :, ch] * fg_f, (0, 0), 3.0) / np.maximum(wb, 1e-4), 0.0)

    d_dy, d_dx = np.gradient(xyz_s, axis=(0, 1))
    nrm = np.cross(d_dx, -d_dy)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)

    az, el = math.radians(args.cam_front), math.radians(args.cam_pitch)
    cam_tw = np.array([args.cam_dist * math.sin(az) * math.cos(el), args.cam_dist * math.sin(el), args.cam_dist * math.cos(az) * math.cos(el)], dtype=np.float32)
    cam_dir_b = np.array([cam_tw[0], -cam_tw[2], cam_tw[1]], dtype=np.float32)
    cam_dir_b /= np.linalg.norm(cam_dir_b)
    cam_b = np.array([cam_tw[0], -cam_tw[2], cam_tw[1]], dtype=np.float32) + np.array([target_s[0], target_s[2], -target_s[1]], dtype=np.float32)

    nrm[np.sum(nrm * cam_dir_b[None, None, :], axis=-1) < 0] *= -1.0
    wn = cv2.GaussianBlur(fg_f, (0, 0), 2.8)
    for ch in range(3):
        nrm[:, :, ch] = np.where(fg, cv2.GaussianBlur(nrm[:, :, ch] * fg_f, (0, 0), 2.8) / np.maximum(wn, 1e-4), 0.0)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)

    fwd = -cam_tw / np.linalg.norm(cam_tw)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], np.float32)); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    fx = fy = (H * 0.5) / math.tan(math.radians(args.cam_fov * 0.5))
    u0, v0 = (W - 1) * 0.5, (H - 1) * 0.5
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def proj_2d(p_b):
        tw = np.array([p_b[0] - target_s[0], p_b[2] - (-target_s[1]), -(p_b[1] - target_s[2])], dtype=np.float32)
        rel_v = tw - cam_tw
        d_cam = float(np.dot(rel_v, fwd))
        return float(np.dot(rel_v, right) / d_cam * fx + u0), float(-np.dot(rel_v, up) / d_cam * fy + v0)

    # 3. Build static background median to cleanly separate character pixels from stage pixels
    sample_ids = np.linspace(0, n_frs - 1, min(4, n_frs)).round().astype(int)
    static_bg = np.median(
        np.stack([np.asarray(Image.open(pngs[k]).convert("RGB"), dtype=np.float32) / 255.0 for k in sample_ids], axis=0),
        axis=0,
    )

    static_lights = []
    for li, lt in enumerate(frame_cfg.get("lights", [])):
        col = np.array(DEFAULT_LIGHT_COLORS.get(lt.get("kind", "soft_ivory"), [1.0, 0.78, 0.34]), dtype=np.float32)
        static_lights.append((np.array([lt["x"], lt["y"], lt.get("z", 1.2)], dtype=np.float32), col, 0.36, 0.42 + 0.07 * li, float(li) * 1.1))

    # 4. Real-time 3D relighting + slab occlusion + character self-shadow + emissive bloom
    out_frs = []
    lan_col = np.array([1.35, 0.86, 0.34], dtype=np.float32)
    px_z = xyz_s[:, :, 2]

    for fi in range(n_frs):
        base_im = np.asarray(Image.open(pngs[fi]).convert("RGB"), dtype=np.float32) / 255.0
        t = traj[fi]
        yr_c = math.radians(float(t["yaw_deg"]))
        char_fwd = np.array([math.sin(yr_c), -math.cos(yr_c), 0.0], dtype=np.float32)
        # Approximate handheld lantern world position ~0.35m in front of character at chest height
        lw = np.array([float(t["x"]), float(t["y"]), float(t["z"]) + args.char_height_m * 0.56], dtype=np.float32) + char_fwd * 0.34
        pulse = 1.0 + 0.09 * math.sin(fi * 0.65)

        is_char = np.abs(base_im - static_bg).max(axis=-1) > 0.045
        stage_only_f = fg_f * (~is_char).astype(np.float32)
        back_factor = float(np.clip(-float(np.dot(char_fwd, cam_dir_b)), 0.0, 1.0))

        lv = lw[None, None, :] - xyz_s
        d3 = np.linalg.norm(lv, axis=-1)
        ld = lv / np.maximum(d3[:, :, None], 1e-4)
        nd = np.clip(np.sum(nrm * ld, axis=-1), 0.0, 1.0)
        wrap = 0.28 + 0.72 * nd

        # Multi-layer slab occlusion
        if lw[2] < args.slab_z - 0.15:
            slab_vis = np.clip((args.slab_z - px_z) / 0.24, 0.0, 1.0)
        elif lw[2] > args.slab_z + 0.12:
            slab_vis = np.clip((px_z - (args.slab_z - 0.85)) / 0.65, 0.10, 1.0)
        else:
            slab_vis = np.ones((H, W), dtype=np.float32)

        atten = (args.lantern_strength * pulse) * np.exp(-(d3 / 2.25) ** 2) / (0.42 + d3 * d3 * 0.85) * slab_vis
        relight_add = base_im * lan_col[None, None, :] * (wrap * atten * stage_only_f)[:, :, None] * 1.15

        for sp_w, sc_rgb, _, sf, sph in static_lights:
            flk = 0.14 * math.sin(fi * sf + sph)
            d_s = np.linalg.norm(sp_w[None, None, :] - xyz_s, axis=-1)
            relight_add += base_im * sc_rgb[None, None, :] * (np.exp(-(d_s / 1.55) ** 2) * flk * stage_only_f)[:, :, None]

        # Character back-facing self-shadow
        char_shaded = base_im
        if back_factor > 0.05:
            cool_back = np.array([0.52, 0.62, 0.84], dtype=np.float32)
            char_shaded = np.where(
                is_char[:, :, None],
                base_im * (1.0 - 0.55 * back_factor) + (base_im * cool_back[None, None, :]) * (0.55 * back_factor),
                base_im,
            )

        lit = np.clip(char_shaded + relight_add, 0.0, 1.0)
        em_allow = (~(is_char & (back_factor > 0.15))).astype(np.float32)
        em_mask = np.maximum(lit - 0.64, 0.0) * 1.40 * em_allow[:, :, None]
        em_bloom = cv2.GaussianBlur(em_mask, (0, 0), 5.5) * 0.55 + cv2.GaussianBlur(em_mask, (0, 0), 14.0) * 0.35

        halo = np.zeros_like(lit)
        for sp_w, sc_rgb, sg, sf, sph in static_lights:
            u_s, v_s = proj_2d(sp_w)
            rs = np.hypot(xx - u_s, yy - v_s)
            flk_m = 1.0 + 0.10 * math.sin(fi * sf + sph)
            halo += (
                np.exp(-(rs / 12.0) ** 2)[:, :, None] * sc_rgb[None, None, :] * (sg * 0.72 * flk_m)
                + np.exp(-(rs / 46.0) ** 2)[:, :, None] * sc_rgb[None, None, :] * (sg * 0.30 * flk_m)
            )

        u_l, v_l = proj_2d(lw)
        iu, iv = int(np.clip(round(u_l), 0, W - 1)), int(np.clip(round(v_l), 0, H - 1))
        d_lan_cam = float(np.linalg.norm(lw - cam_b))
        d_surf_cam = float(np.linalg.norm(xyz_s[iv, iu] - cam_b)) if fg[iv, iu] else 999.0
        if d_surf_cam >= d_lan_cam - 0.25 and back_factor <= 0.35:
            rl = np.hypot(xx - u_l, yy - v_l)
            halo += (
                np.exp(-(rl / 12.0) ** 2)[:, :, None] * np.array([0.42, 0.30, 0.12], np.float32) * pulse * slab_vis[:, :, None]
                + np.exp(-(rl / 46.0) ** 2)[:, :, None] * np.array([0.18, 0.11, 0.04], np.float32) * pulse * slab_vis[:, :, None]
            )

        comp = lit + em_bloom + halo
        comp = np.clip(comp / (1.0 + 0.22 * np.maximum(comp - 0.82, 0.0)), 0.0, 1.0)
        out_frs.append(Image.fromarray((comp * 255.0 + 0.5).astype(np.uint8), mode="RGB"))

    args.out_gif.parent.mkdir(parents=True, exist_ok=True)
    strip = Image.new("RGB", (W * 4, H))
    for k, idx in enumerate(np.linspace(0, n_frs - 1, 4).round().astype(int)):
        strip.paste(out_frs[idx], (k * W, 0))
    pal = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    qfrs = [f.resize((480, int(round(480 * H / W))), Image.Resampling.LANCZOS).quantize(palette=pal, dither=Image.Dither.NONE) for f in out_frs]
    qfrs[0].save(args.out_gif, save_all=True, append_images=qfrs[1:], duration=args.duration_ms, loop=0, optimize=True)
    print(f"Saved real-time 3D relit GIF ({args.out_gif.stat().st_size/1024:.1f} KB) in {time.time()-t0:.1f}s -> {args.out_gif}", flush=True)


if __name__ == "__main__":
    main()
