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
    ap.add_argument("--fireflies", type=int, default=0, help="Number of drifting 3D firefly micro-lights (real 3D surface lighting + depth-occluded orb).")
    ap.add_argument("--firefly-color", default="1.0,0.86,0.42", help="RGB color for firefly micro-lights (comma-separated 0..1).")
    ap.add_argument("--firefly-strength", type=float, default=0.85, help="3D surface illumination strength per firefly.")
    ap.add_argument("--firefly-radius", type=float, default=1.45, help="3D lighting falloff radius (m) per firefly.")
    ap.add_argument("--sun-shadow", action="store_true", help="Cast real 3D directional sun shadows (static stage + moving character) from sky light.")
    ap.add_argument("--sun-dir", default="-0.52,-0.45,0.72", help="Directional sun vector (x,y,z pointing toward sky sun).")
    ap.add_argument("--sun-shadow-strength", type=float, default=0.52, help="Shadow darkening strength (0..1).")
    ap.add_argument("--sun-shadow-tint", default="0.54,0.58,0.74", help="Cool sky-lit shadow tint RGB (0..1).")
    ap.add_argument("--seed", type=int, default=42)
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

    # 3b. Procedural seamlessly-looping 3D firefly micro-lights
    firefly_col = np.array([float(c) for c in args.firefly_color.split(",")], dtype=np.float32)
    fireflies = []
    if args.fireflies > 0:
        rng = np.random.default_rng(args.seed)
        bx0, bx1 = float(lo[0]) + 0.85, float(hi[0]) - 0.85
        by0, by1 = float(lo[1]) + 0.85, float(hi[1]) - 0.85
        for k in range(args.fireflies):
            upper = (k % 2 == 1)
            cx = float(rng.uniform(bx0, bx1))
            cy = float(rng.uniform(by0, by1))
            cz = float(rng.uniform(args.slab_z + 0.35, args.slab_z + 1.45)) if upper else float(rng.uniform(0.45, args.slab_z - 0.30))
            fireflies.append({
                "c": np.array([cx, cy, cz], dtype=np.float32),
                "amp": np.array([float(rng.uniform(0.30, 0.70)), float(rng.uniform(0.30, 0.70)), float(rng.uniform(0.14, 0.34))], dtype=np.float32),
                "freq": (int(rng.choice([1, 2])), int(rng.choice([1, 2])), int(rng.choice([1, 2]))),
                "phase": (float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0, 2 * math.pi))),
                "pfreq": int(rng.choice([1, 2, 3])),
                "pphase": float(rng.uniform(0, 2 * math.pi)),
            })

    # 3c. Optional 3D Directional Sun Shadow Map (static stage cast shadows from sky light)
    sun_dir = np.array([float(v) for v in args.sun_dir.split(",")], dtype=np.float32)
    sun_dir /= max(float(np.linalg.norm(sun_dir)), 1e-6)
    shadow_tint = np.array([float(v) for v in args.sun_shadow_tint.split(",")], dtype=np.float32)
    static_sun_shadow = np.zeros((H, W), dtype=np.float32)
    sun_ndotl = np.ones((H, W), dtype=np.float32)

    if args.sun_shadow:
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(sun_dir[2]) < 0.95 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u_sun = np.cross(up_ref, sun_dir); u_sun /= np.linalg.norm(u_sun)
        v_sun = np.cross(sun_dir, u_sun); v_sun /= np.linalg.norm(v_sun)

        su_b = xyz_b @ u_sun
        sv_b = xyz_b @ v_sun
        sz_b = xyz_b @ sun_dir
        res_s = 0.045
        u_min, u_max = float(su_b.min()) - 0.3, float(su_b.max()) + 0.3
        v_min, v_max = float(sv_b.min()) - 0.3, float(sv_b.max()) + 0.3
        nu = int(math.ceil((u_max - u_min) / res_s)) + 2
        nv = int(math.ceil((v_max - v_min) / res_s)) + 2
        smap = np.full((nv, nu), -1e6, dtype=np.float32)
        iu_b = np.clip(((su_b - u_min) / res_s).astype(np.int32), 0, nu - 1)
        iv_b = np.clip(((sv_b - v_min) / res_s).astype(np.int32), 0, nv - 1)
        np.maximum.at(smap, (iv_b, iu_b), sz_b)
        # 1-texel morphological max-dilate closes tiny gaps between splats
        smap = cv2.dilate(smap, np.ones((3, 3), np.uint8))

        su_px = xyz_s[:, :, 0] * u_sun[0] + xyz_s[:, :, 1] * u_sun[1] + xyz_s[:, :, 2] * u_sun[2]
        sv_px = xyz_s[:, :, 0] * v_sun[0] + xyz_s[:, :, 1] * v_sun[1] + xyz_s[:, :, 2] * v_sun[2]
        sz_px = xyz_s[:, :, 0] * sun_dir[0] + xyz_s[:, :, 1] * sun_dir[1] + xyz_s[:, :, 2] * sun_dir[2]
        iu_px = np.clip(((su_px - u_min) / res_s).astype(np.int32), 0, nu - 1)
        iv_px = np.clip(((sv_px - v_min) / res_s).astype(np.int32), 0, nv - 1)
        z_occ = smap[iv_px, iu_px]
        # Soft cast shadow where a higher solid occluder blocks the sun ray
        raw_sh = np.clip((z_occ - sz_px - 0.24) / 0.22, 0.0, 1.0) * fg_f
        # Also softly shade surfaces facing away from the sun
        sun_ndotl = np.clip(np.sum(nrm * sun_dir[None, None, :], axis=-1), 0.0, 1.0)
        nrm_sh = np.clip((0.18 - sun_ndotl) / 0.25, 0.0, 1.0) * 0.45 * fg_f
        static_sun_shadow = np.maximum(cv2.GaussianBlur(raw_sh, (0, 0), 2.2), cv2.GaussianBlur(nrm_sh, (0, 0), 1.5)) * fg_f

    # 4. Real-time 3D relighting + slab occlusion + character self-shadow + fireflies + sun shadow + emissive bloom
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

        relight_add = np.zeros_like(base_im)
        slab_vis = np.ones((H, W), dtype=np.float32)
        if args.lantern_strength > 1e-4:
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

            atten = (args.lantern_strength * pulse) * np.exp(-(d3 / 2.25) ** 2) / (0.42 + d3 * d3 * 0.85) * slab_vis
            relight_add += base_im * lan_col[None, None, :] * (wrap * atten * stage_only_f)[:, :, None] * 1.15

        is_day_sun = args.sun_shadow and args.lantern_strength <= 1e-4

        if not is_day_sun:
            for sp_w, sc_rgb, _, sf, sph in static_lights:
                flk = 0.14 * math.sin(fi * sf + sph)
                d_s = np.linalg.norm(sp_w[None, None, :] - xyz_s, axis=-1)
                relight_add += base_im * sc_rgb[None, None, :] * (np.exp(-(d_s / 1.55) ** 2) * flk * stage_only_f)[:, :, None]

        # Dynamic 3D firefly micro-lights (real localized 3D surface illumination + slab occlusion)
        theta = 2.0 * math.pi * (fi / max(n_frs, 1))
        ff_frame = []
        for ff in fireflies:
            fx_w = ff["c"][0] + ff["amp"][0] * math.sin(ff["freq"][0] * theta + ff["phase"][0])
            fy_w = ff["c"][1] + ff["amp"][1] * math.cos(ff["freq"][1] * theta + ff["phase"][1])
            fz_w = ff["c"][2] + ff["amp"][2] * math.sin(ff["freq"][2] * theta + ff["phase"][2])
            fw = np.array([fx_w, fy_w, fz_w], dtype=np.float32)
            glow = 0.28 + 0.72 * ((0.5 + 0.5 * math.sin(ff["pfreq"] * theta + ff["pphase"])) ** 1.6)
            ff_frame.append((fw, glow))
            if args.firefly_strength > 1e-4:
                fv = fw[None, None, :] - xyz_s
                df = np.linalg.norm(fv, axis=-1)
                fd = fv / np.maximum(df[:, :, None], 1e-4)
                ndf = 0.35 + 0.65 * np.clip(np.sum(nrm * fd, axis=-1), 0.0, 1.0)
                if fw[2] < args.slab_z - 0.12:
                    f_slab = np.clip((args.slab_z - px_z) / 0.24, 0.0, 1.0)
                elif fw[2] > args.slab_z + 0.12:
                    f_slab = np.clip((px_z - (args.slab_z - 0.75)) / 0.55, 0.12, 1.0)
                else:
                    f_slab = 1.0
                f_att = (args.firefly_strength * glow) * np.exp(-(df / args.firefly_radius) ** 2) / (0.35 + df * df * 1.2) * f_slab
                relight_add += base_im * firefly_col[None, None, :] * (ndf * f_att * stage_only_f)[:, :, None]

        # Character shading & optional 3D Directional Sun Shadows (static stage + dynamic character cast shadow)
        char_shaded = base_im.copy()
        if args.sun_shadow:
            cx, cy, cz = float(t["x"]), float(t["y"]), float(t["z"])
            sc_h = args.char_height_m / 1.40
            # Dynamic character silhouette cast shadow along -sun_dir
            char_cast = np.zeros((H, W), dtype=np.float32)
            if cz < args.slab_z - 0.15:
                c_slab = np.clip((args.slab_z - px_z) / 0.22, 0.0, 1.0)
            else:
                c_slab = np.clip((px_z - (args.slab_z - 0.55)) / 0.45, 0.15, 1.0)
            for hk, rk in ((0.28 * sc_h, 0.27), (0.68 * sc_h, 0.26), (1.08 * sc_h, 0.22)):
                sp_c = np.array([cx, cy, cz + hk], dtype=np.float32)
                dv = sp_c[None, None, :] - xyz_s
                t_ray = dv[:, :, 0] * sun_dir[0] + dv[:, :, 1] * sun_dir[1] + dv[:, :, 2] * sun_dir[2]
                perp = dv - t_ray[:, :, None] * sun_dir[None, None, :]
                d_perp = np.linalg.norm(perp, axis=-1)
                hit = np.clip(1.0 - (d_perp / rk) ** 2, 0.0, 1.0) * np.clip((t_ray - 0.02) / 0.18, 0.0, 1.0) * np.clip((2.8 - t_ray) / 0.8, 0.0, 1.0)
                char_cast = np.maximum(char_cast, hit)
            # Soft contact AO shadow directly under feet
            d_ft = np.hypot(xyz_s[:, :, 0] - cx, xyz_s[:, :, 1] - cy)
            dz_ft = np.abs(px_z - cz)
            contact_ao = np.exp(-(d_ft / 0.28) ** 2) * np.exp(-(dz_ft / 0.14) ** 2) * 0.72
            char_cast = np.maximum(char_cast * c_slab, contact_ao) * stage_only_f

            total_sh = np.maximum(static_sun_shadow * stage_only_f, char_cast)
            sh_w = np.clip(total_sh * args.sun_shadow_strength, 0.0, 0.85)[:, :, None]
            # Cool sky-lit shadow tint on shaded stage pixels + subtle warm sunlit pop on unshaded sun-facing surfaces
            stage_sun = base_im * (1.0 - sh_w) + (base_im * shadow_tint[None, None, :]) * sh_w
            sun_pop = np.clip(sun_ndotl - 0.35, 0.0, 1.0) * (1.0 - total_sh) * stage_only_f
            stage_sun = np.clip(stage_sun + base_im * np.array([0.08, 0.06, 0.02], np.float32)[None, None, :] * sun_pop[:, :, None], 0.0, 1.0)

            # Also shade the character when she walks into a static stage shadow (e.g. under the skybridge!)
            char_mid = np.array([cx, cy, cz + 0.70 * sc_h], dtype=np.float32)
            c_iu = int(np.clip((float(np.dot(char_mid, u_sun)) - u_min) / res_s, 0, nu - 1))
            c_iv = int(np.clip((float(np.dot(char_mid, v_sun)) - v_min) / res_s, 0, nv - 1))
            c_sz = float(np.dot(char_mid, sun_dir))
            c_in_sh = float(np.clip((smap[c_iv, c_iu] - c_sz - 0.24) / 0.35, 0.0, 1.0)) * args.sun_shadow_strength * 0.85
            char_lit = base_im * (1.0 - c_in_sh) + (base_im * shadow_tint[None, None, :]) * c_in_sh
            char_shaded = np.where(is_char[:, :, None], char_lit, stage_sun)
        elif args.lantern_strength > 1e-4 and back_factor > 0.05:
            cool_back = np.array([0.52, 0.62, 0.84], dtype=np.float32)
            char_shaded = np.where(
                is_char[:, :, None],
                base_im * (1.0 - 0.55 * back_factor) + (base_im * cool_back[None, None, :]) * (0.55 * back_factor),
                base_im,
            )

        lit = np.clip(char_shaded + relight_add, 0.0, 1.0)
        if is_day_sun:
            em_bloom = np.zeros_like(lit)
        else:
            em_allow = (~(is_char & (back_factor > 0.15))).astype(np.float32)
            em_mask = np.maximum(lit - 0.64, 0.0) * 1.40 * em_allow[:, :, None]
            em_bloom = cv2.GaussianBlur(em_mask, (0, 0), 5.5) * 0.55 + cv2.GaussianBlur(em_mask, (0, 0), 14.0) * 0.35

        halo = np.zeros_like(lit)
        if not is_day_sun:
            for sp_w, sc_rgb, sg, sf, sph in static_lights:
                u_s, v_s = proj_2d(sp_w)
                rs = np.hypot(xx - u_s, yy - v_s)
                flk_m = 1.0 + 0.10 * math.sin(fi * sf + sph)
                halo += (
                    np.exp(-(rs / 12.0) ** 2)[:, :, None] * sc_rgb[None, None, :] * (sg * 0.72 * flk_m)
                    + np.exp(-(rs / 46.0) ** 2)[:, :, None] * sc_rgb[None, None, :] * (sg * 0.30 * flk_m)
                )

        if args.lantern_strength > 1e-4:
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

        # Depth-occluded firefly orbs (properly pass behind sandstone walls & arches!)
        for fw, glow in ff_frame:
            u_f, v_f = proj_2d(fw)
            if -20 <= u_f < W + 20 and -20 <= v_f < H + 20:
                iuf, ivf = int(np.clip(round(u_f), 0, W - 1)), int(np.clip(round(v_f), 0, H - 1))
                d_ff_cam = float(np.linalg.norm(fw - cam_b))
                d_s_cam = float(np.linalg.norm(xyz_s[ivf, iuf] - cam_b)) if fg[ivf, iuf] else 999.0
                if d_s_cam >= d_ff_cam - 0.18:
                    rf = np.hypot(xx - u_f, yy - v_f)
                    halo += (
                        np.exp(-(rf / 2.4) ** 2)[:, :, None] * np.array([1.0, 0.98, 0.88], np.float32) * (0.95 * glow)
                        + np.exp(-(rf / 10.5) ** 2)[:, :, None] * firefly_col[None, None, :] * (0.42 * glow)
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
