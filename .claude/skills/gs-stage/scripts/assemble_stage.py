"""Assemble a 3DGS stage from a Volumetric 3D Frame JSON + Theme JSON + standalone v1_prop/ directory.

Outputs:
  - map_<theme>.ply                : Unified 3DGS stage (Blender Z-up, meters)
  - nav_surface.json               : Multi-layer 3D walkable spans for 3DGS walkthroughs & 2D/HD-2D games
  - nav_multilayer_3d_preview.png  : Layer 0 / Layer 1 separation & 3D elevation profile
  - walk_3d_trajectory.json        : Planned 3D A* trajectory (when --waypoints is supplied)
  - showcase_<theme>_4view.png     : Spark 4-angle QA showcase (when --spark-dir is supplied)
  - decoupled_outputs_triptych.png : Albedo / Depth / Surface Normal maps (when --export-maps is set)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from ply_io import read_ply, write_ply
from volumetric_frame_3d import VolumetricFrame3D

SH_C0 = 0.28209479177387814


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    out = np.empty_like(q2)
    out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return out


def blender_to_spark_data(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    x, y, z = data["x"].copy(), data["y"].copy(), data["z"].copy()
    out["x"], out["y"], out["z"] = x, -z, y
    s = math.sqrt(0.5)
    w0, x0, y0, z0 = data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]
    out["rot_0"] = s * (w0 + x0)
    out["rot_1"] = s * (x0 - w0)
    out["rot_2"] = s * (y0 + z0)
    out["rot_3"] = s * (z0 - y0)
    return out


def make_splats(xyz, rgb, scales, opacity_prob, dtype) -> np.ndarray:
    a = np.zeros(len(xyz), dtype=dtype)
    a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    fdc = (np.clip(rgb, 0.0, 1.0) - 0.5) / SH_C0
    a["f_dc_0"], a["f_dc_1"], a["f_dc_2"] = fdc[:, 0], fdc[:, 1], fdc[:, 2]
    op_c = np.clip(opacity_prob, 1e-4, 1.0 - 1e-4)
    a["opacity"] = np.log(op_c / (1.0 - op_c)).astype(np.float32)
    a["scale_0"] = np.log(np.maximum(scales[:, 0], 1e-4)).astype(np.float32)
    a["scale_1"] = np.log(np.maximum(scales[:, 1], 1e-4)).astype(np.float32)
    a["scale_2"] = np.log(np.maximum(scales[:, 2], 1e-4)).astype(np.float32)
    a["rot_0"] = 1.0
    return a


def shade_splats(xyz: np.ndarray, rgb: np.ndarray, frame: dict, theme: dict, normals: np.ndarray | None = None) -> np.ndarray:
    lcfg = theme["lighting"]
    amb = np.array(lcfg["ambient"], dtype=np.float32)
    sdir = np.array(lcfg["sun_dir"], dtype=np.float32)
    sdir /= np.linalg.norm(sdir)
    scol = np.array(lcfg["sun_color"], dtype=np.float32)
    nd = np.full(len(xyz), 0.72, dtype=np.float32) if normals is None else np.clip(np.sum(normals * sdir[None, :], axis=-1), 0.0, 1.0)
    acc = amb[None, :] + scol[None, :] * (0.35 + 0.65 * nd)[:, None]
    p_str = float(lcfg["point_light_strength"])
    for lt in frame.get("lights", []):
        lp = np.array([lt["x"], lt["y"], lt.get("z", 1.0)], dtype=np.float32)
        rad = float(lt["radius_m"])
        d2 = np.sum((xyz - lp[None, :]) ** 2, axis=-1)
        fall = np.exp(-d2 / (rad * rad * 0.5)) / (0.45 + d2 * 0.6)
        lc = np.array(lcfg["light_colors"].get(lt["kind"], [0.9, 0.9, 0.85]), dtype=np.float32)
        acc += lc[None, :] * (fall * p_str)[:, None]
    out = rgb * acc
    return np.clip(out / (1.0 + 0.22 * np.maximum(out - 0.80, 0.0)), 0.0, 1.0)


def synth_stone_box(cx, cy, z0, z1, hx, hy, step, rng, surf_top, surf_wall, moss_rgb, frame, theme, dtype, arch_camber=0.0):
    pts, cols, scs, nrms = [], [], [], []
    c_top0, c_top1 = np.array(surf_top["base_rgb"], np.float32), np.array(surf_top["alt_rgb"], np.float32)
    c_w0, c_w1 = np.array(surf_wall["base_rgb"], np.float32), np.array(surf_wall["alt_rgb"], np.float32)
    c_m0, c_m1 = np.array(moss_rgb["base_rgb"], np.float32), np.array(moss_rgb["alt_rgb"], np.float32)
    c_mort = np.array(theme["surfaces"].get("mortar", {}).get("base_rgb", [0.16, 0.19, 0.17]), np.float32)

    xs = np.arange(-hx, hx + 1e-4, step * 0.85, dtype=np.float32)
    ys = np.arange(-hy, hy + 1e-4, step * 0.85, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    cam = arch_camber * (1.0 - (gx / max(hx, 1e-4)) ** 2)

    row_i = np.floor((gy + cy) / 0.42)
    col_u = ((gx + cx) + (row_i % 2) * 0.24) / 0.48
    row_v = (gy + cy) / 0.42
    fu, fv = col_u - np.floor(col_u), row_v - np.floor(row_v)
    is_joint = (fu < 0.09) | (fu > 0.91) | (fv < 0.09) | (fv > 0.91)
    blk_h = ((np.floor(col_u) * 17.3 + row_i * 31.7) % 1.0).astype(np.float32)

    gz = np.full_like(gx, z1) + cam + np.where(is_joint, -0.022, (blk_h - 0.5) * 0.024)
    edge_d = np.maximum(np.abs(gx) / max(hx, 1e-4), np.abs(gy) / max(hy, 1e-4))
    patch_n = 0.5 + 0.5 * np.sin((gx + cx) * 3.1) * np.cos((gy + cy) * 2.9 + 0.8)
    m_mix = np.clip((edge_d - 0.74) * 3.2 * patch_n, 0.0, 0.85)[:, None]

    col_stone = c_top0[None, :] * (1.0 - blk_h[:, None]) + c_top1[None, :] * blk_h[:, None]
    col_t = np.where(is_joint[:, None], c_mort[None, :] * 0.7 + c_m0[None, :] * 0.3, col_stone)
    col_t = col_t * (1.0 - m_mix) + (c_m0[None, :] * 0.65 + c_m1[None, :] * 0.35) * m_mix
    pts.append(np.column_stack([gx + cx, gy + cy, gz]))
    cols.append(col_t)
    scs.append(np.tile(np.array([step * 0.62, step * 0.62, 0.028], np.float32), (len(gx), 1)))
    nrms.append(np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(gx), 1)))

    zs_v = np.arange(z0, z1 + 1e-4, step * 0.78, dtype=np.float32)
    for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
        us = ys if axis == 0 else xs
        uu, zz = np.meshgrid(us, zs_v)
        uu, zz = uu.ravel(), zz.ravel()
        course = np.floor((zz - z0) / 0.34)
        u_blk = (uu + (course % 2) * 0.28) / 0.56
        v_blk = (zz - z0) / 0.34
        fu_w, fv_w = u_blk - np.floor(u_blk), v_blk - np.floor(v_blk)
        joint_w = (fu_w < 0.08) | (fu_w > 0.92) | (fv_w < 0.10) | (fv_w > 0.90)
        bh_w = ((np.floor(u_blk) * 19.1 + course * 37.3) % 1.0).astype(np.float32)
        relief = np.where(joint_w, -0.028, (bh_w - 0.45) * 0.038).astype(np.float32)

        if axis == 0:
            wx = np.full_like(uu, cx + sign * hx) + sign * relief
            wy = cy + uu
            nw = np.array([float(sign), 0.0, 0.0], np.float32)
            cam_w = arch_camber * (1.0 - (sign * hx / max(hx, 1e-4)) ** 2)
        else:
            wx = cx + uu
            wy = np.full_like(uu, cy + sign * hy) + sign * relief
            nw = np.array([0.0, float(sign), 0.0], np.float32)
            cam_w = arch_camber * (1.0 - (uu / max(hx, 1e-4)) ** 2)
        wz = zz + cam_w
        u_z = np.clip((zz - z0) / max(z1 - z0, 1e-4), 0.0, 1.0)
        mp_w = 0.5 + 0.5 * np.sin(uu * 2.8 + course * 1.4) * np.cos(zz * 3.2)
        m_w = np.clip(((u_z > 0.84).astype(np.float32) * 0.65 + (u_z < 0.14).astype(np.float32) * 0.45) * mp_w, 0.0, 0.78)[:, None]
        col_sw = c_w0[None, :] * (1.0 - bh_w[:, None]) + c_w1[None, :] * bh_w[:, None]
        col_w = np.where(joint_w[:, None], c_mort[None, :], col_sw)
        col_w = col_w * (1.0 - m_w) + c_m0[None, :] * m_w
        pts.append(np.column_stack([wx, wy, wz]))
        cols.append(col_w)
        scs.append(np.tile(np.array([step * 0.55, step * 0.55, step * 0.55], np.float32), (len(uu), 1)))
        nrms.append(np.tile(nw, (len(uu), 1)))

    if z0 > 0.2:
        pts.append(np.column_stack([gx + cx, gy + cy, np.full_like(gx, z0) + cam]))
        cols.append(np.tile(c_w1 * 0.72, (len(gx), 1)))
        scs.append(np.tile(np.array([step * 0.68, step * 0.68, 0.036], np.float32), (len(gx), 1)))
        nrms.append(np.tile(np.array([0.0, 0.0, -1.0], np.float32), (len(gx), 1)))

    xyz_all = np.vstack(pts).astype(np.float32)
    rgb_all = shade_splats(xyz_all, np.vstack(cols).astype(np.float32), frame, theme, normals=np.vstack(nrms).astype(np.float32))
    return make_splats(xyz_all, rgb_all, np.vstack(scs).astype(np.float32), np.full(len(xyz_all), 0.97, np.float32), dtype)


def synth_stone_stairs(cx, cy, hx, hy, z0, z1, n_steps, rng, surf_st, surf_wall, moss_rgb, frame, theme, dtype):
    chunks = []
    step_dx = (2.0 * hx) / n_steps
    for i in range(n_steps):
        sx_mid = cx - hx + (i + 0.5) * step_dx
        sz_top = z0 + (z1 - z0) * ((i + 1) / n_steps)
        chunks.append(synth_stone_box(sx_mid, cy, 0.0, sz_top, step_dx * 0.52, hy, 0.085, rng, surf_st, surf_wall, moss_rgb, frame, theme, dtype))
    return np.concatenate(chunks, axis=0)


def synth_stone_pillar(cx, cy, r, z0, z1, rng, surf_wall, moss_rgb, frame, theme, dtype):
    c_w0, c_w1 = np.array(surf_wall["base_rgb"], np.float32), np.array(surf_wall["alt_rgb"], np.float32)
    c_m = np.array(moss_rgb["base_rgb"], np.float32)
    ang = np.linspace(0, 2 * math.pi, 22, endpoint=False, dtype=np.float32)
    zz = np.linspace(z0, z1, 26, dtype=np.float32)
    aa, zg = np.meshgrid(ang, zz)
    aa, zg = aa.ravel(), zg.ravel()
    u_z = (zg - z0) / max(z1 - z0, 1e-4)
    rad = r * (1.0 + 0.18 * np.exp(-(u_z / 0.12) ** 2) + 0.18 * np.exp(-((1.0 - u_z) / 0.12) ** 2))
    xyz = np.column_stack([cx + rad * np.cos(aa), cy + rad * np.sin(aa), zg]).astype(np.float32)
    nrm = np.column_stack([np.cos(aa), np.sin(aa), np.zeros_like(aa)]).astype(np.float32)
    m_mix = np.clip(np.exp(-(u_z / 0.25) ** 2) * 0.7 + rng.uniform(-0.1, 0.15, len(aa)), 0.0, 1.0)[:, None]
    col = (c_w0[None, :] * 0.65 + c_w1[None, :] * 0.35) * (1.0 - m_mix) + c_m[None, :] * m_mix
    rgb = shade_splats(xyz, col.astype(np.float32), frame, theme, normals=nrm)
    return make_splats(xyz, rgb, np.full((len(xyz), 3), 0.068, np.float32), np.full(len(xyz), 0.96, np.float32), dtype)


def assemble_map(frame: dict, theme: dict, vframe: VolumetricFrame3D, props_root: Path | None, out_ply: Path) -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(202)
    canonical_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])

    b = frame["bounds"]
    surfs = theme["surfaces"]
    step = 0.088
    xs = np.arange(b["x_min"], b["x_max"] + 1e-4, step, dtype=np.float32)
    ys = np.arange(b["y_min"], b["y_max"] + 1e-4, step, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    gz = vframe._eval_ground_z(gx, gy)
    gt = vframe._eval_ground_type(gx, gy)
    gx_f, gy_f, gz_f = gx.ravel(), gy.ravel(), gz.ravel()
    gt_f = [gt[iy][ix] for iy in range(gx.shape[0]) for ix in range(gx.shape[1])]

    noise = np.clip(0.5 + 0.5 * np.sin(gx_f * 2.3 + gy_f * 1.7) + rng.normal(0.0, 0.12, len(gx_f)), 0.0, 1.0).astype(np.float32)
    g_rgb = np.zeros((len(gx_f), 3), dtype=np.float32)
    c_mort = np.array(surfs.get("mortar", {}).get("base_rgb", [0.16, 0.19, 0.17]), np.float32)
    for k in ("forest_floor", "stone_path"):
        m = np.array([t == k for t in gt_f])
        if not m.any():
            continue
        c0, c1 = np.array(surfs[k]["base_rgb"], np.float32), np.array(surfs[k]["alt_rgb"], np.float32)
        if k == "stone_path":
            ri = np.floor(gy_f[m] / 0.44)
            cu = (gx_f[m] + (ri % 2) * 0.22) / 0.48
            rv = gy_f[m] / 0.44
            j_p = ((cu - np.floor(cu)) < 0.10) | ((cu - np.floor(cu)) > 0.90) | ((rv - np.floor(rv)) < 0.10) | ((rv - np.floor(rv)) > 0.90)
            bh = ((np.floor(cu) * 13.7 + ri * 29.1) % 1.0).astype(np.float32)
            col_p = c0[None, :] * (1.0 - bh[:, None]) + c1[None, :] * bh[:, None]
            g_rgb[m] = np.where(j_p[:, None], c_mort[None, :] * 0.65 + np.array(surfs["moss"]["base_rgb"], np.float32)[None, :] * 0.35, col_p)
        else:
            g_rgb[m] = c0[None, :] * (1.0 - noise[m, None]) + c1[None, :] * noise[m, None]
    g_xyz = np.column_stack([gx_f, gy_f, gz_f]).astype(np.float32)
    chunks = [make_splats(g_xyz, shade_splats(g_xyz, g_rgb, frame, theme), np.tile(np.array([0.072, 0.072, 0.026], np.float32), (len(gx_f), 1)), np.full(len(gx_f), 0.97, np.float32), canonical_dtype)]

    for vol in frame.get("volumes", []):
        vk = vol["kind"]
        if vk == "platform":
            chunks.append(synth_stone_box(vol["cx"], vol["cy"], 0.0, vol["z_top"], vol["hx"], vol["hy"], 0.095, rng, surfs["ruin_stone"], surfs["ruin_wall"], surfs["moss"], frame, theme, canonical_dtype))
        elif vk == "overpass":
            z0_ov = vol["z_top"] - vol["thickness"]
            chunks.append(synth_stone_box(vol["cx"], vol["cy"], z0_ov, vol["z_top"], vol["hx"], vol["hy"], 0.085, rng, surfs["ruin_stone"], surfs["ruin_wall"], surfs["moss"], frame, theme, canonical_dtype, arch_camber=vol.get("arch_camber", 0.0)))
            for side in (-1.0, 1.0):
                chunks.append(synth_stone_box(vol["cx"], vol["cy"] + side * (vol["hy"] - 0.10), vol["z_top"], vol["z_top"] + 0.34, vol["hx"] * 0.96, 0.09, 0.075, rng, surfs["ruin_stone"], surfs["ruin_wall"], surfs["moss"], frame, theme, canonical_dtype, arch_camber=vol.get("arch_camber", 0.0)))
        elif vk == "ramp":
            chunks.append(synth_stone_stairs(vol["cx"], vol["cy"], vol["hx"], vol["hy"], vol["z_start"], vol["z_end"], 10, rng, surfs["stairs"], surfs["ruin_wall"], surfs["moss"], frame, theme, canonical_dtype))
        elif vk == "pillar":
            chunks.append(synth_stone_pillar(vol["cx"], vol["cy"], vol["radius"], vol["z_min"], vol["z_max"], rng, surfs["ruin_wall"], surfs["moss"], frame, theme, canonical_dtype))

    if props_root is not None:
        grade = theme["prop_grade"]
        g_tint = np.array(grade["tint"], np.float32)
        g_sat, g_bri = float(grade["saturation"]), float(grade["brightness"])
        for p in frame.get("props", []):
            ply_p = props_root / p["asset"] / "v1_prop" / "prop.ply"
            if not ply_p.exists():
                continue
            d_raw, _ = read_ply(ply_p)
            max_pts = int(p.get("max_splats", 22000))
            if len(d_raw) > max_pts:
                op_r = 1.0 / (1.0 + np.exp(-np.clip(d_raw["opacity"], -20.0, 20.0)))
                order = np.argsort(op_r)[::-1]
                sel = rng.choice(order[: min(len(d_raw), int(max_pts * 2.0))], size=max_pts, replace=False)
                d_sub = d_raw[sel]
                s_boost = math.log((len(d_raw) / max_pts) ** 0.22)
            else:
                d_sub = d_raw
                s_boost = 0.0

            d = np.zeros(len(d_sub), dtype=canonical_dtype)
            for k in canonical_dtype.names:
                d[k] = d_sub[k]

            sc_p = float(p.get("scale", 1.0))
            yaw_rad = math.radians(float(p.get("yaw_deg", 0.0)))
            R_z = Rotation.from_euler("z", yaw_rad).as_matrix().astype(np.float32)
            q_xyzw = Rotation.from_euler("z", yaw_rad).as_quat()
            q_z = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

            px, py = float(p["x"]), float(p["y"])
            pz = vframe.query_layer_z(px, py, p.get("layer", "ground")) + float(p.get("z_offset", 0.0))
            xyz_p = (np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32) * sc_p) @ R_z.T + np.array([px, py, pz], np.float32)[None, :]
            rot_p = quat_mul_wxyz(q_z, np.column_stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]]).astype(np.float32))
            rot_p /= np.maximum(np.linalg.norm(rot_p, axis=1, keepdims=True), 1e-8)

            rgb_p = np.clip(np.column_stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]]).astype(np.float32) * SH_C0 + 0.5, 0.0, 1.0)
            lum = (0.299 * rgb_p[:, 0] + 0.587 * rgb_p[:, 1] + 0.114 * rgb_p[:, 2])[:, None]
            rgb_p = np.clip((lum + (rgb_p - lum) * g_sat) * g_tint[None, :] * g_bri, 0.0, 1.0)
            rgb_lit = shade_splats(xyz_p, rgb_p, frame, theme)

            d["x"], d["y"], d["z"] = xyz_p[:, 0], xyz_p[:, 1], xyz_p[:, 2]
            d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"] = rot_p[:, 0], rot_p[:, 1], rot_p[:, 2], rot_p[:, 3]
            for si in range(3):
                d[f"scale_{si}"] = d[f"scale_{si}"] + math.log(sc_p) + s_boost
            fdc_p = (rgb_lit - 0.5) / SH_C0
            d["f_dc_0"], d["f_dc_1"], d["f_dc_2"] = fdc_p[:, 0], fdc_p[:, 1], fdc_p[:, 2]
            chunks.append(d)

    atm = theme.get("atmosphere", {})
    n_m = int(atm.get("mist_count", 0))
    if n_m > 0:
        mx = rng.uniform(b["x_min"], b["x_max"], n_m).astype(np.float32)
        my = rng.uniform(b["y_min"], b["y_max"], n_m).astype(np.float32)
        mz = rng.uniform(0.25, 1.35, n_m).astype(np.float32)
        chunks.append(make_splats(
            np.column_stack([mx, my, mz]), np.tile(np.array(atm["mist_rgb"], np.float32), (n_m, 1)),
            np.tile(np.array([atm["mist_radius"], atm["mist_radius"], atm["mist_radius"] * 0.4], np.float32), (n_m, 1)),
            np.full(n_m, atm["mist_opacity"], np.float32), canonical_dtype,
        ))

    merged = np.concatenate(chunks, axis=0)
    write_ply(out_ply, merged)
    print(f"Assembled {theme['name']} ({len(merged):,} splats) in {time.time()-t0:.1f}s -> {out_ply}", flush=True)
    return merged


def parse_waypoints(raw: str) -> list[tuple[float, float, float]]:
    pts = []
    for tok in raw.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        x, y, z = [float(v.strip()) for v in tok.split(",")]
        pts.append((x, y, z))
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble a 3DGS stage from Frame JSON + Theme JSON + v1_prop/ assets.")
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--theme", type=Path, required=True)
    ap.add_argument("--props-root", type=Path, default=None, help="Directory containing <asset>/v1_prop/prop.ply.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--grid-res", type=float, default=0.15)
    ap.add_argument("--waypoints", default=None, help='Semicolon-separated 3D waypoints "x,y,z; x,y,z; ..." for 3D A* walk planning.')
    ap.add_argument("--num-frames", type=int, default=64)
    ap.add_argument("--spark-dir", type=Path, default=None, help="Optional spark_demo directory for 4-view showcase & decoupled maps.")
    ap.add_argument("--export-maps", action="store_true", help="Also export decoupled Albedo/Depth/Surface Normal triptych.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = json.loads(args.frame.read_text(encoding="utf-8"))
    theme = json.loads(args.theme.read_text(encoding="utf-8"))

    vf = VolumetricFrame3D(frame, props_root=args.props_root, grid_res=args.grid_res)
    vf.export_nav_json(args.out_dir / "nav_surface.json")

    traj = None
    if args.waypoints:
        wps = parse_waypoints(args.waypoints)
        traj = vf.plan_3d_trajectory(wps, n_frames=args.num_frames)
        (args.out_dir / "walk_3d_trajectory.json").write_text(json.dumps(traj, indent=2), encoding="utf-8")
        print(f"Planned {len(traj)}-frame 3D walk (len={traj[-1]['arc_s']:.2f}m)", flush=True)

    vf.save_multilayer_preview(traj, args.out_dir / "nav_multilayer_3d_preview.png")
    out_ply = args.out_dir / f"map_{theme['name']}.ply"
    map_d = assemble_map(frame, theme, vf, args.props_root, out_ply)

    if args.spark_dir:
        tag = f"stage_{theme['name']}"
        tmp_dir = args.spark_dir / "frames" / tag
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        map_rel = f"frames/{tag}/map.ply"
        ds = blender_to_spark_data(map_d)
        write_ply(args.spark_dir / map_rel, ds)

        views = [
            ("v0_sw", -26.0, 22.0, 12.6, [0.0, -1.05, 0.0]),
            ("v1_se",  28.0, 22.0, 12.6, [0.0, -1.05, 0.0]),
            ("v2_low",  4.0, 11.0,  9.2, [0.0, -1.10, -0.3]),
            ("v3_top",  0.0, 54.0, 13.0, [0.0, -0.80, 0.0]),
        ]
        v_imgs = []
        for vn, yaw, pitch, dist, tgt in views:
            cam = {"width": 640, "height": 420, "background": theme["background_hex"], "target": tgt, "distance": dist, "pitch": pitch, "fov": 35.0, "front": yaw, "side": yaw + 90.0}
            crel = f"frames/{tag}/cam_{vn}.json"
            (args.spark_dir / crel).write_text(json.dumps(cam), encoding="utf-8")
            mrel = f"frames/{tag}/man_{vn}.json"
            (args.spark_dir / mrel).write_text(json.dumps({"static": [map_rel], "review": [], "frames": [{"blenderFrame": 0, "ply": [], "png": f"{vn}.png"}]}), encoding="utf-8")
            subprocess.run(["node", "capture.mjs", f"{tag}_{vn}", f"{tag}_out", crel, mrel], cwd=str(args.spark_dir), check=True, capture_output=True)
            v_imgs.append(Image.open(args.spark_dir / f"{tag}_out" / f"{vn}.png").convert("RGB"))
        sh = Image.new("RGB", (1280, 840))
        for i, im in enumerate(v_imgs):
            sh.paste(im, ((i % 2) * 640, (i // 2) * 420))
        sh.save(args.out_dir / f"showcase_{theme['name']}_4view.png")
        print(f"Saved 4-view showcase -> {args.out_dir / f'showcase_{theme[\"name\"]}_4view.png'}", flush=True)

        if args.export_maps:
            op_d = 1.0 / (1.0 + np.exp(-np.clip(map_d["opacity"], -20.0, 20.0)))
            sol = op_d > 0.25
            d_sol, ds_sol = map_d[sol], ds[sol]
            xyz_b = np.column_stack([d_sol["x"], d_sol["y"], d_sol["z"]]).astype(np.float32)
            lo, hi = xyz_b.min(axis=0) - 0.1, xyz_b.max(axis=0) + 0.1
            span = np.maximum(hi - lo, 1e-4)
            pos_s = ds_sol.copy()
            fdc_p = (np.clip((xyz_b - lo[None, :]) / span[None, :], 0.0, 1.0) * 0.90 + 0.05 - 0.5) / SH_C0
            pos_s["f_dc_0"], pos_s["f_dc_1"], pos_s["f_dc_2"] = fdc_p[:, 0], fdc_p[:, 1], fdc_p[:, 2]
            pos_rel = f"frames/{tag}/map_pos.ply"
            write_ply(args.spark_dir / pos_rel, pos_s)
            cam_p = {"width": 640, "height": 420, "background": "#000000", "target": [0.0, -1.05, 0.0], "distance": 12.6, "pitch": 22.0, "fov": 35.0, "front": -26.0, "side": 64.0}
            (args.spark_dir / f"frames/{tag}/cam_pos.json").write_text(json.dumps(cam_p), encoding="utf-8")
            (args.spark_dir / f"frames/{tag}/man_pos.json").write_text(json.dumps({"static": [pos_rel], "review": [], "frames": [{"blenderFrame": 0, "ply": [], "png": "pos.png"}]}), encoding="utf-8")
            subprocess.run(["node", "capture.mjs", f"{tag}_pos", f"{tag}_out", f"frames/{tag}/cam_pos.json", f"frames/{tag}/man_pos.json"], cwd=str(args.spark_dir), check=True, capture_output=True)

            pos_im = np.asarray(Image.open(args.spark_dir / f"{tag}_out" / "pos.png").convert("RGB"), dtype=np.float32) / 255.0
            fg = pos_im.max(axis=-1) > 0.03
            fg_f = fg.astype(np.float32)
            xyz_px = lo[None, None, :] + np.clip((pos_im - 0.05) / 0.90, 0.0, 1.0) * span[None, None, :]
            wb = cv2.GaussianBlur(fg_f, (0, 0), 2.2)
            xyz_s = np.zeros_like(xyz_px)
            for ch in range(3):
                xyz_s[:, :, ch] = np.where(fg, cv2.GaussianBlur(xyz_px[:, :, ch] * fg_f, (0, 0), 2.2) / np.maximum(wb, 1e-4), 0.0)
            d_dy, d_dx = np.gradient(xyz_s, axis=(0, 1))
            nrm = np.cross(d_dx, -d_dy)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
            nrm[nrm[:, :, 2] < 0] *= -1.0
            nrm_vis = np.where(fg[:, :, None], np.clip(nrm * 0.5 + 0.5, 0.0, 1.0), 0.0)
            az, el = math.radians(-26.0), math.radians(22.0)
            cam_b = np.array([12.6 * math.sin(az) * math.cos(el), -12.6 * math.cos(az) * math.cos(el), 12.6 * math.sin(el) + 1.05], dtype=np.float32)
            dist_m = np.linalg.norm(xyz_s - cam_b[None, None, :], axis=-1)
            d_lo, d_hi = np.percentile(dist_m[fg], 2), np.percentile(dist_m[fg], 98)
            dep_n = np.where(fg, 1.0 - np.clip((dist_m - d_lo) / max(d_hi - d_lo, 1e-4), 0.0, 1.0), 0.0)
            dep_vis = np.stack([dep_n, dep_n, dep_n], axis=-1)
            trip = Image.new("RGB", (640 * 3, 420), (10, 12, 18))
            trip.paste(v_imgs[0], (0, 0))
            trip.paste(Image.fromarray((dep_vis * 255).astype(np.uint8)), (640, 0))
            trip.paste(Image.fromarray((nrm_vis * 255).astype(np.uint8)), (1280, 0))
            trip.save(args.out_dir / "decoupled_outputs_triptych.png")
            print(f"Saved decoupled Albedo/Depth/Normal triptych -> {args.out_dir / 'decoupled_outputs_triptych.png'}", flush=True)


if __name__ == "__main__":
    main()
