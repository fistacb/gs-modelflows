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


def _hash2(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    n = np.sin(ix * 127.1 + iy * 311.7) * 43758.5453123
    return (n - np.floor(n)).astype(np.float32)


def _value_noise_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy)
    return (_hash2(ix, iy) * (1 - ux) + _hash2(ix + 1, iy) * ux) * (1 - uy) + (
        _hash2(ix, iy + 1) * (1 - ux) + _hash2(ix + 1, iy + 1) * ux
    ) * uy


def _fbm2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        0.55 * _value_noise_2d(x, y)
        + 0.30 * _value_noise_2d(x * 2.13 + 7.1, y * 2.13 - 3.4)
        + 0.15 * _value_noise_2d(x * 4.37 - 1.9, y * 4.37 + 5.2)
    ).astype(np.float32)


def _voronoi_stone_field(u: np.ndarray, v: np.ndarray, scale_u: float, scale_v: float, jitter: float):
    su, sv = u * scale_u, v * scale_v
    iu, iv = np.floor(su), np.floor(sv)
    d1 = np.full_like(su, 1e9, dtype=np.float32)
    d2 = np.full_like(su, 1e9, dtype=np.float32)
    cid = np.zeros_like(su, dtype=np.float32)
    best_dv = np.zeros_like(su, dtype=np.float32)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            nu, nv = iu + du, iv + dv
            stagger = (nv % 2) * 0.42
            rx = nu + 0.5 + (_hash2(nu, nv) - 0.5) * jitter * 2.0 + (stagger - 0.21) * 0.35
            ry = nv + 0.5 + (_hash2(nu + 19.3, nv + 7.7) - 0.5) * jitter * 1.4
            dist = np.hypot(su - rx, sv - ry)
            closer = dist < d1
            d2 = np.where(closer, d1, np.minimum(d2, dist))
            d1 = np.where(closer, dist, d1)
            cid = np.where(closer, _hash2(nu * 3.1, nv * 5.7), cid)
            best_dv = np.where(closer, sv - ry, best_dv)
    return np.clip(d2 - d1, 0.0, 1.0), cid, best_dv


def synth_stone_box(cx, cy, z0, z1, hx, hy, step, rng, surf_top, surf_wall, moss_rgb, frame, theme, dtype, arch_camber=0.0):
    """Synthesize theme-controlled stonework: supports irregular_ruin (Voronoi), crisp_brick, and ashlar_cut
    with strong floor-vs-wall value separation, stone coping curb, and base/lip contact AO.
    """
    pts, cols, scs, nrms = [], [], [], []
    mcfg = theme.get("masonry", {})
    style = mcfg.get("style", "irregular_ruin")
    jitter = float(mcfg.get("stone_jitter", 0.0 if style == "crisp_brick" else (0.10 if style == "ashlar_cut" else 0.32)))
    moss_amt = float(mcfg.get("moss_amount", 0.08 if style == "crisp_brick" else 0.55))
    floor_gain = float(mcfg.get("floor_wall_contrast", 1.20))

    c_top0, c_top1 = np.array(surf_top["base_rgb"], np.float32), np.array(surf_top["alt_rgb"], np.float32)
    c_w0, c_w1 = np.array(surf_wall["base_rgb"], np.float32), np.array(surf_wall["alt_rgb"], np.float32)
    c_m0, c_m1 = np.array(moss_rgb["base_rgb"], np.float32), np.array(moss_rgb["alt_rgb"], np.float32)
    c_mort = np.array(theme["surfaces"].get("mortar", {}).get("base_rgb", [0.22, 0.19, 0.17]), np.float32)
    c_cream = np.array(theme.get("palette", {}).get("sunbeam_cream", [0.96, 0.90, 0.76]), np.float32)

    # 1. Walkable Top Deck (bright, warm sunlit flagstones + crisp raised coping trim)
    s_top = step * 0.76
    xs = np.arange(-hx, hx + 1e-4, s_top, dtype=np.float32)
    ys = np.arange(-hy, hy + 1e-4, s_top, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    if jitter > 0.05:
        gx = gx + rng.uniform(-s_top * 0.22, s_top * 0.22, len(gx)).astype(np.float32)
        gy = gy + rng.uniform(-s_top * 0.22, s_top * 0.22, len(gy)).astype(np.float32)
    wx, wy = gx + cx, gy + cy
    cam = arch_camber * (1.0 - (gx / max(hx, 1e-4)) ** 2)
    grain = _fbm2(wx * 5.0, wy * 5.0)

    if jitter > 0.05:
        ed, cid, loc_v = _voronoi_stone_field(wx, wy, 1.85, 1.85, jitter)
        is_joint = ed < 0.085
        bevel = np.clip((ed - 0.085) / 0.14, 0.0, 1.0)
    else:
        ri = np.floor(wy / 0.42)
        cu = (wx + (ri % 2) * 0.24) / 0.48
        rv = wy / 0.42
        fu, fv = cu - np.floor(cu), rv - np.floor(rv)
        is_joint = (fu < 0.08) | (fu > 0.92) | (fv < 0.08) | (fv > 0.92)
        cid = _hash2(np.floor(cu), ri)
        bevel = np.where(is_joint, 0.0, 1.0).astype(np.float32)
        loc_v = np.zeros_like(cid)

    edge_rim = np.maximum(np.abs(gx) / max(hx, 1e-4), np.abs(gy) / max(hy, 1e-4))
    is_coping = edge_rim > 0.86
    col_stone = np.clip((c_top0[None, :] * (1.0 - cid[:, None]) + c_top1[None, :] * cid[:, None]) * floor_gain * (0.90 + 0.20 * grain[:, None]), 0.0, 1.0)
    bevel_hl = np.clip((1.0 - bevel) * (~is_joint) * (0.5 - loc_v), 0.0, 0.55)[:, None]
    col_stone = col_stone * (0.78 + 0.22 * bevel[:, None]) + c_cream[None, :] * (bevel_hl * 0.34)
    col_stone = np.where(is_coping[:, None], np.clip(col_stone * 1.12 + c_cream[None, :] * 0.14, 0.0, 1.0), col_stone)

    moss_n = _fbm2(wx * 2.0 + 3.1, wy * 2.0 - 2.4)
    crack_m = np.clip(moss_n * 1.1 - 0.35, 0.0, 0.75)[:, None] * moss_amt
    crack_col = c_mort[None, :] * (1.0 - crack_m * 0.55) + c_m0[None, :] * (crack_m * 0.55)
    rim_m = np.clip((edge_rim - 0.80) * 2.2 * np.maximum(moss_n - 0.45, 0.0), 0.0, 0.45)[:, None] * moss_amt
    col_t = np.where(is_joint[:, None], crack_col, col_stone)
    col_t = col_t * (1.0 - rim_m) + (c_m0[None, :] * 0.6 + c_m1[None, :] * 0.4) * rim_m

    gz = np.full_like(gx, z1) + cam + np.where(is_joint, -0.026, (cid - 0.5) * 0.022 + bevel * 0.014 + np.where(is_coping, 0.020, 0.0))
    pts.append(np.column_stack([wx, wy, gz]))
    cols.append(col_t)
    scs.append(np.tile(np.array([s_top * 0.82, s_top * 0.82, 0.028], np.float32), (len(gx), 1)))
    nrms.append(np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(gx), 1)))

    # 2. 4 Vertical Walls (1 value step darker & cooler + contact AO at base & coping overhang!)
    s_w = step * 0.72
    zs_v = np.arange(z0, z1 + 1e-4, s_w, dtype=np.float32)
    for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1)):
        us = np.arange(-(hy if axis == 0 else hx), (hy if axis == 0 else hx) + 1e-4, s_w, dtype=np.float32)
        uu, zz = np.meshgrid(us, zs_v)
        uu, zz = uu.ravel(), zz.ravel()
        if jitter > 0.05:
            uu = uu + rng.uniform(-s_w * 0.20, s_w * 0.20, len(uu)).astype(np.float32)
            zz = zz + rng.uniform(-s_w * 0.20, s_w * 0.20, len(zz)).astype(np.float32)
            ed_w, bh_w, dv_w = _voronoi_stone_field(uu + (cx if axis == 1 else cy), zz - z0, 1.65, 2.75, jitter)
            joint_w = ed_w < 0.10
            bev_w = np.clip((ed_w - 0.10) / 0.14, 0.0, 1.0)
        else:
            course = np.floor((zz - z0) / 0.34)
            ub = (uu + (course % 2) * 0.28) / 0.56
            vb = (zz - z0) / 0.34
            fu_w, fv_w = ub - np.floor(ub), vb - np.floor(vb)
            joint_w = (fu_w < 0.08) | (fu_w > 0.92) | (fv_w < 0.10) | (fv_w > 0.90)
            bh_w = _hash2(np.floor(ub), course)
            bev_w = np.where(joint_w, 0.0, 1.0).astype(np.float32)
            dv_w = (fv_w - 0.5).astype(np.float32)

        gr_w = _fbm2(uu * 4.8 + axis * 11.0, zz * 4.8)
        relief = np.where(joint_w, -0.032, (bh_w - 0.45) * 0.040 + bev_w * 0.018).astype(np.float32)
        if axis == 0:
            wx_w = np.full_like(uu, cx + sign * hx) + sign * relief
            wy_w = cy + uu
            nw = np.array([float(sign), 0.0, 0.0], np.float32)
            cam_w = arch_camber * (1.0 - (sign * hx / max(hx, 1e-4)) ** 2)
        else:
            wx_w = cx + uu
            wy_w = np.full_like(uu, cy + sign * hy) + sign * relief
            nw = np.array([0.0, float(sign), 0.0], np.float32)
            cam_w = arch_camber * (1.0 - (uu / max(hx, 1e-4)) ** 2)
        wz_w = zz + cam_w

        u_z = np.clip((zz - z0) / max(z1 - z0, 1e-4), 0.0, 1.0)
        ao_bot = (0.50 + 0.50 * np.clip(u_z / 0.28, 0.0, 1.0))[:, None]
        ao_lip = (1.0 - 0.24 * np.clip((u_z - 0.88) / 0.12, 0.0, 1.0))[:, None]
        top_hl = np.clip(dv_w * 1.1, 0.0, 0.32)[:, None] * (~joint_w)[:, None]
        col_sw = (c_w0[None, :] * (1.0 - bh_w[:, None]) + c_w1[None, :] * bh_w[:, None]) * np.array([0.74, 0.78, 0.82], np.float32)
        col_sw = col_sw * (0.86 + 0.26 * gr_w[:, None]) * ao_bot * ao_lip + c_cream[None, :] * (top_hl * 0.22)

        drape = 0.06 + 0.18 * np.maximum(_fbm2(uu * 1.6 + axis * 5.0, np.zeros_like(uu)) - 0.35, 0.0)
        top_m = np.clip((u_z - (1.0 - drape)) / np.maximum(drape, 0.04), 0.0, 1.0)[:, None]
        bot_m = np.clip((0.14 - u_z) / 0.14, 0.0, 1.0)[:, None] * (0.25 + 0.75 * gr_w[:, None]) * 0.45
        m_w = np.clip(top_m * 0.65 + bot_m, 0.0, 0.68) * moss_amt
        col_w = np.where(joint_w[:, None], c_mort[None, :], col_sw)
        col_w = col_w * (1.0 - m_w) + (c_m0[None, :] * 0.65 + c_m1[None, :] * 0.35) * m_w

        pts.append(np.column_stack([wx_w, wy_w, wz_w]))
        cols.append(col_w)
        scs.append(np.tile(np.array([s_w * 0.72, s_w * 0.72, s_w * 0.72], np.float32), (len(uu), 1)))
        nrms.append(np.tile(nw, (len(uu), 1)))

    if z0 > 0.2:
        pts.append(np.column_stack([wx, wy, np.full_like(gx, z0) + cam]))
        cols.append(np.tile(c_mort * 1.15, (len(gx), 1)))
        scs.append(np.tile(np.array([s_top * 0.78, s_top * 0.78, 0.036], np.float32), (len(gx), 1)))
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

    # Optional theme decorations: overhanging moss ledge tufts, wildflower scatter, volumetric sunbeams & sparkles
    om = theme.get("overhang_moss")
    if om:
        c_ma, c_mb = np.array(om["rgb_a"], np.float32), np.array(om["rgb_b"], np.float32)
        ov_pts, ov_cols, ov_scs = [], [], []
        for vol in frame.get("volumes", []):
            if vol["kind"] not in ("platform", "overpass"):
                continue
            cx, cy, hx, hy, zt = vol["cx"], vol["cy"], vol["hx"], vol["hy"], vol["z_top"]
            cam = float(vol.get("arch_camber", 0.0))
            for side_axis in (0, 1):
                for sign in (-1.0, 1.0):
                    span_len = 2.0 * (hy if side_axis == 0 else hx)
                    n_tuft = int(span_len * 28)
                    u = rng.uniform(-1.0, 1.0, n_tuft).astype(np.float32)
                    if side_axis == 0:
                        px = cx + sign * (hx + rng.uniform(-0.08, 0.05, n_tuft))
                        py = cy + u * hy
                        pz = zt + rng.uniform(-0.10, 0.06, n_tuft)
                    else:
                        px = cx + u * hx
                        py = cy + sign * (hy + rng.uniform(-0.08, 0.05, n_tuft))
                        pz = zt + cam * (1.0 - u * u) + rng.uniform(-0.10, 0.06, n_tuft)
                    t_c = rng.uniform(0.0, 1.0, (n_tuft, 1)).astype(np.float32)
                    ov_pts.append(np.column_stack([px, py, pz]))
                    ov_cols.append(c_ma[None, :] * (1.0 - t_c) + c_mb[None, :] * t_c)
                    ov_scs.append(np.column_stack([
                        rng.uniform(0.05, 0.11, n_tuft),
                        rng.uniform(0.05, 0.11, n_tuft),
                        rng.uniform(0.04, 0.09, n_tuft),
                    ]))
        if ov_pts:
            xyz_o = np.vstack(ov_pts).astype(np.float32)
            rgb_o = shade_splats(xyz_o, np.vstack(ov_cols).astype(np.float32), frame, theme)
            chunks.append(make_splats(xyz_o, rgb_o, np.vstack(ov_scs).astype(np.float32), np.full(len(xyz_o), 0.94, np.float32), canonical_dtype))

    fl = theme.get("flowers")
    if fl and int(fl.get("count", 0)) > 0:
        fcols = np.array(fl["colors"], dtype=np.float32)
        f_pts, f_cols, f_scs = [], [], []
        for _ in range(int(fl["count"])):
            fx = float(rng.uniform(b["x_min"] + 0.3, b["x_max"] - 0.3))
            fy = float(rng.uniform(b["y_min"] + 0.3, b["y_max"] - 0.3))
            ix = int(np.clip(round((fx - vframe.x_min) / vframe.res), 0, vframe.nx - 1))
            iy = int(np.clip(round((fy - vframe.y_min) / vframe.res), 0, vframe.ny - 1))
            sps = vframe.spans[iy][ix]
            if not sps:
                continue
            sp = sps[-1]
            if sp["type"] == "stairs" or (sp["type"] == "stone_path" and abs(fx) < 0.55):
                continue
            f_pts.append([fx, fy, sp["z"] + float(rng.uniform(0.03, 0.08))])
            f_cols.append(fcols[rng.integers(0, len(fcols))])
            r = float(rng.uniform(0.032, 0.055))
            f_scs.append([r, r, r * 0.75])
        if f_pts:
            xyz_f = np.array(f_pts, dtype=np.float32)
            rgb_f = shade_splats(xyz_f, np.array(f_cols, dtype=np.float32), frame, theme)
            chunks.append(make_splats(xyz_f, rgb_f, np.array(f_scs, dtype=np.float32), np.full(len(xyz_f), 0.95, np.float32), canonical_dtype))

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

    n_beams = int(atm.get("sunbeam_count", 0))
    if n_beams > 0:
        beam_col = np.array(atm.get("sunbeam_rgb", [1.0, 0.92, 0.76]), dtype=np.float32)
        beam_origins = [(-1.8, 0.4, 5.2), (0.6, 0.2, 5.4), (2.4, -0.5, 5.2), (-3.2, -1.2, 5.0), (1.2, 1.8, 5.2)][:n_beams]
        beam_dir = np.array([0.42, -0.18, -0.88], dtype=np.float32)
        beam_dir /= np.linalg.norm(beam_dir)
        b_pts, b_cols, b_scs, b_ops = [], [], [], []
        for orig in beam_origins:
            o = np.array(orig, dtype=np.float32)
            for t_s in np.linspace(0.0, 4.6, 18):
                b_pts.append(o + beam_dir * t_s + rng.normal(0.0, 0.06, 3).astype(np.float32))
                rad = 0.24 + 0.06 * t_s
                b_cols.append(beam_col)
                b_scs.append([rad, rad, rad * 1.35])
                b_ops.append(0.042 * math.sin(math.pi * np.clip(t_s / 4.6, 0.05, 0.95)))
        chunks.append(make_splats(
            np.array(b_pts, np.float32), np.array(b_cols, np.float32),
            np.array(b_scs, np.float32), np.array(b_ops, np.float32), canonical_dtype,
        ))

    n_sp = int(atm.get("sparkle_count", 0))
    if n_sp > 0:
        spx = rng.uniform(b["x_min"] + 0.5, b["x_max"] - 0.5, n_sp).astype(np.float32)
        spy = rng.uniform(b["y_min"] + 0.5, b["y_max"] - 0.5, n_sp).astype(np.float32)
        spz = rng.uniform(0.35, 3.8, n_sp).astype(np.float32)
        chunks.append(make_splats(
            np.column_stack([spx, spy, spz]),
            np.tile(np.array(atm.get("sparkle_rgb", [1.0, 0.92, 0.72]), np.float32), (n_sp, 1)),
            np.full((n_sp, 3), 0.032, dtype=np.float32),
            np.full(n_sp, 0.95, np.float32), canonical_dtype,
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
        sh_path = args.out_dir / f"showcase_{theme['name']}_4view.png"
        sh.save(sh_path)
        print(f"Saved 4-view showcase -> {sh_path}", flush=True)

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
