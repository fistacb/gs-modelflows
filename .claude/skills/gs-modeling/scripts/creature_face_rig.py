"""Creature / Non-Humanoid Face Rigging Module (Doll Eyes + Eyelid Shells + Muzzle Jaw).

Extends the humanoid doll-eye pipeline to work on furred / stylized / non-humanoid faces
(white fur, brown fur, fantasy colors, angled animal cheeks, protruding muzzles):
  1. `fit_creature_face`:
     - Auto-samples dominant face fur/skin color from the front head shell (no human skin-color hardcode)
     - Detects L/R eyes via chromatic/luminance contrast against face fur
     - Fits local 2D quadratic face surface per eye (wide cheek_yaw_deg up to +/-35 deg)
     - Samples character-specific iris gradient (top/mid/bot/glow) & upper-lash color directly from 3DGS
     - Refines `jaw` bone & mouth `seam_cut` anchored to the true face midline & snout tip
  2. `carve_creature_sockets`:
     - Carves clean ellipsoidal eye sockets on the body PLY
  3. `build_creature_eyeball`:
     - Generates high-density glossy anime/creature 3DGS eyeball aligned to the eye's cheek normal
  4. `build_inner_mouth_patch`:
     - Generates a compact dark-rose inner-mouth cavity patch so opening `jaw` never reveals a hollow void
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

from harmonic_dqs import SH_C0

IDENTITY_Q = [1.0, 0.0, 0.0, 0.0]


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    denom = edge1 - edge0
    if abs(denom) < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    t = np.clip((x - edge0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]).astype(np.float32)


def _fibonacci_front_cap(n: int, max_angle_deg: float = 76.0) -> np.ndarray:
    cos_min = float(np.cos(np.radians(max_angle_deg)))
    i = np.arange(n, dtype=np.float64) + 0.5
    cos_a = 1.0 - (1.0 - cos_min) * (i / n)
    sin_a = np.sqrt(np.maximum(0.0, 1.0 - cos_a * cos_a))
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([sin_a * np.cos(theta), -cos_a, sin_a * np.sin(theta)]).astype(np.float32)


def _normal_to_quat_wxyz(normals: np.ndarray) -> np.ndarray:
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    q = np.zeros((len(n), 4), dtype=np.float32)
    q[:, 0] = 1.0 + n[:, 2]
    q[:, 1] = -n[:, 1]
    q[:, 2] = n[:, 0]
    anti = n[:, 2] < -0.9999
    q[anti] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    return q


def _fit_local_face_quad(
    xyz: np.ndarray,
    fur_mask: np.ndarray,
    surf_center: np.ndarray,
    radius: float,
    max_yaw_deg: float = 34.0,
) -> dict[str, Any]:
    """Robust Tikhonov-regularized IRLS quadratic face fit y(dx, dz) around one creature eye."""
    cx, cy, cz = float(surf_center[0]), float(surf_center[1]), float(surf_center[2])
    m = (
        fur_mask
        & (np.hypot(xyz[:, 0] - cx, xyz[:, 2] - cz) <= max(radius * 2.6, 0.065))
        & (np.hypot(xyz[:, 0] - cx, xyz[:, 2] - cz) >= radius * 0.72)
        & (xyz[:, 1] <= cy + 0.045)
        & (xyz[:, 1] >= cy - 0.045)
    )
    if int(m.sum()) < 20:
        m = fur_mask & (np.hypot(xyz[:, 0] - cx, xyz[:, 2] - cz) <= max(radius * 3.8, 0.11)) & (xyz[:, 1] <= cy + 0.06)
    pts = xyz[m]
    if len(pts) < 12:
        return {
            "skin_surface_y": cy,
            "quad_coef": [cy, 0.0, 0.0, 0.0, 0.0, 0.0],
            "normal": [0.0, -1.0, 0.0],
            "cheek_yaw_deg": 0.0,
            "cheek_pitch_deg": 0.0,
        }

    rx_n = max(radius, 0.015)
    dx = pts[:, 0] - cx
    dz = pts[:, 2] - cz
    u = dx / rx_n
    v = dz / rx_n
    A = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    w_dist = np.exp(-(np.hypot(dx, dz) / max(rx_n * 2.2, 0.045)) ** 2)
    reg = np.diag([0.0, 0.05, 0.10, 0.35, 0.45, 0.35])
    w = w_dist.copy()
    y_vals = pts[:, 1].astype(np.float64)
    for _ in range(5):
        Aw = A * w[:, None]
        coef_uv = np.linalg.solve(Aw.T @ A + reg, Aw.T @ y_vals)
        err = np.abs(A @ coef_uv - y_vals)
        w = w_dist * np.where(err < 0.004, 1.0, 0.002 / np.maximum(err, 1e-5))

    c0 = float(coef_uv[0])
    c1 = float(coef_uv[1] / rx_n)
    c2 = float(coef_uv[2] / rx_n)
    c3 = float(coef_uv[3] / (rx_n * rx_n))
    c4 = float(coef_uv[4] / (rx_n * rx_n))
    c5 = float(coef_uv[5] / (rx_n * rx_n))

    nrm = np.array([c1, -1.0, c2], dtype=np.float64)
    nrm /= max(float(np.linalg.norm(nrm)), 1e-8)
    cheek_yaw = float(np.clip(np.degrees(np.arctan2(nrm[0], -nrm[1])) * 0.72, -max_yaw_deg, max_yaw_deg))
    cheek_pitch = float(np.clip(np.degrees(np.arctan2(nrm[2], -nrm[1])) * 0.65, -20.0, 20.0))
    return {
        "skin_surface_y": c0,
        "quad_coef": [c0, c1, c2, c3, c4, c5],
        "normal": [float(nrm[0]), float(nrm[1]), float(nrm[2])],
        "cheek_yaw_deg": cheek_yaw,
        "cheek_pitch_deg": cheek_pitch,
    }


def fit_creature_face(
    data: np.ndarray,
    bones: dict[str, Any],
) -> dict[str, Any]:
    """Detect L/R creature eyes, sample fur/iris/lash colors, and refine `eye_L`, `eye_R`, and `jaw` in `bones`.

    Mutates/updates `bones` with `eye_L`, `eye_R`, and refined `jaw` and returns a `face_meta` dict.
    """
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    fdc = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    op = 1.0 / (1.0 + np.exp(-np.clip(data["opacity"], -20, 20)))

    hb = bones["head"]
    h_head = np.asarray(hb["head"], dtype=np.float32)
    t_head = np.asarray(hb["tail"], dtype=np.float32)
    head_cx = float(h_head[0])
    head_y_ref = float(min(h_head[1], t_head[1]))
    head_z_lo = float(min(h_head[2], t_head[2])) - 0.05
    head_z_hi = float(max(h_head[2], t_head[2])) + 0.14

    # 1. Isolate front face shell & auto-sample dominant fur/skin RGB
    m_face = (
        (op > 0.25)
        & (xyz[:, 2] >= head_z_lo)
        & (xyz[:, 2] <= head_z_hi)
        & (xyz[:, 1] <= head_y_ref + 0.12)
        & (np.abs(xyz[:, 0] - head_cx) <= 0.22)
    )
    fxyz = xyz[m_face]
    frgb = rgb[m_face]
    if len(fxyz) < 100:
        raise RuntimeError(f"Insufficient front face splats ({len(fxyz)}) around head bone")

    # Dominant face fur color from bright cheek/forehead fur near eyes (excludes darker orange markings)
    m_cheek = (np.abs(fxyz[:, 0] - head_cx) <= 0.09) & (fxyz[:, 2] <= float(np.percentile(fxyz[:, 2], 70)))
    c_sub = frgb[m_cheek] if int(m_cheek.sum()) >= 20 else frgb
    c_lum = 0.299 * c_sub[:, 0] + 0.587 * c_sub[:, 1] + 0.114 * c_sub[:, 2]
    m_bright_fur = c_lum >= float(np.percentile(c_lum, 65))
    fur_rgb = np.median(c_sub[m_bright_fur], axis=0).astype(np.float32) if np.any(m_bright_fur) else np.median(c_sub, axis=0).astype(np.float32)
    d_fur_all = np.linalg.norm(rgb - fur_rgb[None, :], axis=1)
    fur_mask = (op > 0.25) & (d_fur_all < 0.26)

    # 2. Detect L/R eyes as chromatic/dark clusters distinct from face fur in the mid-upper front face
    sat = frgb.max(axis=1) - frgb.min(axis=1)
    lum = 0.299 * frgb[:, 0] + 0.587 * frgb[:, 1] + 0.114 * frgb[:, 2]
    d_fur_f = np.linalg.norm(frgb - fur_rgb[None, :], axis=1)
    y_front_face = float(np.percentile(fxyz[:, 1], 8))
    z_mid_lo = float(np.percentile(fxyz[:, 2], 35))
    z_mid_hi = float(np.percentile(fxyz[:, 2], 88))

    m_eye_cand = (
        (d_fur_f > 0.28)
        & ((sat > 0.28) | (lum < 0.42))
        & (lum < 0.78)
        & (fxyz[:, 2] >= z_mid_lo)
        & (fxyz[:, 2] <= z_mid_hi)
        & (fxyz[:, 1] <= y_front_face + 0.14)
        & (np.abs(fxyz[:, 0] - head_cx) >= 0.025)
        & (np.abs(fxyz[:, 0] - head_cx) <= 0.16)
    )
    ep = fxyz[m_eye_cand]
    ec = frgb[m_eye_cand]
    if len(ep) < 20:
        raise RuntimeError(f"Could not detect creature eyes on front face (only {len(ep)} candidate points)")

    cx_eyes = float(np.median(ep[:, 0]))
    mL = ep[:, 0] > cx_eyes + 0.015
    mR = ep[:, 0] < cx_eyes - 0.015
    if int(mL.sum()) < 8 or int(mR.sum()) < 8:
        raise RuntimeError(f"Asymmetric eye detection failed (L={int(mL.sum())}, R={int(mR.sum())})")

    # Keep only the true eye connected component per side (rejects forehead/cheek orange markings!)
    from scipy.spatial import cKDTree as _cKDTree
    from scipy.sparse.csgraph import connected_components as _cc

    def _pick_eye_component(mask_side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts_s = ep[mask_side]
        cols_s = ec[mask_side]
        if len(pts_s) < 16:
            return pts_s, cols_s
        adj = _cKDTree(pts_s).sparse_distance_matrix(_cKDTree(pts_s), max_distance=0.018, output_type="coo_matrix")
        n_c, lbls = _cc(adj, directed=False)
        if n_c <= 1:
            return pts_s, cols_s
        lum_s = 0.299 * cols_s[:, 0] + 0.587 * cols_s[:, 1] + 0.114 * cols_s[:, 2]
        best_cid, best_score = 0, -1.0
        for cid in range(n_c):
            mc = lbls == cid
            if int(mc.sum()) < 8:
                continue
            # True eyes contain dark pupil/lash splats (lum < 0.36); forehead fur markings are brighter!
            n_dark = int((lum_s[mc] < 0.36).sum())
            score = float(n_dark) * 4.0 + float(mc.sum()) * 0.2
            if score > best_score:
                best_score, best_cid = score, cid
        m_keep = lbls == best_cid
        return pts_s[m_keep], cols_s[m_keep]

    pts_L, col_L = _pick_eye_component(mL)
    pts_R, col_R = _pick_eye_component(mR)

    # Sample iris gradient palette from detected eye splats
    all_ec = np.concatenate([col_L, col_R], axis=0)
    all_lum = 0.299 * all_ec[:, 0] + 0.587 * all_ec[:, 1] + 0.114 * all_ec[:, 2]
    q25, q55, q80 = np.percentile(all_lum, [25, 55, 80])
    top_c = np.median(all_ec[all_lum <= q25], axis=0) if np.any(all_lum <= q25) else np.median(all_ec, axis=0) * 0.45
    mid_c = np.median(all_ec[(all_lum > q25) & (all_lum <= q55)], axis=0) if np.any((all_lum > q25) & (all_lum <= q55)) else np.median(all_ec, axis=0)
    bot_c = np.median(all_ec[(all_lum > q55) & (all_lum <= q80)], axis=0) if np.any((all_lum > q55) & (all_lum <= q80)) else np.clip(mid_c * 1.3, 0, 1)
    glow_c = np.percentile(all_ec[all_lum >= q80], 75, axis=0) if np.any(all_lum >= q80) else np.clip(bot_c * 1.25 + 0.15, 0, 1)
    lash_c = np.clip(top_c * 0.65, 0.06, 0.45)

    iris_palette = {
        "top": np.clip(top_c, 0.05, 0.85).astype(np.float32).tolist(),
        "mid": np.clip(mid_c, 0.08, 0.92).astype(np.float32).tolist(),
        "bot": np.clip(bot_c, 0.12, 0.98).astype(np.float32).tolist(),
        "glow": np.clip(glow_c, 0.20, 1.00).astype(np.float32).tolist(),
        "pupil": np.clip(top_c * 0.45, 0.03, 0.30).astype(np.float32).tolist(),
        "rim": np.clip(top_c * 0.72, 0.05, 0.40).astype(np.float32).tolist(),
    }

    eye_metas: dict[str, dict] = {}
    for side, pts_s in (("eye_L", pts_L), ("eye_R", pts_R)):
        p10, p50, p90 = np.percentile(pts_s, [10, 50, 90], axis=0)
        span = p90 - p10
        # Anchor depth strictly on the actual eye surface Y (prevents protruding muzzle from pulling eye Y forward!)
        y_eye_actual = float(np.percentile(pts_s[:, 1], 45))
        surf = np.array([float(p50[0]), y_eye_actual, float(0.5 * (p10[2] + p90[2]))], dtype=np.float32)

        r_eye = float(np.clip(0.42 * max(float(span[0]), float(span[2])), 0.015, 0.025))
        fit = _fit_local_face_quad(xyz, fur_mask, surf, r_eye, max_yaw_deg=26.0)
        # Clamp fitted quad surface Y within +/-2mm of actual eye surface so muzzle NEVER pulls eyeballs out!
        fitted_y = float(np.clip(fit["skin_surface_y"], y_eye_actual - 0.002, y_eye_actual + 0.002))
        quad_coef = list(fit["quad_coef"])
        quad_coef[0] = fitted_y
        surf[1] = fitted_y

        # Exact 1:1 co-axial alignment between carved socket center (surf) and eyeball iris pole:
        # Let g_hat = R_align @ [0, -1, 0] be the eyeball's painted iris direction.
        # Placing eb_ctr = surf - (r_eye + recess) * g_hat guarantees:
        #   iris_pole = eb_ctr + r_eye * g_hat == surf - recess * g_hat  (0.0 mm lateral/vertical drift!)
        yaw_deg = float(fit["cheek_yaw_deg"])
        pitch_deg = float(fit["cheek_pitch_deg"])
        R_align = (
            Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
        ).astype(np.float32)
        g_hat = (R_align @ np.array([0.0, -1.0, 0.0], dtype=np.float32)).astype(np.float32)
        g_hat /= max(float(np.linalg.norm(g_hat)), 1e-8)

        recess = max(r_eye * 0.14, 0.0025)
        eb_ctr = (surf - g_hat * (r_eye + recess)).astype(np.float32)
        carve_ctr = (surf - g_hat * (r_eye * 0.10)).astype(np.float32)
        excl_r = [float(max(span[0] * 0.44, r_eye * 0.74)), float(r_eye * 1.05), float(max(span[2] * 0.36, r_eye * 0.50))]

        meta_s = {
            "detected_iris_points": int(len(pts_s)),
            "surface_center": surf.tolist(),
            "eyeball_center": eb_ctr.tolist(),
            "eyeball_radius": r_eye,
            "cheek_yaw_deg": yaw_deg,
            "cheek_pitch_deg": pitch_deg,
            "gaze_dir": g_hat.tolist(),
            "face_quad_coef": quad_coef,
            "normal": fit["normal"],
            "carve_center": carve_ctr.tolist(),
            "exclusion_radius": excl_r,
            "sclera_rgb": [0.97, 0.96, 0.94],
        }
        eye_metas[side] = meta_s
        bones[side] = {
            "parent": "head",
            "head": eb_ctr.tolist(),
            "tail": (eb_ctr + g_hat * r_eye).tolist(),
            "rest_quat_wxyz": list(IDENTITY_Q),
            "limits_deg": {"x": [-15, 14], "y": [0, 0], "z": [-24, 24]},
            "non_deforming": True,
            "meta": meta_s,
        }

    # 3. Refine `jaw` bone & mouth `seam_cut` anchored to true face midline (midpoint between eyes) and snout tip
    sL = np.asarray(eye_metas["eye_L"]["surface_center"], dtype=np.float32)
    sR = np.asarray(eye_metas["eye_R"]["surface_center"], dtype=np.float32)
    face_cx = 0.5 * float(sL[0] + sR[0])
    d_eye = float(np.linalg.norm(sL - sR))
    eye_z = 0.5 * float(sL[2] + sR[2])
    eye_y = 0.5 * float(sL[1] + sR[1])

    # Locate frontmost snout tip below the eyes
    m_snout = (
        (op > 0.25)
        & (np.abs(xyz[:, 0] - face_cx) <= d_eye * 0.42)
        & (xyz[:, 2] >= eye_z - d_eye * 0.75)
        & (xyz[:, 2] <= eye_z - d_eye * 0.05)
        & (xyz[:, 1] <= eye_y)
    )
    if int(m_snout.sum()) >= 20:
        spts = xyz[m_snout]
        y_snout_front = float(np.percentile(spts[:, 1], 4))
        z_snout_mid = float(np.percentile(spts[spts[:, 1] <= y_snout_front + 0.035, 2], 45))
        z_chin_bot = float(np.percentile(spts[:, 2], 12))
    else:
        y_snout_front = eye_y - 0.08
        z_snout_mid = eye_z - 0.045
        z_chin_bot = eye_z - 0.085

    # Place mouth split right below nose tip so lower muzzle + chin open cleanly as a single jaw unit
    mouth_z = 0.72 * z_snout_mid + 0.28 * z_chin_bot
    p_jaw_hinge = np.array([face_cx, float(0.40 * eye_y + 0.60 * y_snout_front), mouth_z + 0.006], dtype=np.float32)
    p_jaw_tip = np.array([face_cx, y_snout_front + 0.004, z_chin_bot], dtype=np.float32)
    jaw_len = max(float(np.linalg.norm(p_jaw_tip - p_jaw_hinge)), 0.055)
    cut_r = max(jaw_len * 1.25, d_eye * 0.50)

    jaw_seam = {
        "origin": [face_cx, float(0.30 * p_jaw_hinge[1] + 0.70 * p_jaw_tip[1]), float(mouth_z)],
        "normal": [0.0, 0.0, 1.0],  # +Z = upper muzzle (head), -Z = lower jaw (jaw)
        "radius": float(cut_r),
        "keep_side": -1.0,
        "affect_parent": True,
    }
    jaw_box = {
        "min": [face_cx - cut_r, y_snout_front - 0.04, z_chin_bot - 0.055],
        "max": [face_cx + cut_r, float(p_jaw_hinge[1]) + 0.045, float(mouth_z) + 0.002],
    }
    bones["jaw"] = {
        "parent": "head",
        "head": p_jaw_hinge.tolist(),
        "tail": p_jaw_tip.tolist(),
        "rest_quat_wxyz": list(IDENTITY_Q),
        "skin_group": "jaw",
        "role": "jaw",
        "pin_radius": max(jaw_len * 0.65, 0.032),
        "influence_radius_scale": 1.85,
        "region_box": jaw_box,
        "seam_cut": jaw_seam,
        # In right-handed Z-up, -Y-pointing jaw opens DOWNWARD with positive Rx(+)!
        "limits_deg": {"x": [-4.0, 38.0], "y": [-3.0, 3.0], "z": [-5.0, 5.0]},
    }

    return {
        "face_cx": face_cx,
        "d_eye": d_eye,
        "eye_z": eye_z,
        "eye_y": eye_y,
        "fur_rgb": fur_rgb.tolist(),
        "lash_rgb": lash_c.tolist(),
        "iris_palette": iris_palette,
        "eye_L": eye_metas["eye_L"],
        "eye_R": eye_metas["eye_R"],
        "mouth_center": [face_cx, float(0.45 * p_jaw_hinge[1] + 0.55 * p_jaw_tip[1]), float(mouth_z)],
        "mouth_width": float(min(d_eye * 0.42, 0.065)),
    }


def carve_creature_sockets(data: np.ndarray, face_meta: dict[str, Any]) -> np.ndarray:
    """Remove original painted eye splats inside the L/R socket exclusion ellipsoids."""
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    keep = np.ones(len(data), dtype=bool)
    for k in ("eye_L", "eye_R"):
        em = face_meta[k]
        cc = np.asarray(em["carve_center"], dtype=np.float32)
        er = np.asarray(em["exclusion_radius"], dtype=np.float32)
        rel = (xyz - cc[None, :]) / np.maximum(er[None, :], 1e-4)
        inside = np.sum(rel * rel, axis=1) < 1.0
        keep &= ~inside
    return data[keep].copy()


def build_creature_eyeball(
    eye_meta: dict[str, Any],
    iris_palette: dict[str, list[float]],
    dtype: np.dtype,
    n_back: int = 1200,
    n_front: int = 3200,
) -> np.ndarray:
    """Generate a high-density glossy anime/creature 3DGS eyeball in eyeball-local coordinates."""
    radius = float(eye_meta["eyeball_radius"])
    sclera_rgb = np.asarray(eye_meta.get("sclera_rgb", [0.97, 0.96, 0.94]), dtype=np.float32)
    pal = {k: np.asarray(v, dtype=np.float32) for k, v in iris_palette.items()}

    n_full = _fibonacci_sphere(n_back)
    n_b = n_full[n_full[:, 1] > -float(np.cos(np.radians(74.0)))]
    n_f = _fibonacci_front_cap(n_front, max_angle_deg=76.0)
    normals = np.concatenate([n_b, n_f], axis=0)
    n_total = len(normals)

    # Rotate eyeball rest gaze (-Y) toward the local cheek surface yaw/pitch so iris sits dead-center on curved animal cheeks
    yaw_deg = float(eye_meta.get("cheek_yaw_deg", 0.0))
    pitch_deg = float(eye_meta.get("cheek_pitch_deg", 0.0))
    R_align = (
        Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
    ).astype(np.float32)

    # Paint iris in local canonical (-Y forward) coordinates, then orient into cheek frame
    front_y = np.maximum(-normals[:, 1], 1e-5)
    az = np.degrees(np.arctan2(normals[:, 0], front_y))
    el = np.degrees(np.arcsin(np.clip(normals[:, 2], -1.0, 1.0)))

    # Large, soft anime/fox iris (eliminates wide-eyed white-ring / sanpaku stare!)
    rx_deg, rz_deg = 62.0, 60.0
    r_iris = np.hypot(az / rx_deg, el / rz_deg)
    in_front = normals[:, 1] < -0.10

    v_t = np.clip((el + rz_deg) / (2.0 * rz_deg), 0.0, 1.0)
    iris_rgb = np.where(
        (v_t < 0.5)[:, None],
        pal["bot"][None, :] * (1.0 - (v_t * 2.0)[:, None]) + pal["mid"][None, :] * (v_t * 2.0)[:, None],
        pal["mid"][None, :] * (1.0 - ((v_t - 0.5) * 2.0)[:, None]) + pal["top"][None, :] * ((v_t - 0.5) * 2.0)[:, None],
    )
    # Warm golden lower-iris glow crescent
    glow_m = _smoothstep(0.48, 0.14, v_t) * _smoothstep(0.95, 0.48, r_iris)
    iris_rgb = iris_rgb * (1.0 - 0.62 * glow_m[:, None]) + pal["glow"][None, :] * (0.62 * glow_m[:, None])

    # Soft upper-eyelid shadow across top of iris & sclera for gentle, cute expression
    top_shadow = _smoothstep(0.55, 0.92, v_t)
    iris_rgb = iris_rgb * (1.0 - 0.42 * top_shadow[:, None])

    # Soft outer iris rim & cute rounded-oval pupil
    rim_m = _smoothstep(0.78, 0.96, r_iris) * _smoothstep(1.06, 0.98, r_iris)
    iris_rgb = iris_rgb * (1.0 - 0.65 * rim_m[:, None]) + pal["rim"][None, :] * (0.65 * rim_m[:, None])
    r_pupil = np.hypot(az / 17.0, (el - 2.0) / 22.0)
    pupil_m = _smoothstep(1.05, 0.55, r_pupil)
    iris_rgb = iris_rgb * (1.0 - 0.80 * pupil_m[:, None]) + pal["pupil"][None, :] * (0.80 * pupil_m[:, None])

    # Big warm anime catchlights
    h1 = _smoothstep(15.0, 5.5, np.hypot(az - 16.0, el - 14.0))
    h2 = _smoothstep(8.5, 3.0, np.hypot(az + 16.0, el + 14.0))
    hl = np.clip(h1 + 0.78 * h2, 0.0, 1.0)
    iris_rgb = iris_rgb * (1.0 - hl[:, None]) + np.array([1.0, 0.99, 0.97], dtype=np.float32)[None, :] * hl[:, None]

    # Soften sclera slightly with upper lid shadow so whites never look harsh
    sclera_shaded = sclera_rgb[None, :] * (1.0 - 0.16 * top_shadow[:, None])
    iris_blend = np.where(in_front, _smoothstep(1.04, 0.90, r_iris), 0.0)
    rgb = sclera_shaded * (1.0 - iris_blend[:, None]) + iris_rgb * iris_blend[:, None]
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    normals_w = (normals @ R_align.T).astype(np.float32)
    xyz = normals_w * radius
    rot = _normal_to_quat_wxyz(normals_w)

    r_disc_b = float(np.sqrt((4.0 * np.pi * (radius ** 2) / n_back) / np.pi) * 1.20)
    cap_area = 2.0 * np.pi * (radius ** 2) * (1.0 - np.cos(np.radians(76.0)))
    r_disc_f = float(np.sqrt((cap_area / n_front) / np.pi) * 1.15)
    r_disc = np.concatenate([np.full(len(n_b), r_disc_b, dtype=np.float32), np.full(len(n_f), r_disc_f, dtype=np.float32)])

    out = np.zeros(n_total, dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["f_dc_0"] = (rgb[:, 0] - 0.5) / SH_C0
    out["f_dc_1"] = (rgb[:, 1] - 0.5) / SH_C0
    out["f_dc_2"] = (rgb[:, 2] - 0.5) / SH_C0
    out["opacity"] = 5.0
    out["scale_0"] = np.log(r_disc)
    out["scale_1"] = np.log(r_disc)
    out["scale_2"] = np.log(r_disc * 0.25)
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    return out


def build_creature_eyelid_shell(
    eye_meta: dict[str, Any],
    fur_rgb: list[float],
    lash_rgb: list[float],
    blink: float,
    dtype: np.dtype,
) -> np.ndarray:
    """Generate a curved creature eyelid curtain + arched lash line flush with the local face quad."""
    if blink <= 0.02:
        return np.zeros(0, dtype=dtype)

    sc = np.asarray(eye_meta["surface_center"], dtype=np.float32)
    c0, c1, c2, c3, c4, c5 = [float(v) for v in eye_meta["face_quad_coef"]]
    er = eye_meta["exclusion_radius"]
    rx = float(er[0]) * 0.98
    rz = float(er[2]) * 0.96

    # Grid across the eye opening; eyelid curtain descends from +rz down to z_edge as blink goes 0 -> 1
    z_top = rz * 1.02
    z_bot = -rz * 1.02
    z_edge = z_top - float(np.clip(blink, 0.0, 1.0)) * (z_top - z_bot)

    nu, nv = 34, 26
    us = np.linspace(-rx, rx, nu, dtype=np.float32)
    vs = np.linspace(z_bot, z_top, nv, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    dx = uu.ravel()
    dz = vv.ravel()

    # Elliptical eye aperture + curved upper lash arch
    in_ell = (dx / rx) ** 2 + (dz / rz) ** 2 <= 1.04
    arch = 0.22 * rz * (1.0 - (dx / rx) ** 2) * (1.0 - 0.75 * float(blink))
    z_cur_edge = z_edge + arch
    on_lid = in_ell & (dz >= z_cur_edge - 0.16 * rz)
    if not np.any(on_lid):
        return np.zeros(0, dtype=dtype)

    dx = dx[on_lid]
    dz = dz[on_lid]
    z_cur_edge = z_cur_edge[on_lid]

    # Evaluate local face quad y(dx, dz) and step 1.5 mm outward (-Y) so eyelid cleanly covers eyeball
    y_surf = c0 + c1 * dx + c2 * dz + c3 * dx * dx + c4 * dx * dz + c5 * dz * dz - 0.0022
    pts = np.column_stack([sc[0] + dx, y_surf, sc[2] + dz]).astype(np.float32)

    # Color: face fur on upper curtain, gentle dark lash stroke along the descending lower rim
    dist_to_edge = np.abs(dz - z_cur_edge)
    lash_thick = max(rz * 0.22, 0.0026)
    is_lash = _smoothstep(lash_thick, lash_thick * 0.35, dist_to_edge)
    fur_c = np.asarray(fur_rgb, dtype=np.float32)
    lash_c = np.asarray(lash_rgb, dtype=np.float32)
    rgb = fur_c[None, :] * (1.0 - is_lash[:, None]) + lash_c[None, :] * is_lash[:, None]

    # Surface normals from quad gradient
    nx = c1 + 2.0 * c3 * dx + c4 * dz
    nz = c2 + c4 * dx + 2.0 * c5 * dz
    normals = np.column_stack([nx, np.full_like(nx, -1.0), nz]).astype(np.float32)
    rot = _normal_to_quat_wxyz(normals)

    s_xy = max(rx / nu * 1.45, 0.0028)
    out = np.zeros(len(pts), dtype=dtype)
    out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    out["f_dc_0"] = (rgb[:, 0] - 0.5) / SH_C0
    out["f_dc_1"] = (rgb[:, 1] - 0.5) / SH_C0
    out["f_dc_2"] = (rgb[:, 2] - 0.5) / SH_C0
    out["opacity"] = 4.5
    out["scale_0"] = math.log(s_xy)
    out["scale_1"] = math.log(s_xy)
    out["scale_2"] = math.log(s_xy * 0.28)
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    return out


def build_inner_mouth_patch(
    face_meta: dict[str, Any],
    dtype: np.dtype,
    n_pts: int = 420,
) -> np.ndarray:
    """Build a dark-rose inner mouth cavity dome sitting slightly behind the muzzle opening
    so opening the `jaw` bone reveals a warm mouth interior instead of an empty void.
    """
    mc = np.asarray(face_meta["mouth_center"], dtype=np.float32)
    mw = float(face_meta.get("mouth_width", 0.045))
    mh = mw * 0.52

    rng = np.random.default_rng(7)
    u = rng.uniform(-1.0, 1.0, size=n_pts)
    v = rng.uniform(-1.0, 1.0, size=n_pts)
    keep = (u * u + v * v) <= 1.0
    u, v = u[keep], v[keep]
    depth = 0.012 + 0.014 * (1.0 - np.clip(u * u + v * v, 0.0, 1.0))

    pts = np.column_stack([
        mc[0] + u * (mw * 0.55),
        mc[1] + depth,
        mc[2] + v * (mh * 0.55),
    ]).astype(np.float32)

    # Warm dark burgundy-rose at back, lighter pink tongue hint at bottom
    t_bot = np.clip((-v + 0.3) / 1.3, 0.0, 1.0)
    dark_mouth = np.array([0.36, 0.10, 0.14], dtype=np.float32)
    tongue_pink = np.array([0.82, 0.36, 0.40], dtype=np.float32)
    rgb = dark_mouth[None, :] * (1.0 - 0.55 * t_bot[:, None]) + tongue_pink[None, :] * (0.55 * t_bot[:, None])

    out = np.zeros(len(pts), dtype=dtype)
    out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    out["f_dc_0"] = (rgb[:, 0] - 0.5) / SH_C0
    out["f_dc_1"] = (rgb[:, 1] - 0.5) / SH_C0
    out["f_dc_2"] = (rgb[:, 2] - 0.5) / SH_C0
    out["opacity"] = 4.2
    s_r = max(mw * 0.09, 0.003)
    out["scale_0"] = math.log(s_r)
    out["scale_1"] = math.log(s_r)
    out["scale_2"] = math.log(s_r * 0.5)
    out["rot_0"] = 1.0
    return out
