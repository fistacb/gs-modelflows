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


def detect_eye_contour_and_brow(
    xyz: np.ndarray,
    rgb: np.ndarray,
    fur_rgb: np.ndarray,
    eye_meta: dict[str, Any],
) -> dict[str, Any]:
    """Separately detect the eye's upper contour (upper eyelash) vs an optional eyebrow above it.

    Scans dark-splat density along +Z in the eye's local cheek frame:
      - Peak 1 (dz in [0.35*rz, 1.15*rz]): upper eye contour / eyelash -> ALWAYS carved & moved by dynamic eyelid.
      - Valley above Peak 1: fur/skin gap (`dz_lash_top`).
      - Peak 2 above valley (dz in [1.20*rz, 2.20*rz]): optional eyebrow -> preserved strictly above `dz_brow_floor`.
    """
    sc = np.asarray(eye_meta["surface_center"], dtype=np.float32)
    er = np.asarray(eye_meta["exclusion_radius"], dtype=np.float32)
    rx, rz = float(er[0]), max(float(er[2]), 0.008)
    yaw_deg = float(eye_meta.get("cheek_yaw_deg", 0.0))
    pitch_deg = float(eye_meta.get("cheek_pitch_deg", 0.0))
    R_align = (
        Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
    ).astype(np.float32)

    loc = (xyz - sc[None, :]) @ R_align
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    sat = rgb.max(axis=1) - rgb.min(axis=1)
    d_fur = np.linalg.norm(rgb - fur_rgb[None, :], axis=1)
    # Exclude bright saturated fur markings (e.g. orange forehead patterns) from dark lash/brow detection
    is_bright_marking = (sat > 0.42) & (rgb[:, 0] > 0.52)
    is_dark = (lum < 0.40) & (~is_bright_marking) & (d_fur > 0.22) & (np.abs(loc[:, 0]) <= rx * 1.15) & (np.abs(loc[:, 1]) <= max(er[1] * 1.25, 0.025))

    dzs = np.linspace(0.65 * rz, 2.45 * rz, 22)
    step = float(dzs[1] - dzs[0]) * 0.90
    counts = np.array([int((is_dark & (np.abs(loc[:, 2] - dz) <= step)).sum()) for dz in dzs], dtype=np.float32)

    # Peak 1: upper eye contour / lash in [0.65*rz, 1.45*rz] (never inside the pupil!)
    m_p1 = dzs <= 1.45 * rz
    p1_idx = int(np.argmax(counts[m_p1]))
    p1_val = float(counts[p1_idx])

    # Valley immediately above Peak 1 (must be above the upper lash line)
    search_end = min(len(counts), p1_idx + 7)
    if p1_idx + 1 < search_end:
        val_rel = int(np.argmin(counts[p1_idx + 1 : search_end]))
        val_idx = p1_idx + 1 + val_rel
    else:
        val_idx = min(len(counts) - 1, p1_idx + 2)
    dz_lash_top = float(np.clip(dzs[val_idx], 1.25 * rz, 1.65 * rz))

    # Peak 2: distinct eyebrow STRICTLY above the fur/skin valley (dz >= 1.65 * rz)
    has_brow = False
    m_p2 = (np.arange(len(counts)) > val_idx) & (dzs >= max(dz_lash_top + 0.22 * rz, 1.65 * rz))
    if np.any(m_p2) and p1_val >= 4.0:
        p2_val = float(np.max(counts[m_p2]))
        val_val = float(counts[val_idx])
        if p2_val >= max(8.0, p1_val * 0.35) and val_val <= p2_val * 0.45:
            has_brow = True

    r_eye = float(eye_meta.get("eyeball_radius", rz * 1.4))
    dz_brow_floor = dz_lash_top if has_brow else float(max(1.85 * rz, 1.45 * r_eye))
    return {
        "has_brow": has_brow,
        "dz_lash_peak": float(dzs[p1_idx]),
        "dz_lash_top": dz_lash_top,
        "dz_brow_floor": dz_brow_floor,
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
    eye_raw_surfs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for side, pts_s, col_s in (("eye_L", pts_L, col_L), ("eye_R", pts_R, col_R)):
        # Include BOTH dark upper iris/pupil AND bright colored lower iris (lum < 0.64),
        # while rejecting pale cheek blush so the center never shifts upward into the upper lash!
        lum_s = 0.299 * col_s[:, 0] + 0.587 * col_s[:, 1] + 0.114 * col_s[:, 2]
        m_iris_s = lum_s < 0.64
        pts_core = pts_s[m_iris_s] if int(m_iris_s.sum()) >= 12 else pts_s

        p10, p50, p90 = np.percentile(pts_core, [10, 50, 90], axis=0)
        span = p90 - p10
        y_eye_actual = float(np.percentile(pts_core[:, 1], 28))
        surf = np.array([
            float(0.5 * (p10[0] + p90[0])),
            y_eye_actual,
            float(0.5 * (p10[2] + p90[2])),
        ], dtype=np.float32)
        eye_raw_surfs[side] = (surf, span, pts_core)

    # Bilateral eye-height consistency: prevent asymmetric dark upper lash on one side from pulling
    # one eye center abnormally higher than the other (allows natural head tilt up to 0.045 * D_eye)
    d_eye_est = float(np.linalg.norm(eye_raw_surfs["eye_L"][0] - eye_raw_surfs["eye_R"][0]))
    max_dz_asym = max(0.045 * d_eye_est, 0.006)
    zL = float(eye_raw_surfs["eye_L"][0][2])
    zR = float(eye_raw_surfs["eye_R"][0][2])
    if abs(zL - zR) > max_dz_asym:
        # Pull the higher eye down toward the lower (less lash-biased) eye
        if zL > zR:
            eye_raw_surfs["eye_L"][0][2] = zR + max_dz_asym
        else:
            eye_raw_surfs["eye_R"][0][2] = zL + max_dz_asym

    for side, pts_s, col_s in (("eye_L", pts_L, col_L), ("eye_R", pts_R, col_R)):
        surf, span, pts_core = eye_raw_surfs[side]
        y_eye_actual = float(surf[1])

        r_eye = float(np.clip(0.44 * max(float(span[0]), float(span[2])), 0.016, 0.026))
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

        recess = max(r_eye * 0.04, 0.0008)
        eb_ctr = (surf - g_hat * (r_eye + recess)).astype(np.float32)
        carve_ctr = (surf - g_hat * (r_eye * 0.08)).astype(np.float32)
        excl_r = [
            float(max(span[0] * 0.48, r_eye * 0.85)),
            float(r_eye * 1.15),
            float(max(span[2] * 0.48, r_eye * 0.62)),
        ]

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
        contour_info = detect_eye_contour_and_brow(xyz[op > 0.25], rgb[op > 0.25], fur_rgb, meta_s)
        meta_s.update(contour_info)
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

    # 3. Refine `jaw` bone & mouth `seam_cut` anchored directly to the detected dark nose tip dot!
    sL = np.asarray(eye_metas["eye_L"]["surface_center"], dtype=np.float32)
    sR = np.asarray(eye_metas["eye_R"]["surface_center"], dtype=np.float32)
    face_cx = 0.5 * float(sL[0] + sR[0])
    d_eye = float(np.linalg.norm(sL - sR))
    eye_z = 0.5 * float(sL[2] + sR[2])
    eye_y = 0.5 * float(sL[1] + sR[1])

    # Search full snout height up to eye_z + 0.03 so upward-tilted noses are never missed!
    m_snout = (
        (op > 0.25)
        & (np.abs(xyz[:, 0] - face_cx) <= d_eye * 0.42)
        & (xyz[:, 2] >= eye_z - d_eye * 0.85)
        & (xyz[:, 2] <= eye_z + 0.03)
        & (xyz[:, 1] <= eye_y - 0.015)
    )
    if int(m_snout.sum()) >= 20:
        spts = xyz[m_snout]
        srgb = rgb[m_snout]
        slum = 0.299 * srgb[:, 0] + 0.587 * srgb[:, 1] + 0.114 * srgb[:, 2]
        y_snout_front = float(np.percentile(spts[:, 1], 2))
        m_nose = (spts[:, 1] <= y_snout_front + 0.025) & (slum < 0.48)
        if int(m_nose.sum()) >= 5:
            nose_pt = np.median(spts[m_nose], axis=0).astype(np.float32)
            nose_x, y_nose, z_nose = float(nose_pt[0]), float(nose_pt[1]), float(nose_pt[2])
        else:
            nose_x = face_cx
            y_nose = y_snout_front
            z_nose = float(np.percentile(spts[spts[:, 1] <= y_snout_front + 0.025, 2], 65))
        z_chin_bot = float(np.percentile(spts[:, 2], 12))
    else:
        nose_x = face_cx
        y_nose = eye_y - 0.08
        z_nose = eye_z - 0.02
        z_chin_bot = eye_z - 0.10

    # True mouth smile line: detect actual painted closed-mouth splats below the nose tip if present,
    # otherwise fall back to ~1.8 cm below nose tip.
    mouth_z_fallback = float(max(z_chin_bot + 0.025, z_nose - min(d_eye * 0.13, 0.020)))
    mouth_x = nose_x
    mouth_y = float(y_nose + 0.006)
    mouth_z = mouth_z_fallback
    mw_est = float(min(d_eye * 0.36, 0.055))
    m_mouth_line = (
        (op > 0.20)
        & (np.abs(xyz[:, 0] - nose_x) <= mw_est * 0.55)
        & (xyz[:, 1] <= y_nose + 0.028)
        & (xyz[:, 2] >= z_nose - min(d_eye * 0.32, 0.050))
        & (xyz[:, 2] <= z_nose - 0.010)
        & (d_fur_all > 0.15)
        & ((0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]) < 0.78)
    )
    if int(m_mouth_line.sum()) >= 6:
        mp_med = np.median(xyz[m_mouth_line], axis=0).astype(np.float32)
        mouth_x = float(0.5 * nose_x + 0.5 * float(mp_med[0]))
        mouth_y = float(mp_med[1])
        mouth_z = float(np.clip(mp_med[2], z_chin_bot + 0.018, z_nose - 0.014))
        mouth_line_rgb = np.clip(np.median(rgb[m_mouth_line], axis=0), 0.20, 0.85).astype(np.float32)
    else:
        mouth_line_rgb = np.array([0.74, 0.44, 0.42], dtype=np.float32)

    p_jaw_hinge = np.array([mouth_x, float(0.45 * eye_y + 0.55 * y_nose), mouth_z + 0.006], dtype=np.float32)
    p_jaw_tip = np.array([mouth_x, mouth_y, max(z_chin_bot, mouth_z - 0.032)], dtype=np.float32)
    jaw_len = max(float(np.linalg.norm(p_jaw_tip - p_jaw_hinge)), 0.048)
    cut_r = max(jaw_len * 1.15, d_eye * 0.42)

    jaw_seam = {
        "origin": [mouth_x, float(0.25 * p_jaw_hinge[1] + 0.75 * p_jaw_tip[1]), mouth_z],
        "normal": [0.0, 0.0, 1.0],  # +Z = upper muzzle (head), -Z = lower jaw (jaw)
        "radius": float(cut_r),
        "keep_side": -1.0,
        "affect_parent": True,
    }
    jaw_box = {
        "min": [mouth_x - cut_r, y_nose - 0.035, z_chin_bot - 0.04],
        "max": [mouth_x + cut_r, float(p_jaw_hinge[1]) + 0.035, mouth_z + 0.002],
    }
    bones["jaw"] = {
        "parent": "head",
        "head": p_jaw_hinge.tolist(),
        "tail": p_jaw_tip.tolist(),
        "rest_quat_wxyz": list(IDENTITY_Q),
        "skin_group": "jaw",
        "role": "jaw",
        "pin_radius": max(jaw_len * 0.60, 0.028),
        "influence_radius_scale": 1.80,
        "region_box": jaw_box,
        "seam_cut": jaw_seam,
        "limits_deg": {"x": [-4.0, 38.0], "y": [-3.0, 3.0], "z": [-5.0, 5.0]},
    }

    return {
        "face_cx": face_cx,
        "d_eye": d_eye,
        "eye_z": eye_z,
        "eye_y": eye_y,
        "nose_pos": [nose_x, y_nose, z_nose],
        "fur_rgb": fur_rgb.tolist(),
        "lash_rgb": lash_c.tolist(),
        "mouth_line_rgb": mouth_line_rgb.tolist(),
        "iris_palette": iris_palette,
        "eye_L": eye_metas["eye_L"],
        "eye_R": eye_metas["eye_R"],
        "mouth_center": [mouth_x, mouth_y, mouth_z],
        "mouth_width": mw_est,
    }


def carve_creature_sockets(data: np.ndarray, face_meta: dict[str, Any]) -> np.ndarray:
    """Color-guided socket carving: removes original painted iris/sclera splats while
    strictly preserving surrounding face fur and dark upper-eyelash contour lines!
    """
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    fdc = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    fur_rgb = np.asarray(face_meta.get("fur_rgb", [0.92, 0.84, 0.78]), dtype=np.float32)
    d_fur = np.linalg.norm(rgb - fur_rgb[None, :], axis=1)
    sat = rgb.max(axis=1) - rgb.min(axis=1)
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

    keep = np.ones(len(data), dtype=bool)
    for k in ("eye_L", "eye_R"):
        em = face_meta[k]
        cc = np.asarray(em["carve_center"], dtype=np.float32)
        er = np.asarray(em["exclusion_radius"], dtype=np.float32)

        # Rotate into the eye's local cheek frame so the carving ellipsoid aligns with angled animal cheeks
        yaw_deg = float(em.get("cheek_yaw_deg", 0.0))
        pitch_deg = float(em.get("cheek_pitch_deg", 0.0))
        R_align = (
            Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
        ).astype(np.float32)
        loc = (xyz - cc[None, :]) @ R_align  # local X=width, Y=depth, Z=height

        # 1) Tight inner pupil/iris core (unconditional carve)
        rel_core = loc / np.maximum(er[None, :] * np.array([[0.78, 1.0, 0.72]], dtype=np.float32), 1e-4)
        in_core = np.sum(rel_core * rel_core, axis=1) < 1.0

        # 2) Outer iris + upper eye-contour ring: remove remaining iris AND the static upper eye-contour
        #    (below dz_brow_floor) so the dynamic eyelid moves the contour itself when blinking!
        #    Face fur (d_fur <= 0.24), cheek blush below eye, and any detected eyebrow above dz_brow_floor are preserved!
        has_brow = bool(em.get("has_brow", False))
        r_eye = float(em.get("eyeball_radius", er[2] * 1.4))
        dz_brow_floor = float(em.get("dz_brow_floor", er[2] * 1.50))
        z_outer = float(er[2] * 1.65) if has_brow else float(max(er[2] * 2.15, r_eye * 1.65))
        rel_outer = loc / np.maximum(np.array([[er[0] * 1.72, er[1] * 1.45, z_outer]], dtype=np.float32), 1e-4)
        in_outer = np.sum(rel_outer * rel_outer, axis=1) < 1.0
        below_brow = loc[:, 2] <= dz_brow_floor
        # Preserve light orange cheek blush below eye (only carve bright saturated iris inside tighter lower bound)
        is_dark_contour = (lum < 0.46) & (d_fur > 0.22)
        is_bright_iris = (d_fur > 0.28) & (sat > 0.30) & (lum < 0.78) & (loc[:, 2] >= -er[2] * 1.10) & (np.abs(loc[:, 0]) <= er[0] * 1.25)
        is_iris_or_contour = (is_dark_contour | is_bright_iris) & below_brow

        keep &= ~(in_core | (in_outer & is_iris_or_contour))

    out = data[keep].copy()

    # 3) Soft halo fur-inpainting (recolor, NOT delete!) around each carved socket:
    #    Eliminates leftover painterly eye-shadow / eyeliner halo splats without creating holes in the face,
    #    while strictly preserving lower cheek blush and any forehead markings / eyebrows above dz_brow_floor!
    from scipy.spatial import cKDTree as _cKDTree
    xyz_o = np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32)
    fdc_o = np.column_stack([out["f_dc_0"], out["f_dc_1"], out["f_dc_2"]]).astype(np.float32)
    rgb_o = np.clip(0.5 + SH_C0 * fdc_o, 0.0, 1.0)
    d_fur_o = np.linalg.norm(rgb_o - fur_rgb[None, :], axis=1)

    for k in ("eye_L", "eye_R"):
        em = face_meta[k]
        cc = np.asarray(em["carve_center"], dtype=np.float32)
        er = np.asarray(em["exclusion_radius"], dtype=np.float32)
        has_brow = bool(em.get("has_brow", False))
        r_eye = float(em.get("eyeball_radius", er[2] * 1.4))
        dz_brow_floor = float(em.get("dz_brow_floor", er[2] * 1.50))

        yaw_deg = float(em.get("cheek_yaw_deg", 0.0))
        pitch_deg = float(em.get("cheek_pitch_deg", 0.0))
        R_align = (
            Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
            @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
        ).astype(np.float32)
        loc = (xyz_o - cc[None, :]) @ R_align

        z_halo = float(er[2] * 1.75) if has_brow else float(max(er[2] * 2.25, r_eye * 1.65))
        # (a) General soft halo around upper/side eye border
        r_halo = np.sqrt(
            (loc[:, 0] / max(er[0] * 2.15, 1e-4)) ** 2
            + (loc[:, 1] / max(er[1] * 1.75, 1e-4)) ** 2
            + (loc[:, 2] / max(z_halo, 1e-4)) ** 2
        )
        w_rad = _smoothstep(1.08, 0.65, r_halo)
        w_low = _smoothstep(-er[2] * 1.35, -er[2] * 0.45, loc[:, 2])
        w_top = _smoothstep(dz_brow_floor * 1.02, dz_brow_floor * 0.72, loc[:, 2])
        w_col = _smoothstep(0.12, 0.24, d_fur_o)
        alpha_halo = w_rad * w_low * w_top * w_col * 0.94

        # (b) Outer/inner dark eye-corner liner remnants (lum < 0.60): clean thoroughly into fur
        lum_o = 0.299 * rgb_o[:, 0] + 0.587 * rgb_o[:, 1] + 0.114 * rgb_o[:, 2]
        r_corner = np.sqrt(
            (loc[:, 0] / max(er[0] * 2.45, 1e-4)) ** 2
            + (loc[:, 1] / max(er[1] * 1.85, 1e-4)) ** 2
            + (loc[:, 2] / max(z_halo, 1e-4)) ** 2
        )
        w_dark_corner = (
            _smoothstep(1.05, 0.72, r_corner)
            * _smoothstep(-er[2] * 1.75, -er[2] * 1.15, loc[:, 2])
            * w_top
            * _smoothstep(0.62, 0.44, lum_o)
            * _smoothstep(0.16, 0.26, d_fur_o)
            * 0.97
        )

        # (c) Directly-under-eye lower-lid highlight / shadow rim (preserves outer cheek blush!)
        w_under_x = _smoothstep(er[0] * 1.55, er[0] * 1.05, np.abs(loc[:, 0]))
        w_under_y = _smoothstep(er[1] * 1.75, er[1] * 1.15, np.abs(loc[:, 1]))
        w_under_z = _smoothstep(-er[2] * 1.85, -er[2] * 1.15, loc[:, 2]) * _smoothstep(er[2] * 0.25, -er[2] * 0.15, loc[:, 2])
        w_under = w_under_x * w_under_y * w_under_z * _smoothstep(0.14, 0.26, d_fur_o) * 0.90

        alpha = np.maximum.reduce([alpha_halo, w_dark_corner, w_under]).astype(np.float32)

        m_blend = alpha > 0.03
        if not np.any(m_blend):
            continue

        # Sample local surrounding bright fur color with subtle natural fur luminance variation
        m_local_fur = (d_fur_o < 0.18) & (np.linalg.norm(xyz_o - cc[None, :], axis=1) < max(er[0] * 3.8, 0.10))
        if int(m_local_fur.sum()) >= 16:
            tree = _cKDTree(xyz_o[m_local_fur])
            dists, nn = tree.query(xyz_o[m_blend], k=min(6, int(m_local_fur.sum())))
            if nn.ndim == 1:
                nn = nn[:, None]
                dists = dists[:, None]
            w_inv = 1.0 / np.maximum(dists, 1e-4)
            w_inv /= np.sum(w_inv, axis=1, keepdims=True)
            target_rgb = np.sum(rgb_o[m_local_fur][nn] * w_inv[:, :, None], axis=1).astype(np.float32)
        else:
            target_rgb = np.broadcast_to(fur_rgb[None, :], (int(m_blend.sum()), 3)).copy()

        # Preserve subtle +/-2.5% natural fur shading variation so recolored area looks like soft fur, not flat plastic
        orig_lum_m = lum_o[m_blend]
        tgt_lum_m = 0.299 * target_rgb[:, 0] + 0.587 * target_rgb[:, 1] + 0.114 * target_rgb[:, 2]
        micro_var = np.clip((orig_lum_m - tgt_lum_m) * 0.08, -0.025, 0.025)[:, None]
        target_rgb = np.clip(target_rgb + micro_var, 0.0, 1.0)

        a_m = alpha[m_blend][:, None]
        rgb_o[m_blend] = rgb_o[m_blend] * (1.0 - a_m) + target_rgb * a_m

    # 4) Soft fur-inpainting of the original static closed-mouth smile line so opening the dynamic mouth
    #    NEVER produces a creepy double-mouth (strictly protects the dark nose tip above z_nose - 0.009)!
    if "mouth_center" in face_meta and "nose_pos" in face_meta:
        mc = np.asarray(face_meta["mouth_center"], dtype=np.float32)
        np_pos = np.asarray(face_meta["nose_pos"], dtype=np.float32)
        mw = float(face_meta.get("mouth_width", 0.050))
        d_fur_m = np.linalg.norm(rgb_o - fur_rgb[None, :], axis=1)
        lum_m = 0.299 * rgb_o[:, 0] + 0.587 * rgb_o[:, 1] + 0.114 * rgb_o[:, 2]

        w_mx = _smoothstep(mw * 0.70, mw * 0.40, np.abs(xyz_o[:, 0] - mc[0]))
        w_my = _smoothstep(mc[1] + 0.032, mc[1] + 0.012, xyz_o[:, 1])
        w_mz_top = _smoothstep(float(np_pos[2]) - 0.008, float(np_pos[2]) - 0.015, xyz_o[:, 2])
        w_mz_bot = _smoothstep(mc[2] - 0.026, mc[2] - 0.012, xyz_o[:, 2])
        w_mcol = _smoothstep(0.10, 0.20, d_fur_m) * _smoothstep(0.84, 0.68, lum_m)
        alpha_m = (w_mx * w_my * w_mz_top * w_mz_bot * w_mcol * 0.96).astype(np.float32)
        m_mblend = alpha_m > 0.03
        if np.any(m_mblend):
            m_mfur = (d_fur_m < 0.18) & (np.linalg.norm(xyz_o - mc[None, :], axis=1) < max(mw * 1.8, 0.08))
            if int(m_mfur.sum()) >= 12:
                tree_m = _cKDTree(xyz_o[m_mfur])
                dm, nnm = tree_m.query(xyz_o[m_mblend], k=min(6, int(m_mfur.sum())))
                if nnm.ndim == 1:
                    nnm = nnm[:, None]
                    dm = dm[:, None]
                wm = 1.0 / np.maximum(dm, 1e-4)
                wm /= np.sum(wm, axis=1, keepdims=True)
                tgt_m = np.sum(rgb_o[m_mfur][nnm] * wm[:, :, None], axis=1).astype(np.float32)
            else:
                tgt_m = np.broadcast_to(fur_rgb[None, :], (int(m_mblend.sum()), 3)).copy()
            am = alpha_m[m_mblend][:, None]
            rgb_o[m_mblend] = rgb_o[m_mblend] * (1.0 - am) + tgt_m * am

    out["f_dc_0"] = (rgb_o[:, 0] - 0.5) / SH_C0
    out["f_dc_1"] = (rgb_o[:, 1] - 0.5) / SH_C0
    out["f_dc_2"] = (rgb_o[:, 2] - 0.5) / SH_C0
    return out


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
    body_xyz: np.ndarray | None = None,
    body_rgb: np.ndarray | None = None,
    always_draw_lash: bool = True,
) -> np.ndarray:
    """Generate a solid, fur-matched creature eyelid curtain + crisp arched upper eye-contour.

    - Draws the upper contour even at blink=0 (`always_draw_lash=True`) so the entire eye contour
      descends smoothly as one piece when blinking (no floating leftover upper line on browless animals!).
    - Clamps the top curtain strictly below `dz_brow_floor` so detected eyebrows are never covered.
    """
    if blink <= 0.02 and not always_draw_lash:
        return np.zeros(0, dtype=dtype)

    sc = np.asarray(eye_meta["surface_center"], dtype=np.float32)
    r_eye = float(eye_meta["eyeball_radius"])
    yaw_deg = float(eye_meta.get("cheek_yaw_deg", 0.0))
    pitch_deg = float(eye_meta.get("cheek_pitch_deg", 0.0))
    R_align = (
        Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        @ Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
    ).astype(np.float32)

    er = eye_meta["exclusion_radius"]
    rx = float(er[0]) * 1.22
    rz = float(er[2]) * 1.22
    dz_brow_floor = float(eye_meta.get("dz_brow_floor", rz * 1.35))
    b_t = float(np.clip(blink, 0.0, 1.0))

    # Eyelid contour curve z_lash(u_norm) across the FULL width of the eye:
    # - At blink=0.0: sits right on the upper eye contour rim (+0.68*rz at center, gently curving down at corners)
    # - At blink=1.0: descends cleanly to the lower-mid closed-eye crease (-0.38*rz) with a gentle closed-eye arc
    #   so the contour NEVER gets clipped by the bottom ellipse or buried behind lower cheek fur!
    z_top = min(rz * 1.18, dz_brow_floor)
    z_open_edge = min(rz * 0.84, z_top - rz * 0.06)
    z_closed_edge = -rz * 0.34
    z_edge_base = z_open_edge - b_t * (z_open_edge - z_closed_edge)
    arch_amp = rz * (0.24 * (1.0 - b_t) - 0.14 * b_t)  # arches UP when open, gently curves into closed smile arc when shut

    # Wrap radius slightly larger than eyeball so eyelid sits cleanly flush over the socket
    r_wrap = max(r_eye * 1.12, 0.022)

    def _to_world_pts_and_nrm(lx: np.ndarray, lz: np.ndarray, y_push: float) -> tuple[np.ndarray, np.ndarray]:
        rad_sq = np.maximum(r_wrap * r_wrap - lx * lx - lz * lz, (r_wrap * 0.42) ** 2)
        dy_local = (r_wrap - np.sqrt(rad_sq)) - y_push
        loc_p = np.column_stack([lx, dy_local, lz]).astype(np.float32)
        w_pts = sc[None, :] + (loc_p @ R_align.T)
        loc_n = np.column_stack([lx / r_wrap, -np.sqrt(rad_sq) / r_wrap, lz / r_wrap]).astype(np.float32)
        w_nrm = (loc_n @ R_align.T).astype(np.float32)
        return w_pts, w_nrm

    # Layer 1: Solid fur-matched eyelid curtain (covers eyeball above the descending lash contour)
    curt_pts = np.zeros((0, 3), dtype=np.float32)
    curt_nrm = np.zeros((0, 3), dtype=np.float32)
    curt_rgb = np.zeros((0, 3), dtype=np.float32)
    if b_t > 0.04:
        nu, nv = 38, 28
        us = np.linspace(-rx, rx, nu, dtype=np.float32)
        vs = np.linspace(-rz * 1.05, z_top, nv, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        dx = uu.ravel()
        dz = vv.ravel()
        u_n = dx / max(rx, 1e-5)
        z_curve = z_edge_base + arch_amp * (1.0 - u_n * u_n)
        in_ell = (dx / rx) ** 2 + (dz / (rz * 1.08)) ** 2 <= 1.04
        # At blink >= 0.85, fill all the way down to -rz*0.92 below the crease too so lower sclera never peeks out!
        z_min_curt = z_curve if b_t < 0.82 else np.minimum(z_curve, -rz * 0.88 * np.sqrt(np.maximum(0.0, 1.0 - u_n * u_n)))
        on_lid = in_ell & (dz >= z_min_curt)
        if np.any(on_lid):
            dx_c, dz_c = dx[on_lid], dz[on_lid]
            curt_pts, curt_nrm = _to_world_pts_and_nrm(dx_c, dz_c, y_push=0.0052)
            fur_c = np.asarray(fur_rgb, dtype=np.float32)
            if body_xyz is not None and body_rgb is not None and len(body_xyz) >= 50:
                from scipy.spatial import cKDTree as _cKDTree
                d_fur = np.linalg.norm(body_rgb - fur_c[None, :], axis=1)
                m_f = (d_fur < 0.24) & (np.linalg.norm(body_xyz - sc[None, :], axis=1) < max(rx * 2.5, 0.07))
                if int(m_f.sum()) >= 20:
                    tree = _cKDTree(body_xyz[m_f])
                    _, nn = tree.query(curt_pts, k=min(8, int(m_f.sum())))
                    if nn.ndim == 1:
                        nn = nn[:, None]
                    curt_rgb = np.median(body_rgb[m_f][nn], axis=1).astype(np.float32)
                else:
                    curt_rgb = np.broadcast_to(fur_c[None, :], (len(curt_pts), 3)).copy()
            else:
                curt_rgb = np.broadcast_to(fur_c[None, :], (len(curt_pts), 3)).copy()

    # Layer 2: Soft-feathered full-width upper eye-contour / lash ribbon (ALWAYS drawn when always_draw_lash=True!)
    # Feathers smoothly into surrounding fur along top/bottom edges & corners so it matches painterly/soft 3DGS styles!
    n_u_lash, n_v_lash = 44, 7
    u_ln = np.linspace(-0.96, 0.96, n_u_lash, dtype=np.float32)
    v_ln = np.linspace(-1.0, 1.0, n_v_lash, dtype=np.float32)
    uu_l, vv_l = np.meshgrid(u_ln, v_ln)
    u_l = uu_l.ravel()
    v_l = vv_l.ravel()
    # Taper ribbon smoothly at inner/outer eye corners; slightly slimmer when open, gently fuller when closed
    taper = np.sqrt(np.maximum(0.15, 1.0 - (u_l * 0.94) ** 2))
    thick_scale = 0.115 + 0.035 * b_t
    half_thick = max(rz * thick_scale, 0.0020) * taper
    dx_l = u_l * rx
    dz_l = (z_edge_base + arch_amp * (1.0 - u_l * u_l)) + v_l * half_thick
    lash_pts, lash_nrm = _to_world_pts_and_nrm(dx_l, dz_l, y_push=0.0060 + 0.0010 * b_t)

    # Sample local fur color right at the lash points so feather edges blend seamlessly into the face
    fur_c = np.asarray(fur_rgb, dtype=np.float32)
    if body_xyz is not None and body_rgb is not None and len(body_xyz) >= 50:
        from scipy.spatial import cKDTree as _cKDTree
        d_fur_l = np.linalg.norm(body_rgb - fur_c[None, :], axis=1)
        m_fl = (d_fur_l < 0.24) & (np.linalg.norm(body_xyz - sc[None, :], axis=1) < max(rx * 2.5, 0.07))
        if int(m_fl.sum()) >= 20:
            tree_l = _cKDTree(body_xyz[m_fl])
            _, nn_l = tree_l.query(lash_pts, k=min(8, int(m_fl.sum())))
            if nn_l.ndim == 1:
                nn_l = nn_l[:, None]
            edge_fur_rgb = np.median(body_rgb[m_fl][nn_l], axis=1).astype(np.float32)
        else:
            edge_fur_rgb = np.broadcast_to(fur_c[None, :], (len(lash_pts), 3)).copy()
    else:
        edge_fur_rgb = np.broadcast_to(fur_c[None, :], (len(lash_pts), 3)).copy()

    # Keep warm original contour tone when open; deepen gently when closed so the closed-eye arc reads clearly
    lash_c = np.clip(np.asarray(lash_rgb, dtype=np.float32) * (0.92 - 0.22 * b_t), 0.06, 0.48)
    w_v = np.exp(-(v_l / (0.62 + 0.14 * b_t)) ** 2)
    w_u = _smoothstep(1.0, 0.68, np.abs(u_l))
    w_lash = np.clip(w_v * (0.55 + 0.45 * w_u), 0.0, 1.0).astype(np.float32)
    lash_rgb_arr = edge_fur_rgb * (1.0 - w_lash[:, None]) + lash_c[None, :] * w_lash[:, None]
    # Soft opacity tapering on feather edges so the contour melts naturally into the face
    lash_op = (1.6 + (3.0 + 0.9 * b_t) * w_lash).astype(np.float32)

    pts = np.concatenate([curt_pts, lash_pts], axis=0) if len(curt_pts) > 0 else lash_pts
    normals = np.concatenate([curt_nrm, lash_nrm], axis=0) if len(curt_nrm) > 0 else lash_nrm
    rgb = np.concatenate([curt_rgb, lash_rgb_arr], axis=0) if len(curt_rgb) > 0 else lash_rgb_arr
    if len(pts) == 0:
        return np.zeros(0, dtype=dtype)

    rot = _normal_to_quat_wxyz(normals)
    s_curt = max(rx / 38.0 * 1.85, 0.0032)
    s_lash = max(rx / n_u_lash * 1.65, 0.0022)
    s_xy = np.concatenate([
        np.full(len(curt_pts), s_curt, dtype=np.float32),
        np.full(len(lash_pts), s_lash, dtype=np.float32),
    ]) if len(curt_pts) > 0 else np.full(len(lash_pts), s_lash, dtype=np.float32)
    op_arr = np.concatenate([
        np.full(len(curt_pts), 5.5, dtype=np.float32),
        lash_op,
    ]) if len(curt_pts) > 0 else lash_op

    out = np.zeros(len(pts), dtype=dtype)
    out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    out["f_dc_0"] = (rgb[:, 0] - 0.5) / SH_C0
    out["f_dc_1"] = (rgb[:, 1] - 0.5) / SH_C0
    out["f_dc_2"] = (rgb[:, 2] - 0.5) / SH_C0
    out["opacity"] = op_arr
    out["scale_0"] = np.log(s_xy)
    out["scale_1"] = np.log(s_xy)
    out["scale_2"] = np.log(s_xy * 0.30)
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    return out


def build_creature_mouth_patch(
    face_meta: dict[str, Any],
    open_amt: float,
    dtype: np.dtype,
    body_xyz: np.ndarray | None = None,
    always_draw_closed: bool = True,
) -> np.ndarray:
    """Build a unified animated creature mouth (from closed smile line at open_amt=0.0 to open mouth at 1.0).

    Paired with static mouth-line fur-inpainting in `carve_creature_sockets`, so there is ALWAYS
    exactly ONE mouth on the face (never a creepy double-mouth when opening!).
    """
    if open_amt <= 0.02 and not always_draw_closed:
        return np.zeros(0, dtype=dtype)

    mc = np.asarray(face_meta["mouth_center"], dtype=np.float32).copy()
    mw = float(face_meta.get("mouth_width", 0.045))
    fur_c = np.asarray(face_meta.get("fur_rgb", [0.94, 0.84, 0.78]), dtype=np.float32)
    smile_c = np.asarray(face_meta.get("mouth_line_rgb", [0.74, 0.44, 0.42]), dtype=np.float32) * 0.86

    # Snap mouth surface Y to the actual frontmost muzzle surface right at (mc[0], mc[2])
    if body_xyz is not None and len(body_xyz) >= 50:
        m_m = (np.abs(body_xyz[:, 0] - mc[0]) <= mw * 0.45) & (np.abs(body_xyz[:, 2] - mc[2]) <= mw * 0.45)
        if int(m_m.sum()) >= 8:
            mc[1] = float(np.percentile(body_xyz[m_m, 1], 12))

    o = float(np.clip(open_amt, 0.0, 1.0))

    # Closed smile-line state (open_amt <= 0.04): draw a delicate curved smile line right at mouth_center
    if o <= 0.04:
        nu_s, nv_s = 28, 5
        us = np.linspace(-1.0, 1.0, nu_s, dtype=np.float32)
        vs = np.linspace(-1.0, 1.0, nv_s, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        u = uu.ravel()
        v = vv.ravel()
        rx_s = mw * 0.21
        thick = max(mw * 0.032, 0.0014) * np.sqrt(np.maximum(0.15, 1.0 - (u * 0.92) ** 2))
        # Cute gentle upward-curved fox smile arc (`u*u` rises at corners)
        z_arc = mc[2] + mw * 0.045 * (u * u - 0.35) + v * thick
        y_arc = mc[1] - 0.0046 + 0.0012 * (u * u)
        pts = np.column_stack([mc[0] + u * rx_s, y_arc, z_arc]).astype(np.float32)

        w_v = np.exp(-(v / 0.65) ** 2)
        w_u = _smoothstep(1.0, 0.65, np.abs(u))
        w_s = np.clip(w_v * w_u, 0.0, 1.0).astype(np.float32)
        rgb = fur_c[None, :] * (1.0 - w_s[:, None]) + smile_c[None, :] * w_s[:, None]
        op_arr = (1.4 + 2.8 * w_s).astype(np.float32)

        nrm = np.broadcast_to(np.array([[0.0, -1.0, 0.0]], dtype=np.float32), (len(pts), 3))
        rot = _normal_to_quat_wxyz(nrm)
        s_xy = max(rx_s / nu_s * 2.0, 0.0016)
        out = np.zeros(len(pts), dtype=dtype)
        out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        out["f_dc_0"] = (rgb[:, 0] - 0.5) / SH_C0
        out["f_dc_1"] = (rgb[:, 1] - 0.5) / SH_C0
        out["f_dc_2"] = (rgb[:, 2] - 0.5) / SH_C0
        out["opacity"] = op_arr
        out["scale_0"] = math.log(s_xy)
        out["scale_1"] = math.log(s_xy)
        out["scale_2"] = math.log(s_xy * 0.30)
        out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
        return out

    # Open mouth state (o > 0.04): grows smoothly from the exact same smile-line position downward!
    rx = mw * (0.20 + 0.16 * (o ** 0.75))
    rz = mw * (0.025 + 0.21 * (o ** 1.10))

    nu, nv = 26, 22
    us = np.linspace(-1.0, 1.0, nu, dtype=np.float32)
    vs = np.linspace(-1.0, 1.0, nv, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    u = uu.ravel()
    v = vv.ravel()
    # Cute rounded-triangle / fox "a/o" mouth shape (wider at top, softly rounded at bottom)
    r2 = (u / np.maximum(0.86 + 0.16 * v, 0.4)) ** 2 + v * v
    keep = r2 <= 1.0
    u, v, r2 = u[keep], v[keep], r2[keep]

    # Anchor upper lip right on the closed smile line and expand downward as jaw opens!
    z_center = float(mc[2]) - rz * 0.55
    y_bowl = mc[1] - 0.0052 + 0.0008 * (1.0 - r2)
    pts = np.column_stack([
        mc[0] + u * rx,
        y_bowl,
        z_center + v * rz,
    ]).astype(np.float32)

    # Warm painterly rose mouth interior + soft coral-pink tongue + feathered fur-blended outer lip edge
    dark_in = np.array([0.45, 0.14, 0.18], dtype=np.float32)
    tongue = np.array([0.94, 0.48, 0.52], dtype=np.float32)
    rim_c = np.array([0.62, 0.30, 0.32], dtype=np.float32)
    t_tongue = _smoothstep(0.05, -0.55, v) * _smoothstep(0.92, 0.42, r2) * _smoothstep(0.12, 0.40, o)
    t_rim = _smoothstep(0.65, 0.92, r2)
    t_feather = _smoothstep(0.76, 1.0, r2)
    rgb = dark_in[None, :] * (1.0 - t_tongue[:, None]) + tongue[None, :] * t_tongue[:, None]
    rgb = rgb * (1.0 - 0.55 * t_rim[:, None]) + rim_c[None, :] * (0.55 * t_rim[:, None])
    rgb = rgb * (1.0 - 0.65 * t_feather[:, None]) + fur_c[None, :] * (0.65 * t_feather[:, None])

    # Orient disc splats facing forward (-Y) so they render smoothly with zero scanline striping!
    nrm = np.broadcast_to(np.array([[0.0, -1.0, 0.0]], dtype=np.float32), (len(pts), 3))
    rot = _normal_to_quat_wxyz(nrm)

    # Smooth opacity ramp + feathered rim so the mouth transitions seamlessly from smile line to open
    fade_in = float(_smoothstep(0.02, 0.22, np.array([o], dtype=np.float32))[0])
    op_arr = (1.6 + (3.2 * fade_in) * (1.0 - 0.55 * t_feather)).astype(np.float32)

    s_xy = max(rx / nu * 2.2, 0.0022)
    out = np.zeros(len(pts), dtype=dtype)
    out["x"], out["y"], out["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    out["f_dc_0"] = (rgb[:, 0] - 0.5) / SH_C0
    out["f_dc_1"] = (rgb[:, 1] - 0.5) / SH_C0
    out["f_dc_2"] = (rgb[:, 2] - 0.5) / SH_C0
    out["opacity"] = op_arr
    out["scale_0"] = math.log(s_xy)
    out["scale_1"] = math.log(s_xy)
    out["scale_2"] = math.log(s_xy * 0.30)
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
