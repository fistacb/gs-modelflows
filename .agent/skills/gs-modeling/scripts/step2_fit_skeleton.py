"""Step 2: Fit minimal doll skeleton using config proportions (no character-specific hardcodes).

Includes:
- Facial-skin-bounded iris detection (prevents brown hair / hood from inflating D_eye).
- Robust IRLS 2D quadratic face-surface fitting per eye, outputting exact local normal,
  data-driven cheek_yaw_deg / cheek_pitch_deg, quadratic surface coefficients (face_quad_coef),
  fit_rmse_mm, and rim_flush_err_mm so eyeballs/eyelids/lashes sit flush with the face.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ply_io import read_ply
from config_loader import load_config

SH_C0 = 0.28209479177387814


def fit_eye_surface(
    xyz: np.ndarray,
    skin: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    face_cx: float | None = None,
) -> dict | None:
    """Estimate the local quadratic skin surface y(dx, dz) around one eye, excluding its opening.

    If heavy bangs cover the forehead above the eye (insufficient local upper-rim skin cells),
    automatically falls back to a robust IRLS quadratic fit over the full front-face skin shell
    re-centered at (center[0], center[2]) so every character gets flush normal & depth.
    """
    local = (xyz - center) / radii
    x, y, z = local.T
    samples = local[
        skin
        & (np.abs(x) <= 1.8)
        & (z >= -2.2)
        & (z <= 1.35)
        & (np.abs(y) <= 1.0)
        & (x * x + z * z >= 0.85 ** 2)
    ]
    use_fallback = True
    if len(samples) >= 20:
        cells = np.floor(samples[:, [0, 2]] * 5).astype(int)
        order = np.lexsort((cells[:, 1], cells[:, 0]))
        splits = np.flatnonzero(np.any(np.diff(cells[order], axis=0), axis=1)) + 1
        cell_pts = np.array(
            [np.median(group, axis=0) for group in np.split(samples[order], splits) if len(group) >= 2]
        )
        if len(cell_pts) >= 18:
            cx_s, cy_s, cz_s = cell_pts.T
            if cx_s.min() < -0.45 and cx_s.max() > 0.45 and cz_s.min() < -0.75 and cz_s.max() > -0.25:
                matrix = np.column_stack((np.ones_like(cx_s), cx_s, cz_s, cx_s * cx_s, cx_s * cz_s, cz_s * cz_s))
                if np.linalg.cond(matrix) <= 65:
                    weights = np.ones_like(cy_s)
                    for _ in range(5):
                        coefficients, *_ = np.linalg.lstsq(matrix * weights[:, None], cy_s * weights, rcond=None)
                        error = np.abs(matrix @ coefficients - cy_s)
                        weights = np.sqrt(np.minimum(1.0, 0.08 / np.maximum(error, 1e-6)))
                    if abs(coefficients[0]) <= 0.40 and np.median(error) <= 0.16:
                        rx, ry, rz = float(radii[0]), float(radii[1]), float(radii[2])
                        c0 = float(center[1] + coefficients[0] * ry)
                        c1 = float(coefficients[1] * ry / max(rx, 1e-6))
                        c2 = float(coefficients[2] * ry / max(rz, 1e-6))
                        c3 = float(coefficients[3] * ry / max(rx * rx, 1e-10))
                        c4 = float(coefficients[4] * ry / max(rx * rz, 1e-10))
                        c5 = float(coefficients[5] * ry / max(rz * rz, 1e-10))
                        fit_rmse_mm = float(np.sqrt(np.mean((error * ry) ** 2)) * 1000.0)
                        n_cells = int(len(cell_pts))
                        use_fallback = False

    if use_fallback:
        # Robust IRLS quadratic fit over the FRONTMOST facial skin shell (immune to heavy bangs & rear skin layers)
        fcx = float(face_cx) if face_cx is not None else float(center[0])
        m_fb = (
            skin
            & (np.abs(xyz[:, 0] - fcx) < 0.055)
            & (xyz[:, 2] >= center[2] - 0.055)
            & (xyz[:, 2] <= center[2] + 0.022)
            & (xyz[:, 1] <= center[1] + 0.016)
            & (xyz[:, 1] >= center[1] - 0.020)
        )
        if m_fb.sum() < 25:
            return None
        pts_raw = xyz[m_fb]
        # Bin into 4 mm (x, z) cells and take frontmost (p20 in Y) per cell to exclude interior/back-wall skin
        c_xz = np.floor(pts_raw[:, [0, 2]] / 0.004).astype(int)
        ord_c = np.lexsort((c_xz[:, 1], c_xz[:, 0]))
        sp_c = np.flatnonzero(np.any(np.diff(c_xz[ord_c], axis=0), axis=1)) + 1
        pts_fb = np.array(
            [np.array([np.median(g[:, 0]), np.percentile(g[:, 1], 20), np.median(g[:, 2])])
             for g in np.split(pts_raw[ord_c], sp_c) if len(g) >= 1],
            dtype=np.float64,
        )
        if len(pts_fb) < 15:
            pts_fb = pts_raw.astype(np.float64)
        # Dimensionless Tikhonov-regularized IRLS quadratic fit over the frontmost facial skin shell.
        # Automatically damps vertical slope & curvature when upper-eye skin is occluded by bangs,
        # preventing upward (-Y) extrapolation at the upper eyelash height.
        rx_n = max(float(radii[0]), 0.010)
        rz_n = max(float(radii[2]), 0.010)
        dx = pts_fb[:, 0] - float(center[0])
        dz = pts_fb[:, 2] - float(center[2])
        u = dx / rx_n
        v = dz / rz_n
        A = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
        w_dist = np.exp(-(np.hypot(dx, dz) / max(rx_n * 3.0, 0.036)) ** 2)
        # Measure upper-eye support fraction: when bangs hide v > 0.25, increase vertical regularization
        upper_frac = float(np.mean(v > 0.25))
        reg_v = 0.08 + 1.8 * max(0.0, 0.25 - upper_frac)
        reg_curv = 0.25 + 3.5 * max(0.0, 0.25 - upper_frac)
        reg_diag = np.array([0.0, 0.04, reg_v, 0.20, reg_curv, reg_curv], dtype=np.float64)

        w = w_dist.copy()
        y_vals = pts_fb[:, 1]
        for _ in range(5):
            Aw = A * w[:, None]
            AtA = Aw.T @ A + np.diag(reg_diag)
            Atb = Aw.T @ y_vals
            coef_uv = np.linalg.solve(AtA, Atb)
            err_w = np.abs(A @ coef_uv - y_vals)
            w = w_dist * np.where(err_w < 0.0020, 1.0, 0.0008 / np.maximum(err_w, 1e-5))

        # Convert dimensionless (u, v) coefficients back to world (dx, dz)
        c0 = float(coef_uv[0])
        c1 = float(coef_uv[1] / rx_n)
        c2 = float(coef_uv[2] / rz_n)
        c3 = float(coef_uv[3] / (rx_n * rx_n))
        c4 = float(coef_uv[4] / (rx_n * rz_n))
        c5 = float(coef_uv[5] / (rz_n * rz_n))
        fit_rmse_mm = float(np.median(err_w) * 1000.0)
        n_cells = int(len(pts_fb))

    # Outward surface normal pointing toward camera (-Y forward): n = normalize([dy/dx, -1, dy/dz])
    nrm = np.array([c1, -1.0, c2], dtype=np.float64)
    nrm /= max(float(np.linalg.norm(nrm)), 1e-8)
    # Clamp cheek_yaw_deg to +/-14 deg so painted iris stays centered toward front camera on curved chibi cheeks
    cheek_yaw_deg = float(np.clip(np.degrees(np.arctan2(nrm[0], -nrm[1])) * 0.55, -14.0, 14.0))
    cheek_pitch_deg = float(np.clip(np.degrees(np.arctan2(nrm[2], -nrm[1])), -18.0, 18.0))

    return {
        "skin_surface_y": c0,
        "quad_coef": [c0, c1, c2, c3, c4, c5],
        "normal": [float(nrm[0]), float(nrm[1]), float(nrm[2])],
        "cheek_yaw_deg": cheek_yaw_deg,
        "cheek_pitch_deg": cheek_pitch_deg,
        "fit_rmse_mm": round(fit_rmse_mm, 3),
        "sample_cells": n_cells,
        "used_full_face_fallback": use_fallback,
    }


def compute_skin_mask(xyz: np.ndarray, rgb: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    red, green, blue = rgb.T
    lum = 0.299 * red + 0.587 * green + 0.114 * blue
    return (
        (opacity > 0.0)
        & (red > 0.76)
        & (green > 0.56)
        & (lum > 0.68)
        & (green > blue + 0.012)
        & (red > green + 0.015)
    )


def estimate_face_cx_and_skin_width(
    xyz: np.ndarray, rgb: np.ndarray, skin: np.ndarray, prop: dict
) -> tuple[float, float, float]:
    """Estimate true face midline X, facial-skin half-width, and frontmost nose-plane Y from front lower-face skin
    (immune to peach hood linings, hats, collars, or side hair)."""
    fz0, fz1 = float(prop["face_z_range"][0]), float(prop["face_z_range"][1])
    z_eye0 = float(prop["eye_z_range"][0])
    # Anchor on central nose/cheek strip (|X| < 0.045) to find true frontmost face plane Y
    m_center = skin & (xyz[:, 2] >= fz0) & (xyz[:, 2] <= max(z_eye0 + 0.02, fz1)) & (np.abs(xyz[:, 0]) < 0.045)
    if m_center.sum() < 20:
        m_center = (xyz[:, 2] >= fz0) & (xyz[:, 2] <= max(z_eye0 + 0.02, fz1)) & (np.abs(xyz[:, 0]) < 0.06)
    y_nose = float(np.percentile(xyz[m_center, 1], 10)) if m_center.sum() >= 10 else -0.065

    # Only keep front-shell skin within 18 mm of nose plane (excludes deeper hood lining & side hair)
    m_skin = skin & (xyz[:, 2] >= fz0) & (xyz[:, 2] <= z_eye0 + 0.015) & (xyz[:, 1] <= y_nose + 0.018) & (np.abs(xyz[:, 0]) < 0.085)
    if m_skin.sum() < 25:
        return 0.0, 0.052, y_nose
    face_cx = float(np.median(xyz[m_skin, 0]))
    w_skin_half = float(np.percentile(np.abs(xyz[m_skin, 0] - face_cx), 95))
    return face_cx, float(np.clip(w_skin_half, 0.028, 0.068)), y_nose


def detect_eye(
    xyz: np.ndarray,
    rgb: np.ndarray,
    skin: np.ndarray,
    side: str,
    prop: dict,
    face_cx_est: float = 0.0,
    w_skin_half: float = 0.052,
    y_nose: float = -0.065,
) -> dict:
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    sat = rgb.max(axis=1) - rgb.min(axis=1)
    z0, z1 = float(prop["eye_z_range"][0]), float(prop["eye_z_range"][1])

    # Restrict search strictly inside facial skin half-width & front-face plane (excludes hair/hood!)
    x_min = max(0.010, w_skin_half * 0.28)
    x_max = min(0.062, w_skin_half * 0.96)
    y_max = y_nose + 0.024

    dx_rel = xyz[:, 0] - face_cx_est
    mask_x = (dx_rel > x_min) & (dx_rel < x_max) if side == "eye_L" else (dx_rel < -x_min) & (dx_rel > -x_max)
    eye_win = mask_x & (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1) & (xyz[:, 1] <= y_max) & (~skin)

    # Include BOTH dark upper iris AND bright glowing warm lower iris (prevents upper-lash Z bias!)
    m_iris_dark = eye_win & (lum < 0.52) & (sat > 0.08)
    m_iris_glow = eye_win & (lum >= 0.52) & (lum < 0.84) & (sat > 0.24)
    m_iris = m_iris_dark | m_iris_glow
    m_sclera_init = eye_win & (lum >= 0.78) & (sat < 0.16)
    m_dark = eye_win & (lum < 0.48)
    pts = xyz[m_iris] if m_iris.sum() >= 20 else (xyz[m_sclera_init] if m_sclera_init.sum() >= 20 else xyz[m_dark])
    if len(pts) < 10:
        raise RuntimeError(f"Failed to detect {side} eye points in Z=[{z0},{z1}] (found {len(pts)})")

    p05, p10, p50, p90, p95 = np.percentile(pts, [5, 10, 50, 90, 95], axis=0)
    # Use vertical midpoint 0.5*(p10+p90) instead of p50 so dense dark upper-eyelash splats do not pull eye Z upward!
    z_mid = float(0.5 * (p10[2] + p90[2]))
    surf_center = np.array([float(p50[0]), float(np.percentile(pts[:, 1], 45)), z_mid], dtype=np.float32)

    sgn = +1.0 if side == "eye_L" else -1.0
    m_slope = (
        (np.abs(xyz[:, 0] - surf_center[0]) < 0.022)
        & (np.abs(xyz[:, 2] - surf_center[2]) < 0.012)
        & (xyz[:, 1] <= surf_center[1] + 0.022)
        & (xyz[:, 1] >= surf_center[1] - 0.018)
    )
    cheek_yaw_deg = sgn * 10.0
    if m_slope.sum() >= 15:
        slope = float(np.polyfit(xyz[m_slope, 0] - surf_center[0], xyz[m_slope, 1], 1)[0])
        raw_deg = float(np.degrees(np.arctan(slope)))
        if raw_deg * sgn > 0:
            cheek_yaw_deg = float(np.clip(raw_deg * 0.65, -16.0, 16.0))

    r_eye = float(prop["eyeball_radius"])
    excl = [float(v) for v in prop["socket_exclusion_radius"]]
    d_xy = np.hypot(xyz[:, 0] - surf_center[0], xyz[:, 2] - surf_center[2])
    m_sclera = (
        (d_xy <= max(r_eye * 1.15, 0.010))
        & (xyz[:, 1] <= surf_center[1] + r_eye * 0.6)
        & (xyz[:, 1] >= surf_center[1] - r_eye * 0.6)
        & (lum >= 0.78)
        & (sat < 0.15)
    )
    sclera_rgb = (
        np.median(rgb[m_sclera], axis=0)
        if m_sclera.sum() >= 5
        else np.array([0.95, 0.93, 0.91], dtype=np.float32)
    )

    rad = float(np.radians(cheek_yaw_deg))
    nx, ny = float(np.sin(rad)), -float(np.cos(rad))
    eye_cz = float(surf_center[2])
    ctr = [float(surf_center[0] - nx * r_eye), float(surf_center[1] - ny * r_eye), eye_cz]
    return {
        "detected_iris_points": int(len(pts)),
        "surface_center": surf_center.tolist(),
        "iris_span": (p95 - p05).tolist(),
        "eyeball_center": ctr,
        "eyeball_radius": r_eye,
        "cheek_yaw_deg": float(cheek_yaw_deg),
        "cheek_pitch_deg": 0.0,
        "carve_center": [float(surf_center[0]), float(surf_center[1] + r_eye * 0.10), eye_cz],
        "exclusion_radius": excl,
        "sclera_sample_count": int(m_sclera.sum()),
        "sclera_rgb": [float(v) for v in sclera_rgb],
    }


def extract_eye_style_from_2d(
    lm_path: Path | None,
    lm: dict | None,
    xyz: np.ndarray,
    rgb: np.ndarray,
    skin: np.ndarray,
    eye_L: dict,
    eye_R: dict,
) -> dict:
    """Extract character-specific eyelash color, span, thickness & arch directly from 2D design art.

    Calibrates 2D RGB into 3DGS lighting space using the 3D/2D cheek skin luminance/channel ratio.
    Falls back cleanly to 3D body.ply upper-socket dark splats when no 2D image is available.
    """
    scL = np.asarray(eye_L["surface_center"], dtype=np.float32)
    scR = np.asarray(eye_R["surface_center"], dtype=np.float32)
    d_eye_3d = max(float(abs(scL[0] - scR[0])), 0.025)
    r_excl = np.asarray(eye_L["exclusion_radius"], dtype=np.float32)
    mid_x = 0.5 * float(scL[0] + scR[0])
    mid_y = 0.5 * float(scL[1] + scR[1])
    mid_z = 0.5 * float(scL[2] + scR[2])

    # 3D cheek skin reference right below both eyes
    m_ch3d = (
        skin
        & (np.abs(xyz[:, 0] - mid_x) < d_eye_3d * 0.90)
        & (xyz[:, 2] >= mid_z - d_eye_3d * 0.55)
        & (xyz[:, 2] <= mid_z - d_eye_3d * 0.18)
        & (xyz[:, 1] <= mid_y + 0.025)
    )
    skin_rgb_3d = (
        np.median(rgb[m_ch3d], axis=0).astype(np.float32)
        if m_ch3d.sum() >= 10
        else np.array([0.92, 0.80, 0.72], dtype=np.float32)
    )

    # 3D fallback lash color from original upper-socket dark splats
    lum3 = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    skin_lum_3d = float(0.299 * skin_rgb_3d[0] + 0.587 * skin_rgb_3d[1] + 0.114 * skin_rgb_3d[2])
    m_l3 = np.zeros(len(xyz), dtype=bool)
    for sc in (scL, scR):
        m_l3 |= (
            (np.abs(xyz[:, 0] - sc[0]) < r_excl[0] * 1.15)
            & (xyz[:, 2] >= sc[2] + r_excl[2] * 0.40)
            & (xyz[:, 2] <= sc[2] + r_excl[2] * 1.40)
            & (xyz[:, 1] < sc[1] + 0.015)
            & (lum3 < skin_lum_3d * 0.55)
        )
    fallback_lash_rgb = (
        np.percentile(rgb[m_l3], 25, axis=0).astype(np.float32)
        if m_l3.sum() >= 8
        else np.array([0.28, 0.16, 0.14], dtype=np.float32)
    )
    style = {
        "source": "3d_fallback",
        "lash_rgb": [round(float(v), 4) for v in np.clip(fallback_lash_rgb, 0.06, 0.70)],
        "skin_rgb_3d": [round(float(v), 4) for v in skin_rgb_3d],
        "lash_span_over_deye": 0.44,
        "lash_thick_over_deye": 0.036,
        "lash_arch_ratio": 0.34,
        "d_eye_3d": round(float(d_eye_3d), 5),
    }

    if lm is None or "marks" not in lm:
        return style
    marks = lm["marks"]
    if "eye_l" not in marks or "eye_r" not in marks:
        return style

    # Locate the 2D reference image referenced by landmarks.json (or sibling front_rgba.png / front_raw.png)
    img_candidates = []
    if lm.get("source"):
        p_src = Path(lm["source"])
        img_candidates.append(p_src)
        if lm_path is not None:
            img_candidates.append(lm_path.parent / p_src.name)
    if lm_path is not None:
        img_candidates.extend([lm_path.parent / "front_rgba.png", lm_path.parent / "front_raw.png"])
    img_path = next((p for p in img_candidates if p.exists()), None)
    if img_path is None:
        return style

    try:
        from PIL import Image
        im = np.asarray(Image.open(img_path).convert("RGBA"), dtype=np.float32) / 255.0
    except Exception:
        return style

    H, W = im.shape[:2]
    im_rgb = im[..., :3]
    im_a = im[..., 3]
    im_lum = 0.299 * im_rgb[..., 0] + 0.587 * im_rgb[..., 1] + 0.114 * im_rgb[..., 2]

    el_x, el_y = float(marks["eye_l"]["x_px"]), float(marks["eye_l"]["y_px"])
    er_x, er_y = float(marks["eye_r"]["x_px"]), float(marks["eye_r"]["y_px"])
    d_eye_px = max(float(np.hypot(er_x - el_x, er_y - el_y)), 12.0)

    # Sample 2D cheek skin directly below both eyes
    cheek_px = []
    for ex, ey in ((el_x, el_y), (er_x, er_y)):
        x0, x1 = max(0, int(ex - 0.24 * d_eye_px)), min(W, int(ex + 0.24 * d_eye_px))
        y0, y1 = max(0, int(ey + 0.22 * d_eye_px)), min(H, int(ey + 0.42 * d_eye_px))
        if x1 > x0 and y1 > y0:
            patch = im_rgb[y0:y1, x0:x1]
            pa = im_a[y0:y1, x0:x1]
            plum = im_lum[y0:y1, x0:x1]
            ok = (pa > 0.8) & (plum > 0.55)
            if ok.any():
                cheek_px.append(patch[ok])
    if not cheek_px:
        return style
    skin_rgb_2d = np.median(np.concatenate(cheek_px, axis=0), axis=0).astype(np.float32)
    skin_lum_2d = float(0.299 * skin_rgb_2d[0] + 0.587 * skin_rgb_2d[1] + 0.114 * skin_rgb_2d[2])
    gain = np.clip(skin_rgb_3d / np.maximum(skin_rgb_2d, 0.05), 0.72, 1.32)

    lash_colors, span_ratios, thick_ratios, arch_ratios = [], [], [], []
    for ex, ey in ((el_x, el_y), (er_x, er_y)):
        x0, x1 = max(0, int(ex - 0.34 * d_eye_px)), min(W, int(ex + 0.34 * d_eye_px))
        y0, y1 = max(0, int(ey - 0.26 * d_eye_px)), min(H, int(ey - 0.02 * d_eye_px))
        if x1 <= x0 + 3 or y1 <= y0 + 2:
            continue
        p_rgb = im_rgb[y0:y1, x0:x1]
        p_lum = im_lum[y0:y1, x0:x1]
        p_a = im_a[y0:y1, x0:x1]
        # Dark upper-lash pixels (strictly darker than 52% of character's own 2D cheek skin luminance)
        dark = (p_a > 0.8) & (p_lum < skin_lum_2d * 0.52)
        if dark.sum() < 6:
            continue
        # Restrict to the darkest 40% of dark pixels to avoid lighter bangs above the eye
        thr = float(np.percentile(p_lum[dark], 45))
        lash_m = dark & (p_lum <= max(thr, skin_lum_2d * 0.32))
        if lash_m.sum() < 5:
            lash_m = dark
        # Sample lash color from the central 60% horizontally so hair edges at the sides do not bias color
        cw = x1 - x0
        cx0, cx1 = int(cw * 0.20), int(cw * 0.80)
        center_m = lash_m[:, cx0:cx1]
        if center_m.sum() >= 4:
            lash_colors.append(np.percentile(p_rgb[:, cx0:cx1][center_m], 35, axis=0))
        else:
            lash_colors.append(np.percentile(p_rgb[lash_m], 35, axis=0))

        # Horizontal span of the upper lash band (using lower 65% of the upper-eye window to avoid bangs)
        ys_idx, xs_idx = np.nonzero(lash_m)
        span_px = float(np.percentile(xs_idx, 94) - np.percentile(xs_idx, 6))
        span_ratios.append(span_px / d_eye_px)

        # Vertical thickness in the central columns
        col_thicks = []
        col_y_mid = []
        for c_i in range(cw):
            ys_c = np.nonzero(lash_m[:, c_i])[0]
            if len(ys_c) > 0:
                if cx0 <= c_i < cx1:
                    col_thicks.append(float(len(ys_c)))
                col_y_mid.append((c_i - 0.5 * cw, float(np.median(ys_c))))
        if col_thicks:
            thick_ratios.append(float(np.median(col_thicks)) / d_eye_px)
        if len(col_y_mid) >= 6:
            arr_cm = np.asarray(col_y_mid, dtype=np.float32)
            mid_mask = np.abs(arr_cm[:, 0]) < 0.12 * d_eye_px
            flank_mask = (np.abs(arr_cm[:, 0]) >= 0.14 * d_eye_px) & (np.abs(arr_cm[:, 0]) <= 0.26 * d_eye_px)
            if mid_mask.any() and flank_mask.any():
                # In image Y-down coords, flanks are LOWER (larger y) than center arch top
                drop_px = float(np.median(arr_cm[flank_mask, 1]) - np.median(arr_cm[mid_mask, 1]))
                arch_ratios.append(drop_px / max(0.24 * d_eye_px, 4.0))

    if lash_colors:
        lash_2d = np.median(np.asarray(lash_colors, dtype=np.float32), axis=0)
        lash_3d_cal = np.clip(lash_2d * gain, 0.06, 0.65)
        # Blend 70% calibrated 2D design-art lash color + 30% 3D body lash color for seamless 3DGS integration
        lash_final = np.clip(0.70 * lash_3d_cal + 0.30 * fallback_lash_rgb, 0.06, 0.65)
        style["source"] = str(img_path)
        style["lash_rgb"] = [round(float(v), 4) for v in lash_final]
        style["lash_rgb_2d_raw"] = [round(float(v), 4) for v in lash_2d]
        if span_ratios:
            style["lash_span_over_deye"] = round(float(np.clip(np.median(span_ratios), 0.30, 0.52)), 4)
        if thick_ratios:
            style["lash_thick_over_deye"] = round(float(np.clip(np.median(thick_ratios), 0.022, 0.055)), 4)
        if arch_ratios:
            style["lash_arch_ratio"] = round(float(np.clip(np.median(arch_ratios), 0.18, 0.50)), 4)

    # Also extract 2D iris vertical gradient (top dark -> mid -> bright bottom glow) & iris aspect ratio
    iris_tops, iris_mids, iris_bots, iris_glows = [], [], [], []
    iris_w_ratios, iris_h_ratios = [], []
    for ex, ey in ((el_x, el_y), (er_x, er_y)):
        x0, x1 = max(0, int(ex - 0.18 * d_eye_px)), min(W, int(ex + 0.18 * d_eye_px))
        y0, y1 = max(0, int(ey - 0.16 * d_eye_px)), min(H, int(ey + 0.16 * d_eye_px))
        if x1 <= x0 + 4 or y1 <= y0 + 4:
            continue
        p_rgb = im_rgb[y0:y1, x0:x1]
        p_lum = im_lum[y0:y1, x0:x1]
        p_a = im_a[y0:y1, x0:x1]
        p_sat = p_rgb.max(axis=-1) - p_rgb.min(axis=-1)
        d_skin_2d = np.linalg.norm(p_rgb - skin_rgb_2d[None, None, :], axis=-1)
        # Strictly iris pixels: distinct from cheek skin, darker than white sclera, with real chromatic saturation
        m_ir = (p_a > 0.8) & (d_skin_2d > 0.24) & (p_sat > 0.18) & (p_lum > 0.08) & (p_lum < 0.78)
        if m_ir.sum() < 10:
            continue
        ys_i, xs_i = np.nonzero(m_ir)
        y_min_i, y_max_i = float(np.percentile(ys_i, 5)), float(np.percentile(ys_i, 95))
        x_min_i, x_max_i = float(np.percentile(xs_i, 5)), float(np.percentile(xs_i, 95))
        h_i = max(y_max_i - y_min_i, 3.0)
        w_i = max(x_max_i - x_min_i, 3.0)
        iris_w_ratios.append(w_i / d_eye_px)
        iris_h_ratios.append(h_i / d_eye_px)

        # Split iris vertically into top third (dark), mid third, bottom third (bright warm glow)
        yy_grid = np.broadcast_to(np.arange(y1 - y0, dtype=np.float32)[:, None], p_lum.shape)
        m_top = m_ir & (yy_grid <= y_min_i + 0.35 * h_i)
        m_mid = m_ir & (yy_grid > y_min_i + 0.32 * h_i) & (yy_grid <= y_min_i + 0.68 * h_i)
        m_bot = m_ir & (yy_grid > y_min_i + 0.65 * h_i)
        if m_top.sum() >= 3:
            iris_tops.append(np.percentile(p_rgb[m_top], 35, axis=0))
        if m_mid.sum() >= 3:
            iris_mids.append(np.median(p_rgb[m_mid], axis=0))
        if m_bot.sum() >= 3:
            iris_bots.append(np.percentile(p_rgb[m_bot], 60, axis=0))
            iris_glows.append(np.percentile(p_rgb[m_bot], 85, axis=0))

    if iris_mids and iris_bots:
        t_c = np.clip(np.median(iris_tops, axis=0) * gain, 0.05, 0.85) if iris_tops else np.clip(np.median(iris_mids, axis=0) * 0.45 * gain, 0.05, 0.85)
        m_c = np.clip(np.median(iris_mids, axis=0) * gain, 0.08, 0.90)
        b_c = np.clip(np.median(iris_bots, axis=0) * gain, 0.12, 0.98)
        g_c = np.clip(np.median(iris_glows, axis=0) * gain * 1.08, 0.18, 1.00)
        style["iris_top_rgb"] = [round(float(v), 4) for v in t_c]
        style["iris_mid_rgb"] = [round(float(v), 4) for v in m_c]
        style["iris_bottom_rgb"] = [round(float(v), 4) for v in b_c]
        style["iris_glow_rgb"] = [round(float(v), 4) for v in g_c]
        if iris_w_ratios and iris_h_ratios:
            r_ball_est = float(eye_L.get("eyeball_radius", d_eye_3d * 0.26))
            rx_world = 0.5 * float(np.median(iris_w_ratios)) * d_eye_3d
            rz_world = 0.5 * float(np.median(iris_h_ratios)) * d_eye_3d
            # Ensure vertical iris radius >= horizontal so anime eyes stay softly oval/tall (never squished wide!)
            rx_deg = float(np.clip(np.degrees(np.arcsin(np.clip(rx_world / max(r_ball_est, 1e-4), 0.35, 0.85))) * 1.05, 42.0, 52.0))
            rz_deg = float(np.clip(np.degrees(np.arcsin(np.clip(rz_world / max(r_ball_est, 1e-4), 0.40, 0.90))) * 1.12, max(rx_deg + 2.0, 48.0), 58.0))
            style["iris_rx_rz_deg"] = [round(rx_deg, 1), round(rz_deg, 1)]

    if lash_colors:
        print(
            f"[EyeStyle2D] lash_rgb={style['lash_rgb']} span/D_eye={style['lash_span_over_deye']:.3f} "
            f"thick/D_eye={style['lash_thick_over_deye']:.3f} arch={style['lash_arch_ratio']:.2f} "
            f"iris_deg={style.get('iris_rx_rz_deg')} iris_bot={style.get('iris_bottom_rgb')} iris_glow={style.get('iris_glow_rgb')}"
        )
    return style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--landmarks", default=None, help="Optional 01_input/landmarks.json for 1:1 eye height & D_eye anchoring")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])
    prop = cfg["proportions"]
    src = Path(args.src) if args.src else (work_dir / "outputs" / "step1_normalized.ply")
    dst = Path(args.dst) if args.dst else (work_dir / "outputs" / "step2_skeleton.json")
    lm_path = Path(args.landmarks) if args.landmarks else (work_dir / "01_input" / "landmarks.json")

    d, _ = read_ply(src)
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
    fdc = np.column_stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    skin = compute_skin_mask(xyz, rgb, d["opacity"])

    face_cx_est, w_skin_half, y_nose = estimate_face_cx_and_skin_width(xyz, rgb, skin, prop)
    eye_L = detect_eye(xyz, rgb, skin, "eye_L", prop, face_cx_est, w_skin_half, y_nose)
    eye_R = detect_eye(xyz, rgb, skin, "eye_R", prop, face_cx_est, w_skin_half, y_nose)

    # Cross-check & anchor eye Z (via chin-to-top head ratio) and horizontal D_eye against 01_input/landmarks.json
    lm_data = None
    if lm_path.exists():
        try:
            lm = json.loads(lm_path.read_text(encoding="utf-8"))
            lm_data = lm
            marks = lm.get("marks", {})
            if "eye_l" in marks and "eye_r" in marks:
                op_m = d["opacity"] > 0.0
                z_top_3d = float(np.percentile(xyz[op_m, 2], 99.8))
                z_bot_3d = float(np.percentile(xyz[op_m, 2], 0.2))
                h_3d = z_top_3d - z_bot_3d
                d_eye_2d = float(lm.get("D_eye", 0.0)) * h_3d
                d_eye_3d = abs(eye_L["surface_center"][0] - eye_R["surface_center"][0])
                if d_eye_2d > 0.02 and abs(d_eye_3d - d_eye_2d) > 0.004:
                    target_half = min(0.5 * d_eye_2d, w_skin_half * 0.72)
                    fcx_cur = 0.5 * (eye_L["surface_center"][0] + eye_R["surface_center"][0])
                    print(f"[LandmarkAnchor] D_eye drift {(d_eye_3d-d_eye_2d)*1000:+.1f}mm (3D={d_eye_3d:.4f} vs 2D={d_eye_2d:.4f}); anchoring half-span={target_half:.4f}")
                    eye_L["surface_center"][0] = fcx_cur + target_half
                    eye_R["surface_center"][0] = fcx_cur - target_half

                # Anchor Z using head-internal ratio (chin -> top) so bangs/upper-lash NEVER pull eyes upward!
                if "chin" in marks:
                    fz0_c = float(prop["face_z_range"][0])
                    fz1_c = float(prop["face_z_range"][1])
                    fs_c = skin & (xyz[:, 2] >= fz0_c - 0.01) & (xyz[:, 2] <= fz1_c) & (np.abs(xyz[:, 0] - face_cx_est) < 0.05) & (xyz[:, 1] <= y_nose + 0.022)
                    if fs_c.sum() >= 25:
                        z_chin_3d = float(np.percentile(xyz[fs_c, 2], 1.0))
                        t_chin_2d = float(marks["chin"]["t"])
                        t_eye_2d = 0.5 * (float(marks["eye_l"]["t"]) + float(marks["eye_r"]["t"]))
                        frac_in_head = (t_eye_2d - t_chin_2d) / max(1.0 - t_chin_2d, 0.05)
                        z_eye_head = z_chin_3d + frac_in_head * (z_top_3d - z_chin_3d)
                        z_eye_3d = 0.5 * (eye_L["surface_center"][2] + eye_R["surface_center"][2])
                        if abs(z_eye_3d - z_eye_head) > 0.002:
                            print(f"[LandmarkAnchor] Eye Z drift {(z_eye_3d-z_eye_head)*1000:+.1f}mm (3D={z_eye_3d:.4f} vs head-ratio={z_eye_head:.4f}); anchoring Z")
                            eye_L["surface_center"][2] = z_eye_head
                            eye_R["surface_center"][2] = z_eye_head
                else:
                    z_sym = 0.5 * (eye_L["surface_center"][2] + eye_R["surface_center"][2])
                    eye_L["surface_center"][2] = z_sym
                    eye_R["surface_center"][2] = z_sym
        except Exception as exc:
            print(f"[LandmarkAnchor] Skipped landmarks.json ({exc})")

    # Fit each eye to its own robust local quadratic face surface so eyeball/eyelid/lashes sit flush
    face_cx = 0.5 * (eye_L["surface_center"][0] + eye_R["surface_center"][0])
    recess_frac = float(prop.get("eyeball_recess_frac", 0.06))

    for name, eye in (("eye_L", eye_L), ("eye_R", eye_R)):
        surface = eye["surface_center"]
        detected_y = float(surface[1])
        radii = np.asarray(eye["exclusion_radius"], dtype=np.float32)
        fit = fit_eye_surface(xyz, skin, np.asarray(surface, dtype=np.float32), radii, face_cx=face_cx_est)
        if fit is None:
            eye["depth_fit"] = {"detected_y": detected_y, "skin_surface_y": None}
            print(f"{name}: insufficient local skin support; review detected depth before carving")
            continue

        fitted_y = fit["skin_surface_y"]
        r_eye = float(eye["eyeball_radius"])
        recess = r_eye * recess_frac

        surface[1] = fitted_y
        # Keep eyeball center X & Z strictly locked to surface_center[0, 2] so the front iris pole
        # sits dead-center on the eye opening (never shifted inward/upward by steep cheek normals!).
        ctr = np.array([float(surface[0]), float(fitted_y + r_eye + recess), float(surface[2])], dtype=np.float64)
        eye["eyeball_center"] = [float(ctr[0]), float(ctr[1]), float(ctr[2])]
        eye["cheek_yaw_deg"] = fit["cheek_yaw_deg"]
        eye["cheek_pitch_deg"] = fit["cheek_pitch_deg"]
        eye["face_quad_coef"] = fit["quad_coef"]
        eye["carve_center"] = [float(surface[0]), float(fitted_y + r_eye * 0.12), float(surface[2])]

        # Measure rim flush error (distance between eyeball front cap and fitted face surface at 4 rim points)
        c0, c1, c2, c3, c4, c5 = fit["quad_coef"]
        rim_errs = []
        for dx_r, dz_r in ((radii[0] * 0.75, 0.0), (-radii[0] * 0.75, 0.0), (0.0, radii[2] * 0.75), (0.0, -radii[2] * 0.75)):
            y_face_r = c0 + c1 * dx_r + c2 * dz_r + c3 * dx_r * dx_r + c4 * dx_r * dz_r + c5 * dz_r * dz_r
            rad_sq = r_eye * r_eye - (surface[0] + dx_r - ctr[0]) ** 2 - (surface[2] + dz_r - ctr[2]) ** 2
            if rad_sq > 0:
                y_eye_r = ctr[1] - float(np.sqrt(rad_sq))
                rim_errs.append(abs(y_eye_r - y_face_r) * 1000.0)
        rim_flush_err_mm = round(float(max(rim_errs)) if rim_errs else 0.0, 3)

        eye["depth_fit"] = {
            "detected_y": detected_y,
            "skin_surface_y": fitted_y,
            "normal": fit["normal"],
            "fit_rmse_mm": fit["fit_rmse_mm"],
            "rim_flush_err_mm": rim_flush_err_mm,
            "sample_cells": fit["sample_cells"],
        }
        warn = ""
        if fit["fit_rmse_mm"] > 1.5 or rim_flush_err_mm > 2.0:
            warn = " [WARNING: review +60/+90 deg flushness!]"
        print(
            f"{name}: surf=({surface[0]:.4f},{fitted_y:.4f},{surface[2]:.4f}) "
            f"yaw={fit['cheek_yaw_deg']:+.1f}deg pitch={fit['cheek_pitch_deg']:+.1f}deg "
            f"rmse={fit['fit_rmse_mm']:.2f}mm rim_err={rim_flush_err_mm:.2f}mm{warn}"
        )

    # Gently symmetrize L/R eyeball depth when one side's local skin fit drifts > 2 mm,
    # WITHOUT shifting face_quad_coef[0] so each eyelid/eyelash stays flush with its OWN real cheek surface!
    if eye_L["depth_fit"].get("skin_surface_y") is not None and eye_R["depth_fit"].get("skin_surface_y") is not None:
        yL, yR = float(eye_L["surface_center"][1]), float(eye_R["surface_center"][1])
        if abs(yL - yR) > 0.0020:
            y_avg = 0.5 * (yL + yR)
            for eye in (eye_L, eye_R):
                dy_shift = y_avg - float(eye["surface_center"][1])
                eye["eyeball_center"][1] = float(eye["eyeball_center"][1]) + dy_shift
                eye["carve_center"][1] = float(eye["carve_center"][1]) + dy_shift
            print(f"[SymDepth] Aligned L/R eyeball Y ({yL:.4f}, {yR:.4f}) -> {y_avg:.4f} (kept per-eye face_quad_coef flush)")

    eye_style = extract_eye_style_from_2d(lm_path if lm_path.exists() else None, lm_data, xyz, rgb, skin, eye_L, eye_R)

    fz0, fz1 = float(prop["face_z_range"][0]), float(prop["face_z_range"][1])
    neck_z0 = float(prop["neck_z_range"][0])
    m_torso_y = (
        (xyz[:, 2] >= neck_z0 - 0.10)
        & (xyz[:, 2] <= neck_z0 + 0.05)
        & (np.abs(xyz[:, 0] - face_cx) < 0.075)
        & (xyz[:, 1] < 0.05)
    )
    head_y = (
        float(0.5 * (np.percentile(xyz[m_torso_y, 1], 10) + np.percentile(xyz[m_torso_y, 1], 90)))
        if m_torso_y.sum() >= 20
        else float(np.median(xyz[(xyz[:, 2] >= fz0) & (xyz[:, 2] <= fz1), 1]))
    )
    id_q = [1.0, 0.0, 0.0, 0.0]

    skeleton = {
        "convention": {
            "world": "Z-up, -Y-forward, +X-character-left",
            "bone_local_axes": "Unified across all bones: +X=left, -Y=forward(gaze), +Z=up (rest quat=[1,0,0,0])",
        },
        "eye_style": eye_style,
        "bones": {
            "root": {"parent": None, "head": [face_cx, 0.0, 0.0], "tail": [face_cx, 0.0, 0.12], "rest_quat_wxyz": id_q},
            "spine": {"parent": "root", "head": [face_cx, head_y, neck_z0 * 0.65], "tail": [face_cx, head_y, neck_z0], "rest_quat_wxyz": id_q},
            "head": {"parent": "spine", "head": [face_cx, head_y, neck_z0 + 0.02], "tail": [face_cx, head_y, 0.90], "rest_quat_wxyz": id_q},
            "eye_L": {"parent": "head", "head": eye_L["eyeball_center"], "tail": [eye_L["eyeball_center"][0], eye_L["eyeball_center"][1] - eye_L["eyeball_radius"], eye_L["eyeball_center"][2]], "rest_quat_wxyz": id_q, "meta": eye_L},
            "eye_R": {"parent": "head", "head": eye_R["eyeball_center"], "tail": [eye_R["eyeball_center"][0], eye_R["eyeball_center"][1] - eye_R["eyeball_radius"], eye_R["eyeball_center"][2]], "rest_quat_wxyz": id_q, "meta": eye_R},
        },
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(skeleton, indent=2), encoding="utf-8")
    print(f"Saved skeleton -> {dst}")


if __name__ == "__main__":
    main()
