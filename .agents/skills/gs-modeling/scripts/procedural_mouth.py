"""Reusable Character-Agnostic Procedural 3DGS Dynamic Mouth & Lip-Sync Module.

Provides:
- detect_mouth_landmarks: data-driven detection of mouth center, width, local face curvature,
  lip line color, and cheek/chin skin color from a normalized head PLY.
- carve_mouth_pocket: cleanly neutralizes the original blurry TripoSplat mouth line and carves
  a tight recessed pocket behind the aperture so body_base.ply stays static on GPU.
- MOUTH_PRESETS: standard anime lip-sync shapes ('closed', 'A', 'I', 'U', 'E', 'O', 'smile').
- build_procedural_mouth: builds silky-smooth tapered anime lip lines, skin-matched dynamic
  closing veil, 3D recessed oral cavity, subtle upper teeth, and tongue splats.
"""
import math
import numpy as np
from harmonic_dqs import SH_C0

# Standard anime lip-sync & expression presets: (open_amt, width_scale, smile, roundness)
# roundness: 0.0 = wide/flat (I/E), 0.5 = natural (A), 1.0 = soft rounded oval (U/O)
MOUTH_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "closed": (0.00, 0.86, 0.30, 0.30),
    "smile":  (0.00, 1.00, 0.62, 0.30),
    "A":      (0.82, 0.86, 0.10, 0.65),
    "I":      (0.28, 1.04, 0.32, 0.25),
    "U":      (0.42, 0.44, 0.00, 1.00),
    "E":      (0.50, 0.94, 0.20, 0.45),
    "O":      (0.68, 0.52, 0.00, 1.00),
}


def _make_splats(xyz, rgb, normals, scales, dtype, opacity_logit=4.2):
    n = len(xyz)
    out = np.zeros(n, dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    f_dc = (np.clip(rgb, 0.0, 1.0) - 0.5) / SH_C0
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    out["opacity"] = float(opacity_logit)
    out["scale_0"] = math.log(max(scales[0], 1e-5))
    out["scale_1"] = math.log(max(scales[1], 1e-5))
    out["scale_2"] = math.log(max(scales[2], 1e-5))
    # Orient local +Z along normal
    z = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    up = np.zeros_like(z)
    up[:, 2] = 1.0
    alt = np.abs(z[:, 2]) > 0.92
    up[alt] = np.array([1.0, 0.0, 0.0], np.float32)
    x = np.cross(up, z)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    y = np.cross(z, x)
    tr = x[:, 0] + y[:, 1] + z[:, 2]
    qw = np.sqrt(np.maximum(1e-8, 1.0 + tr)) * 0.5
    qx = (y[:, 2] - z[:, 1]) / (4.0 * qw)
    qy = (z[:, 0] - x[:, 2]) / (4.0 * qw)
    qz = (x[:, 1] - y[:, 0]) / (4.0 * qw)
    q = np.column_stack([qw, qx, qy, qz]).astype(np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return out


def detect_mouth_landmarks(
    head_ply: np.ndarray,
    face_cx: float,
    z_eye: float,
    sy: float,
    D_eye: float,
    r_eye: float,
) -> dict:
    """Detect mouth center, width, local face curvature, skin RGB, and lip line RGB."""
    xyz = np.column_stack([head_ply["x"], head_ply["y"], head_ply["z"]]).astype(np.float32)
    fdc = np.column_stack([head_ply["f_dc_0"], head_ply["f_dc_1"], head_ply["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    op = 1.0 / (1.0 + np.exp(-np.clip(head_ply["opacity"], -20, 20)))
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

    # 1. Sample local lower-face skin color & luminance first (data-driven -> works for any skin tone!)
    m_cheek = (
        (op > 0.30)
        & (np.abs(xyz[:, 0] - face_cx) < 0.55 * D_eye)
        & (xyz[:, 2] > z_eye - 1.05 * D_eye)
        & (xyz[:, 2] < z_eye - 0.35 * D_eye)
        & (xyz[:, 1] < sy + 0.25 * r_eye)
    )
    if int(m_cheek.sum()) >= 30:
        # Exclude darkest 15% (mouth/nostril lines) and brightest 10% (specular) to estimate true skin tone
        c_lum = lum[m_cheek]
        lo_l, hi_l = float(np.percentile(c_lum, 20)), float(np.percentile(c_lum, 90))
        m_sk_core = m_cheek & (lum >= lo_l) & (lum <= hi_l)
        skin_rgb = np.median(rgb[m_sk_core], axis=0).astype(np.float32) if int(m_sk_core.sum()) >= 10 else np.median(rgb[m_cheek], axis=0).astype(np.float32)
    else:
        skin_rgb = np.array([0.93, 0.85, 0.80], np.float32)
    skin_lum = float(0.299 * skin_rgb[0] + 0.587 * skin_rgb[1] + 0.114 * skin_rgb[2])

    # 2. Search lower-face zone strictly below nose tip (z < z_eye - 0.52 * D_eye) using relative luminance
    m_zone = (
        (op > 0.30)
        & (np.abs(xyz[:, 0] - face_cx) < 0.36 * D_eye)
        & (xyz[:, 2] > z_eye - 1.08 * D_eye)
        & (xyz[:, 2] < z_eye - 0.52 * D_eye)
        & (xyz[:, 1] < sy + 0.25 * r_eye)
    )
    m_dark = m_zone & (lum < skin_lum * 0.78)

    if int(m_dark.sum()) >= 6:
        p = xyz[m_dark]
        mx = float(np.median(p[:, 0]))
        mz = float(np.median(p[:, 2]))
        half_w = 0.5 * float(np.percentile(p[:, 0], 95) - np.percentile(p[:, 0], 5))
        rx_mouth = float(np.clip(half_w * 1.12, 0.145 * D_eye, 0.23 * D_eye))
        lip_rgb = np.clip(np.median(rgb[m_dark], axis=0) * 0.72, 0.10, 0.50).astype(np.float32)
    else:
        mx = float(face_cx)
        mz = float(z_eye - 0.74 * D_eye)
        rx_mouth = float(0.165 * D_eye)
        lip_rgb = np.clip(skin_rgb * 0.42, 0.10, 0.45).astype(np.float32)

    # Clamp mz safely inside measured chin-to-eye span (prevents chibi mouth dropping below chin into neck)
    m_chin_skin = (
        (op > 0.30)
        & (np.abs(xyz[:, 0] - face_cx) < 0.45 * D_eye)
        & (xyz[:, 2] > z_eye - 1.15 * D_eye) & (xyz[:, 2] < z_eye - 0.12 * D_eye)
        & (xyz[:, 1] < sy + 0.65 * r_eye)
        & (lum >= skin_lum * 0.82) & (rgb[:, 0] > rgb[:, 2] + 0.03)
    )
    if int(m_chin_skin.sum()) >= 20:
        z_chin = float(np.percentile(xyz[m_chin_skin, 2], 4))
        mz_lo = z_chin + 0.22 * max(z_eye - z_chin, 0.01)
        mz_hi = z_eye - 0.32 * D_eye
        if mz_lo < mz_hi:
            mz = float(np.clip(mz, mz_lo, mz_hi))

    # Refine skin color right around detected mouth center using relative brightness
    m_skin = (
        (op > 0.30)
        & (np.abs(xyz[:, 0] - mx) < 0.55 * D_eye)
        & (np.abs(xyz[:, 2] - mz) < 0.28 * D_eye)
        & (xyz[:, 1] < sy + 0.65 * r_eye)
        & (lum >= skin_lum * 0.86)
    )
    if int(m_skin.sum()) >= 20:
        skin_rgb = np.median(rgb[m_skin], axis=0).astype(np.float32)
        skin_lum = float(0.299 * skin_rgb[0] + 0.587 * skin_rgb[1] + 0.114 * skin_rgb[2])

    # Fit frontmost skin surface y_surf(dx, dz) = y0 + ky * dx^2 + kz * dz
    # Use the actual frontmost envelope (p15) so the mouth sits 100% flush on the 3D lip surface (never recessed/floating)
    m_surf = (
        (op > 0.35)
        & (np.abs(xyz[:, 0] - mx) < 0.34 * D_eye)
        & (np.abs(xyz[:, 2] - mz) < 0.18 * D_eye)
        & (xyz[:, 1] < sy + 0.65 * r_eye)
    )
    if int(m_surf.sum()) >= 20:
        ps = xyz[m_surf]
        y_cut = float(np.percentile(ps[:, 1], 18))
        pf = ps[ps[:, 1] <= y_cut]
        if len(pf) < 10:
            pf = ps
        dx = pf[:, 0] - mx
        dz = pf[:, 2] - mz
        A = np.column_stack([np.ones_like(dx), dx * dx, dz])
        coef, *_ = np.linalg.lstsq(A, pf[:, 1], rcond=None)
        y0 = float(coef[0]) - 0.0012
        ky = float(np.clip(coef[1], 10.0, 120.0))
        kz = float(np.clip(coef[2], -0.45, 0.45))
    else:
        y0 = float(sy - 0.95 * r_eye)
        ky = 42.0
        kz = -0.10

    rz_max = float(0.088 * D_eye)  # graceful anime max vertical half-opening (never crowds chin)
    return {
        "center": [mx, y0, mz],
        "rx": rx_mouth,
        "rz_max": rz_max,
        "ky": ky,
        "kz": kz,
        "D_eye": float(D_eye),
        "r_eye": float(r_eye),
        "skin_rgb": skin_rgb.tolist(),
        "lip_rgb": lip_rgb.tolist(),
    }


def carve_mouth_pocket(head_ply: np.ndarray, meta: dict) -> np.ndarray:
    """Cleanly neutralize original blurry mouth line and carve a tight pocket behind the aperture."""
    out = head_ply.copy()
    mx, y0, mz = meta["center"]
    rx = float(meta["rx"])
    rz_max = float(meta["rz_max"])
    ky = float(meta["ky"])
    kz = float(meta["kz"])
    skin_rgb = np.asarray(meta["skin_rgb"], np.float32)

    xyz = np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32)
    fdc = np.column_stack([out["f_dc_0"], out["f_dc_1"], out["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

    dx = xyz[:, 0] - mx
    dz = xyz[:, 2] - mz
    y_surf = y0 + ky * (dx * dx) + kz * dz
    front_shell = (xyz[:, 1] > y_surf - 0.005) & (xyz[:, 1] < y_surf + 0.0065)

    # 1. Neutralize any dark/shaded original mouth-line & ghost shadow splats in a tight rim around the mouth
    skin_lum = float(0.299 * skin_rgb[0] + 0.587 * skin_rgb[1] + 0.114 * skin_rgb[2])
    u_rim = dx / (rx * 1.35)
    v_rim = (dz - 0.15 * rz_max) / (rz_max * 1.75)
    in_rim = front_shell & ((u_rim * u_rim + v_rim * v_rim) < 1.0) & (lum < skin_lum * 0.94)
    if np.any(in_rim):
        skin_fdc = (skin_rgb - 0.5) / SH_C0
        out["f_dc_0"][in_rim] = skin_fdc[0]
        out["f_dc_1"][in_rim] = skin_fdc[1]
        out["f_dc_2"][in_rim] = skin_fdc[2]

    # 2. Carve tight interior pocket so oral cavity sits cleanly behind the opening
    u_pkt = dx / (rx * 0.94)
    v_pkt = (dz + 0.22 * rz_max) / (rz_max * 1.15)
    r2 = u_pkt * u_pkt + v_pkt * v_pkt
    in_pkt = front_shell & (r2 < 1.0)
    if np.any(in_pkt):
        out["opacity"][in_pkt] = -14.0

    return out


def _mouth_contour_z(u: np.ndarray, open_amt: float, rz_max: float, smile: float, roundness: float = 0.5):
    """Return smooth anime (dz_upper, dz_lower) relative to mz for u in [-1, 1]."""
    uc = np.clip(u, -1.0, 1.0)
    ell = np.sqrt(np.maximum(0.0, 1.0 - uc * uc))
    soft_arch = ell ** 0.78
    round_arch = ell ** 0.95
    # Smooth parabolic smile arc (no flat-center kink!)
    smile_arc = smile * (0.28 * rz_max) * (uc * uc - 0.50)
    up_h = (0.14 + 0.36 * roundness) * open_amt * rz_max
    dz_up = up_h * (soft_arch * (1.0 - roundness) + round_arch * roundness) + smile_arc
    lo_h = (1.05 - 0.12 * roundness) * open_amt * rz_max
    dz_lo = -lo_h * (soft_arch * (1.0 - 0.5 * roundness) + round_arch * (0.5 * roundness)) + smile_arc * (1.0 - 0.60 * open_amt)
    return dz_up, dz_lo


def build_procedural_mouth(
    meta: dict,
    open_amt: float,
    dtype,
    width_scale: float = 1.0,
    smile: float = 0.0,
    roundness: float = 0.5,
) -> np.ndarray:
    """Build delicate, natural anime-style procedural 3DGS mouth splats in Blender coordinates."""
    open_amt = float(np.clip(open_amt, 0.0, 1.0))
    width_scale = float(np.clip(width_scale, 0.40, 1.25))
    smile = float(np.clip(smile, -0.5, 1.0))
    roundness = float(np.clip(roundness, 0.0, 1.0))

    mx, y0, mz = meta["center"]
    rx_base = float(meta["rx"])
    rx = rx_base * width_scale
    rz_max = float(meta["rz_max"])
    ky = float(meta["ky"])
    kz = float(meta["kz"])
    skin_rgb = np.asarray(meta["skin_rgb"], np.float32)
    lip_rgb = np.asarray(meta["lip_rgb"], np.float32)
    # Crisp yet warm rose-brown line (stays clearly visible at 640px video resolution while blending with anime skin)
    lip_soft = np.clip(lip_rgb * 0.76 + skin_rgb * 0.24, 0.22, 0.58).astype(np.float32)
    s_unit = max(rx_base, 1e-4)

    parts = []

    def surf_y(dx_arr, dz_arr):
        return y0 + ky * (dx_arr * dx_arr) + kz * dz_arr

    # 1. Dynamic skin veil sealing the carved pocket outside the current opening
    ug = np.linspace(-1.14, 1.14, 38).astype(np.float32)
    vg = np.linspace(-1.45, 1.15, 32).astype(np.float32)
    U, V = np.meshgrid(ug, vg, indexing="ij")
    U, V = U.ravel(), V.ravel()
    dx_s = U * rx_base
    dz_s = V * rz_max - 0.18 * rz_max
    in_ell = ((dx_s / (rx_base * 1.12)) ** 2 + ((dz_s + 0.22 * rz_max) / (rz_max * 1.28)) ** 2) <= 1.0
    u_cur = dx_s / max(rx, 1e-5)
    dz_u_s, dz_l_s = _mouth_contour_z(u_cur, open_amt, rz_max, smile, roundness)
    in_open = (open_amt > 0.03) & (np.abs(u_cur) < 0.94) & (dz_s < dz_u_s) & (dz_s > dz_l_s)
    keep_veil = in_ell & (~in_open)
    if np.any(keep_veil):
        dx_v, dz_v = dx_s[keep_veil], dz_s[keep_veil]
        y_v = surf_y(dx_v, dz_v) + 0.00004
        v_xyz = np.column_stack([mx + dx_v, y_v, mz + dz_v]).astype(np.float32)
        v_rgb = np.tile(skin_rgb[None, :], (len(dx_v), 1))
        parts.append(_make_splats(
            v_xyz, v_rgb, np.tile([[0, -1, 0]], (len(dx_v), 1)),
            (0.080 * s_unit, 0.050 * s_unit, 0.054 * s_unit), dtype, opacity_logit=4.9,
        ))

    # 2. Soft pastel-muted anime oral cavity strictly INSIDE the tapered corners (ZERO corner bleed!)
    if open_amt > 0.05:
        uc = np.linspace(-0.88, 0.88, 34).astype(np.float32)
        vc = np.linspace(0.06, 0.94, 20).astype(np.float32)
        UC, VC = np.meshgrid(uc, vc, indexing="ij")
        UC, VC = UC.ravel(), VC.ravel()
        dz_u_c, dz_l_c = _mouth_contour_z(UC, open_amt, rz_max, smile, roundness)
        span = np.maximum(0.0, dz_u_c - dz_l_c)
        valid_c = span > (0.10 * rz_max)
        if np.any(valid_c):
            UC, VC = UC[valid_c], VC[valid_c]
            dz_c = dz_l_c[valid_c] + VC * span[valid_c]
            dx_c = UC * rx
            depth = (0.045 + 0.035 * (1.0 - UC * UC)) * s_unit
            y_c = surf_y(dx_c, dz_c) + depth
            c_xyz = np.column_stack([mx + dx_c, y_c, mz + dz_c]).astype(np.float32)

            # Soft muted coral-rose anime shading (blends naturally with pale anime skin)
            top_col = np.array([0.58, 0.34, 0.36], np.float32)
            mid_col = np.array([0.74, 0.48, 0.49], np.float32)
            t_v = np.clip(VC, 0.0, 1.0)[:, None]
            cav_col = mid_col[None, :] * (1.0 - t_v) + top_col[None, :] * t_v

            if open_amt > 0.25:
                tongue_dome = np.clip((0.46 - VC) / 0.28, 0.0, 1.0) * np.clip((0.68 - np.abs(UC)) / 0.30, 0.0, 1.0)
                tongue_dome *= np.clip((open_amt - 0.25) / 0.40, 0.0, 1.0)
                tongue_col = np.array([0.88, 0.63, 0.62], np.float32)
                cav_col = cav_col * (1.0 - tongue_dome[:, None]) + tongue_col[None, :] * tongue_dome[:, None]

            if open_amt > 0.22 and roundness < 0.65:
                tooth_w = np.clip((0.65 - roundness) / 0.35, 0.0, 1.0)
                tooth_mask = np.clip((VC - 0.76) / 0.14, 0.0, 1.0) * np.clip((0.65 - np.abs(UC)) / 0.22, 0.0, 1.0) * tooth_w
                tooth_col = np.array([0.96, 0.93, 0.91], np.float32)
                cav_col = cav_col * (1.0 - 0.72 * tooth_mask[:, None]) + tooth_col[None, :] * (0.72 * tooth_mask[:, None])

            # Shrink cavity splats smoothly near corners & when opening is narrow (zero corner bleed on 'I'!)
            ell_c = np.sqrt(np.maximum(0.0, 1.0 - UC * UC))
            sp_c = _make_splats(
                c_xyz, cav_col, np.tile([[0, -1, 0]], (len(UC), 1)),
                (0.056 * s_unit, 0.042 * s_unit, 0.042 * s_unit), dtype, opacity_logit=5.2,
            )
            sz_mod = np.clip(0.35 + 0.65 * ell_c, 0.35, 1.0) * np.clip(span[valid_c] / (0.65 * rz_max), 0.35, 1.0)
            sp_c["scale_0"] = np.log(np.maximum(1e-5, (0.056 * s_unit) * sz_mod)).astype(np.float32)
            sp_c["scale_2"] = np.log(np.maximum(1e-5, (0.042 * s_unit) * sz_mod)).astype(np.float32)
            parts.append(sp_c)

    # 3. Delicate tapered anime upper lip line (matches cavity width exactly; NO overhanging gray roof!)
    n_lip = 84
    # On open rounded vowels (U/O), keep upper lip line tucked neatly over the arch (0.86x width)
    u_lim = 0.96 - 0.12 * open_amt * roundness
    ul = np.linspace(-u_lim, u_lim, n_lip).astype(np.float32)
    dz_u_l, _ = _mouth_contour_z(ul, open_amt, rz_max, smile, roundness)
    t_u = np.clip((ul + u_lim) / max(2.0 * u_lim, 1e-5), 0.0, 1.0)
    taper = 0.12 + 0.88 * (np.maximum(0.0, np.sin(math.pi * t_u)) ** 1.10)

    dx_ul = ul * rx
    y_ul = surf_y(dx_ul, dz_u_l) - 0.00040
    ul_xyz = np.column_stack([mx + dx_ul, y_ul, mz + dz_u_l]).astype(np.float32)
    # Clearly readable dainty anime lip line even at 640px video resolution (thinner when open on rounded vowels)
    base_thick = max(0.048 * s_unit, 0.00050)
    line_thick = base_thick * (1.0 - 0.45 * open_amt * roundness)
    line_op = 4.4 - 1.0 * open_amt * roundness
    sp_ul = _make_splats(
        ul_xyz, np.tile(lip_soft[None, :], (n_lip, 1)), np.tile([[0, -1, 0]], (n_lip, 1)),
        (max(0.055 * s_unit, 0.00055), max(0.025 * s_unit, 0.00025), line_thick), dtype, opacity_logit=float(line_op),
    )
    sp_ul["scale_2"] = np.log(np.maximum(1e-5, line_thick * taper)).astype(np.float32)
    parts.append(sp_ul)

    return np.concatenate(parts, axis=0)
