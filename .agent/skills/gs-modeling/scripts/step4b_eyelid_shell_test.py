"""Step 4b: Procedural Face-Surface-Fitted Independent Eyelid Shells (data-driven lash-ceiling & natural anime arch)."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from ply_io import read_ply, write_ply
from config_loader import load_config, resolve_spark_dir

SH_C0 = 0.28209479177387814


def smoothstep(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normal_to_quat_wxyz(normals: np.ndarray) -> np.ndarray:
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    q = np.zeros((len(n), 4), dtype=np.float32)
    q[:, 0] = 1.0 + n[:, 2]
    q[:, 1] = -n[:, 1]
    q[:, 2] = n[:, 0]
    anti = n[:, 2] < -0.9999
    q[anti] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    return q


def detect_lash_ceiling_dz(xyz: np.ndarray, rgb: np.ndarray, is_body: np.ndarray, surf_c: np.ndarray, rx: float, rz: float) -> float:
    """Data-driven detection of the skin gap between the static upper eyelash and the eyebrow/bangs."""
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    dzs = np.linspace(rz * 0.45, rz * 1.45, 16)
    step = (dzs[1] - dzs[0]) * 0.85
    counts = []
    for dz in dzs:
        m = (
            is_body
            & (np.abs(xyz[:, 0] - surf_c[0]) < rx * 0.85)
            & (xyz[:, 1] < surf_c[1] + rz * 0.85)
            & (np.abs(xyz[:, 2] - (surf_c[2] + dz)) <= step)
            & (lum < 0.40)
        )
        counts.append(int(m.sum()))
    # Find the upper eyelash peak in the lower half of the scan, then the local minimum right above it
    half = max(3, len(counts) // 2)
    peak_i = int(np.argmax(counts[:half]))
    if peak_i + 2 < len(counts):
        valley_i = peak_i + 1 + int(np.argmin(counts[peak_i + 1 : min(len(counts), peak_i + 7)]))
        return float(dzs[valley_i])
    return float(rz * 1.12)


def _fit_robust_face_quad(skin_xyz: np.ndarray, cx: float, cz: float, rx: float) -> np.ndarray:
    """2-stage IRLS robust quadratic fit y(dx, dz) that rejects deep socket-wall or foreground-bangs outliers."""
    d_local = np.hypot(skin_xyz[:, 0] - cx, skin_xyz[:, 2] - cz)
    local_m = d_local < max(rx * 2.4, 0.028)
    if local_m.sum() < 16:
        local_m = d_local < max(rx * 4.2, 0.055)
    pts_fit = skin_xyz[local_m]
    dx_f, dz_f = pts_fit[:, 0] - cx, pts_fit[:, 2] - cz
    A = np.column_stack([np.ones_like(dx_f), dx_f, dz_f, dx_f * dx_f, dx_f * dz_f, dz_f * dz_f])
    w_base = np.exp(-(d_local[local_m] / max(rx * 1.5, 0.020)) ** 2)
    w = w_base.copy()
    for _ in range(4):
        coef, _, _, _ = np.linalg.lstsq(A * w[:, None], pts_fit[:, 1] * w, rcond=None)
        resid = np.abs(A @ coef - pts_fit[:, 1])
        # Huber / Tukey-like downweight for outliers > 2.5 mm off the true front skin surface
        rob = np.where(resid < 0.0025, 1.0, 0.0012 / np.maximum(resid, 1e-5))
        w = w_base * rob
    return coef


def build_eyelid_shell(
    eye_center: np.ndarray,
    surf_center: np.ndarray,
    skin_xyz: np.ndarray,
    skin_fdc: np.ndarray,
    skin_scale_med: float,
    blink: float,
    dtype: np.dtype,
    rx_rz: tuple[float, float] = (0.0385, 0.0315),
    lash_top_bot: tuple[float, float] = (0.0115, -0.0195),
    lash_rgb: tuple[float, float, float] | None = None,
    dz_top_cap: float | None = None,
    face_quad_coef: list[float] | np.ndarray | None = None,
    always_draw_lash: bool = False,
    eye_style: dict | None = None,
    eye_radius: float | None = None,
    ref_xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a robust quadratic face surface and generate a flush anime eyelid + design-matched eyelash stroke.

    - Uses 2D design-art eye_style (lash_rgb, lash_span_over_deye, lash_thick_over_deye, lash_arch_ratio)
      when available so color, width, thickness and arch match the character's design art automatically.
    - Projects every lash/eyelid point onto the actual 3D face/eyeball support surface (scale-invariant in
      skin_scale_med) and clips endpoints lacking 3D support so lashes NEVER float or protrude obliquely.
    - Skin curtain stays strictly ABOVE the moving lash edge during half-blink (sealing below only for blink > 0.80).
    """
    if blink <= 0.02 and not always_draw_lash:
        return np.zeros(0, dtype=dtype)

    cx, cy, cz = float(surf_center[0]), float(surf_center[1]), float(surf_center[2])
    rx, rz = float(rx_rz[0]), float(rx_rz[1])
    rz_up = min(rz * 1.06, float(dz_top_cap) * 1.02) if dz_top_cap is not None else rz * 1.06
    rz_dn = rz * 1.02

    if face_quad_coef is not None and len(face_quad_coef) == 6:
        coef = np.asarray(face_quad_coef, dtype=np.float64)
    else:
        coef = _fit_robust_face_quad(skin_xyz, cx, cz, rx)

    # Pull character-adaptive lash parameters from 2D design-art eye_style when present
    style = eye_style or {}
    d_eye_3d = float(style.get("d_eye_3d", rx * 3.6))
    arch_ratio = float(style.get("lash_arch_ratio", 0.36))
    ec = np.asarray(eye_center, dtype=np.float64)
    r_ball = float(eye_radius) if eye_radius is not None else float(max(abs(ec[1] - cy) * 0.94, rx * 0.72))

    # Automatic sanpaku prevention: ensure open upper lash hugs the upper iris rim (no staring white gap above iris!)
    z_top_mid = float(lash_top_bot[0])
    z_bot_mid = float(lash_top_bot[1])
    iris_deg = style.get("iris_rx_rz_deg")
    if iris_deg is not None and len(iris_deg) == 2:
        iris_top_z = r_ball * float(np.sin(np.radians(float(iris_deg[1]))))
        # Hug just slightly over the top edge of the iris (~92%) for a soft, cute, non-staring anime gaze
        z_top_mid = min(z_top_mid, iris_top_z * 0.94)
    else:
        z_top_mid = min(z_top_mid, r_ball * 0.74)

    def _lash_z(u_arr: np.ndarray) -> np.ndarray:
        # Soft dome arch wrapping gently down around the iris contour (eliminates flat/stern bar look)
        arch_drop = rz * (0.44 + 0.36 * arch_ratio)
        z_op = z_top_mid - arch_drop * (u_arr * u_arr)
        z_cl = z_bot_mid + (rz * 0.14) * (u_arr * u_arr)
        return z_op * (1.0 - blink) + z_cl * blink

    # Build 2D (x, z) KD-tree of actual front-facing head surface around this eye (includes forehead/upper rim
    # that detect_face_skin_mask excludes!) for exact local surface projection & silhouette guard.
    surf_pts = ref_xyz if (ref_xyz is not None and len(ref_xyz) >= 20) else skin_xyz
    m_loc_ref = (
        (np.abs(surf_pts[:, 0] - cx) < rx * 2.2)
        & (np.abs(surf_pts[:, 2] - cz) < rz * 2.2)
        & (surf_pts[:, 1] < cy + max(rx * 1.4, 0.025))
        & (surf_pts[:, 1] > cy - max(rx * 1.2, 0.020))
    )
    loc_ref = surf_pts[m_loc_ref] if m_loc_ref.sum() >= 12 else surf_pts
    ref_xz_tree = cKDTree(loc_ref[:, [0, 2]]) if len(loc_ref) >= 8 else None
    r_tan = float(np.clip(skin_scale_med * 1.15, max(rx * 0.065, 0.0009), max(rx * 0.10, 0.0016)))
    skin_depth_thick = r_tan * 0.28

    def _project_to_true_surface(x_arr: np.ndarray, z_arr: np.ndarray, y_quad_arr: np.ndarray, is_1d_curve: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Project (x, z) onto the exact visible frontmost 3D surface (actual head splats + eyeball cap).

        - Where actual head/forehead/cheek splats exist within local radius, tracks their frontmost Y directly
          (never burying behind forehead skin and never protruding 7-11 mm forward!).
        - Over the carved eye socket opening, smoothly bridges across the eyeball front cap.
        - Clips endpoints that lack both head and eyeball support (eliminates oblique air protrusion).
        """
        y_out = y_quad_arr.astype(np.float64).copy()
        has_support = np.ones(len(x_arr), dtype=bool)

        # Eyeball front cap Y at (x, z)
        rad_sq = (r_ball * r_ball) - (x_arr - ec[0]) ** 2 - (z_arr - ec[2]) ** 2
        in_ball = rad_sq > 0.0
        y_ball = np.full(len(x_arr), np.inf, dtype=np.float64)
        y_ball[in_ball] = ec[1] - np.sqrt(np.maximum(rad_sq[in_ball], 0.0))

        if ref_xz_tree is not None:
            pts_xz = np.column_stack([x_arr, z_arr])
            k_q = min(8, len(loc_ref))
            dists, idxs = ref_xz_tree.query(pts_xz, k=k_q)
            if k_q == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
            y_nbrs = loc_ref[idxs, 1]
            # Frontmost local head splats (15th percentile in Y = frontmost toward -Y camera)
            y_head_loc = np.percentile(y_nbrs, 15, axis=1)
            min_d = dists[:, 0]
            # Support exists if inside eyeball disk OR within 0.35*rx of real head surface
            has_support = in_ball | (min_d < max(rx * 0.35, skin_scale_med * 3.0))

            # Tight local trust weight: trust y_head_loc when real splats exist right at (x, z) (< 3.5 mm),
            # and smoothly hand off to y_ball (or regularized y_quad) inside the carved socket opening.
            r_trust = max(skin_scale_med * 2.8, 0.0038)
            w_head = np.clip(1.0 - min_d / r_trust, 0.0, 1.0)
            y_socket = np.where(in_ball, np.minimum(y_ball, y_quad_arr), y_quad_arr)
            # Take frontmost of local head and eyeball when both are present
            y_head_eff = np.where(in_ball, np.minimum(y_head_loc, y_ball), y_head_loc)
            y_out = w_head * y_head_eff + (1.0 - w_head) * y_socket
            # Guarantee clearance in front of eyeball front cap everywhere inside eyeball disk
            y_out = np.where(in_ball, np.minimum(y_out, y_ball), y_out)
        else:
            y_out = np.where(in_ball, np.minimum(y_out, y_ball), y_out)

        if is_1d_curve and len(y_out) >= 12:
            # Smooth 1D depth curve so lash glides cleanly across the socket-to-forehead transition
            ker = np.array([0.08, 0.16, 0.24, 0.24, 0.16, 0.08], dtype=np.float64)
            ker /= ker.sum()
            y_pad = np.pad(y_out, (2, 3), mode="edge")
            y_sm = np.convolve(y_pad, ker, mode="valid")
            # Keep frontmost of smoothed and eyeball cap so eyeball never pokes through the middle of the lash!
            y_out = np.where(in_ball, np.minimum(y_sm, y_ball), y_sm)

        return y_out.astype(np.float32), has_support

    parts = []

    # 1. Skin curtain (only drawn when blink > 0.03 so open eyes NEVER overwrite cheek blush!)
    if blink > 0.03:
        GX, GZ = np.meshgrid(np.linspace(-rx, rx, 44), np.linspace(-rz_dn, rz_up, 40))
        dx, dz = GX.ravel().astype(np.float32), GZ.ravel().astype(np.float32)
        u = dx / max(rx, 1e-5)
        v = np.where(dz >= 0.0, dz / max(rz_up, 1e-5), dz / max(rz_dn, 1e-5))
        r_ell = np.sqrt(u * u + v * v)
        z_cur_edge = _lash_z(u)

        # Seal below the lash ONLY when nearly closed (blink > 0.80) so half-blink cleanly reveals lower iris
        seal_below = float(smoothstep(0.80, 0.98, blink))
        lower_margin = rz_dn * 0.52 * seal_below
        keep = (r_ell <= 1.0) & (dz >= z_cur_edge - lower_margin)
        if keep.any():
            dx_k, dz_k, r_k = dx[keep], dz[keep], r_ell[keep]
            y_quad = coef[0] + coef[1] * dx_k + coef[2] * dz_k + coef[3] * dx_k * dx_k + coef[4] * dx_k * dz_k + coef[5] * dz_k * dz_k
            y_clamped, sup_k = _project_to_true_surface(cx + dx_k, cz + dz_k, y_quad)
            # Uniform flush offset along surface normal (scale-invariant)
            y_surf = y_clamped - max(skin_depth_thick * 0.85, 0.00035)

            dy_dx = coef[1] + 2.0 * coef[3] * dx_k + coef[4] * dz_k
            dy_dz = coef[2] + coef[4] * dx_k + 2.0 * coef[5] * dz_k
            normals = np.column_stack([dy_dx, -np.ones_like(dx_k), dy_dz]).astype(np.float32)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

            xyz = np.column_stack([cx + dx_k, y_surf, cz + dz_k]).astype(np.float32)
            rot = normal_to_quat_wxyz(normals)

            # Match color strictly to surrounding cheek skin below the eye + local IDW (avoids pale/grey patch!)
            skin_rgb_all = np.clip(0.5 + SH_C0 * skin_fdc, 0.0, 1.0)
            cheek_m = (
                (np.abs(skin_xyz[:, 0] - cx) < rx * 1.4)
                & (skin_xyz[:, 2] < cz - rz_dn * 0.45)
                & (skin_xyz[:, 2] > cz - rz_dn * 2.4)
            )
            cheek_rgb = np.median(skin_rgb_all[cheek_m], axis=0).astype(np.float32) if cheek_m.sum() >= 8 else np.median(skin_rgb_all, axis=0).astype(np.float32)

            tree = cKDTree(skin_xyz)
            dists, idx = tree.query(xyz, k=min(14, len(skin_xyz)))
            w_idw = 1.0 / np.maximum(dists, 1e-4)
            w_idw /= w_idw.sum(axis=1, keepdims=True)
            idw_rgb = np.sum(w_idw[:, :, None] * skin_rgb_all[idx], axis=1).astype(np.float32)
            rgb_skin = np.clip(0.55 * cheek_rgb[None, :] + 0.45 * idw_rgb, 0.0, 1.0)
            fdc = (rgb_skin - 0.5) / SH_C0

            blink_vis = float(smoothstep(0.03, 0.16, blink))
            alpha_edge = (1.0 - smoothstep(0.74, 1.00, r_k)) * blink_vis * np.where(sup_k, 1.0, 0.15)
            alpha_prob = np.clip(alpha_edge * 0.992, 0.02, 0.995)
            opacity_logit = np.log(alpha_prob / (1.0 - alpha_prob)).astype(np.float32)

            skin_out = np.zeros(len(dx_k), dtype=dtype)
            skin_out["x"], skin_out["y"], skin_out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            skin_out["f_dc_0"], skin_out["f_dc_1"], skin_out["f_dc_2"] = fdc[:, 0], fdc[:, 1], fdc[:, 2]
            skin_out["opacity"] = opacity_logit
            skin_out["scale_0"] = np.log(r_tan)
            skin_out["scale_1"] = np.log(r_tan)
            skin_out["scale_2"] = np.log(skin_depth_thick)
            skin_out["rot_0"], skin_out["rot_1"], skin_out["rot_2"], skin_out["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
            parts.append(skin_out)

    # 2. Design-art-matched anime eyelash stroke (soft natural color & alpha taper, clamped to 3D face surface!)
    n_lash = 76
    # Derive horizontal span from 2D design-art lash_span_over_deye (fallback ~0.78*rx)
    span_half_world = 0.5 * float(style.get("lash_span_over_deye", 0.42)) * d_eye_3d
    u_max = float(np.clip(span_half_world / max(rx, 1e-5), 0.62, 0.86))
    u_l = np.linspace(-u_max, u_max, n_lash, dtype=np.float32)
    dx_l = u_l * rx
    dz_l_base = _lash_z(u_l)

    # Use 2D-extracted character lash color directly (NO forced near-black clamp!)
    if "lash_rgb" in style and len(style["lash_rgb"]) == 3:
        ink_rgb = np.asarray(style["lash_rgb"], dtype=np.float32)
    elif lash_rgb is not None:
        ink_rgb = np.asarray(lash_rgb, dtype=np.float32)
    else:
        ink_rgb = np.array([0.28, 0.16, 0.14], dtype=np.float32)
    ink_rgb = np.clip(ink_rgb, 0.05, 0.65)
    fdc_l = (ink_rgb - 0.5) / SH_C0

    # Derive stroke thickness from 2D design-art lash_thick_over_deye (scale-invariant)
    thick_world = float(style.get("lash_thick_over_deye", 0.036)) * d_eye_3d
    r_thick_base = float(np.clip(0.5 * thick_world, rx * 0.022, rx * 0.052))

    # Scale-invariant depth clearance: sits cleanly on top of eyelid skin curtain when blinking,
    # and flush on face/eyeball surface when open (never floating, never buried!)
    lash_proud = max(skin_depth_thick * (1.8 + 1.8 * blink), 0.00075 + 0.00065 * blink)

    # Primary stroke + subtle soft upper feather row (clean, continuous, soft-blended anime lash)
    for row_idx, (dz_off, thick_mul, op_center) in enumerate(((0.0, 1.0, 4.0), (r_thick_base * 0.50 * (1.0 - 0.5 * blink), 0.65, 2.4))):
        dz_l = dz_l_base + dz_off
        y_quad_l = coef[0] + coef[1] * dx_l + coef[2] * dz_l + coef[3] * dx_l * dx_l + coef[4] * dx_l * dz_l + coef[5] * dz_l * dz_l
        y_surf_l, sup_l = _project_to_true_surface(cx + dx_l, cz + dz_l, y_quad_l, is_1d_curve=True)
        if not sup_l.any():
            continue
        # Clip endpoints that would protrude into empty air past the curved cheek silhouette
        dx_k, dz_k, u_k = dx_l[sup_l], dz_l[sup_l], u_l[sup_l]
        y_l = y_surf_l[sup_l] - lash_proud

        dy_dx_l = coef[1] + 2.0 * coef[3] * dx_k + coef[4] * dz_k
        dy_dz_l = coef[2] + coef[4] * dx_k + 2.0 * coef[5] * dz_k
        nrm_l = np.column_stack([dy_dx_l, -np.ones_like(dx_k), dy_dz_l]).astype(np.float32)
        nrm_l /= np.maximum(np.linalg.norm(nrm_l, axis=1, keepdims=True), 1e-8)
        rot_l = normal_to_quat_wxyz(nrm_l)

        u_norm = np.abs(u_k) / max(u_max, 1e-5)
        taper_l = 1.0 - smoothstep(0.50, 0.98, u_norm)
        r_lash_thick = (r_thick_base * (0.55 + 0.65 * taper_l) * thick_mul).astype(np.float32)
        r_lash_along = np.full(len(dx_k), max(rx * 0.044, 0.00055), dtype=np.float32)
        r_lash_depth = (r_lash_thick * 0.34).astype(np.float32)
        # Soft natural opacity taper at inner/outer tips so stroke blends into skin instead of looking pasted-on
        op_l = (op_center * (0.35 + 0.65 * taper_l)).astype(np.float32)

        lash_out = np.zeros(len(dx_k), dtype=dtype)
        lash_out["x"] = cx + dx_k
        lash_out["y"] = y_l
        lash_out["z"] = cz + dz_k
        lash_out["f_dc_0"], lash_out["f_dc_1"], lash_out["f_dc_2"] = fdc_l[0], fdc_l[1], fdc_l[2]
        lash_out["opacity"] = op_l
        lash_out["scale_0"] = np.log(r_lash_along)
        lash_out["scale_1"] = np.log(r_lash_thick)
        lash_out["scale_2"] = np.log(r_lash_depth)
        lash_out["rot_0"], lash_out["rot_1"], lash_out["rot_2"], lash_out["rot_3"] = rot_l[:, 0], rot_l[:, 1], rot_l[:, 2], rot_l[:, 3]
        parts.append(lash_out)

    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=dtype)


def detect_face_skin_mask(xyz: np.ndarray, rgb: np.ndarray, is_body: np.ndarray, surf_L: np.ndarray, surf_R: np.ndarray, face_z_range: list[float]) -> np.ndarray:
    """Data-driven facial skin detection: sample median cheek/nose-bridge color scaled to inter-ocular distance D_eye."""
    d_eye = max(float(abs(surf_L[0] - surf_R[0])), 0.015)
    dL = np.hypot(xyz[:, 0] - surf_L[0], xyz[:, 2] - surf_L[2])
    dR = np.hypot(xyz[:, 0] - surf_R[0], xyz[:, 2] - surf_R[2])
    mid_x = 0.5 * (surf_L[0] + surf_R[0])
    mid_y = 0.5 * (surf_L[1] + surf_R[1])
    mid_z = 0.5 * (surf_L[2] + surf_R[2])
    seed_m = (
        is_body
        & (np.abs(xyz[:, 0] - mid_x) < d_eye * 0.25)
        & (xyz[:, 1] < mid_y + d_eye * 0.35)
        & (xyz[:, 2] >= mid_z - d_eye * 0.45) & (xyz[:, 2] <= mid_z + d_eye * 0.15)
    )
    skin_ref = np.median(rgb[seed_m], axis=0) if seed_m.sum() >= 10 else np.array([0.95, 0.85, 0.78], dtype=np.float32)
    col_dist = np.linalg.norm(rgb - skin_ref[None, :], axis=1)
    return (
        is_body & (dL > d_eye * 0.30) & (dR > d_eye * 0.30)
        & (xyz[:, 1] < mid_y + d_eye * 0.45)
        & (xyz[:, 2] >= float(face_z_range[0])) & (xyz[:, 2] <= float(face_z_range[1]))
        & (np.abs(xyz[:, 0] - mid_x) < d_eye * 1.35)
        & (col_dist < 0.16)
    )


def apply_blink_to_master(
    master: np.ndarray,
    bones: dict,
    prop: dict,
    cols: dict,
    bL: float,
    bR: float,
) -> np.ndarray:
    """Apply clean blink/wink to master PLY: hides static upper eyelash/socket rim & generates moving eyelid shell."""
    xyz = np.column_stack([master["x"], master["y"], master["z"]]).astype(np.float32)
    fdc = np.column_stack([master["f_dc_0"], master["f_dc_1"], master["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

    pos_L, pos_R = np.array(bones["eye_L"]["head"], dtype=np.float32), np.array(bones["eye_R"]["head"], dtype=np.float32)
    surf_L, surf_R = np.array(bones["eye_L"]["meta"]["surface_center"], dtype=np.float32), np.array(bones["eye_R"]["meta"]["surface_center"], dtype=np.float32)
    r_eye = float(bones["eye_L"]["meta"]["eyeball_radius"])
    is_eye_L = np.linalg.norm(xyz - pos_L[None, :], axis=1) <= (r_eye + 0.0015)
    is_eye_R = np.linalg.norm(xyz - pos_R[None, :], axis=1) <= (r_eye + 0.0015)
    is_body = ~(is_eye_L | is_eye_R)

    is_skin = detect_face_skin_mask(xyz, rgb, is_body, surf_L, surf_R, prop["face_z_range"])
    skin_xyz, skin_fdc = xyz[is_skin], fdc[is_skin]
    skin_sc = float(np.exp(np.median(master["scale_0"][is_skin])))
    rx_e, rz_e = float(prop["eyelid_shell_rx_rz"][0]), float(prop["eyelid_shell_rx_rz"][1])

    dz_cap_L = detect_lash_ceiling_dz(xyz, rgb, is_body, surf_L, rx_e, rz_e)
    dz_cap_R = detect_lash_ceiling_dz(xyz, rgb, is_body, surf_R, rx_e, rz_e)
    dz_cap = 0.5 * (dz_cap_L + dz_cap_R)

    base = master.copy()
    for bval, sc_c, is_eye_m in ((bL, surf_L, is_eye_L), (bR, surf_R, is_eye_R)):
        if bval > 0.05:
            # Retract eyeball slightly back during blink, hide completely when closed
            base["y"][is_eye_m] += max(r_eye * 0.30, 0.0030) * bval
            if bval >= 0.92:
                base["opacity"][is_eye_m] = -12.0
            # Hide the FULL static upper eyelash & socket rim up to dz_cap * 1.12 (strictly below eyebrow!)
            # so zero frozen double-lash or dark upper-rim outline remains when closing!
            dx_c = (xyz[:, 0] - sc_c[0]) / max(rx_e * 1.08, 1e-5)
            dz_rel = xyz[:, 2] - sc_c[2]
            in_static_lash = (
                is_body
                & (np.abs(dx_c) <= 1.0)
                & (xyz[:, 1] < sc_c[1] + r_eye * 0.95)
                & (dz_rel >= -rz_e * 0.90)
                & (dz_rel <= dz_cap * 1.12)
                & ((~is_skin) | (lum < 0.62))
            )
            base["opacity"][in_static_lash] = -12.0

    lid_L = build_eyelid_shell(pos_L, surf_L, skin_xyz, skin_fdc, skin_sc, bL, master.dtype, prop["eyelid_shell_rx_rz"], prop["eyelid_lash_top_bottom_z"], cols["lash_rgb"], dz_top_cap=dz_cap)
    lid_R = build_eyelid_shell(pos_R, surf_R, skin_xyz, skin_fdc, skin_sc, bR, master.dtype, prop["eyelid_shell_rx_rz"], prop["eyelid_lash_top_bottom_z"], cols["lash_rgb"], dz_top_cap=dz_cap)
    return np.concatenate([base, lid_L, lid_R], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])
    ver_dir = work_dir / "packages" / cfg["asset_id"] / cfg["version"]
    frames_dir = resolve_spark_dir(work_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    prop, cols = cfg["proportions"], cfg["colors"]

    master, _ = read_ply(ver_dir / "master.ply")
    skel = json.loads((ver_dir / "rig_skeleton.json").read_text(encoding="utf-8"))
    bones = skel["bones"]

    R_spark_obj = Rotation.from_euler("x", 90.0, degrees=True)
    R_spark = R_spark_obj.as_matrix().astype(np.float32)
    q_s = R_spark_obj.as_quat()
    q_spark = np.array([q_s[3], q_s[0], q_s[1], q_s[2]], dtype=np.float32)

    def to_spark(arr):
        out = arr.copy()
        p = (np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32) @ R_spark.T).astype(np.float32)
        q = np.column_stack([out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"]]).astype(np.float32)
        w1, x1, y1, z1 = q_spark
        w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        qn = np.column_stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2]).astype(np.float32)
        qn /= np.maximum(np.linalg.norm(qn, axis=1, keepdims=True), 1e-8)
        out["x"], out["y"], out["z"] = p[:, 0], p[:, 1], p[:, 2]
        out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
        return out

    for fn, bL, bR in [("shell_open.ply", 0.0, 0.0), ("shell_closed.ply", 1.0, 1.0), ("shell_wink.ply", 1.0, 0.0)]:
        merged = apply_blink_to_master(master, bones, prop, cols, bL, bR)
        write_ply(frames_dir / fn, to_spark(merged))
        print(f"Wrote {fn}: total={len(merged)}")


if __name__ == "__main__":
    main()
