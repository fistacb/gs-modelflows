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
    lash_rgb: tuple[float, float, float] = (0.14, 0.11, 0.12),
    dz_top_cap: float | None = None,
) -> np.ndarray:
    """Fit a smooth quadratic surface to surrounding facial skin + subtle eyeball dome and generate a natural anime eyelid."""
    if blink <= 0.02:
        return np.zeros(0, dtype=dtype)

    cx, cy, cz = float(surf_center[0]), float(surf_center[1]), float(surf_center[2])
    rx, rz = float(rx_rz[0]), float(rx_rz[1])
    rz_up = min(rz * 1.06, float(dz_top_cap) * 1.02) if dz_top_cap is not None else rz * 1.06
    rz_dn = rz * 1.02

    d_local = np.hypot(skin_xyz[:, 0] - cx, skin_xyz[:, 2] - cz)
    local_m = d_local < max(rx * 2.2, 0.025)
    if local_m.sum() < 12:
        local_m = d_local < max(rx * 4.0, 0.055)
    pts_fit = skin_xyz[local_m]
    dx_f, dz_f = pts_fit[:, 0] - cx, pts_fit[:, 2] - cz
    A = np.column_stack([np.ones_like(dx_f), dx_f, dz_f, dx_f * dx_f, dx_f * dz_f, dz_f * dz_f])
    w_fit = np.exp(-(d_local[local_m] / max(rx * 1.4, 0.018)) ** 2)
    coef, _, _, _ = np.linalg.lstsq(A * w_fit[:, None], pts_fit[:, 1] * w_fit, rcond=None)

    GX, GZ = np.meshgrid(np.linspace(-rx, rx, 44), np.linspace(-rz_dn, rz_up, 40))
    dx, dz = GX.ravel().astype(np.float32), GZ.ravel().astype(np.float32)
    u = dx / max(rx, 1e-5)
    v = np.where(dz >= 0.0, dz / max(rz_up, 1e-5), dz / max(rz_dn, 1e-5))
    r_ell = np.sqrt(u * u + v * v)

    # Natural anime lash curve:
    # Open (blink=0): arches UP in the middle and curves down at inner/outer corners
    # Closed (blink=1): clean low horizontal/gentle-smile curve covering the entire iris
    z_top_mid = float(lash_top_bot[0]) + rz * 0.10
    z_bot_mid = float(lash_top_bot[1]) * 0.96
    z_lash_open = z_top_mid - (rz * 0.30) * (u * u)
    z_lash_closed = z_bot_mid + (rz * 0.14) * (u * u)
    z_cur_edge = z_lash_open * (1.0 - blink) + z_lash_closed * blink

    # Keep skin splats from upper cap down past the moving lash edge (at full blink cover full lower socket too)
    lower_margin = rz * (0.10 + 0.25 * blink)
    keep = (r_ell <= 1.0) & (dz >= z_cur_edge - lower_margin)
    dx, dz, u, r_ell, z_edge = dx[keep], dz[keep], u[keep], r_ell[keep], z_cur_edge[keep]
    n = len(dx)
    if n == 0:
        return np.zeros(0, dtype=dtype)

    # Flush quadratic face surface (flat anime cheek continuity, no puffy bulge)
    y_quad = coef[0] + coef[1] * dx + coef[2] * dz + coef[3] * dx * dx + coef[4] * dx * dz + coef[5] * dz * dz
    dome = np.maximum(0.0, 1.0 - r_ell * r_ell) * (rz * 0.04)
    y_surf = y_quad - dome - max(rz * 0.03, 0.0004)

    dy_dx = coef[1] + 2.0 * coef[3] * dx + coef[4] * dz
    dy_dz = coef[2] + coef[4] * dx + 2.0 * coef[5] * dz
    normals = np.column_stack([dy_dx, -np.ones_like(dx), dy_dz]).astype(np.float32)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)

    xyz = np.column_stack([cx + dx, y_surf, cz + dz]).astype(np.float32)
    rot = normal_to_quat_wxyz(normals)

    # Pure flat anime skin color matched seamlessly to surrounding neutral facial skin
    tree = cKDTree(skin_xyz)
    dists, idx = tree.query(xyz, k=min(16, len(skin_xyz)))
    w_idw = 1.0 / np.maximum(dists, 1e-4)
    w_idw /= w_idw.sum(axis=1, keepdims=True)
    fdc = np.sum(w_idw[:, :, None] * skin_fdc[idx], axis=1).astype(np.float32)
    rgb_skin = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
    fdc = (rgb_skin - 0.5) / SH_C0

    # Pure opaque skin shell (NO blurry dark lash blending onto coarse skin splats!)
    blink_vis = float(smoothstep(0.04, 0.20, blink))
    alpha_edge = (1.0 - smoothstep(0.84, 1.00, r_ell)) * blink_vis
    alpha_prob = np.clip(alpha_edge * 0.995, 0.02, 0.996)
    opacity_logit = np.log(alpha_prob / (1.0 - alpha_prob)).astype(np.float32)

    r_tan = float(np.clip(skin_scale_med * 1.25, max(rx * 0.06, 0.0008), max(rx * 0.11, 0.0018)))
    skin_out = np.zeros(n, dtype=dtype)
    skin_out["x"], skin_out["y"], skin_out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    skin_out["f_dc_0"], skin_out["f_dc_1"], skin_out["f_dc_2"] = fdc[:, 0], fdc[:, 1], fdc[:, 2]
    skin_out["opacity"] = opacity_logit
    skin_out["scale_0"] = np.log(r_tan)
    skin_out["scale_1"] = np.log(r_tan)
    skin_out["scale_2"] = np.log(r_tan * 0.28)
    skin_out["rot_0"], skin_out["rot_1"], skin_out["rot_2"], skin_out["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    # Dedicated high-res razor-sharp anime eyelash stroke (96 tiny crisp ink splats sitting flush in front of the lid)
    n_lash = 96
    u_l = np.linspace(-0.82, 0.82, n_lash, dtype=np.float32)
    dx_l = u_l * rx
    z_l_open = z_top_mid - (rz * 0.30) * (u_l * u_l)
    z_l_closed = z_bot_mid + (rz * 0.14) * (u_l * u_l)
    dz_l = z_l_open * (1.0 - blink) + z_l_closed * blink
    v_l = np.where(dz_l >= 0.0, dz_l / max(rz_up, 1e-5), dz_l / max(rz_dn, 1e-5))
    r_l = np.sqrt(u_l * u_l + v_l * v_l)

    y_quad_l = coef[0] + coef[1] * dx_l + coef[2] * dz_l + coef[3] * dx_l * dx_l + coef[4] * dx_l * dz_l + coef[5] * dz_l * dz_l
    dome_l = np.maximum(0.0, 1.0 - r_l * r_l) * (rz * 0.04)
    # Sit 0.55 mm in front of the skin lid so it renders razor-sharp with zero Z-fighting or skin bleed
    y_l = y_quad_l - dome_l - max(rz * 0.03, 0.0004) - 0.00055

    dy_dx_l = coef[1] + 2.0 * coef[3] * dx_l + coef[4] * dz_l
    dy_dz_l = coef[2] + coef[4] * dx_l + 2.0 * coef[5] * dz_l
    nrm_l = np.column_stack([dy_dx_l, -np.ones_like(dx_l), dy_dz_l]).astype(np.float32)
    nrm_l /= np.maximum(np.linalg.norm(nrm_l, axis=1, keepdims=True), 1e-8)
    rot_l = normal_to_quat_wxyz(nrm_l)

    taper_l = 1.0 - smoothstep(0.50, 0.86, np.abs(u_l))
    r_lash_thick = np.clip(rx * (0.014 + 0.018 * taper_l), 0.00022, 0.00058).astype(np.float32)
    r_lash_along = np.full(n_lash, max(rx * 0.028, 0.00045), dtype=np.float32)
    r_lash_depth = (r_lash_thick * 0.40).astype(np.float32)

    alpha_l = np.clip((0.88 + 0.11 * taper_l) * blink_vis, 0.02, 0.995)
    op_l = np.log(alpha_l / (1.0 - alpha_l)).astype(np.float32)
    # Deep crisp anime ink color
    ink_rgb = np.array([min(lash_rgb[0], 0.09), min(lash_rgb[1], 0.07), min(lash_rgb[2], 0.08)], dtype=np.float32)
    fdc_l = (ink_rgb - 0.5) / SH_C0

    lash_out = np.zeros(n_lash, dtype=dtype)
    lash_out["x"] = cx + dx_l
    lash_out["y"] = y_l
    lash_out["z"] = cz + dz_l
    lash_out["f_dc_0"], lash_out["f_dc_1"], lash_out["f_dc_2"] = fdc_l[0], fdc_l[1], fdc_l[2]
    lash_out["opacity"] = op_l
    lash_out["scale_0"] = np.log(r_lash_along)
    lash_out["scale_1"] = np.log(r_lash_thick)
    lash_out["scale_2"] = np.log(r_lash_depth)
    lash_out["rot_0"], lash_out["rot_1"], lash_out["rot_2"], lash_out["rot_3"] = rot_l[:, 0], rot_l[:, 1], rot_l[:, 2], rot_l[:, 3]

    return np.concatenate([skin_out, lash_out], axis=0)


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
