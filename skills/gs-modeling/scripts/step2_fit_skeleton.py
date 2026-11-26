"""Step 2: Fit minimal doll skeleton using config proportions (no character-specific hardcodes)."""
import argparse
import json
from pathlib import Path
import numpy as np
from ply_io import read_ply
from config_loader import load_config

SH_C0 = 0.28209479177387814


def fit_eye_surface(xyz: np.ndarray, skin: np.ndarray, center: np.ndarray,
                    radii: np.ndarray) -> float | None:
    """Estimate the skin surface around one eye, excluding its opening."""
    # Work in units of the socket radii. Include the outer rim and the cheek
    # below it, then give each occupied cell similar weight despite splat density.
    local = (xyz - center) / radii
    x, y, z = local.T
    samples = local[skin & (np.abs(x) <= 1.7) & (z >= -2.1) & (z <= 1.3)
                    & (np.abs(y) <= 1.0) & (x*x + z*z >= 0.85**2)]
    cells = np.floor(samples[:, [0, 2]] * 5).astype(int)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    splits = np.flatnonzero(np.any(np.diff(cells[order], axis=0), axis=1)) + 1
    samples = np.array([np.median(group, axis=0) for group in np.split(samples[order], splits)
                        if len(group) >= 2])
    if len(samples) < 20:
        return None
    x, y, z = samples.T
    if not (x.min() < -0.5 and x.max() > 0.5 and z.min() < -0.8 and z.max() > -0.25):
        return None
    matrix = np.column_stack((np.ones_like(x), x, z, x*x, x*z, z*z))
    if np.linalg.cond(matrix) > 50:
        return None
    weights = np.ones_like(y)
    for _ in range(4):
        coefficients, *_ = np.linalg.lstsq(matrix * weights[:, None], y * weights, rcond=None)
        error = np.abs(matrix @ coefficients - y)
        weights = np.sqrt(np.minimum(1.0, 0.10 / np.maximum(error, 1e-6)))
    if abs(coefficients[0]) > 0.35 or np.median(error) > 0.15:
        return None
    return float(center[1] + coefficients[0] * radii[1])


def estimate_face_cx(xyz: np.ndarray, rgb: np.ndarray, prop: dict) -> float:
    """Estimate true face midline X from lower-face skin splats (immune to asymmetric side ponytails)."""
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    fz0, fz1 = float(prop["face_z_range"][0]), float(prop["face_z_range"][1])
    z_eye0 = float(prop["eye_z_range"][0])
    m_band = (xyz[:, 2] >= fz0) & (xyz[:, 2] <= z_eye0)
    if m_band.sum() < 30:
        return 0.0
    y_front = float(np.percentile(xyz[m_band, 1], 10))
    m_skin = m_band & (xyz[:, 1] <= y_front + 0.035) & (lum > 0.72) & (rgb[:, 0] > rgb[:, 2] + 0.04) & (np.abs(xyz[:, 0]) < 0.14)
    return float(np.median(xyz[m_skin, 0])) if m_skin.sum() >= 25 else 0.0


def detect_eye(xyz: np.ndarray, rgb: np.ndarray, side: str, prop: dict, face_cx_est: float = 0.0) -> dict:
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    sat = rgb.max(axis=1) - rgb.min(axis=1)
    z0, z1 = float(prop["eye_z_range"][0]), float(prop["eye_z_range"][1])

    # Data-driven front-face plane & half-width relative to face_cx_est (scale-invariant across 2.5-8 head-tall characters)
    band_m = (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1) & (np.abs(xyz[:, 0] - face_cx_est) < 0.14)
    y_front = float(np.percentile(xyz[band_m, 1], 8))
    front_band = band_m & (xyz[:, 1] <= y_front + 0.035)
    w_face_half = float(np.percentile(np.abs(xyz[front_band, 0] - face_cx_est), 90)) if front_band.sum() >= 20 else 0.075
    x_min = max(0.006, w_face_half * 0.14)
    x_max = min(0.085, w_face_half * 0.72)
    y_max = y_front + max(0.018, w_face_half * 0.30)

    dx_rel = xyz[:, 0] - face_cx_est
    mask_x = (dx_rel > x_min) & (dx_rel < x_max) if side == "eye_L" else (dx_rel < -x_min) & (dx_rel > -x_max)
    eye_win = mask_x & (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1) & (xyz[:, 1] <= y_max)

    # Prefer colored/dark iris below upper lash or bright sclera on the front face plane
    m_iris = eye_win & (lum < 0.55) & (sat > 0.08)
    m_sclera_init = eye_win & (lum >= 0.78) & (sat < 0.16)
    m_dark = eye_win & (lum < 0.52)
    pts = xyz[m_iris] if m_iris.sum() >= 25 else (xyz[m_sclera_init] if m_sclera_init.sum() >= 20 else xyz[m_dark])
    if len(pts) < 10:
        raise RuntimeError(f"Failed to detect {side} eye points in Z=[{z0},{z1}] (found {len(pts)})")

    p05, p50, p95 = np.percentile(pts, [5, 50, 95], axis=0)
    surf_center = np.array([float(p50[0]), float(np.percentile(pts[:, 1], 45)), float(p50[2])], dtype=np.float32)

    # Estimate gentle cheek curvature yaw (outer corner recedes along face sphere)
    sgn = +1.0 if side == "eye_L" else -1.0
    m_slope = (np.abs(xyz[:, 0] - surf_center[0]) < 0.022) & (np.abs(xyz[:, 2] - surf_center[2]) < 0.012) & (xyz[:, 1] <= surf_center[1] + 0.022) & (xyz[:, 1] >= surf_center[1] - 0.018)
    cheek_yaw_deg = sgn * 10.0
    if m_slope.sum() >= 15:
        slope = float(np.polyfit(xyz[m_slope, 0] - surf_center[0], xyz[m_slope, 1], 1)[0])
        raw_deg = float(np.degrees(np.arctan(slope)))
        if raw_deg * sgn > 0:
            cheek_yaw_deg = float(np.clip(raw_deg * 0.65, -14.0, 14.0))

    r_eye = float(prop["eyeball_radius"])
    excl = [float(v) for v in prop["socket_exclusion_radius"]]
    d_xy = np.hypot(xyz[:, 0] - surf_center[0], xyz[:, 2] - surf_center[2])
    m_sclera = (
        (d_xy <= max(r_eye * 1.15, 0.010))
        & (xyz[:, 1] <= surf_center[1] + r_eye * 0.6) & (xyz[:, 1] >= surf_center[1] - r_eye * 0.6)
        & (lum >= 0.78) & (sat < 0.15)
    )
    sclera_rgb = np.median(rgb[m_sclera], axis=0) if m_sclera.sum() >= 5 else np.array([0.93, 0.91, 0.89], dtype=np.float32)

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
        "carve_center": [float(surf_center[0]), float(surf_center[1] + r_eye * 0.10), eye_cz],
        "exclusion_radius": excl,
        "sclera_sample_count": int(m_sclera.sum()),
        "sclera_rgb": [float(v) for v in sclera_rgb],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])
    prop = cfg["proportions"]
    src = Path(args.src) if args.src else (work_dir / "outputs" / "step1_normalized.ply")
    dst = Path(args.dst) if args.dst else (work_dir / "outputs" / "step2_skeleton.json")

    d, _ = read_ply(src)
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
    fdc = np.column_stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)

    face_cx_est = estimate_face_cx(xyz, rgb, prop)
    eye_L = detect_eye(xyz, rgb, "eye_L", prop, face_cx_est)
    eye_R = detect_eye(xyz, rgb, "eye_R", prop, face_cx_est)

    # Fit each eye to its own opening. Look-at already uses each eye's origin.
    face_cx = 0.5 * (eye_L["surface_center"][0] + eye_R["surface_center"][0])
    red, green, blue = rgb.T
    skin = (red > 0.76) & (green > 0.56) & (green > blue + 0.015) & (red > green + 0.018)
    skin &= d["opacity"] > 0
    for name, eye in (("eye_L", eye_L), ("eye_R", eye_R)):
        surface = eye["surface_center"]
        detected_y = surface[1]
        radii = np.asarray(eye["exclusion_radius"])
        fitted_y = fit_eye_surface(xyz, skin, np.asarray(surface), radii)
        eye["depth_fit"] = {"detected_y": detected_y, "skin_surface_y": fitted_y}
        if fitted_y is None:
            print(f"{name}: insufficient local skin support; review detected depth before carving")
            continue
        # Keep the carve centered on the baked iris splats in the original opening.
        # Align the globe's forward pole with the fitted skin, even when gaze moves.
        surface[1] = fitted_y
        eye["eyeball_center"][1] = fitted_y + eye["eyeball_radius"]
    fz0, fz1 = float(prop["face_z_range"][0]), float(prop["face_z_range"][1])
    neck_z0 = float(prop["neck_z_range"][0])
    # Exclude rear ponytail/cape when estimating torso/head Y depth
    m_torso_y = (xyz[:, 2] >= neck_z0 - 0.10) & (xyz[:, 2] <= neck_z0 + 0.05) & (np.abs(xyz[:, 0] - face_cx) < 0.075) & (xyz[:, 1] < 0.05)
    head_y = float(0.5 * (np.percentile(xyz[m_torso_y, 1], 10) + np.percentile(xyz[m_torso_y, 1], 90))) if m_torso_y.sum() >= 20 else float(np.median(xyz[(xyz[:, 2] >= fz0) & (xyz[:, 2] <= fz1), 1]))
    id_q = [1.0, 0.0, 0.0, 0.0]

    skeleton = {
        "convention": {
            "world": "Z-up, -Y-forward, +X-character-left",
            "bone_local_axes": "Unified across all bones: +X=left, -Y=forward(gaze), +Z=up (rest quat=[1,0,0,0])",
        },
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
