"""Step 2: Fit minimal doll skeleton using config proportions (no character-specific hardcodes)."""
import argparse
import json
from pathlib import Path
import numpy as np
from ply_io import read_ply
from config_loader import load_config

SH_C0 = 0.28209479177387814


def detect_eye(xyz: np.ndarray, rgb: np.ndarray, side: str, prop: dict) -> dict:
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    sat = rgb.max(axis=1) - rgb.min(axis=1)
    z0, z1 = float(prop["eye_z_range"][0]), float(prop["eye_z_range"][1])
    fz0, fz1 = float(prop["face_z_range"][0]), float(prop["face_z_range"][1])

    # Data-driven front-face plane & half-width (scale-invariant across 2.5-8 head-tall characters)
    band_m = (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
    y_front = float(np.percentile(xyz[band_m, 1], 5))
    front_band = band_m & (xyz[:, 1] <= y_front + 0.025)
    w_face_half = float(np.percentile(np.abs(xyz[front_band, 0]), 90)) if front_band.sum() >= 20 else 0.075
    x_min = max(0.006, w_face_half * 0.12)
    x_max = min(0.085, w_face_half * 0.72)
    y_max = y_front + max(0.012, w_face_half * 0.22)

    mask_x = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) if side == "eye_L" else (xyz[:, 0] < -x_min) & (xyz[:, 0] > -x_max)
    eye_win = mask_x & (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1) & (xyz[:, 1] <= y_max)

    # Prefer bright sclera + colored iris on the front face plane to avoid hair/eyebrow contamination
    m_sclera_init = eye_win & (lum >= 0.78) & (sat < 0.16)
    m_dark = eye_win & (lum < 0.52)
    pts = xyz[m_sclera_init] if m_sclera_init.sum() >= 20 else xyz[m_dark]
    if len(pts) < 10:
        raise RuntimeError(f"Failed to detect {side} eye points in Z=[{z0},{z1}] (found {len(pts)})")

    p05, p50, p95 = np.percentile(pts, [5, 50, 95], axis=0)
    surf_center = np.array([0.5 * (p05[0] + p95[0]), float(np.percentile(pts[:, 1], 35)), 0.5 * (p05[2] + p95[2])], dtype=np.float32)

    r_eye = float(prop["eyeball_radius"])
    excl = [float(v) for v in prop["socket_exclusion_radius"]]
    d_xy = np.hypot(xyz[:, 0] - surf_center[0], xyz[:, 2] - surf_center[2])
    m_sclera = (
        (d_xy <= max(r_eye * 1.15, 0.010))
        & (xyz[:, 1] <= surf_center[1] + r_eye * 0.6) & (xyz[:, 1] >= surf_center[1] - r_eye * 0.6)
        & (lum >= 0.78) & (sat < 0.15)
    )
    sclera_rgb = np.median(rgb[m_sclera], axis=0) if m_sclera.sum() >= 5 else np.array([0.93, 0.91, 0.89], dtype=np.float32)

    eye_cz = float(surf_center[2])
    return {
        "detected_iris_points": int(len(pts)),
        "surface_center": surf_center.tolist(),
        "iris_span": (p95 - p05).tolist(),
        "eyeball_center": [float(surf_center[0]), float(surf_center[1] + r_eye * 0.88), eye_cz],
        "eyeball_radius": r_eye,
        "carve_center": [float(surf_center[0]), float(surf_center[1] + r_eye * 0.12), eye_cz],
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

    eye_L = detect_eye(xyz, rgb, "eye_L", prop)
    eye_R = detect_eye(xyz, rgb, "eye_R", prop)

    # Symmetrize eye_L / eye_R in Y, Z, and |X - face_cx| so Look-At vergence is strictly symmetric
    face_cx = 0.5 * (eye_L["eyeball_center"][0] + eye_R["eyeball_center"][0])
    ex_half = 0.5 * (abs(eye_L["eyeball_center"][0] - face_cx) + abs(eye_R["eyeball_center"][0] - face_cx))
    ey_sym = 0.5 * (eye_L["eyeball_center"][1] + eye_R["eyeball_center"][1])
    ez_sym = 0.5 * (eye_L["eyeball_center"][2] + eye_R["eyeball_center"][2])
    sy_sym = 0.5 * (eye_L["surface_center"][1] + eye_R["surface_center"][1])
    for ed, sgn in ((eye_L, +1.0), (eye_R, -1.0)):
        ed["eyeball_center"] = [float(face_cx + sgn * ex_half), float(ey_sym), float(ez_sym)]
        ed["surface_center"] = [float(face_cx + sgn * ex_half), float(sy_sym), float(ez_sym)]
        ed["carve_center"] = [float(face_cx + sgn * ex_half), float(sy_sym + ed["eyeball_radius"] * 0.12), float(ez_sym)]
    fz0, fz1 = float(prop["face_z_range"][0]), float(prop["face_z_range"][1])
    neck_z0 = float(prop["neck_z_range"][0])
    head_y = float(np.median(xyz[(xyz[:, 2] >= fz0) & (xyz[:, 2] <= fz1), 1]))
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
