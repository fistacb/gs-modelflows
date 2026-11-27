"""Step 4: Generate high-density soft anime/doll 3DGS eyeballs using config palette (or auto-sampled from source iris)."""
import argparse
import json
from pathlib import Path
import numpy as np
from ply_io import read_ply, write_ply
from config_loader import load_config

SH_C0 = 0.28209479177387814


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]).astype(np.float32)


def fibonacci_front_cap(n: int, max_angle_deg: float = 72.0) -> np.ndarray:
    cos_min = float(np.cos(np.radians(max_angle_deg)))
    i = np.arange(n, dtype=np.float64) + 0.5
    cos_a = 1.0 - (1.0 - cos_min) * (i / n)
    sin_a = np.sqrt(np.maximum(0.0, 1.0 - cos_a * cos_a))
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([sin_a * np.cos(theta), -cos_a, sin_a * np.sin(theta)]).astype(np.float32)


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


def derive_iris_palette(cfg_colors: dict, src_ply: Path | None, skel: dict) -> dict:
    """Use 2D design-art eye_style palette from skeleton.json (when auto_sample enabled), else config/3D fallback."""
    pal = {
        "bot": np.array(cfg_colors["iris_bottom_rgb"], dtype=np.float32),
        "mid": np.array(cfg_colors["iris_mid_rgb"], dtype=np.float32),
        "top": np.array(cfg_colors["iris_top_rgb"], dtype=np.float32),
        "glow": np.array(cfg_colors["iris_glow_rgb"], dtype=np.float32),
        "pupil": np.array(cfg_colors["pupil_rgb"], dtype=np.float32),
        "rim": np.array(cfg_colors["rim_rgb"], dtype=np.float32),
    }
    eye_style = skel.get("eye_style", {})
    if cfg_colors.get("auto_sample_iris_from_source", True) and "iris_bottom_rgb" in eye_style and "iris_mid_rgb" in eye_style:
        pal["top"] = np.array(eye_style["iris_top_rgb"], dtype=np.float32)
        pal["mid"] = np.array(eye_style["iris_mid_rgb"], dtype=np.float32)
        pal["bot"] = np.array(eye_style["iris_bottom_rgb"], dtype=np.float32)
        pal["glow"] = np.array(eye_style["iris_glow_rgb"], dtype=np.float32)
        pal["pupil"] = np.clip(pal["top"] * 0.65, 0.04, 0.40)
        pal["rim"] = np.clip(pal["top"] * 0.85, 0.06, 0.45)
        print(f"[AutoIris2D] Using design-art gradient top={pal['top'].round(3).tolist()} mid={pal['mid'].round(3).tolist()} bot={pal['bot'].round(3).tolist()} glow={pal['glow'].round(3).tolist()}")
        return pal
    if cfg_colors.get("auto_sample_iris_from_source") and src_ply and src_ply.exists():
        d, _ = read_ply(src_ply)
        xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
        fdc = np.column_stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]]).astype(np.float32)
        rgb = np.clip(0.5 + SH_C0 * fdc, 0.0, 1.0)
        lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        sat = rgb.max(axis=1) - rgb.min(axis=1)
        samples = []
        for k in ("eye_L", "eye_R"):
            sc = np.array(skel["bones"][k]["meta"]["surface_center"], dtype=np.float32)
            r_sample = max(float(skel["bones"][k]["meta"].get("eyeball_radius", 0.02)) * 0.72, 0.006)
            m = (np.linalg.norm(xyz - sc[None, :], axis=1) < r_sample) & (lum > 0.10) & (lum < 0.56) & (sat > 0.24)
            if m.sum() >= 5:
                samples.append(rgb[m])
        if samples:
            all_s = np.concatenate(samples, axis=0)
            base = np.median(all_s, axis=0).astype(np.float32)
            pal["mid"] = base
            pal["bot"] = np.clip(base * 1.35 + 0.12, 0.0, 1.0)
            pal["top"] = np.clip(base * 0.45, 0.0, 1.0)
            pal["glow"] = np.clip(base * 1.45 + 0.28, 0.0, 1.0)
            pal["pupil"] = np.clip(base * 0.30, 0.0, 1.0)
            pal["rim"] = np.clip(base * 0.42, 0.0, 1.0)
            print(f"[AutoIrisPalette] Sampled mid_rgb={base.round(3).tolist()} from {len(all_s)} eye splats")
    return pal


def build_eyeball_ply(
    radius: float,
    sclera_rgb: np.ndarray,
    pal: dict,
    n_back: int = 1600,
    n_front: int = 3600,
    white_only: bool = False,
    cheek_yaw_deg: float = 0.0,
    cheek_pitch_deg: float = 0.0,
    iris_rx_rz_deg: tuple[float, float] = (40.0, 48.0),
) -> np.ndarray:
    n_full = fibonacci_sphere(n_back)
    n_b = n_full[n_full[:, 1] > -float(np.cos(np.radians(74.0)))]
    n_f = fibonacci_front_cap(n_front, max_angle_deg=76.0)

    normals = np.concatenate([n_b, n_f], axis=0)
    n_total = len(normals)
    xyz = normals * float(radius)

    r_disc_b = float(np.sqrt((4.0 * np.pi * (radius ** 2) / n_back) / np.pi) * 1.20)
    cap_area = 2.0 * np.pi * (radius ** 2) * (1.0 - np.cos(np.radians(76.0)))
    r_disc_f = float(np.sqrt((cap_area / n_front) / np.pi) * 1.15)
    r_disc = np.concatenate([np.full(len(n_b), r_disc_b, dtype=np.float32), np.full(len(n_f), r_disc_f, dtype=np.float32)])

    rot = normal_to_quat_wxyz(normals)
    scale = np.column_stack([np.log(r_disc), np.log(r_disc), np.log(r_disc * 0.25)]).astype(np.float32)

    front_y = np.maximum(-normals[:, 1], 1e-5)
    # Paint iris straight forward (-Y) so rest gaze is strictly parallel and look_at_eye_rot converges cleanly on camera
    # (offsetting ax by +/-cheek_yaw_deg caused outward wall-eyed divergence!).
    ax = np.degrees(np.arctan2(normals[:, 0], front_y))
    az = np.degrees(np.arctan2(normals[:, 2], front_y))
    is_front = normals[:, 1] < 0.0

    rgb = np.tile(sclera_rgb.astype(np.float32)[None, :], (n_total, 1))
    rgb *= (1.0 - 0.16 * smoothstep(8.0, 44.0, az))[:, None]

    if not white_only:
        az_c = 0.0
        rx_iris, rz_iris = float(iris_rx_rz_deg[0]), float(iris_rx_rz_deg[1])
        d_iris = np.sqrt((ax / rx_iris) ** 2 + ((az - az_c) / rz_iris) ** 2)
        t_v = np.clip((az - (az_c - rz_iris)) / (2.0 * rz_iris), 0.0, 1.0)
        iris_col = np.where(
            (t_v < 0.45)[:, None],
            pal["bot"][None, :] * (1.0 - (t_v / 0.45)[:, None]) + pal["mid"][None, :] * (t_v / 0.45)[:, None],
            pal["mid"][None, :] * (1.0 - ((t_v - 0.45) / 0.55)[:, None]) + pal["top"][None, :] * ((t_v - 0.45) / 0.55)[:, None],
        )
        # Rich luminous lower-iris anime crescent glow (gives warm, gentle, sparkling look!)
        d_glow = np.sqrt((ax / (rx_iris * 0.72)) ** 2 + ((az - (az_c - rz_iris * 0.40)) / (rz_iris * 0.44)) ** 2)
        w_glow = (1.0 - smoothstep(0.15, 1.0, d_glow)) * 0.84
        iris_col = iris_col * (1.0 - w_glow[:, None]) + pal["glow"][None, :] * w_glow[:, None]

        # Slightly smaller, softly-blended pupil so the eyes look warm and friendly (not stark/intense)
        d_pupil = np.sqrt((ax / (rx_iris * 0.30)) ** 2 + ((az - (az_c + 1.5)) / (rz_iris * 0.32)) ** 2)
        w_pupil = (1.0 - smoothstep(0.55, 1.05, d_pupil)) * 0.90
        iris_col = iris_col * (1.0 - w_pupil[:, None]) + pal["pupil"][None, :] * w_pupil[:, None]

        w_rim = smoothstep(0.78, 0.94, d_iris) * (1.0 - smoothstep(0.96, 1.06, d_iris))
        iris_col = iris_col * (1.0 - 0.62 * w_rim[:, None]) + pal["rim"][None, :] * (0.62 * w_rim[:, None])

        w_iris = (1.0 - smoothstep(0.95, 1.05, d_iris)) * is_front.astype(np.float32)
        rgb = rgb * (1.0 - w_iris[:, None]) + iris_col * w_iris[:, None]

        for hx, hz, r0, r1, col in [
            (-rx_iris * 0.34, az_c + rz_iris * 0.28, 7.2, 11.2, [1.00, 0.99, 0.98]),
            (rx_iris * 0.30, az_c - rz_iris * 0.32, 3.8, 6.2, [1.00, 0.96, 0.92]),
        ]:
            w_hl = (1.0 - smoothstep(r0, r1, np.hypot(ax - hx, az - hz))) * is_front.astype(np.float32)
            rgb = rgb * (1.0 - w_hl[:, None]) + np.array(col, dtype=np.float32)[None, :] * w_hl[:, None]

    # Keep wide sclera coverage (~74 deg H / ~64 deg V) so gaze +/-18 deg & +60 deg never show jagged edges,
    # while trimming rear hemisphere so it never pokes through cheek/temple.
    keep = (normals[:, 1] < -0.15) & (np.abs(ax) < 74.0) & (np.abs(az) < 64.0)
    xyz, rgb, scale, rot = xyz[keep], rgb[keep], scale[keep], rot[keep]
    n_kept = len(xyz)

    fdc = (np.clip(rgb, 0.0, 1.0) - 0.5) / SH_C0
    out_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"), ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    out = np.zeros(n_kept, dtype=out_dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = fdc[:, 0], fdc[:, 1], fdc[:, 2]
    out["opacity"] = np.full(n_kept, 7.5, dtype=np.float32)
    out["scale_0"], out["scale_1"], out["scale_2"] = scale[:, 0], scale[:, 1], scale[:, 2]
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skeleton", default=None)
    ap.add_argument("--source-ply", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--white-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])
    skel_path = Path(args.skeleton) if args.skeleton else (work_dir / "outputs" / "step2_skeleton.json")
    src_ply = Path(args.source_ply) if args.source_ply else (work_dir / "outputs" / "step1_normalized.ply")
    out_dir = Path(args.out_dir) if args.out_dir else (work_dir / "outputs")

    skel = json.loads(skel_path.read_text(encoding="utf-8"))
    pal = derive_iris_palette(cfg["colors"], src_ply, skel)

    rgb_L = np.array(skel["bones"]["eye_L"]["meta"]["sclera_rgb"], dtype=np.float32)
    rgb_R = np.array(skel["bones"]["eye_R"]["meta"]["sclera_rgb"], dtype=np.float32)
    sclera_rgb = np.clip(0.5 * (0.5 * (rgb_L + rgb_R)) + 0.5 * np.array([0.93, 0.91, 0.89], dtype=np.float32), 0.0, 0.96)

    eye_style = skel.get("eye_style", {})
    cfg_iris = cfg.get("proportions", {}).get("iris_rx_rz_deg")
    if cfg.get("colors", {}).get("auto_sample_iris_from_source", True) and "iris_rx_rz_deg" in eye_style:
        iris_rx_rz = tuple(float(v) for v in eye_style["iris_rx_rz_deg"])
    elif cfg_iris is not None:
        iris_rx_rz = tuple(float(v) for v in cfg_iris)
    else:
        iris_rx_rz = (44.0, 52.0)
    suffix = "_white" if args.white_only else ""
    for eye_name in ("eye_L", "eye_R"):
        meta = skel["bones"][eye_name]["meta"]
        r = float(meta["eyeball_radius"])
        cyaw = float(meta.get("cheek_yaw_deg", 0.0))
        cpitch = float(meta.get("cheek_pitch_deg", 0.0))
        arr = build_eyeball_ply(
            r, sclera_rgb, pal, white_only=args.white_only,
            cheek_yaw_deg=cyaw, cheek_pitch_deg=cpitch, iris_rx_rz_deg=iris_rx_rz,
        )
        out_path = out_dir / f"step4_{eye_name}{suffix}_local.ply"
        write_ply(out_path, arr)
        print(f"Saved {out_path} ({len(arr)} splats, r={r:.4f}, iris={iris_rx_rz})")


if __name__ == "__main__":
    main()
