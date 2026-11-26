"""Step 3: Carve eye sockets from normalized base PLY using ellipsoid exclusion around eye_L and eye_R."""
import json
import sys
from pathlib import Path
import numpy as np
from ply_io import read_ply, write_ply


def main():
    if len(sys.argv) < 4:
        raise SystemExit("Usage: python step3_carve_sockets.py <normalized.ply> <skeleton.json> <out_carved.ply>")
    src = Path(sys.argv[1])
    skel_path = Path(sys.argv[2])
    dst = Path(sys.argv[3])

    data, _ = read_ply(src)
    skel = json.loads(skel_path.read_text(encoding="utf-8"))
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    fdc = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(np.float32)
    rgb = np.clip(0.5 + 0.28209479177387814 * fdc, 0.0, 1.0)
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    is_reddish_dark = (lum < 0.60) & ((rgb[:, 0] - rgb[:, 2]) > 0.07)

    exclude = np.zeros(len(data), dtype=bool)
    counts = {}
    for eye_name in ("eye_L", "eye_R"):
        meta = skel["bones"][eye_name]["meta"]
        c = np.array(meta["carve_center"], dtype=np.float32)
        r = np.array(meta["exclusion_radius"], dtype=np.float32)
        norm_dist2 = np.sum(((xyz - c[None, :]) / r[None, :]) ** 2, axis=1)
        # Protect original thick upper eyelashes (z_top) and lower eyelid cheek rim (z_bot)
        z_top_protect = c[2] + r[2] * 0.72
        z_bot_protect = c[2] - r[2] * 0.85
        m_core = (norm_dist2 <= 1.0) & (xyz[:, 2] <= z_top_protect) & (xyz[:, 2] >= z_bot_protect)
        # Also clean stray reddish-dark iris remnants just outside lower/outer socket rim while protecting upper eyelashes
        r_ext = r * np.array([1.15, 1.15, 1.10], dtype=np.float32)
        norm_ext2 = np.sum(((xyz - c[None, :]) / r_ext[None, :]) ** 2, axis=1)
        m_remnant = (norm_ext2 <= 1.0) & is_reddish_dark & (xyz[:, 2] < c[2] + r[2] * 0.50) & (xyz[:, 2] >= z_bot_protect)
        m = m_core | m_remnant
        counts[eye_name] = int(m.sum())
        exclude |= m

    carved = data[~exclude]

    # Add smooth cheek-fitted lower-eyelid blend cushion so there is zero indentation under the eyes
    op_c = 1.0 / (1.0 + np.exp(-np.clip(carved["opacity"], -20, 20)))
    cx_c, cy_c, cz_c = carved["x"], carved["y"], carved["z"]
    fdc_c = np.column_stack([carved["f_dc_0"], carved["f_dc_1"], carved["f_dc_2"]]).astype(np.float32)
    rgb_c = np.clip(0.5 + 0.28209479177387814 * fdc_c, 0.0, 1.0)
    lum_c = 0.299 * rgb_c[:, 0] + 0.587 * rgb_c[:, 1] + 0.114 * rgb_c[:, 2]
    cushions = []
    for eye_name, side_sgn in (("eye_L", +1.0), ("eye_R", -1.0)):
        meta = skel["bones"][eye_name]["meta"]
        sc = np.array(meta["surface_center"], dtype=np.float32)
        r = np.array(meta["exclusion_radius"], dtype=np.float32)
        sx, sy, sz = float(sc[0]), float(sc[1]), float(sc[2])
        m_ch = (
            (op_c > 0.30) & (lum_c > 0.72) & (rgb_c[:, 0] > rgb_c[:, 2] + 0.04)
            & (np.abs(cx_c - sx) < r[0] * 1.6)
            & (cz_c > sz - r[2] * 2.4) & (cz_c < sz - r[2] * 0.55)
            & (cy_c < sy + 0.018) & (cy_c > sy - 0.022)
        )
        if m_ch.sum() >= 20:
            skin_rgb = np.median(rgb_c[m_ch], axis=0).astype(np.float32)
            dx_s, dz_s = cx_c[m_ch] - sx, cz_c[m_ch] - sz
            A = np.column_stack([np.ones_like(dx_s), dx_s, dz_s, dx_s ** 2, dz_s ** 2])
            coef, *_ = np.linalg.lstsq(A, cy_c[m_ch], rcond=None)
            us = np.linspace(-1.0, 1.0, 36)
            vs = np.linspace(0.0, 1.0, 6)
            pts_xyz, pts_rgb, pts_op = [], [], []
            rx_c, rz_c = float(r[0] * 0.98), float(r[2])
            for v in vs:
                for u in us:
                    dx_p = side_sgn * u * rx_c
                    z_top = sz - rz_c * 0.60 - rz_c * 0.10 * (1.0 - u ** 2) + rz_c * 0.06 * u
                    z_bot = z_top - rz_c * 0.45
                    dz_p = (z_bot + v * (z_top - z_bot)) - sz
                    y_fit = float(coef[0] + coef[1] * dx_p + coef[2] * dz_p + coef[3] * (dx_p ** 2) + coef[4] * (dz_p ** 2))
                    pts_xyz.append([sx + dx_p, y_fit - 0.0006 * v, sz + dz_p])
                    pts_rgb.append(skin_rgb * (1.0 - 0.03 * (v ** 2)))
                    pts_op.append(6.0 * float(np.clip(1.0 - (u / 1.06) ** 4, 0.25, 1.0)))
            pts_xyz = np.asarray(pts_xyz, dtype=np.float32)
            pts_rgb = np.asarray(pts_rgb, dtype=np.float32)
            c_arr = np.zeros(len(pts_xyz), dtype=carved.dtype)
            c_arr["x"], c_arr["y"], c_arr["z"] = pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2]
            cfdc = (pts_rgb - 0.5) / 0.28209479177387814
            c_arr["f_dc_0"], c_arr["f_dc_1"], c_arr["f_dc_2"] = cfdc[:, 0], cfdc[:, 1], cfdc[:, 2]
            c_arr["opacity"] = np.asarray(pts_op, dtype=np.float32)
            sc_v = max(rz_c * 0.13, 0.0012)
            c_arr["scale_0"] = np.log(sc_v)
            c_arr["scale_1"] = np.log(sc_v * 0.45)
            c_arr["scale_2"] = np.log(sc_v)
            c_arr["rot_0"] = 1.0
            cushions.append(c_arr)
    if cushions:
        carved = np.concatenate([carved] + cushions, axis=0)
    write_ply(dst, carved)

    report = {
        "source": str(src),
        "skeleton": str(skel_path),
        "output": str(dst),
        "n_before": len(data),
        "n_after": len(carved),
        "n_carved_total": int(exclude.sum()),
        "n_carved_per_eye": counts,
    }
    dst.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
