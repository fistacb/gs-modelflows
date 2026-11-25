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
        m_core = norm_dist2 <= 1.0
        # Also clean stray reddish-dark iris remnants just outside lower/outer socket rim while protecting upper eyelashes
        r_ext = r * np.array([1.18, 1.15, 1.20], dtype=np.float32)
        norm_ext2 = np.sum(((xyz - c[None, :]) / r_ext[None, :]) ** 2, axis=1)
        m_remnant = (norm_ext2 <= 1.0) & is_reddish_dark & (xyz[:, 2] < c[2] + r[2] * 0.55)
        m = m_core | m_remnant
        counts[eye_name] = int(m.sum())
        exclude |= m

    carved = data[~exclude]
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
