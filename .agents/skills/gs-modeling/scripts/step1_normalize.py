"""Step 1: Normalize 3DGS PLY (schema unification, outlier removal, Z-up / -Y-forward alignment, height=1.0, feet at Z=0)."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from ply_io import read_ply, write_ply


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 in [w, x, y, z] format. q1 can be (4,), q2 (N,4)."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    out = np.empty_like(q2)
    out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python step1_normalize.py <input_raw.ply> <output_normalized.ply>")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    data, _ = read_ply(src)
    n_orig = len(data)

    # 1. Schema normalization: ensure opacity in logit, scale in log, rot normalized [w,x,y,z], SH degree 0
    op = data["opacity"].astype(np.float32).copy()
    if op.min() >= 0.0 and op.max() <= 1.0:
        op_clamped = np.clip(op, 1e-5, 1.0 - 1e-5)
        op = np.log(op_clamped / (1.0 - op_clamped))
    op_prob = 1.0 / (1.0 + np.exp(-np.clip(op, -20.0, 20.0)))

    sc = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float32)
    if sc.min() >= 0.0:
        sc = np.log(np.maximum(sc, 1e-6))

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    rot = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float32)
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot = np.where(rot_norm > 1e-8, rot / np.maximum(rot_norm, 1e-8), np.array([1, 0, 0, 0], dtype=np.float32))
    fdc = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]).astype(np.float32)

    # 2. Axis alignment: TripoSplat (front=+X, up=-Y) -> Blender standard (front=-Y, up=+Z)
    # R: [x_new, y_new, z_new] = [z_old, -x_old, -y_old], det(R) = +1
    R = np.array([
        [0.0,  0.0,  1.0],
        [-1.0, 0.0,  0.0],
        [0.0, -1.0,  0.0],
    ], dtype=np.float64)
    assert np.isclose(np.linalg.det(R), 1.0), "R must be a proper rotation"
    q_xyzw = Rotation.from_matrix(R).as_quat()
    q_R = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    xyz = (xyz @ R.T).astype(np.float32)
    rot = quat_mul_wxyz(q_R, rot)
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8)

    # 2b. Pitch-lean correction (TripoSplat camera elevation bias leaves ~25-30 deg backward lean)
    z_min_tmp, z_max_tmp = float(np.percentile(xyz[:, 2], 1)), float(np.percentile(xyz[:, 2], 99))
    h_tmp = z_max_tmp - z_min_tmp
    m_feet = (xyz[:, 2] >= z_min_tmp + 0.02 * h_tmp) & (xyz[:, 2] <= z_min_tmp + 0.12 * h_tmp)
    m_head = (xyz[:, 2] >= z_min_tmp + 0.65 * h_tmp) & (xyz[:, 2] <= z_min_tmp + 0.85 * h_tmp)
    c_feet = np.median(xyz[m_feet], axis=0)
    c_head = np.median(xyz[m_head], axis=0)
    pitch_rad = float(np.arctan2(c_head[1] - c_feet[1], c_head[2] - c_feet[2]))
    pitch_deg = float(np.degrees(pitch_rad))
    R_pitch = Rotation.from_euler("x", pitch_rad).as_matrix()
    q_p_xyzw = Rotation.from_matrix(R_pitch).as_quat()
    q_pitch = np.array([q_p_xyzw[3], q_p_xyzw[0], q_p_xyzw[1], q_p_xyzw[2]], dtype=np.float32)

    xyz = (xyz @ R_pitch.T).astype(np.float32)
    rot = quat_mul_wxyz(q_pitch, rot)
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8)
    R_total = R_pitch @ R
    q_tot_xyzw = Rotation.from_matrix(R_total).as_quat()
    q_R = np.array([q_tot_xyzw[3], q_tot_xyzw[0], q_tot_xyzw[1], q_tot_xyzw[2]], dtype=np.float32)

    # 3. Outlier & floater removal
    # (a) Remove very low opacity splats
    keep_op = op_prob >= 0.05
    # (b) Remove extreme scale splats
    max_sc = np.max(sc, axis=1)
    keep_sc = max_sc <= np.percentile(max_sc, 99.8) + 0.5
    # (c) Statistical outlier removal using kNN distance (k=16)
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=16)
    mean_d = dists[:, 1:].mean(axis=1)
    d_thresh = float(np.median(mean_d) + 6.0 * (np.percentile(mean_d, 75) - np.percentile(mean_d, 25)))
    keep_knn = mean_d <= d_thresh

    keep = keep_op & keep_sc & keep_knn
    xyz, sc, rot, fdc, op = xyz[keep], sc[keep], rot[keep], fdc[keep], op[keep]
    n_kept = len(xyz)

    # 4. Scale normalization (height = 1.0) & origin grounding (feet at Z=0, XY centered)
    z_p01, z_p99 = float(np.percentile(xyz[:, 2], 0.5)), float(np.percentile(xyz[:, 2], 99.5))
    raw_height = z_p99 - z_p01
    scale_factor = 1.0 / raw_height

    cx = float(np.median(xyz[:, 0]))
    cy = float(np.median(xyz[:, 1]))
    xyz[:, 0] = (xyz[:, 0] - cx) * scale_factor
    xyz[:, 1] = (xyz[:, 1] - cy) * scale_factor
    xyz[:, 2] = (xyz[:, 2] - z_p01) * scale_factor
    sc = sc + float(np.log(scale_factor))

    # 5. Pack into unified standard 3DGS SH0 schema
    out_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    out = np.zeros(n_kept, dtype=out_dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = fdc[:, 0], fdc[:, 1], fdc[:, 2]
    out["opacity"] = op
    out["scale_0"], out["scale_1"], out["scale_2"] = sc[:, 0], sc[:, 1], sc[:, 2]
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    write_ply(dst, out)

    meta = {
        "source": str(src),
        "output": str(dst),
        "n_orig": n_orig,
        "n_kept": n_kept,
        "n_removed": n_orig - n_kept,
        "removed_low_opacity": int((~keep_op).sum()),
        "removed_extreme_scale": int((~keep_sc).sum()),
        "removed_knn_outliers": int((~keep_knn).sum()),
        "knn_dist_threshold": d_thresh,
        "rotation_matrix": R_total.tolist(),
        "rotation_quat_wxyz": q_R.tolist(),
        "pitch_correction_deg": pitch_deg,
        "raw_height": raw_height,
        "scale_factor": scale_factor,
        "origin_offset_raw": [cx, cy, z_p01],
        "target_convention": "Z-up, -Y-forward, height=1.0, feet at Z=0, SH0, logit opacity, log scale",
        "bbox_min": xyz.min(axis=0).tolist(),
        "bbox_max": xyz.max(axis=0).tolist(),
    }
    meta_path = dst.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
