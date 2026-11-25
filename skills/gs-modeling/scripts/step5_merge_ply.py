"""Step 5: Transform independent eyeball PLYs by bone pose (position + quaternion) and merge with carved head in NumPy."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ply_io import read_ply, write_ply


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 in [w, x, y, z] format."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    out = np.empty_like(q2)
    out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return out


def transform_part(part: np.ndarray, pos: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Apply bone rotation (yaw around local Z, pitch around local X) and translation to a 3DGS structured array."""
    out = part.copy()
    # Local axes: +X=left, -Y=forward(gaze), +Z=up
    # Yaw (+deg turns gaze toward +X = character's left), Pitch (+deg turns gaze downward, -deg upward)
    rot_obj = Rotation.from_euler("zx", [yaw_deg, pitch_deg], degrees=True)
    R = rot_obj.as_matrix().astype(np.float32)
    q_xyzw = rot_obj.as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

    xyz = np.column_stack([out["x"], out["y"], out["z"]]).astype(np.float32)
    xyz_w = (xyz @ R.T) + pos.astype(np.float32)[None, :]
    out["x"], out["y"], out["z"] = xyz_w[:, 0], xyz_w[:, 1], xyz_w[:, 2]

    q_local = np.column_stack([out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"]]).astype(np.float32)
    q_world = quat_mul_wxyz(q_wxyz, q_local)
    q_world /= np.maximum(np.linalg.norm(q_world, axis=1, keepdims=True), 1e-8)
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = q_world[:, 0], q_world[:, 1], q_world[:, 2], q_world[:, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--eye-L", required=True)
    ap.add_argument("--eye-R", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    ap.add_argument("--pitch-deg", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    head, _ = read_ply(args.head)
    eye_L, _ = read_ply(args.eye_L)
    eye_R, _ = read_ply(args.eye_R)
    skel = json.loads(Path(args.skeleton).read_text(encoding="utf-8"))

    assert head.dtype == eye_L.dtype == eye_R.dtype, (
        f"Schema mismatch: head={head.dtype}, eye_L={eye_L.dtype}, eye_R={eye_R.dtype}"
    )

    pos_L = np.array(skel["bones"]["eye_L"]["head"], dtype=np.float32)
    pos_R = np.array(skel["bones"]["eye_R"]["head"], dtype=np.float32)

    eye_L_w = transform_part(eye_L, pos_L, args.yaw_deg, args.pitch_deg)
    eye_R_w = transform_part(eye_R, pos_R, args.yaw_deg, args.pitch_deg)

    merged = np.concatenate([head, eye_L_w, eye_R_w], axis=0)
    write_ply(args.out, merged)
    print(
        f"Merged -> {args.out}: total={len(merged)} "
        f"(head={len(head)}, eye_L={len(eye_L_w)}, eye_R={len(eye_R_w)}, "
        f"yaw={args.yaw_deg:+.1f}deg, pitch={args.pitch_deg:+.1f}deg)"
    )


if __name__ == "__main__":
    main()
