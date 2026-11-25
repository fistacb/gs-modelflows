"""Step 9: High-angle oblique walk + head-turn + blink demo video purely from frozen v1 package (config-driven)."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from ply_io import read_ply, write_ply
from config_loader import load_config, resolve_spark_dir
from step4b_eyelid_shell_test import build_eyelid_shell, detect_face_skin_mask
from step8_v1_package_test_video import (
    apply_weighted_rotvec,
    extract_arm_mask,
    quat_mul_wxyz,
    rot_to_quat_wxyz,
    smoothstep,
)

SH_C0 = 0.28209479177387814


def extract_leg_mask(
    xyz_base: np.ndarray,
    is_body: np.ndarray,
    side: str,
    face_cx: float,
    skirt_hem_z: float,
    hip_z: float,
    thigh_cyl_max_xy: list[float],
) -> np.ndarray:
    """Extract left/right leg via 3D connected components from shoe upward.

    CRITICAL: Below `skirt_hem_z`, do NOT clip XY extent so shoe heels, toes, and bows
    are 100% captured. Only apply the narrow thigh cylinder filter inside the skirt
    overlap band `[skirt_hem_z, hip_z + 0.010]`.
    """
    in_side = (xyz_base[:, 0] > face_cx + 0.002) if side == "L" else (xyz_base[:, 0] < face_cx - 0.002)
    below_skirt = xyz_base[:, 2] < skirt_hem_z
    in_thigh_cyl = (
        (xyz_base[:, 2] >= skirt_hem_z)
        & (xyz_base[:, 2] < hip_z + 0.010)
        & (np.abs(xyz_base[:, 0] - face_cx) < float(thigh_cyl_max_xy[0]))
        & (np.abs(xyz_base[:, 1]) < float(thigh_cyl_max_xy[1]))
    )
    cand = np.where(is_body & in_side & (below_skirt | in_thigh_cyl))[0]
    if len(cand) == 0:
        return np.zeros(len(xyz_base), dtype=bool)
    cxyz = xyz_base[cand]
    adj = cKDTree(cxyz).sparse_distance_matrix(cKDTree(cxyz), max_distance=0.018, output_type="coo_matrix")
    _, labels = connected_components(adj, directed=False)
    shoe_local = int(np.argmin(cxyz[:, 2]))
    leg_idx = cand[labels == labels[shoe_local]]
    m = np.zeros(len(xyz_base), dtype=bool)
    m[leg_idx] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ver-dir", default=None)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--fps", type=int, default=24)
    args = ap.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])
    prop, cols, qa_cfg = cfg["proportions"], cfg["colors"], cfg["qa_thresholds"]
    ver_dir = Path(args.ver_dir) if args.ver_dir else (work_dir / "packages" / cfg["asset_id"] / cfg["version"])
    demo_dir = resolve_spark_dir(work_dir)
    frames_dir = demo_dir / "frames/walk"
    frames_dir.mkdir(parents=True, exist_ok=True)

    master, _ = read_ply(ver_dir / "master.ply")
    skel = json.loads((ver_dir / "rig_skeleton.json").read_text(encoding="utf-8"))
    bones = skel["bones"]

    xyz_base = np.column_stack([master["x"], master["y"], master["z"]]).astype(np.float32)
    q_base = np.column_stack([master["rot_0"], master["rot_1"], master["rot_2"], master["rot_3"]]).astype(np.float32)
    fdc_base = np.column_stack([master["f_dc_0"], master["f_dc_1"], master["f_dc_2"]]).astype(np.float32)
    rgb_base = np.clip(0.5 + SH_C0 * fdc_base, 0.0, 1.0)

    face_cx = float(bones["root"]["head"][0])
    pos_L = np.array(bones["eye_L"]["head"], dtype=np.float32)
    pos_R = np.array(bones["eye_R"]["head"], dtype=np.float32)
    surf_L = np.array(bones["eye_L"]["meta"]["surface_center"], dtype=np.float32)
    surf_R = np.array(bones["eye_R"]["meta"]["surface_center"], dtype=np.float32)
    r_eye = float(bones["eye_L"]["meta"]["eyeball_radius"])

    is_eye_L = np.linalg.norm(xyz_base - pos_L[None, :], axis=1) <= (r_eye + 0.0025)
    is_eye_R = np.linalg.norm(xyz_base - pos_R[None, :], axis=1) <= (r_eye + 0.0025)
    is_body = ~(is_eye_L | is_eye_R)

    # Head weight
    w_head = smoothstep(float(prop["neck_z_range"][0]), float(prop["neck_z_range"][1]), xyz_base[:, 2]).astype(np.float32)
    w_head[is_eye_L | is_eye_R] = 1.0
    head_piv = np.array(bones["head"]["head"], dtype=np.float32)

    # Left & Right arms (connected components + bone-axis weights)
    is_arm_R = extract_arm_mask(xyz_base, "R", prop, face_cx) & is_body
    is_arm_L = extract_arm_mask(xyz_base, "L", prop, face_cx) & is_body
    sh_R = np.array(bones["upper_arm_R"]["head"], dtype=np.float32)
    el_R = np.array(bones["forearm_R"]["head"], dtype=np.float32)
    hand_R = np.array(bones["hand_R"]["tail"], dtype=np.float32)
    sh_L = np.array(bones["upper_arm_L"]["head"], dtype=np.float32)
    el_L = np.array(bones["forearm_L"]["head"], dtype=np.float32)
    hand_L = np.array(bones["hand_L"]["tail"], dtype=np.float32)

    def arm_w(is_a, sh, el, hand):
        va = hand - sh
        la = float(np.linalg.norm(va))
        ua = va / max(la, 1e-6)
        s = np.sum((xyz_base - sh[None, :]) * ua[None, :], axis=1) / max(la, 1e-6)
        s_el = float(np.dot(el - sh, ua) / max(la, 1e-6))
        perp = np.linalg.norm((xyz_base - sh[None, :]) - (s * la)[:, None] * ua[None, :], axis=1)
        frill = 1.0 - 0.45 * (1.0 - smoothstep(0.14, 0.28, s)) * smoothstep(0.024, 0.046, perp)
        wa = (is_a.astype(np.float32) * smoothstep(0.04, 0.14, s) * frill).astype(np.float32)
        wf = (smoothstep(s_el - 0.065, s_el + 0.065, s) * wa).astype(np.float32)
        return wa, wf

    w_arm_R, w_fore_R = arm_w(is_arm_R, sh_R, el_R, hand_R)
    w_arm_L, w_fore_L = arm_w(is_arm_L, sh_L, el_L, hand_L)

    # Left & Right legs via strict 3D connected components from shoes upward
    skirt_hem_z = float(prop.get("skirt_hem_z", 0.215))
    hip_z = float(prop.get("hip_z", 0.285))
    hip_hw = float(prop.get("hip_half_width", 0.042))
    thigh_cyl_max_xy = prop.get("thigh_cyl_max_xy", [0.105, 0.095])
    arm_out_deg = float(prop.get("walk_arm_outward_deg", 8.0))

    hip_L = np.array([face_cx + hip_hw, 0.0, hip_z], dtype=np.float32)
    hip_R = np.array([face_cx - hip_hw, 0.0, hip_z], dtype=np.float32)

    is_leg_L = extract_leg_mask(xyz_base, is_body, "L", face_cx, skirt_hem_z, hip_z, thigh_cyl_max_xy)
    is_leg_R = extract_leg_mask(xyz_base, is_body, "R", face_cx, skirt_hem_z, hip_z, thigh_cyl_max_xy)

    # Automated numeric QA: check shoe/heel orphan ratio below skirt hem
    shoe_band = is_body & (xyz_base[:, 2] < skirt_hem_z * 0.65)
    shoe_orphans = shoe_band & (~(is_leg_L | is_leg_R))
    shoe_orphan_ratio = float(shoe_orphans.sum()) / max(int(shoe_band.sum()), 1)
    min_leg_pts = int(qa_cfg.get("min_leg_points", 500))
    max_orphan = float(qa_cfg.get("max_shoe_orphan_ratio", 0.02))
    leg_qa_pass = (int(is_leg_L.sum()) >= min_leg_pts) and (int(is_leg_R.sum()) >= min_leg_pts) and (shoe_orphan_ratio <= max_orphan)
    print(
        f"[Walk QA] leg_L={int(is_leg_L.sum())} pts, leg_R={int(is_leg_R.sum())} pts, "
        f"shoe_orphan_ratio={shoe_orphan_ratio:.4f} (limit {max_orphan}) -> {'PASS' if leg_qa_pass else 'FAIL'}"
    )

    # Smooth hip blend only deep inside skirt; 1.0 rigid across entire exposed leg & shoe
    w_leg_L = (is_leg_L.astype(np.float32) * (1.0 - smoothstep(hip_z - 0.030, hip_z + 0.005, xyz_base[:, 2]))).astype(np.float32)
    w_leg_R = (is_leg_R.astype(np.float32) * (1.0 - smoothstep(hip_z - 0.030, hip_z + 0.005, xyz_base[:, 2]))).astype(np.float32)

    # Skirt strictly excludes both legs & arms
    skirt_z0, skirt_z1 = float(prop["skirt_z_range"][0]), float(prop["skirt_z_range"][1])
    is_skirt = is_body & (~is_arm_R) & (~is_arm_L) & (~is_leg_L) & (~is_leg_R) & (xyz_base[:, 2] >= skirt_z0) & (xyz_base[:, 2] <= skirt_z1)
    w_skirt = (is_skirt.astype(np.float32) * (1.0 - smoothstep(skirt_z0 + 0.03, skirt_z1 - 0.02, xyz_base[:, 2]))).astype(np.float32)

    # Facial skin pool for independent eyelid shells
    face_skin_mask = detect_face_skin_mask(xyz_base, rgb_base, is_body, surf_L, surf_R, prop["face_z_range"])
    skin_xyz_pool, skin_fdc_pool = xyz_base[face_skin_mask], fdc_base[face_skin_mask]
    skin_scale_med = float(np.exp(np.median(master["scale_0"][face_skin_mask])))
    in_rim_L = (
        is_body & (np.abs(xyz_base[:, 0] - surf_L[0]) < 0.034)
        & (xyz_base[:, 1] < surf_L[1] + 0.016)
        & (xyz_base[:, 2] >= surf_L[2] - 0.018) & (xyz_base[:, 2] <= surf_L[2] + 0.025)
        & (~face_skin_mask)
    )
    in_rim_R = (
        is_body & (np.abs(xyz_base[:, 0] - surf_R[0]) < 0.034)
        & (xyz_base[:, 1] < surf_R[1] + 0.016)
        & (xyz_base[:, 2] >= surf_R[2] - 0.018) & (xyz_base[:, 2] <= surf_R[2] + 0.025)
        & (~face_skin_mask)
    )

    R_spark_obj = Rotation.from_euler("x", 90.0, degrees=True)
    R_spark = R_spark_obj.as_matrix().astype(np.float32)
    q_spark = rot_to_quat_wxyz(R_spark_obj)

    n_frames = args.frames
    fps = args.fps
    manifest_frames = []

    for f in range(n_frames):
        t = f / n_frames
        stride = 2.0 * math.pi * 2.0 * t
        s_val = math.sin(stride)

        head_yaw = 24.0 * math.sin(2.0 * math.pi * t)
        head_roll = 7.5 * math.sin(2.0 * math.pi * t)
        head_pitch = 4.0 * math.sin(2.0 * stride)
        eye_yaw = 14.0 * math.sin(2.0 * math.pi * t + 0.3)
        blink = max(math.exp(-((t - 0.28) / 0.035) ** 2), math.exp(-((t - 0.76) / 0.035) ** 2))

        xyz_c, q_c = xyz_base.copy(), q_base.copy()

        # 1. Eye gaze + blink retraction
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, is_eye_L.astype(np.float32), pos_L, np.radians([0.0, 0.0, eye_yaw]))
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, is_eye_R.astype(np.float32), pos_R, np.radians([0.0, 0.0, eye_yaw]))
        if blink > 0.05:
            xyz_c[is_eye_L, 1] += 0.0045 * blink
            xyz_c[is_eye_R, 1] += 0.0045 * blink

        # 2. Legs walking: clean stride (±16 deg) + gentle foot lift
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_leg_R, hip_R, np.radians([-16.0 * s_val, 0.0, 0.0]))
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_leg_L, hip_L, np.radians([+16.0 * s_val, 0.0, 0.0]))
        xyz_c[:, 2] += w_leg_R * (0.016 * max(0.0, s_val))
        xyz_c[:, 2] += w_leg_L * (0.016 * max(0.0, -s_val))

        # 3. Arms counter-swinging with outward skirt clearance
        arm_fwd = +16.0 * s_val
        rv_sh_R = Rotation.from_euler("zy", [-arm_fwd, +arm_out_deg], degrees=True).as_rotvec().astype(np.float32)
        rv_sh_L = Rotation.from_euler("zy", [-arm_fwd, -arm_out_deg], degrees=True).as_rotvec().astype(np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_arm_R, sh_R, rv_sh_R)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_arm_L, sh_L, rv_sh_L)
        el_R_c = ((el_R - sh_R) @ Rotation.from_rotvec(rv_sh_R).as_matrix().T).astype(np.float32) + sh_R
        el_L_c = ((el_L - sh_L) @ Rotation.from_rotvec(rv_sh_L).as_matrix().T).astype(np.float32) + sh_L
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_fore_R, el_R_c, np.radians([0.0, 8.0, 12.0]))
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_fore_L, el_L_c, np.radians([0.0, -8.0, -12.0]))

        # 4. Head turn & nod
        rv_head = Rotation.from_euler("zxy", [head_yaw, head_pitch, head_roll], degrees=True).as_rotvec().astype(np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_head, head_piv, rv_head)

        # 5. Skirt hem cloth response (push front hem forward/up on stepping side so knee never clips)
        skirt_idx = np.where(is_skirt)[0]
        sx, sy = xyz_base[skirt_idx, 0] - face_cx, xyz_base[skirt_idx, 1]
        push_R = np.exp(-((sx - (-0.06)) / 0.09) ** 2) * (sy < 0.0).astype(np.float32) * max(0.0, s_val)
        push_L = np.exp(-((sx - (+0.06)) / 0.09) ** 2) * (sy < 0.0).astype(np.float32) * max(0.0, -s_val)
        xyz_c[skirt_idx, 1] -= w_skirt[skirt_idx] * (0.018 * (push_R + push_L))
        xyz_c[skirt_idx, 2] += w_skirt[skirt_idx] * (0.008 * (push_R + push_L))
        xyz_c[:, 0] += w_skirt * (0.012 * s_val)
        bounce_z = 0.014 * abs(s_val)
        xyz_c[:, 2] += bounce_z

        out = master.copy()
        out["x"], out["y"], out["z"] = xyz_c[:, 0], xyz_c[:, 1], xyz_c[:, 2]
        out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = q_c[:, 0], q_c[:, 1], q_c[:, 2], q_c[:, 3]

        if blink > 0.05:
            op_f = float(smoothstep(0.15, 0.70, blink))
            out["opacity"][in_rim_L] = (1.0 - op_f) * out["opacity"][in_rim_L] + op_f * (-12.0)
            out["opacity"][in_rim_R] = (1.0 - op_f) * out["opacity"][in_rim_R] + op_f * (-12.0)
            if blink >= 0.90:
                out["opacity"][is_eye_L] = -12.0
                out["opacity"][is_eye_R] = -12.0

        parts = [out]
        if blink > 0.03:
            for p_eye, sc_c in ((pos_L, surf_L), (pos_R, surf_R)):
                lid = build_eyelid_shell(
                    p_eye, sc_c, skin_xyz_pool, skin_fdc_pool, skin_scale_med, blink, master.dtype,
                    prop["eyelid_shell_rx_rz"], prop["eyelid_lash_top_bottom_z"], cols["lash_rgb"],
                )
                if len(lid) > 0:
                    lxyz = np.column_stack([lid["x"], lid["y"], lid["z"]]).astype(np.float32)
                    lq = np.column_stack([lid["rot_0"], lid["rot_1"], lid["rot_2"], lid["rot_3"]]).astype(np.float32)
                    lxyz, lq = apply_weighted_rotvec(lxyz, lq, np.ones(len(lid), dtype=np.float32), head_piv, rv_head)
                    lxyz[:, 2] += bounce_z
                    lid["x"], lid["y"], lid["z"] = lxyz[:, 0], lxyz[:, 1], lxyz[:, 2]
                    lid["rot_0"], lid["rot_1"], lid["rot_2"], lid["rot_3"] = lq[:, 0], lq[:, 1], lq[:, 2], lq[:, 3]
                    parts.append(lid)

        merged = np.concatenate(parts, axis=0)
        m_xyz = (np.column_stack([merged["x"], merged["y"], merged["z"]]).astype(np.float32) @ R_spark.T).astype(np.float32)
        m_q = quat_mul_wxyz(q_spark, np.column_stack([merged["rot_0"], merged["rot_1"], merged["rot_2"], merged["rot_3"]]).astype(np.float32))
        m_q /= np.maximum(np.linalg.norm(m_q, axis=1, keepdims=True), 1e-8)
        merged["x"], merged["y"], merged["z"] = m_xyz[:, 0], m_xyz[:, 1], m_xyz[:, 2]
        merged["rot_0"], merged["rot_1"], merged["rot_2"], merged["rot_3"] = m_q[:, 0], m_q[:, 1], m_q[:, 2], m_q[:, 3]

        rel = f"frames/walk/frame_{f + 1:04d}.ply"
        write_ply(demo_dir / rel, merged)
        manifest_frames.append({"frame": f + 1, "blenderFrame": f + 1, "png": f"frame-{f + 1:04d}.png", "ply": [rel]})

    (frames_dir / "manifest.json").write_text(json.dumps({"fps": fps, "static": [], "review": [], "frames": manifest_frames}, indent=2), encoding="utf-8")
    print(f"Generated {n_frames} oblique walk frames in {frames_dir}")


if __name__ == "__main__":
    main()
