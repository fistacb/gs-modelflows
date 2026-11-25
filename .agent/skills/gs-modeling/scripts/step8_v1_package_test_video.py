"""Step 8: Anatomical 2-Bone IK + Pole Vector + Isolated Forearm Twist + Face-Fitted Eyelid Shells + Lip-Sync (config-driven)."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
from ply_io import read_ply, write_ply
from config_loader import load_config
from step4b_eyelid_shell_test import build_eyelid_shell, detect_face_skin_mask


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    if q1.ndim == 1:
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    else:
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    out = np.empty_like(q2)
    out[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    out[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return out


def rot_to_quat_wxyz(R_obj: Rotation) -> np.ndarray:
    q = R_obj.as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def smoothstep(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rot_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation vector (axis * angle_rad) rotating unit vector a onto unit vector b."""
    a = a / max(float(np.linalg.norm(a)), 1e-8)
    b = b / max(float(np.linalg.norm(b)), 1e-8)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return np.zeros(3, dtype=np.float32)
    if dot < -0.999999:
        ortho = np.cross(a, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(ortho) < 1e-4:
            ortho = np.cross(a, np.array([1.0, 0.0, 0.0]))
        ortho /= np.linalg.norm(ortho)
        return (ortho * math.pi).astype(np.float32)
    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    return (axis * math.acos(dot)).astype(np.float32)


def solve_2bone_ik(S: np.ndarray, T: np.ndarray, P: np.ndarray, L1: float, L2: float) -> tuple[np.ndarray, np.ndarray]:
    """Analytic 2-bone IK using Law of Cosines + Pole Target P (guarantees anatomically valid elbow direction)."""
    d_vec = T - S
    dist_raw = float(np.linalg.norm(d_vec))
    u_d = d_vec / max(dist_raw, 1e-6)
    d = float(np.clip(dist_raw, abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4))

    pole_vec = P - S
    pole_perp = pole_vec - float(np.dot(pole_vec, u_d)) * u_d
    pole_norm = float(np.linalg.norm(pole_perp))
    if pole_norm < 1e-5:
        pole_perp = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        pole_perp -= float(np.dot(pole_perp, u_d)) * u_d
        pole_norm = float(np.linalg.norm(pole_perp))
    u_pole = (pole_perp / pole_norm).astype(np.float32)

    cos_a = float(np.clip((L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d), -1.0, 1.0))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    E = S + L1 * (cos_a * u_d + sin_a * u_pole)
    T_clamped = S + d * u_d
    return E.astype(np.float32), T_clamped.astype(np.float32)


def apply_weighted_rotvec(xyz: np.ndarray, q: np.ndarray, w: np.ndarray, pivot: np.ndarray, rotvec_rad: np.ndarray):
    angle_total = float(np.linalg.norm(rotvec_rad))
    if angle_total < 1e-5:
        return xyz, q
    act = w > 1e-4
    if not np.any(act):
        return xyz, q

    axis = (rotvec_rad / angle_total).astype(np.float32)
    ang = (angle_total * w[act]).astype(np.float32)
    ca = np.cos(ang)[:, None]
    sa = np.sin(ang)[:, None]

    p = xyz[act] - pivot[None, :]
    k = axis[None, :]
    p_rot = p * ca + np.cross(k, p) * sa + k * np.sum(p * k, axis=1, keepdims=True) * (1.0 - ca)

    half = ang * 0.5
    qd = np.concatenate([np.cos(half)[:, None], k * np.sin(half)[:, None]], axis=1).astype(np.float32)

    xyz_o, q_o = xyz.copy(), q.copy()
    xyz_o[act] = p_rot.astype(np.float32) + pivot[None, :]
    qn = quat_mul_wxyz(qd, q[act])
    qn /= np.maximum(np.linalg.norm(qn, axis=1, keepdims=True), 1e-8)
    q_o[act] = qn
    return xyz_o, q_o


def extract_arm_mask(xyz: np.ndarray, side: str, prop: dict, torso_cx: float = 0.0) -> np.ndarray:
    az0, az1 = float(prop["arm_z_range"][0]), float(prop["arm_z_range"][1])
    cut_x = float(prop["arm_cut_abs_x"])
    if side == "R":
        cand = np.where((xyz[:, 0] < torso_cx - cut_x) & (xyz[:, 2] >= az0) & (xyz[:, 2] <= az1))[0]
        tip_fn = np.argmin
    else:
        cand = np.where((xyz[:, 0] > torso_cx + cut_x) & (xyz[:, 2] >= az0) & (xyz[:, 2] <= az1))[0]
        tip_fn = np.argmax
    cxyz = xyz[cand]
    adj = cKDTree(cxyz).sparse_distance_matrix(cKDTree(cxyz), max_distance=0.016, output_type="coo_matrix")
    _, labels = connected_components(adj, directed=False)
    arm_idx = cand[labels == labels[int(tip_fn(cxyz[:, 0]))]]
    m = np.zeros(len(xyz), dtype=bool)
    m[arm_idx] = True
    return m


def compute_anatomical_arm_weights(xyz: np.ndarray, is_arm: np.ndarray, sh: np.ndarray, el: np.ndarray, wr: np.ndarray, hand: np.ndarray):
    """Separate weights: rigid 1.0 on all exposed upper arm (blend hidden inside sleeve!), 1DOF elbow hinge, 0->100% forearm twist, wrist."""
    v_arm = hand - sh
    len_arm = float(np.linalg.norm(v_arm))
    u_arm = v_arm / max(len_arm, 1e-6)
    s = np.sum((xyz - sh[None, :]) * u_arm[None, :], axis=1) / max(len_arm, 1e-6)
    s_el = float(np.dot(el - sh, u_arm) / max(len_arm, 1e-6))
    s_wr = float(np.dot(wr - sh, u_arm) / max(len_arm, 1e-6))

    # Continuous C1 smooth weight across sleeve-to-arm boundary (zero discontinuous tearing!)
    perp = np.linalg.norm((xyz - sh[None, :]) - (s * len_arm)[:, None] * u_arm[None, :], axis=1)
    # Smoothly reduce follow only on outer frill volume near shoulder; exposed core arm stays 1.0 rigid
    frill_attenuation = 1.0 - 0.45 * (1.0 - smoothstep(0.14, 0.28, s)) * smoothstep(0.024, 0.046, perp)
    w_arm = (is_arm.astype(np.float32) * smoothstep(0.04, 0.14, s) * frill_attenuation).astype(np.float32)

    # 1DOF elbow hinge blend: tight enough that forearm stays straight & rigid (no distortion!)
    w_fore = (smoothstep(s_el - 0.065, s_el + 0.065, s) * w_arm).astype(np.float32)
    # Anatomical radioulnar forearm twist: starts AFTER elbow bend finishes (s_el+0.05 -> s_wr) so elbow+twist never interfere!
    w_fore_twist = (smoothstep(s_el + 0.05, s_wr, s) * w_arm).astype(np.float32)
    # Wrist flexion/deviation
    w_hand = (smoothstep(s_wr - 0.05, s_wr + 0.05, s) * w_arm).astype(np.float32)
    return w_arm, w_fore, w_fore_twist, w_hand, u_arm, s, s_el, perp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ver-dir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    work_dir = Path(cfg["work_dir"])
    prop, cols, qa_cfg = cfg["proportions"], cfg["colors"], cfg["qa_thresholds"]
    ver_dir = Path(args.ver_dir) if args.ver_dir else (work_dir / "packages" / cfg["asset_id"] / cfg["version"])
    demo_dir = work_dir / "spark_demo"
    frames_dir = demo_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    master, _ = read_ply(ver_dir / "master.ply")
    skel = json.loads((ver_dir / "rig_skeleton.json").read_text(encoding="utf-8"))
    bones = skel["bones"]

    xyz_base = np.column_stack([master["x"], master["y"], master["z"]]).astype(np.float32)
    q_base = np.column_stack([master["rot_0"], master["rot_1"], master["rot_2"], master["rot_3"]]).astype(np.float32)

    pos_L = np.array(bones["eye_L"]["head"], dtype=np.float32)
    pos_R = np.array(bones["eye_R"]["head"], dtype=np.float32)
    r_eye = float(bones["eye_L"]["meta"]["eyeball_radius"])
    is_eye_L = np.linalg.norm(xyz_base - pos_L[None, :], axis=1) <= (r_eye + 0.0025)
    is_eye_R = np.linalg.norm(xyz_base - pos_R[None, :], axis=1) <= (r_eye + 0.0025)
    is_body = ~(is_eye_L | is_eye_R)

    face_cx = float(bones["root"]["head"][0])
    is_arm_R = extract_arm_mask(xyz_base, "R", prop, face_cx) & is_body
    is_arm_L = extract_arm_mask(xyz_base, "L", prop, face_cx) & is_body
    w_head = smoothstep(float(prop["neck_z_range"][0]), float(prop["neck_z_range"][1]), xyz_base[:, 2]).astype(np.float32)
    w_head[is_eye_L | is_eye_R] = 1.0

    sh_R = np.array(bones["upper_arm_R"]["head"], dtype=np.float32)
    el_R = np.array(bones["forearm_R"]["head"], dtype=np.float32)
    wr_R = np.array(bones["hand_R"]["head"], dtype=np.float32)
    hand_R = np.array(bones["hand_R"]["tail"], dtype=np.float32)

    sh_L = np.array(bones["upper_arm_L"]["head"], dtype=np.float32)
    el_L = np.array(bones["forearm_L"]["head"], dtype=np.float32)
    wr_L = np.array(bones["hand_L"]["head"], dtype=np.float32)
    hand_L = np.array(bones["hand_L"]["tail"], dtype=np.float32)
    head_piv = np.array(bones["head"]["head"], dtype=np.float32)

    L1_R = float(np.linalg.norm(el_R - sh_R))
    L2_R = float(np.linalg.norm(wr_R - el_R))
    u1_R_rest = (el_R - sh_R) / max(L1_R, 1e-6)
    u2_R_rest = (wr_R - el_R) / max(L2_R, 1e-6)
    # Rest palm normal in A-pose points inward-downward (+X, slightly -Y, -Z)
    n_palm_R_rest = np.array([0.45, -0.25, -0.85], dtype=np.float32)
    n_palm_R_rest /= np.linalg.norm(n_palm_R_rest)

    w_arm_R, w_fore_R, w_ftwist_R, w_hand_R, _, s_R, s_el_R, perp_R = compute_anatomical_arm_weights(xyz_base, is_arm_R, sh_R, el_R, wr_R, hand_R)
    w_arm_L, w_fore_L, _, _, _, _, _, _ = compute_anatomical_arm_weights(xyz_base, is_arm_L, sh_L, el_L, wr_L, hand_L)

    # Core bare upper-arm sample mask (excludes sleeve frill) for accurate bone-line deviation QA
    upper_arm_qa_mask = is_arm_R & (perp_R < 0.024) & (s_R >= 0.22) & (s_R <= s_el_R - 0.04)

    # Precompute eyeball vertical coords & head-side upper eyelash masks for single-lash natural blink
    SH_C0 = 0.28209479177387814
    fdc_base = np.column_stack([master["f_dc_0"], master["f_dc_1"], master["f_dc_2"]]).astype(np.float32)
    rgb_base = np.clip(0.5 + SH_C0 * fdc_base, 0.0, 1.0)
    lum_base = 0.299 * rgb_base[:, 0] + 0.587 * rgb_base[:, 1] + 0.114 * rgb_base[:, 2]
    eye_z_L = (xyz_base[is_eye_L, 2] - pos_L[2]) / r_eye
    eye_front_L = (xyz_base[is_eye_L, 1] - pos_L[1]) < 0.0
    eye_z_R = (xyz_base[is_eye_R, 2] - pos_R[2]) / r_eye
    eye_front_R = (xyz_base[is_eye_R, 1] - pos_R[1]) < 0.0

    surf_L = np.array(bones["eye_L"]["meta"]["surface_center"], dtype=np.float32)
    surf_R = np.array(bones["eye_R"]["meta"]["surface_center"], dtype=np.float32)
    face_skin_mask = detect_face_skin_mask(xyz_base, rgb_base, is_body, surf_L, surf_R, prop["face_z_range"])
    skin_xyz_pool = xyz_base[face_skin_mask]
    skin_fdc_pool = fdc_base[face_skin_mask]
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
    # Detect mouth interior (reddish-pink) & surrounding lip skin for smooth procedural lip-sync (open <-> closed)
    cp_path = ver_dir / "control_points.json"
    cp = json.loads(cp_path.read_text(encoding="utf-8"))
    mc = np.array(cp["points"]["mouth_center"]["pos"], dtype=np.float32)
    m_mouth_box = (
        is_body
        & (np.abs(xyz_base[:, 0] - mc[0]) < 0.028)
        & (xyz_base[:, 1] < -0.048)
        & (xyz_base[:, 2] >= mc[2] - 0.016)
        & (xyz_base[:, 2] <= mc[2] + 0.018)
    )
    is_mouth_inner = m_mouth_box & ((rgb_base[:, 0] - rgb_base[:, 1]) > 0.13) & (lum_base < 0.80)
    if is_mouth_inner.sum() >= 5:
        mc = np.median(xyz_base[is_mouth_inner], axis=0).astype(np.float32)
    w_lip_pull = (
        is_body.astype(np.float32)
        * (xyz_base[:, 1] < -0.048).astype(np.float32)
        * (~is_mouth_inner).astype(np.float32)
        * np.exp(-((xyz_base[:, 0] - mc[0]) / 0.024) ** 2 - ((xyz_base[:, 2] - mc[2]) / 0.018) ** 2)
    ).astype(np.float32)
    fdc_lip_line = (np.array(cols["lip_line_rgb"], dtype=np.float32) - 0.5) / SH_C0
    print(f"Detected mouth interior splats for lip-sync: {int(is_mouth_inner.sum())} at {mc.round(3).tolist()}")

    sz0, sz1 = float(prop["skirt_z_range"][0]), float(prop["skirt_z_range"][1])
    is_skirt = is_body & (~is_arm_R) & (~is_arm_L) & (xyz_base[:, 2] >= sz0) & (xyz_base[:, 2] <= sz1)
    w_skirt = (is_skirt.astype(np.float32) * (1.0 - smoothstep(sz0 + 0.04, sz1 - 0.02, xyz_base[:, 2]))).astype(np.float32)

    arm_check_idx = np.where(is_arm_R | is_arm_L)[0][::4]
    _, nbr_idx = cKDTree(xyz_base[arm_check_idx]).query(xyz_base[arm_check_idx], k=4)
    rest_d = np.linalg.norm(xyz_base[arm_check_idx][:, None, :] - xyz_base[arm_check_idx][nbr_idx[:, 1:]], axis=2)

    R_spark_obj = Rotation.from_euler("x", 90.0, degrees=True)
    R_spark = R_spark_obj.as_matrix().astype(np.float32)
    q_spark = rot_to_quat_wxyz(R_spark_obj)

    n_frames = 72
    fps = 24
    max_tear_ratio = 0.0
    peak_pole_dot = 0.0
    peak_palm_dot = 0.0
    peak_bone_dev = 0.0
    manifest_frames = []

    # Pole target for anatomical right elbow: outward (-X), downward (-Z), slightly backward (+Y)!
    pole_R = np.array([-0.28, +0.18, 0.36], dtype=np.float32)

    for f in range(n_frames):
        t = f / n_frames

        eye_yaw = 16.0 * math.sin(2.0 * math.pi * t)
        eye_pitch = -11.0 * math.sin(math.pi * np.clip((t - 0.44) / 0.30, 0.0, 1.0)) ** 2
        head_yaw = 0.45 * eye_yaw
        head_pitch = 0.35 * eye_pitch + 2.5 * math.sin(2.0 * math.pi * t)
        head_roll = 7.0 * math.sin(2.0 * math.pi * t)

        env_R = float(smoothstep(0.06, 0.24, t) * (1.0 - smoothstep(0.76, 0.94, t)))
        wave = math.sin(2.0 * math.pi * 2.5 * t)
        wave_lag = math.sin(2.0 * math.pi * 2.5 * t - 0.55)
        env_L = float(math.sin(2.0 * math.pi * t))

        xyz_c, q_c = xyz_base.copy(), q_base.copy()

        # 1. Local eye gaze rotations
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, is_eye_L.astype(np.float32), pos_L, np.radians([eye_pitch, 0.0, 0.0]))
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, is_eye_L.astype(np.float32), pos_L, np.radians([0.0, 0.0, eye_yaw]))
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, is_eye_R.astype(np.float32), pos_R, np.radians([eye_pitch, 0.0, 0.0]))
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, is_eye_R.astype(np.float32), pos_R, np.radians([0.0, 0.0, eye_yaw]))

        # 2. Right arm: Clavicle micro-lift + Analytic 2-Bone IK (Pole Vector) + Isolated Forearm Twist!
        clav_shift = np.array([0.0, -0.004 * env_R, +0.008 * env_R], dtype=np.float32)
        xyz_c += w_arm_R[:, None] * clav_shift[None, :]
        S_cur = sh_R + clav_shift

        # Target wrist position: smoothly interpolates from rest wrist -> waving arc beside shoulder/cheek
        T_wave = np.array([
            -0.210 + 0.025 * wave,
            -0.080,
            0.655 + 0.008 * math.cos(2.0 * math.pi * 2.5 * t),
        ], dtype=np.float32)
        T_tgt = (1.0 - env_R) * (wr_R + clav_shift) + env_R * T_wave

        # Solve 2-Bone IK -> exact elbow position E_cur facing down-back pole_R
        E_cur, T_cur = solve_2bone_ik(S_cur, T_tgt, pole_R, L1_R, L2_R)
        u1_tgt = (E_cur - S_cur) / max(float(np.linalg.norm(E_cur - S_cur)), 1e-6)
        u2_tgt = (T_cur - E_cur) / max(float(np.linalg.norm(T_cur - E_cur)), 1e-6)

        # (a) Upper arm rotation (shoulder 3DOF)
        rv_upper = rot_between_vectors(u1_R_rest, u1_tgt)
        R_upper = Rotation.from_rotvec(rv_upper).as_matrix().astype(np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_arm_R, S_cur, rv_upper)

        # (b) Elbow 1DOF pure hinge rotation (in the exact plane defined by u1_tgt and u2_tgt)
        u2_after_upper = R_upper @ u2_R_rest
        rv_elbow = rot_between_vectors(u2_after_upper, u2_tgt)
        R_elbow = Rotation.from_rotvec(rv_elbow).as_matrix().astype(np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_fore_R, E_cur, rv_elbow)

        # (c) Isolated Forearm Twist (0% at elbow -> 100% at wrist): automatically aligns palm normal to camera (-Y)!
        n_palm_cur = (R_elbow @ (R_upper @ n_palm_R_rest)).astype(np.float32)
        n_palm_tgt = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        # Project both onto plane perpendicular to forearm axis u2_tgt
        p_cur = n_palm_cur - float(np.dot(n_palm_cur, u2_tgt)) * u2_tgt
        p_tgt = n_palm_tgt - float(np.dot(n_palm_tgt, u2_tgt)) * u2_tgt
        if np.linalg.norm(p_cur) > 1e-4 and np.linalg.norm(p_tgt) > 1e-4:
            p_cur /= np.linalg.norm(p_cur)
            p_tgt /= np.linalg.norm(p_tgt)
            sin_tw = float(np.dot(np.cross(p_cur, p_tgt), u2_tgt))
            cos_tw = float(np.clip(np.dot(p_cur, p_tgt), -1.0, 1.0))
            twist_ang = float(np.clip(math.atan2(sin_tw, cos_tw), -math.radians(58.0), math.radians(58.0))) * env_R
        else:
            twist_ang = 0.0
        rv_ftwist = (twist_ang * u2_tgt).astype(np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_ftwist_R, E_cur, rv_ftwist)

        # (d) Very gentle wrist follow-through (keep elbow as the single clear bend!)
        rv_wrist = np.radians(5.0 * wave_lag * env_R) * np.array([0.0, 1.0, 0.0], dtype=np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_hand_R, T_cur, rv_wrist)

        # Anatomical QA metrics at wave peak (f=39)
        if f == 39:
            elbow_dir = (E_cur - 0.5 * (S_cur + T_cur))
            elbow_dir /= max(float(np.linalg.norm(elbow_dir)), 1e-6)
            down_back = np.array([-0.4, 0.5, -0.75], dtype=np.float32); down_back /= np.linalg.norm(down_back)
            peak_pole_dot = float(np.dot(elbow_dir, down_back))
            R_tw = Rotation.from_rotvec(rv_ftwist).as_matrix().astype(np.float32)
            n_palm_final = R_tw @ n_palm_cur
            peak_palm_dot = float(np.dot(n_palm_final, np.array([0.0, -1.0, 0.0], dtype=np.float32)))
            # Bone-line deviation QA: distance from deformed upper-arm centroid to bone segment S_cur -> E_cur
            ua_c = np.median(xyz_c[upper_arm_qa_mask], axis=0)
            v_se = E_cur - S_cur
            t_proj = float(np.clip(np.dot(ua_c - S_cur, v_se) / max(float(np.dot(v_se, v_se)), 1e-8), 0.0, 1.0))
            peak_bone_dev = float(np.linalg.norm(ua_c - (S_cur + t_proj * v_se)))

        # 3. Left arm: relaxed natural counter-sway
        rv_sh_L = Rotation.from_euler("zy", [-7.0 * env_L, -5.0 * env_L], degrees=True).as_rotvec().astype(np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_arm_L, sh_L, rv_sh_L)
        rv_el_L = np.radians(-7.0 * (0.5 + 0.5 * env_L)) * np.array([0.0, 1.0, 0.0], dtype=np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_fore_L, el_L, rv_el_L)

        # 4. Face-fitted independent eyelid shells (natural blink at t=0.33 + cute left-eye wink at t=0.72!) + lip-sync
        blink_both = math.exp(-((t - 0.33) / 0.035) ** 2)
        wink_L = math.exp(-((t - 0.72) / 0.065) ** 2)
        blink_L = max(blink_both, wink_L)
        blink_R = blink_both

        # Retract closed eyeball slightly behind eyelid shell so iris never pokes through
        if blink_L > 0.05:
            xyz_c[is_eye_L, 1] += 0.0045 * blink_L
        if blink_R > 0.05:
            xyz_c[is_eye_R, 1] += 0.0045 * blink_R

        # mouth_close: 0.0 = natural open smile, 1.0 = closed mouth (closes twice cleanly during wave)
        mouth_close = max(
            math.exp(-((t - 0.18) / 0.075) ** 2),
            math.exp(-((t - 0.56) / 0.075) ** 2),
        )
        if mouth_close > 0.02:
            dz_in = xyz_c[is_mouth_inner, 2] - mc[2]
            xyz_c[is_mouth_inner, 2] = mc[2] + dz_in * (1.0 - 0.80 * mouth_close)
            xyz_c[is_mouth_inner, 1] += 0.002 * mouth_close
            sign_lip = np.sign(xyz_base[:, 2] - mc[2])
            xyz_c[:, 2] -= sign_lip * w_lip_pull * (0.0065 * mouth_close)

        # 5. Head (carries eyes and animated mouth rigidly)
        rv_head = Rotation.from_euler("zxy", [head_yaw, head_pitch, head_roll], degrees=True).as_rotvec().astype(np.float32)
        xyz_c, q_c = apply_weighted_rotvec(xyz_c, q_c, w_head, head_piv, rv_head)

        # 6. Gentle skirt hem sway
        xyz_c[:, 0] += w_skirt * (0.010 * math.sin(2.0 * math.pi * t))

        cur_d = np.linalg.norm(xyz_c[arm_check_idx][:, None, :] - xyz_c[arm_check_idx][nbr_idx[:, 1:]], axis=2)
        tear_ratio = float(np.mean((cur_d / np.maximum(rest_d, 1e-5)) > 2.5))
        max_tear_ratio = max(max_tear_ratio, tear_ratio)

        out = master.copy()
        out["x"], out["y"], out["z"] = xyz_c[:, 0], xyz_c[:, 1], xyz_c[:, 2]
        out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = q_c[:, 0], q_c[:, 1], q_c[:, 2], q_c[:, 3]

        # Fade socket rim & fully-closed eyeball underneath descending eyelid shell
        if blink_L > 0.05:
            op_fade_L = float(smoothstep(0.15, 0.70, blink_L))
            out["opacity"][in_rim_L] = (1.0 - op_fade_L) * out["opacity"][in_rim_L] + op_fade_L * (-12.0)
            if blink_L >= 0.90:
                out["opacity"][is_eye_L] = -12.0
        if blink_R > 0.05:
            op_fade_R = float(smoothstep(0.15, 0.70, blink_R))
            out["opacity"][in_rim_R] = (1.0 - op_fade_R) * out["opacity"][in_rim_R] + op_fade_R * (-12.0)
            if blink_R >= 0.90:
                out["opacity"][is_eye_R] = -12.0

        if mouth_close > 0.02:
            fdc_c = fdc_base.copy()
            w_mc = float(smoothstep(0.25, 0.90, mouth_close))
            fdc_c[is_mouth_inner] = (1.0 - 0.75 * w_mc) * fdc_c[is_mouth_inner] + (0.75 * w_mc) * fdc_lip_line
            out["f_dc_0"], out["f_dc_1"], out["f_dc_2"] = fdc_c[:, 0], fdc_c[:, 1], fdc_c[:, 2]

        # Generate independent face-fitted eyelid shells and rigidly transform them by head bone!
        parts = [out]
        for bval, p_eye, sc_c in ((blink_L, pos_L, surf_L), (blink_R, pos_R, surf_R)):
            if bval > 0.03:
                lid = build_eyelid_shell(p_eye, sc_c, skin_xyz_pool, skin_fdc_pool, skin_scale_med, bval, master.dtype, prop["eyelid_shell_rx_rz"], prop["eyelid_lash_top_bottom_z"], cols["lash_rgb"])
                if len(lid) > 0:
                    lxyz = np.column_stack([lid["x"], lid["y"], lid["z"]]).astype(np.float32)
                    lq = np.column_stack([lid["rot_0"], lid["rot_1"], lid["rot_2"], lid["rot_3"]]).astype(np.float32)
                    lxyz, lq = apply_weighted_rotvec(lxyz, lq, np.ones(len(lid), dtype=np.float32), head_piv, rv_head)
                    lid["x"], lid["y"], lid["z"] = lxyz[:, 0], lxyz[:, 1], lxyz[:, 2]
                    lid["rot_0"], lid["rot_1"], lid["rot_2"], lid["rot_3"] = lq[:, 0], lq[:, 1], lq[:, 2], lq[:, 3]
                    parts.append(lid)

        merged = np.concatenate(parts, axis=0)
        m_xyz = np.column_stack([merged["x"], merged["y"], merged["z"]]).astype(np.float32)
        m_q = np.column_stack([merged["rot_0"], merged["rot_1"], merged["rot_2"], merged["rot_3"]]).astype(np.float32)
        s_xyz = (m_xyz @ R_spark.T).astype(np.float32)
        s_q = quat_mul_wxyz(q_spark, m_q)
        s_q /= np.maximum(np.linalg.norm(s_q, axis=1, keepdims=True), 1e-8)
        merged["x"], merged["y"], merged["z"] = s_xyz[:, 0], s_xyz[:, 1], s_xyz[:, 2]
        merged["rot_0"], merged["rot_1"], merged["rot_2"], merged["rot_3"] = s_q[:, 0], s_q[:, 1], s_q[:, 2], s_q[:, 3]

        rel = f"frames/frame_{f + 1:04d}.ply"
        write_ply(demo_dir / rel, merged)
        manifest_frames.append({"frame": f + 1, "blenderFrame": f + 1, "png": f"frame-{f + 1:04d}.png", "ply": [rel]})

    (frames_dir / "manifest.json").write_text(json.dumps({"fps": fps, "static": [], "review": [], "frames": manifest_frames}, indent=2), encoding="utf-8")
    sh_sym_diff = float(abs(sh_R[2] - sh_L[2]) + abs(sh_R[1] - sh_L[1]))
    qa_ok = (
        (max_tear_ratio < float(qa_cfg["max_arm_tear_ratio"]))
        and (peak_pole_dot > float(qa_cfg["min_elbow_down_back_dot"]))
        and (peak_palm_dot > float(qa_cfg["min_palm_camera_dot"]))
        and (peak_bone_dev < float(qa_cfg["max_bone_deviation"]))
        and (sh_sym_diff < float(qa_cfg["max_shoulder_sym_diff"]))
    )
    print(
        f"v1 Final Anatomical IK + Single-Lash Blink done: 72 frames, "
        f"tear={max_tear_ratio:.5f}, bone_dev={peak_bone_dev:.4f}, sh_sym_diff={sh_sym_diff:.5f}, "
        f"elbow_dot={peak_pole_dot:.3f}, palm_dot={peak_palm_dot:.3f} (QA: {'PASS' if qa_ok else 'FAIL'})"
    )


if __name__ == "__main__":
    main()
