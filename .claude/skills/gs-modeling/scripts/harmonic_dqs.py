"""Reusable Character-Agnostic Harmonic Heat-Diffusion DQS & Anatomical Rig Module.

Provides:
- add_forearm_twist_bones: splits forearm_L/R into proximal hinge + distal pronation/supination twist bone
- compute_harmonic_skin_weights: C-infinity smooth 3D kNN Graph-Laplacian heat-diffusion weights (works for T-pose & A-pose)
- deform_dqs: Dual Quaternion Skinning with antipodal sign alignment & exact unit dual-quaternion projection
- build_stretch_neighborhood & apply_stretch_densification: dynamic midpoint splat insertion for stretched cloth
- look_at_eye_rot: 3D Look-At binocular vergence rotation matrix per eye
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

SH_C0 = 0.28209479177387814


def smooth01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1).astype(np.float32)


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] = -out[..., 1:]
    return out


def rot_by_quat_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[:, 1:4]
    qw = q[:, 0:1]
    t = 2.0 * np.cross(qvec, v)
    return v + qw * t + np.cross(qvec, t)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(R).as_quat()
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
    if q[0] < 0.0:
        q = -q
    return q / max(float(np.linalg.norm(q)), 1e-8)


def rigid_to_dq(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_r = rotmat_to_quat_wxyz(R)
    q_t = np.array([0.0, float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)
    q_d = 0.5 * quat_mul_wxyz(q_t, q_r)
    return q_r, q_d


def axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    u = np.asarray(axis, dtype=np.float32)
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    return Rotation.from_rotvec(u * math.radians(deg)).as_matrix().astype(np.float32)


def euler_xyz(rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0) -> np.ndarray:
    return Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix().astype(np.float32)


def dist_to_segment(pts: np.ndarray, h: np.ndarray, t: np.ndarray) -> np.ndarray:
    v = t - h
    l2 = float(np.dot(v, v))
    if l2 < 1e-8:
        return np.linalg.norm(pts - h[None, :], axis=1).astype(np.float32)
    proj = np.clip(((pts - h[None, :]) @ v) / l2, 0.0, 1.0)
    closest = h[None, :] + proj[:, None] * v[None, :]
    return np.linalg.norm(pts - closest, axis=1).astype(np.float32)


def add_forearm_twist_bones(bones: dict) -> dict:
    """Split forearm_L/R into proximal (forearm_*) + distal twist (forearm_twist_*) for smooth pronation/supination."""
    out = {k: dict(v) for k, v in bones.items()}
    for s in ("L", "R"):
        if f"forearm_{s}" not in out or f"forearm_twist_{s}" in out:
            continue
        fb = out[f"forearm_{s}"]
        h = np.asarray(fb["head"], dtype=np.float32)
        t = np.asarray(fb["tail"], dtype=np.float32)
        mid = (0.5 * (h + t)).tolist()
        out[f"forearm_{s}"] = {"parent": fb["parent"], "head": fb["head"], "tail": mid}
        out[f"forearm_twist_{s}"] = {"parent": f"forearm_{s}", "head": mid, "tail": fb["tail"]}
        if f"hand_{s}" in out:
            out[f"hand_{s}"]["parent"] = f"forearm_twist_{s}"
    return out


def build_fk_hierarchy(bones: dict) -> list[str]:
    order, visited = [], set()

    def visit(bname):
        if bname in visited:
            return
        parent = bones[bname].get("parent")
        if parent and parent in bones:
            visit(parent)
        visited.add(bname)
        order.append(bname)

    for bname in bones:
        visit(bname)
    return order


def compute_fk_transforms(
    bones: dict,
    order: list[str],
    local_rots: dict[str, np.ndarray],
    root_shift: np.ndarray | None = None,
) -> dict:
    world = {}
    for bname in order:
        b = bones[bname]
        h_rest = np.asarray(b["head"], dtype=np.float32)
        parent = b.get("parent")
        R_loc = local_rots.get(bname, np.eye(3, dtype=np.float32))
        if parent is None or parent not in world:
            R_w = R_loc.astype(np.float32)
            h_w = h_rest + (root_shift if root_shift is not None else np.zeros(3, dtype=np.float32))
        else:
            R_p, h_p, _ = world[parent]
            h_p_rest = np.asarray(bones[parent]["head"], dtype=np.float32)
            h_w = h_p + R_p @ (h_rest - h_p_rest)
            R_w = (R_p @ R_loc).astype(np.float32)
        t_w = (h_w - R_w @ h_rest).astype(np.float32)
        world[bname] = (R_w, h_w, t_w)
    return world


def compute_harmonic_skin_weights(
    xyz: np.ndarray,
    bones: dict,
    is_body: np.ndarray,
    max_edge: float = 0.012,
    n_harmonic_iters: int = 90,
    n_smooth_iters: int = 20,
) -> tuple[list[str], np.ndarray]:
    """Data-driven C-infinity smooth Harmonic Graph-Laplacian Heat-Diffusion skinning weights.
    Adapts automatically to both T-pose (75-90 deg) and A-pose (~45 deg) skeletons.
    """
    skin_bones = [b for b in bones if b not in ("root", "eye_L", "eye_R")]
    n_pts = len(xyz)
    n_b = len(skin_bones)
    body_idx = np.where(is_body)[0]
    bxyz = xyz[body_idx]
    nb = len(bxyz)

    face_cx = float(bones["spine_lower"]["head"][0])
    sh_L = np.asarray(bones["upper_arm_L"]["head"], np.float32)
    sh_z = float(sh_L[2])
    sh_hw = abs(float(sh_L[0]) - face_cx)
    hip_z = float(bones["thigh_L"]["head"][2])
    neck_z = float(bones["neck"]["head"][2])
    arm_min_z = min(float(bones["hand_L"]["tail"][2]), float(bones["forearm_L"]["tail"][2])) - 0.05

    tree = cKDTree(bxyz)
    dists_knn, idx_knn = tree.query(bxyz, k=13, workers=-1)
    d_e = dists_knn[:, 1:]
    j_e = idx_knn[:, 1:]
    i_e = np.repeat(np.arange(nb, dtype=np.int32)[:, None], 12, axis=1)
    valid = d_e <= max_edge
    w_edge = np.exp(-(d_e[valid] / (max_edge * 0.60)) ** 2).astype(np.float32)

    A = csr_matrix((w_edge, (i_e[valid], j_e[valid])), shape=(nb, nb))
    A = 0.5 * (A + A.T)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.where(deg > 1e-8, 1.0 / np.maximum(deg, 1e-8), 0.0).astype(np.float32)
    P = diags(inv_deg) @ A

    dists = np.full((nb, n_b), 1e6, dtype=np.float32)
    for bi, bname in enumerate(skin_bones):
        h = np.asarray(bones[bname]["head"], dtype=np.float32)
        t = np.asarray(bones[bname]["tail"], dtype=np.float32)
        d = dist_to_segment(bxyz, h, t)
        if bname.startswith(("upper_arm_", "forearm_", "hand_")):
            sgn = +1.0 if bname.endswith("_L") else -1.0
            wrong_side = (bxyz[:, 0] - face_cx) * sgn < sh_hw * 0.65
            d = np.where(wrong_side, d + 0.80, d)
        elif bname.startswith(("thigh_", "shin_", "foot_")):
            sgn = +1.0 if bname.endswith("_L") else -1.0
            wrong_side = (bxyz[:, 0] - face_cx) * sgn < -0.015
            too_high = bxyz[:, 2] > hip_z + 0.04
            d = np.where(wrong_side | too_high, d + 0.80, d)
        elif bname == "head":
            d = np.where(bxyz[:, 2] < neck_z - 0.01, d + 0.35, d * 0.85)
        dists[:, bi] = d

    # Segment outer arm cores via 3D CC + upper-arm cylinder (automatically adapts to T-pose & A-pose!)
    # Restrict candidates to an arm-bone proximity tube so wide flare skirts near A-pose hands never merge into arm_core
    arm_core_L = np.zeros(nb, dtype=bool)
    arm_core_R = np.zeros(nb, dtype=bool)
    arm_cut_x = max(sh_hw + 0.045, sh_hw * 1.45)
    for side, sgn, mask_out in (("L", +1.0, arm_core_L), ("R", -1.0, arm_core_R)):
        d_arm_side = np.full(nb, 1e6, dtype=np.float32)
        for bn in (f"upper_arm_{side}", f"forearm_{side}", f"forearm_twist_{side}", f"hand_{side}"):
            if bn in bones:
                d_arm_side = np.minimum(d_arm_side, dist_to_segment(bxyz, np.asarray(bones[bn]["head"], np.float32), np.asarray(bones[bn]["tail"], np.float32)))
        hand_tip = np.asarray(bones[f"hand_{side}"]["tail"], np.float32)
        d_tip = np.linalg.norm(bxyz - hand_tip[None, :], axis=1)
        # Tight fingertip capture (0.075) while excluding low skirt petals below the hand
        near_arm_tube = ((d_arm_side <= 0.056) & (bxyz[:, 2] >= hip_z + 0.025)) | (d_arm_side <= 0.042) | ((d_tip <= 0.075) & (bxyz[:, 2] >= hand_tip[2] - 0.025))
        cand = np.where(((bxyz[:, 0] - face_cx) * sgn > arm_cut_x) & (bxyz[:, 2] >= arm_min_z) & (bxyz[:, 2] <= sh_z + 0.09) & near_arm_tube)[0]
        if len(cand) >= 20:
            cxyz = bxyz[cand]
            adj_c = cKDTree(cxyz).sparse_distance_matrix(cKDTree(cxyz), max_distance=0.016, output_type="coo_matrix")
            _, labels = connected_components(adj_c, directed=False)
            tip_i = int(np.argmax(cxyz[:, 0]) if sgn > 0 else np.argmin(cxyz[:, 0]))
            mask_out[cand[labels == labels[tip_i]]] = True
        h_ua = np.asarray(bones[f"upper_arm_{side}"]["head"], np.float32)
        t_ua = np.asarray(bones[f"upper_arm_{side}"]["tail"], np.float32)
        v_ua = t_ua - h_ua
        l_ua = max(float(np.linalg.norm(v_ua)), 1e-6)
        u_ua = v_ua / l_ua
        rel = bxyz - h_ua[None, :]
        s_ua = (rel @ u_ua) / l_ua
        perp_ua = np.linalg.norm(rel - (s_ua * l_ua)[:, None] * u_ua[None, :], axis=1)
        ua_cyl = (s_ua >= 0.28) & (s_ua <= 1.05) & (perp_ua <= 0.072) & ((bxyz[:, 0] - face_cx) * sgn > sh_hw + 0.025)
        mask_out |= ua_cyl

    head_joint_z = float(bones["head"]["head"][2])
    armpit_z = sh_z - 0.068

    # Auto-detect hanging long hair / ponytail / side-tail below head_joint_z via 3D CC from cranium
    # (rear y > torso_back_y or lateral outside shoulders, strictly excluding arms & front torso)
    torso_y_c = float(bones["chest"]["head"][1])
    hair_cand_mask = (
        (bxyz[:, 2] > head_joint_z + 0.004)
        | (
            (bxyz[:, 2] >= max(hip_z + 0.08, head_joint_z - 0.22))
            & (bxyz[:, 2] <= head_joint_z + 0.015)
            & (~(arm_core_L | arm_core_R))
            & ((bxyz[:, 1] > torso_y_c + 0.055) | (np.abs(bxyz[:, 0] - face_cx) > sh_hw + 0.065))
        )
    )
    hanging_hair = np.zeros(nb, dtype=bool)
    hc_idx = np.where(hair_cand_mask)[0]
    if len(hc_idx) >= 50:
        hc_xyz = bxyz[hc_idx]
        adj_h = cKDTree(hc_xyz).sparse_distance_matrix(cKDTree(hc_xyz), max_distance=0.020, output_type="coo_matrix")
        _, h_labels = connected_components(adj_h, directed=False)
        top_seed = int(np.argmax(hc_xyz[:, 2]))
        hanging_hair[hc_idx[h_labels == h_labels[top_seed]]] = True

    torso_core = (
        ((np.abs(bxyz[:, 0] - face_cx) < sh_hw * 1.05) & (bxyz[:, 2] >= hip_z - 0.02) & (bxyz[:, 2] <= neck_z + 0.02))
        | ((np.abs(bxyz[:, 0] - face_cx) < sh_hw + 0.024) & (bxyz[:, 2] >= hip_z - 0.02) & (bxyz[:, 2] <= armpit_z))
    ) & ~(arm_core_L | arm_core_R | hanging_hair)
    # Pin ALL non-arm points in the waist/flare-skirt zone so arm weights never bleed into wide skirts during heat diffusion
    waist_core = (bxyz[:, 2] >= hip_z - 0.02) & (bxyz[:, 2] <= armpit_z - 0.045) & ~(arm_core_L | arm_core_R | hanging_hair)
    head_core = (bxyz[:, 2] > head_joint_z + 0.004) | hanging_hair
    leg_core = (bxyz[:, 2] < hip_z - 0.02) & ~(arm_core_L | arm_core_R)

    pinned = torso_core | waist_core | arm_core_L | arm_core_R | head_core | leg_core
    is_arm_b_L = np.array([b.endswith("_L") and b.startswith(("upper_arm_", "forearm_", "hand_")) for b in skin_bones])
    is_arm_b_R = np.array([b.endswith("_R") and b.startswith(("upper_arm_", "forearm_", "hand_")) for b in skin_bones])
    is_torso_b = np.array([b in ("spine_lower", "chest", "neck", "clavicle_L", "clavicle_R") for b in skin_bones])
    is_head_b = np.array([b == "head" for b in skin_bones])
    is_leg_b = np.array([b.startswith(("thigh_", "shin_", "foot_")) for b in skin_bones])

    dists_pin = dists.copy()
    dists_pin[torso_core | waist_core] = np.where(is_torso_b[None, :], dists_pin[torso_core | waist_core], 10.0)
    dists_pin[arm_core_L] = np.where(is_arm_b_L[None, :], dists_pin[arm_core_L], 10.0)
    dists_pin[arm_core_R] = np.where(is_arm_b_R[None, :], dists_pin[arm_core_R], 10.0)
    dists_pin[head_core] = np.where(is_head_b[None, :], dists_pin[head_core], 10.0)
    dists_pin[leg_core] = np.where(is_leg_b[None, :], dists_pin[leg_core], 10.0)

    dmin = np.min(dists_pin, axis=1, keepdims=True)
    W_b = np.exp(-((dists_pin - dmin) / 0.024) ** 2).astype(np.float32)
    W_b /= np.maximum(W_b.sum(axis=1, keepdims=True), 1e-8)

    free = (~pinned) & (deg > 1e-8)
    for _ in range(n_harmonic_iters):
        W_b[free] = (P @ W_b)[free]

    # Smooth transitions across joints while keeping the rigid cranium + connected long hair locked to head=1.0
    # and preventing final smoothing pass from bleeding arm weights into the pinned skirt/legs
    rigid_cranium = (bxyz[:, 2] > head_joint_z + 0.012) | hanging_hair
    skirt_or_leg = (waist_core | leg_core) & (~(arm_core_L | arm_core_R))
    has_nbr = (deg > 1e-8) & (~rigid_cranium)
    for _ in range(n_smooth_iters):
        W_b[has_nbr] = 0.55 * W_b[has_nbr] + 0.45 * (P @ W_b)[has_nbr]
        if np.any(skirt_or_leg):
            W_b[np.ix_(skirt_or_leg, is_arm_b_L | is_arm_b_R)] = 0.0
            W_b[skirt_or_leg] /= np.maximum(W_b[skirt_or_leg].sum(axis=1, keepdims=True), 1e-8)
    if np.any(is_head_b) and np.any(rigid_cranium):
        W_b[rigid_cranium] = 0.0
        W_b[rigid_cranium, np.where(is_head_b)[0][0]] = 1.0

    top_idx = np.argpartition(-W_b, kth=min(4, n_b - 1), axis=1)[:, :4]
    W_sp = np.zeros_like(W_b)
    r_idx = np.arange(nb)[:, None]
    W_sp[r_idx, top_idx] = W_b[r_idx, top_idx]
    W_sp /= np.maximum(W_sp.sum(axis=1, keepdims=True), 1e-8)

    W_full = np.zeros((n_pts, n_b), dtype=np.float32)
    W_full[body_idx] = W_sp
    return skin_bones, W_full


def deform_dqs(
    base_xyz: np.ndarray,
    base_rot: np.ndarray,
    skin_bones: list[str],
    W: np.ndarray,
    world_xforms: dict,
    is_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_pts = len(base_xyz)
    n_b = len(skin_bones)
    qr_b = np.zeros((n_b, 4), dtype=np.float32)
    qd_b = np.zeros((n_b, 4), dtype=np.float32)
    for bi, bname in enumerate(skin_bones):
        R_w, _, t_w = world_xforms[bname]
        qr_b[bi], qd_b[bi] = rigid_to_dq(R_w, t_w)

    dom = np.argmax(W, axis=1)
    q_dom = qr_b[dom]
    Qr = np.zeros((n_pts, 4), dtype=np.float32)
    Qd = np.zeros((n_pts, 4), dtype=np.float32)
    for bi in range(n_b):
        wb = W[:, bi]
        mask = is_body & (wb > 1e-5)
        if not np.any(mask):
            continue
        dots = np.sum(q_dom[mask] * qr_b[bi][None, :], axis=1)
        sgn = np.where(dots >= 0.0, 1.0, -1.0).astype(np.float32)
        ws = (wb[mask] * sgn)[:, None]
        Qr[mask] += ws * qr_b[bi][None, :]
        Qd[mask] += ws * qd_b[bi][None, :]

    out_xyz = base_xyz.copy()
    out_rot = base_rot.copy()
    m = is_body
    nr = np.maximum(np.linalg.norm(Qr[m], axis=1, keepdims=True), 1e-8)
    qr = Qr[m] / nr
    dot_rd = np.sum(Qr[m] * Qd[m], axis=1, keepdims=True)
    qd = Qd[m] / nr - (dot_rd / (nr * nr * nr)) * Qr[m]

    t_quat = 2.0 * quat_mul_wxyz(qd, quat_conj_wxyz(qr))
    out_xyz[m] = rot_by_quat_wxyz(qr, base_xyz[m]) + t_quat[:, 1:4]
    out_rot[m] = quat_mul_wxyz(qr, base_rot[m])
    out_rot[m] /= np.maximum(np.linalg.norm(out_rot[m], axis=1, keepdims=True), 1e-8)
    return out_xyz, out_rot


def build_stretch_neighborhood(xyz: np.ndarray, is_body: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_idx = np.where(is_body)[0]
    bxyz = xyz[body_idx]
    tree = cKDTree(bxyz)
    dists, idxs = tree.query(bxyz, k=k + 1, workers=-1)
    return body_idx, idxs[:, 1:].astype(np.int32), np.maximum(dists[:, 1:].astype(np.float32), 1e-4)


def apply_stretch_densification(
    data: np.ndarray,
    deformed_xyz: np.ndarray,
    deformed_rot: np.ndarray,
    body_idx: np.ndarray,
    nbr_local: np.ndarray,
    rest_d: np.ndarray,
    sh_z: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dynamic Surface Splat Densification: inserts crisp midpoint splats along stretched cloth edges."""
    out_data = data.copy()
    bxyz_def = deformed_xyz[body_idx]
    brot_def = deformed_rot[body_idx]
    diff = bxyz_def[nbr_local] - bxyz_def[:, None, :]
    def_d = np.linalg.norm(diff, axis=-1)

    valid = (rest_d >= 0.001) & (rest_d <= 0.014)
    ratio = np.where(valid, def_d / rest_d, 1.0)
    stretched = valid & (ratio > 1.32) & (def_d > 0.0045)

    ii, kk = np.where(stretched)
    if len(ii) == 0:
        return out_data, deformed_xyz, deformed_rot
    jj = nbr_local[ii, kk]
    keep_u = ii < jj
    ii, jj, r_e = ii[keep_u], jj[keep_u], ratio[ii[keep_u], kk[keep_u]]
    if len(ii) == 0:
        return out_data, deformed_xyz, deformed_rot

    m1 = r_e <= 1.85
    m2 = ~m1
    t_list, i_list, j_list = [], [], []
    if np.any(m1):
        i_list.append(ii[m1]); j_list.append(jj[m1]); t_list.append(np.full(int(m1.sum()), 0.5, dtype=np.float32))
    if np.any(m2):
        for tv in (0.33, 0.67):
            i_list.append(ii[m2]); j_list.append(jj[m2]); t_list.append(np.full(int(m2.sum()), tv, dtype=np.float32))

    ia, ib, tt = np.concatenate(i_list), np.concatenate(j_list), np.concatenate(t_list)[:, None]
    if len(ia) > 18000:
        sel = np.linspace(0, len(ia) - 1, 18000, dtype=np.int32)
        ia, ib, tt = ia[sel], ib[sel], tt[sel]

    ga, gb = body_idx[ia], body_idx[ib]
    new_xyz = ((1.0 - tt) * bxyz_def[ia] + tt * bxyz_def[ib]).astype(np.float32)
    new_rot = brot_def[ia].copy()

    new_data = out_data[ga].copy()
    for ch in ("f_dc_0", "f_dc_1", "f_dc_2"):
        new_data[ch] = (1.0 - tt[:, 0]) * out_data[ch][ga] + tt[:, 0] * out_data[ch][gb]
    s_crisp = np.minimum(np.minimum(out_data["scale_0"][ga], out_data["scale_0"][gb]) + 0.12, np.log(0.0028)).astype(np.float32)
    new_data["scale_0"] = s_crisp
    new_data["scale_1"] = s_crisp
    new_data["scale_2"] = (s_crisp - 0.85).astype(np.float32)
    new_data["opacity"] = np.maximum(out_data["opacity"][ga], 3.0).astype(np.float32)

    return np.concatenate([out_data, new_data], axis=0), np.concatenate([deformed_xyz, new_xyz], axis=0), np.concatenate([deformed_rot, new_rot], axis=0)


def look_at_eye_rot(eye_center: np.ndarray, target_3d: np.ndarray, max_yaw_deg: float = 22.0, max_pitch_deg: float = 16.0) -> np.ndarray:
    """Compute 3D Look-At binocular vergence rotation matrix for one eye (rest gaze = -Y)."""
    v = target_3d - eye_center
    u = v / max(float(np.linalg.norm(v)), 1e-6)
    yaw_deg = float(np.clip(math.degrees(math.atan2(float(u[0]), float(-u[1]))), -max_yaw_deg, max_yaw_deg))
    pitch_deg = float(np.clip(-math.degrees(math.atan2(float(u[2]), math.hypot(float(u[0]), float(u[1])))), -max_pitch_deg, max_pitch_deg))
    return (Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @
            Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()).astype(np.float32)


def to_spark_ply(data: np.ndarray, xyz: np.ndarray, rot_wxyz: np.ndarray) -> np.ndarray:
    """Convert Blender Z-up (-Y forward) splats to Spark Y-up via Rx(+90 deg)."""
    out = data.copy()
    out["x"], out["y"], out["z"] = xyz[:, 0], -xyz[:, 2], xyz[:, 1]
    q_rx90 = np.array([math.cos(math.pi / 4.0), math.sin(math.pi / 4.0), 0.0, 0.0], dtype=np.float32)
    q_s = quat_mul_wxyz(np.broadcast_to(q_rx90, rot_wxyz.shape), rot_wxyz)
    q_s /= np.maximum(np.linalg.norm(q_s, axis=1, keepdims=True), 1e-8)
    out["rot_0"], out["rot_1"], out["rot_2"], out["rot_3"] = q_s[:, 0], q_s[:, 1], q_s[:, 2], q_s[:, 3]
    return out


_M_BL_TO_SPARK = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def pack_spark_skin_weights(W: np.ndarray) -> np.ndarray:
    """Pack (N, B) skin weight matrix into Spark SplatSkinning RGBA16UI layout (N, 4) uint16."""
    n = W.shape[0]
    k = min(4, W.shape[1])
    top_idx = np.argpartition(-W, kth=k - 1, axis=1)[:, :k]
    top_w = np.take_along_axis(W, top_idx, axis=1)
    order = np.argsort(-top_w, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_w = np.take_along_axis(top_w, order, axis=1)
    w_sum = np.maximum(top_w.sum(axis=1, keepdims=True), 1e-8)
    top_w = top_w / w_sum

    packed = np.zeros((n, 4), dtype=np.uint16)
    w_u8 = np.clip(np.rint(top_w * 255.0), 0, 255).astype(np.uint16)
    # Ensure at least weight 1 on primary bone so GPU dual-quat norm is never 0
    w_u8[:, 0] = np.maximum(w_u8[:, 0], 1)
    packed[:, :k] = w_u8 + (top_idx.astype(np.uint16) << 8)
    return packed


def blender_rigid_to_spark_qp(R_b: np.ndarray, t_b: np.ndarray) -> list[float]:
    """Convert Blender rigid transform (R_b, t_b) to Spark [qx, qy, qz, qw, tx, ty, tz]."""
    R_s = (_M_BL_TO_SPARK @ R_b @ _M_BL_TO_SPARK.T).astype(np.float32)
    q_xyzw = Rotation.from_matrix(R_s).as_quat().astype(np.float32)
    if q_xyzw[3] < 0.0:
        q_xyzw = -q_xyzw
    t_s = (_M_BL_TO_SPARK @ t_b).astype(np.float32)
    return [float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3]),
            float(t_s[0]), float(t_s[1]), float(t_s[2])]


def fk_to_spark_bones(
    bones: dict,
    skin_bones: list[str],
    world: dict,
    dz_offset: float = 0.0,
) -> list[float]:
    """Convert FK world transforms to flat [qx,qy,qz,qw, tx,ty,tz, ...] for Spark SplatSkinning."""
    out: list[float] = []
    dz_vec = np.array([0.0, 0.0, dz_offset], dtype=np.float32)
    for bname in skin_bones:
        R_b, h_w, _ = world[bname]
        h_r = np.asarray(bones[bname]["head"], dtype=np.float32)
        t_b = (h_w + dz_vec) - (R_b @ h_r)
        out.extend(blender_rigid_to_spark_qp(R_b, t_b))
    return out
