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


def _infer_bone_skin_group(bname: str, bone: dict) -> str:
    """Infer a default skin_group for a bone when not explicitly set in rig_skeleton.json."""
    if bone.get("skin_group"):
        return str(bone["skin_group"])
    role = str(bone.get("role", "")).lower()
    side = "_L" if bname.endswith("_L") or "_L_" in bname else ("_R" if bname.endswith("_R") or "_R_" in bname else "")
    if role in ("wing", "ear", "ribbon", "hair", "cape", "accessory", "tail", "jaw"):
        return f"{role}{side}" if side and role in ("wing", "ear", "ribbon", "hair", "cape", "accessory") else role
    if bname.startswith(("wing_", "ear_", "ribbon_", "hair_", "cape_", "acc_")):
        prefix = bname.split("_", 1)[0]
        return f"{prefix}{side}"
    if bname.startswith("tail"):
        return "tail"
    if bname == "jaw" or bname.startswith("jaw_"):
        return "jaw"
    if bname == "head":
        return "head"
    if bname.startswith(("upper_arm_", "forearm_", "hand_")):
        return f"arm{side}"
    if bname.startswith(("front_shoulder", "front_upper", "front_lower", "front_paw")):
        return f"front_leg{side}"
    if bname.startswith(("hind_upper", "hind_lower", "hind_paw", "thigh_", "shin_", "foot_", "toe_")):
        return f"leg{side}"
    return "torso"


def _is_legacy_humanoid_skeleton(bones: dict) -> bool:
    """Return True iff bones match the original v2.0 humanoid rig with no v2.1 extensions."""
    req = ("spine_lower", "upper_arm_L", "upper_arm_R", "forearm_L", "hand_L", "thigh_L", "neck", "head")
    if not all(k in bones for k in req):
        return False
    for bname, b in bones.items():
        if b.get("skin_group") or b.get("role") or b.get("seam_cut") or b.get("dynamics") or b.get("rigid") is not None:
            return False
        if bname.startswith(("wing_", "tail", "ear_", "ribbon_", "hair_", "cape_", "jaw", "front_", "hind_")):
            return False
    return True


def _apply_seam_cuts_to_edges(
    bxyz: np.ndarray,
    i_e: np.ndarray,
    j_e: np.ndarray,
    valid: np.ndarray,
    seam_cuts: list[dict],
) -> np.ndarray:
    """Cut kNN graph edges crossing any declared seam_cut plane (e.g. muzzle mouth split, wing/body seam)."""
    if not seam_cuts:
        return valid
    out_valid = valid.copy()
    ii = i_e[out_valid]
    jj = j_e[out_valid]
    pi = bxyz[ii]
    pj = bxyz[jj]
    mid = 0.5 * (pi + pj)
    keep = np.ones(len(ii), dtype=bool)
    for cut in seam_cuts:
        o = np.asarray(cut["origin"], dtype=np.float32)
        n = np.asarray(cut["normal"], dtype=np.float32)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-8:
            continue
        n = n / n_norm
        si = (pi - o[None, :]) @ n
        sj = (pj - o[None, :]) @ n
        crosses = (si * sj) < 0.0
        in_region = np.ones(len(ii), dtype=bool)
        if cut.get("radius") is not None:
            r = float(cut["radius"])
            in_region &= np.linalg.norm(mid - o[None, :], axis=1) <= r
        if cut.get("bbox_min") is not None and cut.get("bbox_max") is not None:
            bmin = np.asarray(cut["bbox_min"], dtype=np.float32)
            bmax = np.asarray(cut["bbox_max"], dtype=np.float32)
            in_region &= np.all((mid >= bmin[None, :]) & (mid <= bmax[None, :]), axis=1)
        keep &= ~(crosses & in_region)
    out_valid[out_valid] = keep
    return out_valid


def _compute_generic_harmonic_skin_weights(
    xyz: np.ndarray,
    bones: dict,
    is_body: np.ndarray,
    seam_cuts: list[dict] | None = None,
    max_edge: float = 0.012,
    n_harmonic_iters: int = 90,
    n_smooth_iters: int = 20,
    mid_x: float | None = None,
) -> tuple[list[str], np.ndarray]:
    """Archetype-agnostic Harmonic Graph-Laplacian Heat-Diffusion skinning weights.

    Supports humanoid, quadruped, winged, tailed, jaw/muzzle split, secondary-motion chains,
    and arbitrary custom skeletons via optional per-bone metadata:
      - skin_group: isolate weight diffusion between distinct anatomical groups
      - side_lock: 'L' | 'R' to prevent cross-midline leakage
      - rigid: bool to lock cranium / hard accessories to 1.0
      - pin_radius: float override for core pinning radius around the bone segment
      - influence_radius_scale: float (>1 wider reach, <1 tighter reach)
      - region_box: {'min': [x,y,z], 'max': [x,y,z]} soft boundary constraint
      - seam_cut: {'origin': [...], 'normal': [...], 'bbox_min': [...], 'bbox_max': [...], 'radius': float, 'keep_side': -1|+1}
    """
    skin_bones = [
        b for b, bd in bones.items()
        if b not in ("root", "eye_L", "eye_R") and not bd.get("non_deforming", False)
    ]
    n_pts = len(xyz)
    n_b = len(skin_bones)
    if n_b == 0:
        return [], np.zeros((n_pts, 0), dtype=np.float32)

    body_idx = np.where(is_body)[0]
    bxyz = xyz[body_idx]
    nb = len(bxyz)

    if mid_x is None:
        if "root" in bones:
            mid_x = float(bones["root"]["head"][0])
        elif "spine_lower" in bones:
            mid_x = float(bones["spine_lower"]["head"][0])
        elif "pelvis" in bones:
            mid_x = float(bones["pelvis"]["head"][0])
        else:
            mid_x = float(np.median(bxyz[:, 0]))

    # Collect all seam cuts (both top-level and per-bone; per-bone may be a single dict or a list of dicts)
    all_cuts: list[dict] = list(seam_cuts) if seam_cuts else []
    for bname in skin_bones:
        sc = bones[bname].get("seam_cut")
        if isinstance(sc, dict):
            all_cuts.append(sc)
        elif isinstance(sc, list):
            all_cuts.extend(c for c in sc if isinstance(c, dict))

    tree = cKDTree(bxyz)
    dists_knn, idx_knn = tree.query(bxyz, k=min(13, nb), workers=-1)
    d_e = dists_knn[:, 1:]
    j_e = idx_knn[:, 1:]
    k_nbr = d_e.shape[1]
    i_e = np.repeat(np.arange(nb, dtype=np.int32)[:, None], k_nbr, axis=1)
    valid = d_e <= max_edge
    if all_cuts:
        valid = _apply_seam_cuts_to_edges(bxyz, i_e, j_e, valid, all_cuts)

    w_edge = np.exp(-(d_e[valid] / (max_edge * 0.60)) ** 2).astype(np.float32)
    A = csr_matrix((w_edge, (i_e[valid], j_e[valid])), shape=(nb, nb))
    A = 0.5 * (A + A.T)
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv_deg = np.where(deg > 1e-8, 1.0 / np.maximum(deg, 1e-8), 0.0).astype(np.float32)
    P = diags(inv_deg) @ A

    groups = [_infer_bone_skin_group(b, bones[b]) for b in skin_bones]
    unique_groups = list(dict.fromkeys(groups))
    group_to_bidxs = {g: np.array([i for i, gg in enumerate(groups) if gg == g], dtype=np.int32) for g in unique_groups}

    dists = np.full((nb, n_b), 1e6, dtype=np.float32)
    raw_dists = np.full((nb, n_b), 1e6, dtype=np.float32)
    bone_lens = np.zeros(n_b, dtype=np.float32)

    for bi, bname in enumerate(skin_bones):
        b = bones[bname]
        h = np.asarray(b["head"], dtype=np.float32)
        t = np.asarray(b["tail"], dtype=np.float32)
        blen = float(np.linalg.norm(t - h))
        bone_lens[bi] = blen
        d_raw = dist_to_segment(bxyz, h, t)
        raw_dists[:, bi] = d_raw
        r_scale = max(float(b.get("influence_radius_scale", 1.0)), 0.1)
        d = d_raw / r_scale

        # Side lock (explicit or inferred from _L/_R suffix for limbs/wings/ears/ribbons)
        # Note: explicit side_lock="none" disables side locking (e.g. continuous cape cloth sheet).
        side_lock = b.get("side_lock")
        if side_lock is None and (bname.endswith("_L") or "_L_" in bname):
            if not bname.startswith("clavicle_"):
                side_lock = "L"
        elif side_lock is None and (bname.endswith("_R") or "_R_" in bname):
            if not bname.startswith("clavicle_"):
                side_lock = "R"
        if side_lock in ("L", "R"):
            sgn = +1.0 if side_lock == "L" else -1.0
            side_margin = float(b.get("side_margin", 0.010))
            wrong_side = (bxyz[:, 0] - mid_x) * sgn < -side_margin
            d = np.where(wrong_side, d + 0.80, d)

        # Optional bounding box constraint
        rbox = b.get("region_box")
        if isinstance(rbox, dict) and "min" in rbox and "max" in rbox:
            rmin = np.asarray(rbox["min"], dtype=np.float32)
            rmax = np.asarray(rbox["max"], dtype=np.float32)
            outside = np.any((bxyz < rmin[None, :]) | (bxyz > rmax[None, :]), axis=1)
            d = np.where(outside, d + 0.80, d)

        # Per-bone seam_cut sidedness (supports a single dict or a list of dicts)
        sc_raw = b.get("seam_cut")
        sc_list = [sc_raw] if isinstance(sc_raw, dict) else (sc_raw if isinstance(sc_raw, list) else [])
        for sc in sc_list:
            if not isinstance(sc, dict) or "origin" not in sc or "normal" not in sc:
                continue
            o = np.asarray(sc["origin"], dtype=np.float32)
            n = np.asarray(sc["normal"], dtype=np.float32)
            n = n / max(float(np.linalg.norm(n)), 1e-8)
            keep_side = float(sc.get("keep_side", -1.0))
            signed = (bxyz - o[None, :]) @ n
            in_reg = np.ones(nb, dtype=bool)
            if sc.get("radius") is not None:
                in_reg &= np.linalg.norm(bxyz - o[None, :], axis=1) <= float(sc["radius"])
            if sc.get("bbox_min") is not None and sc.get("bbox_max") is not None:
                bmin = np.asarray(sc["bbox_min"], dtype=np.float32)
                bmax = np.asarray(sc["bbox_max"], dtype=np.float32)
                in_reg &= np.all((bxyz >= bmin[None, :]) & (bxyz <= bmax[None, :]), axis=1)
            wrong_half = in_reg & (signed * keep_side < 0.0)
            d = np.where(wrong_half, d + 0.80, d)

        dists[:, bi] = d

    # Also penalize parent bone on the child's side of a seam_cut when `affect_parent` is True (default True for jaw)
    for bi, bname in enumerate(skin_bones):
        b = bones[bname]
        sc_raw = b.get("seam_cut")
        sc_list = [sc_raw] if isinstance(sc_raw, dict) else (sc_raw if isinstance(sc_raw, list) else [])
        parent = b.get("parent")
        if parent not in skin_bones:
            continue
        p_idx = skin_bones.index(parent)
        for sc in sc_list:
            if not isinstance(sc, dict) or not sc.get("affect_parent", True):
                continue
            if "origin" not in sc or "normal" not in sc:
                continue
            o = np.asarray(sc["origin"], dtype=np.float32)
            n = np.asarray(sc["normal"], dtype=np.float32)
            n = n / max(float(np.linalg.norm(n)), 1e-8)
            keep_side = float(sc.get("keep_side", -1.0))
            signed = (bxyz - o[None, :]) @ n
            in_reg = np.ones(nb, dtype=bool)
            if sc.get("radius") is not None:
                in_reg &= np.linalg.norm(bxyz - o[None, :], axis=1) <= float(sc["radius"])
            if sc.get("bbox_min") is not None and sc.get("bbox_max") is not None:
                bmin = np.asarray(sc["bbox_min"], dtype=np.float32)
                bmax = np.asarray(sc["bbox_max"], dtype=np.float32)
                in_reg &= np.all((bxyz >= bmin[None, :]) & (bxyz <= bmax[None, :]), axis=1)
            child_half = in_reg & (signed * keep_side >= 0.0)
            dists[:, p_idx] = np.where(child_half, dists[:, p_idx] + 0.80, dists[:, p_idx])

    # Group-level core pinning: each point whose closest bone is well within that bone's core radius
    # is pinned to that bone's skin_group (preventing cross-group bleed during diffusion).
    best_b = np.argmin(dists, axis=1)
    best_d = dists[np.arange(nb), best_b]
    second_d = np.partition(dists, min(1, n_b - 1), axis=1)[:, min(1, n_b - 1)]

    pin_radii = np.array([
        float(bones[b].get("pin_radius", np.clip(bone_lens[i] * 0.42, 0.022, 0.065)))
        for i, b in enumerate(skin_bones)
    ], dtype=np.float32)

    dists_pin = dists.copy()
    pinned = np.zeros(nb, dtype=bool)

    for g in unique_groups:
        g_bidxs = group_to_bidxs[g]
        in_g = np.isin(best_b, g_bidxs)
        g_core = in_g & (
            (best_d <= pin_radii[best_b])
            | ((second_d - best_d) > 0.025)
        )
        if np.any(g_core):
            g_mask = np.zeros(n_b, dtype=bool)
            g_mask[g_bidxs] = True
            dists_pin[g_core] = np.where(g_mask[None, :], dists_pin[g_core], 10.0)
            pinned |= g_core

    dmin = np.min(dists_pin, axis=1, keepdims=True)
    W_b = np.exp(-((dists_pin - dmin) / 0.024) ** 2).astype(np.float32)
    W_b /= np.maximum(W_b.sum(axis=1, keepdims=True), 1e-8)

    free = (~pinned) & (deg > 1e-8)
    for _ in range(n_harmonic_iters):
        W_b[free] = (P @ W_b)[free]

    # Rigid locks (e.g. head cranium or hard accessories with rigid=True)
    rigid_mask_all = np.zeros(nb, dtype=bool)
    rigid_locks: list[tuple[int, np.ndarray]] = []
    for bi, bname in enumerate(skin_bones):
        b = bones[bname]
        if b.get("rigid", False):
            rm = (best_b == bi) & (W_b[:, bi] >= 0.55)
            if np.any(rm):
                rigid_locks.append((bi, rm))
                rigid_mask_all |= rm

    has_nbr = (deg > 1e-8) & (~rigid_mask_all)
    for _ in range(n_smooth_iters):
        W_b[has_nbr] = 0.55 * W_b[has_nbr] + 0.45 * (P @ W_b)[has_nbr]

    for bi, rm in rigid_locks:
        W_b[rm] = 0.0
        W_b[rm, bi] = 1.0

    k_top = min(4, n_b)
    if n_b > 4:
        top_idx = np.argpartition(-W_b, kth=k_top - 1, axis=1)[:, :k_top]
        W_sp = np.zeros_like(W_b)
        r_idx = np.arange(nb)[:, None]
        W_sp[r_idx, top_idx] = W_b[r_idx, top_idx]
        W_sp /= np.maximum(W_sp.sum(axis=1, keepdims=True), 1e-8)
    else:
        W_sp = W_b / np.maximum(W_b.sum(axis=1, keepdims=True), 1e-8)

    W_full = np.zeros((n_pts, n_b), dtype=np.float32)
    W_full[body_idx] = W_sp
    return skin_bones, W_full


def compute_harmonic_skin_weights(
    xyz: np.ndarray,
    bones: dict,
    is_body: np.ndarray,
    max_edge: float = 0.012,
    n_harmonic_iters: int = 90,
    n_smooth_iters: int = 20,
    seam_cuts: list[dict] | None = None,
    use_generic: bool | None = None,
    mid_x: float | None = None,
) -> tuple[list[str], np.ndarray]:
    """Data-driven C-infinity smooth Harmonic Graph-Laplacian Heat-Diffusion skinning weights.

    - Legacy v2.0 humanoid skeletons automatically use the battle-tested humanoid core pinning
      (100% backward compatible with existing v1 packages such as hood-01).
    - Extended v2.1 / non-humanoid / secondary-motion / winged / quadruped / jaw skeletons
      automatically dispatch to the generic group & seam-cut Harmonic solver.
    """
    if use_generic is True or (use_generic is None and (seam_cuts or not _is_legacy_humanoid_skeleton(bones))):
        return _compute_generic_harmonic_skin_weights(
            xyz=xyz,
            bones=bones,
            is_body=is_body,
            seam_cuts=seam_cuts,
            max_edge=max_edge,
            n_harmonic_iters=n_harmonic_iters,
            n_smooth_iters=n_smooth_iters,
            mid_x=mid_x,
        )

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
    arm_core_L = np.zeros(nb, dtype=bool)
    arm_core_R = np.zeros(nb, dtype=bool)
    arm_cut_x = max(sh_hw + 0.045, sh_hw * 1.45)
    for side, sgn, mask_out in (("L", +1.0, arm_core_L), ("R", -1.0, arm_core_R)):
        cand = np.where(((bxyz[:, 0] - face_cx) * sgn > arm_cut_x) & (bxyz[:, 2] >= arm_min_z) & (bxyz[:, 2] <= sh_z + 0.09))[0]
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
    torso_core = (
        ((np.abs(bxyz[:, 0] - face_cx) < sh_hw * 1.05) & (bxyz[:, 2] >= hip_z - 0.02) & (bxyz[:, 2] <= neck_z + 0.02))
        | ((np.abs(bxyz[:, 0] - face_cx) < sh_hw + 0.024) & (bxyz[:, 2] >= hip_z - 0.02) & (bxyz[:, 2] <= armpit_z))
    ) & ~(arm_core_L | arm_core_R)
    waist_core = (np.abs(bxyz[:, 0] - face_cx) < sh_hw + 0.050) & (bxyz[:, 2] >= hip_z - 0.02) & (bxyz[:, 2] <= armpit_z - 0.065) & ~(arm_core_L | arm_core_R)
    head_core = bxyz[:, 2] > head_joint_z + 0.004
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

    # Smooth transitions across joints while keeping the rigid cranium (face, eyes, mouth, hair, hat) locked to head=1.0
    rigid_cranium = bxyz[:, 2] > head_joint_z + 0.012
    has_nbr = (deg > 1e-8) & (~rigid_cranium)
    for _ in range(n_smooth_iters):
        W_b[has_nbr] = 0.55 * W_b[has_nbr] + 0.45 * (P @ W_b)[has_nbr]
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


def densify_deforming_region(
    data: np.ndarray,
    skin_bones: list[str],
    W: np.ndarray,
    bone_prefixes: tuple[str, ...] = ("tail_",),
    min_weight: float = 0.25,
    max_new: int = 12000,
    max_edge: float = 0.024,
) -> np.ndarray:
    """Pre-densify high-flex 3DGS regions (e.g. fluffy tails, cloth roots) by inserting
    color-interpolated midpoint splats along local kNN edges so GPU skinning never thins out when bent.
    """
    b_idxs = [i for i, b in enumerate(skin_bones) if b.startswith(bone_prefixes)]
    if not b_idxs:
        return data
    w_reg = W[:, b_idxs].sum(axis=1)
    # Include both interior of the region and the root transition band (where stretch is highest)
    reg_idx = np.where(w_reg >= min_weight)[0]
    if len(reg_idx) < 30:
        return data

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    rxyz = xyz[reg_idx]
    tree = cKDTree(rxyz)
    dists, nbrs = tree.query(rxyz, k=min(5, len(reg_idx)), workers=-1)
    d_e = dists[:, 1:]
    j_e = nbrs[:, 1:]
    i_e = np.repeat(np.arange(len(reg_idx), dtype=np.int32)[:, None], d_e.shape[1], axis=1)
    valid = (d_e >= 0.0025) & (d_e <= max_edge) & (i_e < j_e)
    ii, jj = i_e[valid], j_e[valid]
    if len(ii) == 0:
        return data

    # Prioritize wider gaps (higher stretch risk)
    d_sel = d_e[valid]
    if len(ii) > max_new:
        order = np.argsort(-d_sel)[:max_new]
        ii, jj = ii[order], jj[order]

    ga, gb = reg_idx[ii], reg_idx[jj]
    new_d = data[ga].copy()
    for fld in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"):
        new_d[fld] = (0.5 * (data[fld][ga] + data[fld][gb])).astype(np.float32)
    # Slightly plump scale & solid opacity so gaps close cleanly without puffing out silhouette
    for sf in ("scale_0", "scale_1", "scale_2"):
        new_d[sf] = (0.5 * (data[sf][ga] + data[sf][gb]) + 0.08).astype(np.float32)
    new_d["opacity"] = np.maximum(data["opacity"][ga], data["opacity"][gb]).astype(np.float32)
    return np.concatenate([data, new_d], axis=0)


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
