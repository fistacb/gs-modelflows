"""Step 7: Build immutable v1 standard 3DGS asset package (<asset_id>/v1/) driven by character_config.json.

All skeleton joints (spine, chest, neck, head, clavicles, arms, legs, feet) are derived 100% data-driven
from 3DGS cross-section centroids, anatomical width minima (neck, wrist, ankle), and leg bifurcation (crotch).
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from ply_io import read_ply, write_ply
from config_loader import load_config

SH_C0 = 0.28209479177387814


def make_box_mesh(bmin: np.ndarray, bmax: np.ndarray, margin: float = 0.012) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Create an 8-vertex, 6-quad-face low-poly axis-aligned cage box with safety margin."""
    lo = bmin - margin
    hi = bmax + margin
    verts = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ], dtype=np.float32)
    faces = [
        (1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5),
        (3, 4, 8, 7), (1, 5, 8, 4), (2, 3, 7, 6),
    ]
    return verts, faces


def local_slab_median(tree: cKDTree, xyz_op: np.ndarray, p: np.ndarray, axis: np.ndarray, blen: float, perp_r: float = 0.085) -> np.ndarray:
    """Compute local 3DGS cross-section center (hollow-shell aware) around p perpendicular to bone axis."""
    slab_h = max(0.014, min(0.030, blen * 0.18))
    idx = tree.query_ball_point(p, r=perp_r)
    if len(idx) < 12:
        return p.copy()
    pts = xyz_op[idx]
    along = (pts - p[None, :]) @ axis
    sm = np.abs(along) <= slab_h
    spts = pts[sm] if int(sm.sum()) >= 10 else pts
    p05 = np.percentile(spts, 5, axis=0)
    p95 = np.percentile(spts, 95, axis=0)
    med = np.median(spts, axis=0)
    mid = 0.5 * (p05 + p95)
    return np.where((p95 - p05) > 0.11, mid, med).astype(np.float32)


def slice_mid(pts: np.ndarray, z: float, half_dz: float = 0.016) -> np.ndarray:
    """Geometric midpoint (0.5*(p10+p90)) + median blend for hollow 3DGS shell stability."""
    m = np.abs(pts[:, 2] - z) <= half_dz
    spts = pts[m] if m.sum() >= 12 else pts[np.argsort(np.abs(pts[:, 2] - z))[:max(12, min(40, len(pts)))]]
    mid_y = 0.5 * (float(np.percentile(spts[:, 1], 10)) + float(np.percentile(spts[:, 1], 90)))
    med_x = float(np.median(spts[:, 0]))
    return np.array([med_x, mid_y, float(z)], dtype=np.float32)


def detect_crotch_z(xyz_no_arms: np.ndarray, torso_cx: float, legs_max_z_hint: float) -> float:
    """Data-driven crotch (leg bifurcation) Z: highest Z where center density dips below side leg density."""
    leg_zone = xyz_no_arms[np.abs(xyz_no_arms[:, 0] - torso_cx) <= 0.14]
    z_lo = 0.12
    z_hi = min(0.60, max(legs_max_z_hint + 0.06, 0.52))
    zs = np.linspace(z_lo, z_hi, 60)
    dz = (zs[1] - zs[0]) * 1.15
    last_split_z = legs_max_z_hint
    for z in zs:
        m = np.abs(leg_zone[:, 2] - z) <= dz
        if m.sum() < 40:
            continue
        xs = leg_zone[m, 0] - torso_cx
        n_l = int(((xs > 0.020) & (xs < 0.095)).sum())
        n_r = int(((xs < -0.020) & (xs > -0.095)).sum())
        n_mid = int((np.abs(xs) <= 0.014).sum())
        # Dip in center density (tolerates baggy pants lightly brushing at upper inner thigh)
        if n_l >= 20 and n_r >= 20 and n_mid <= 0.35 * min(n_l, n_r):
            last_split_z = float(z)
    return float(last_split_z)


def detect_neck_z(xyz: np.ndarray, z_lo: float, z_hi: float) -> float:
    """Data-driven neck Z: narrowest horizontal X-span between shoulders and head."""
    zs = np.linspace(z_lo, z_hi, 50)
    dz = (zs[1] - zs[0]) * 1.2
    widths = []
    for z in zs:
        m = np.abs(xyz[:, 2] - z) <= dz
        widths.append(1e9 if m.sum() < 20 else float(np.percentile(xyz[m, 0], 95) - np.percentile(xyz[m, 0], 5)))
    w_arr = np.array(widths)
    sm = np.convolve(w_arr, [0.25, 0.5, 0.25], mode="same")
    sm[0], sm[-1] = w_arr[0], w_arr[-1]
    return float(zs[int(np.argmin(sm))])


def extract_arm_component(
    xyz: np.ndarray,
    rgb: np.ndarray,
    side: str,
    prop: dict,
    torso_cx: float,
    torso_y: float,
    z_neck: float,
    override_sh: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract left (+X) or right (-X) arm via 3D kNN connected component and fit shoulder/elbow/wrist/hand."""
    az0, az1 = float(prop["arm_z_range"][0]), float(prop["arm_z_range"][1])
    cut_x = float(prop["arm_cut_abs_x"])
    sh_hw = float(prop["shoulder_half_width"])
    ws0, ws1 = float(prop["wrist_scan_s_range"][0]), float(prop["wrist_scan_s_range"][1])

    if side == "R":
        cand = np.where((xyz[:, 0] < torso_cx - cut_x) & (xyz[:, 2] >= az0) & (xyz[:, 2] <= az1))[0]
        tip_fn, sign_x = np.argmin, -1.0
    else:
        cand = np.where((xyz[:, 0] > torso_cx + cut_x) & (xyz[:, 2] >= az0) & (xyz[:, 2] <= az1))[0]
        tip_fn, sign_x = np.argmax, +1.0

    cxyz = xyz[cand]
    adj = cKDTree(cxyz).sparse_distance_matrix(cKDTree(cxyz), max_distance=0.016, output_type="coo_matrix")
    _, labels = connected_components(adj, directed=False)
    arm_idx = cand[labels == labels[int(tip_fn(cxyz[:, 0]))]]
    pts = xyz[arm_idx]
    cols = rgb[arm_idx]

    if override_sh is not None:
        shoulder = override_sh.astype(np.float32)
    else:
        # True anatomical glenohumeral shoulder joint: ~2.2 cm below the top shoulder contour at |x - torso_cx| = sh_hw
        target_sh_x = torso_cx + sign_x * sh_hw
        sh_col = (np.abs(xyz[:, 0] - target_sh_x) <= 0.015) & (xyz[:, 2] >= 0.65) & (xyz[:, 2] <= z_neck)
        if sh_col.sum() >= 30:
            z_sh_top = float(np.percentile(xyz[sh_col, 2], 96))
            sh_z_est = min(z_sh_top - 0.026, z_neck - 0.060)
        else:
            abs_x = np.abs(pts[:, 0] - torso_cx)
            junc_m = abs_x <= cut_x + 0.030
            junc_pts = pts[junc_m] if junc_m.sum() >= 25 else pts
            sh_z_est = min(float(np.percentile(junc_pts[:, 2], 90)), z_neck - 0.055)
        sh_loc = (np.abs(xyz[:, 0] - target_sh_x) <= 0.022) & (np.abs(xyz[:, 2] - sh_z_est) <= 0.022)
        if sh_loc.sum() >= 20:
            sh_y = 0.5 * (float(np.percentile(xyz[sh_loc, 1], 10)) + float(np.percentile(xyz[sh_loc, 1], 90)))
        else:
            sh_y = torso_y
        shoulder = np.array([target_sh_x, sh_y, sh_z_est], dtype=np.float32)

    hand_tip = pts[int(tip_fn(pts[:, 0]))].astype(np.float32)
    v_arm = hand_tip - shoulder
    len_arm = float(np.linalg.norm(v_arm))
    u_arm = v_arm / max(len_arm, 1e-6)
    s = np.sum((pts - shoulder[None, :]) * u_arm[None, :], axis=1) / max(len_arm, 1e-6)
    perp = np.linalg.norm((pts - shoulder[None, :]) - (s * len_arm)[:, None] * u_arm[None, :], axis=1)

    # Data-driven wrist detection:
    # 1) If sleeved (distinct color between forearm s in [0.30, 0.55] and hand s >= 0.87),
    #    detect exact sleeve-cuff to skin transition boundary in s in [0.58, 0.86].
    # 2) Otherwise (sleeveless), fall back to narrowest perpendicular diameter in [ws0, min(ws1, 0.82)].
    m_hand_ref = s >= 0.87
    m_slv_ref = (s >= 0.30) & (s <= 0.55)
    cuff_s = None
    if m_hand_ref.sum() >= 15 and m_slv_ref.sum() >= 30:
        skin_rgb = np.median(cols[m_hand_ref], axis=0)
        slv_rgb = np.median(cols[m_slv_ref], axis=0)
        if float(np.linalg.norm(skin_rgb - slv_rgb)) > 0.16:
            d_skin = np.linalg.norm(cols - skin_rgb[None, :], axis=1)
            d_slv = np.linalg.norm(cols - slv_rgb[None, :], axis=1)
            is_skin = d_skin < d_slv
            for sc in np.linspace(0.58, 0.86, 29):
                ms = (s >= sc - 0.022) & (s <= sc + 0.022)
                if ms.sum() >= 10 and float(is_skin[ms].mean()) >= 0.50:
                    cuff_s = float(sc)
                    break

    if cuff_s is not None:
        best_s_wr = cuff_s
    else:
        best_s_wr, min_diam = 0.5 * (ws0 + min(ws1, 0.82)), 1e9
        for sc in np.linspace(ws0, min(ws1, 0.82), 14):
            ms = (s >= sc - 0.032) & (s <= sc + 0.032)
            if ms.sum() >= 12:
                diam = float(np.percentile(perp[ms], 85))
                if diam < min_diam:
                    min_diam, best_s_wr = diam, float(sc)

    s_el = 0.48 * best_s_wr
    m_el = (s >= s_el - 0.038) & (s <= s_el + 0.038)
    m_wr = (s >= best_s_wr - 0.030) & (s <= best_s_wr + 0.030)
    m_hd = (s >= max(best_s_wr + 0.06, 0.89)) & (s <= 0.96)
    elbow = np.median(pts[m_el], axis=0).astype(np.float32) if np.any(m_el) else (shoulder + s_el * v_arm).astype(np.float32)
    wrist = np.median(pts[m_wr], axis=0).astype(np.float32) if np.any(m_wr) else (shoulder + best_s_wr * v_arm).astype(np.float32)
    hand = np.median(pts[m_hd], axis=0).astype(np.float32) if np.any(m_hd) else (shoulder + 0.93 * v_arm).astype(np.float32)
    print(f"[AutoArmExtractor-{side}] sh_z={shoulder[2]:.3f}, s_el={s_el:.2f} (el_z={elbow[2]:.3f}), s_wr={best_s_wr:.2f} (wr_z={wrist[2]:.3f}, cuff={'yes' if cuff_s is not None else 'no'}), hand_len={float(np.linalg.norm(hand - wrist)):.3f}")
    return arm_idx, shoulder, elbow, wrist, hand


def extract_leg_joints(xyz_no_arms: np.ndarray, torso_cx: float, z_crotch: float) -> dict:
    """Data-driven symmetric leg joint extraction (hip, knee, ankle, toe) from left/right leg cross-sections."""
    legs = {}
    z_hip = max(0.10, z_crotch - 0.015)
    z_an_lo = max(0.055, z_crotch * 0.12)
    z_an_hi = max(z_an_lo + 0.04, z_crotch * 0.34)
    zs_an = np.linspace(z_an_lo, z_an_hi, 28)

    raw = {}
    for side, sgn in (("L", +1.0), ("R", -1.0)):
        lpts = xyz_no_arms[(xyz_no_arms[:, 2] <= z_crotch + 0.025) & ((xyz_no_arms[:, 0] - torso_cx) * sgn > 0.008)]
        if len(lpts) < 50:
            raise RuntimeError(f"Insufficient leg splats for side {side} below z_crotch={z_crotch:.3f}")
        diams = []
        for z in zs_an:
            m = np.abs(lpts[:, 2] - z) <= 0.012
            if m.sum() < 15:
                diams.append(1e9)
            else:
                dx = float(np.percentile(lpts[m, 0], 92) - np.percentile(lpts[m, 0], 8))
                dy = float(np.percentile(lpts[m, 1], 92) - np.percentile(lpts[m, 1], 8))
                diams.append(dx + dy)
        z_ankle = float(zs_an[int(np.argmin(diams))])
        z_knee = 0.50 * (z_hip + z_ankle)

        hip = slice_mid(lpts, z_hip, half_dz=0.018)
        knee = slice_mid(lpts, z_knee, half_dz=0.018)
        ankle = slice_mid(lpts, z_ankle, half_dz=0.014)

        # Foot bone: starts at ankle, points forward into mid-shoe volume (aligned with shoe centerline)
        shoe_m = lpts[:, 2] <= max(z_ankle * 0.85, 0.065)
        spts = lpts[shoe_m] if shoe_m.sum() >= 25 else lpts[lpts[:, 2] <= z_ankle]
        toe_z = max(float(np.percentile(spts[:, 2], 55)), 0.032)
        toe_y = float(np.percentile(spts[:, 1], 32))
        toe_x = float(np.median(spts[:, 0]))
        toe = np.array([toe_x, toe_y, toe_z], dtype=np.float32)
        raw[side] = {"hip": hip, "knee": knee, "ankle": ankle, "toe": toe}

    for jname in ("hip", "knee", "ankle", "toe"):
        pL, pR = raw["L"][jname], raw["R"][jname]
        ax = 0.5 * (abs(float(pL[0]) - torso_cx) + abs(float(pR[0]) - torso_cx))
        ay = 0.5 * (float(pL[1]) + float(pR[1]))
        az = 0.5 * (float(pL[2]) + float(pR[2]))
        legs[f"{jname}_L"] = np.array([torso_cx + ax, ay, az], dtype=np.float32)
        legs[f"{jname}_R"] = np.array([torso_cx - ax, ay, az], dtype=np.float32)
    print(f"[AutoLegExtractor] z_crotch={z_crotch:.3f}, hip_z={legs['hip_L'][2]:.3f}, knee_z={legs['knee_L'][2]:.3f}, ankle_z={legs['ankle_L'][2]:.3f}, toe_y={legs['toe_L'][1]:.3f}")
    return legs


def refine_joints_to_cross_sections(bones: dict, xyz_op: np.ndarray, torso_cx: float, n_iters: int = 4):
    """Iteratively align bone interiors to local 3DGS cross-section medians while preventing hollow-shell wall drift."""
    tree = cKDTree(xyz_op)
    # Record initial geometric positions so joints never drift far from anatomical landmarks
    orig = {bname: {"head": np.asarray(b["head"], dtype=np.float32).copy(), "tail": np.asarray(b["tail"], dtype=np.float32).copy()} for bname, b in bones.items()}
    # Refine ALL non-root, non-eye bones gently toward their exact hollow-shell-aware cross-section centers
    for _ in range(n_iters):
        shifts = {}
        counts = {}
        for bname, b in bones.items():
            if bname == "root" or bname.startswith("eye_") or bname.startswith("clavicle_"):
                continue
            h = np.asarray(b["head"], dtype=np.float32)
            t = np.asarray(b["tail"], dtype=np.float32)
            dv = t - h
            blen = float(np.linalg.norm(dv))
            if blen < 1e-4:
                continue
            axis = dv / blen
            is_central = (abs(float(h[0])) < 0.06 and abs(float(t[0])) < 0.06 and float(min(h[2], t[2])) > 0.35)
            perp_r = 0.065 if bname.startswith("clavicle_") else (0.135 if is_central else 0.085)
            dh = np.zeros(3, dtype=np.float32)
            dt = np.zeros(3, dtype=np.float32)
            for st in (0.25, 0.50, 0.75):
                p = h + st * dv
                c = local_slab_median(tree, xyz_op, p, axis, blen, perp_r=perp_r)
                err = c - p
                err_perp = err - float(np.dot(err, axis)) * axis
                dh += (1.0 - st) * err_perp
                dt += st * err_perp
            step_scale = 0.38
            key_h = tuple(np.round(h, 4).tolist())
            key_t = tuple(np.round(t, 4).tolist())
            shifts[key_h] = shifts.get(key_h, np.zeros(3, dtype=np.float32)) + dh * step_scale
            counts[key_h] = counts.get(key_h, 0) + 1
            shifts[key_t] = shifts.get(key_t, np.zeros(3, dtype=np.float32)) + dt * step_scale
            counts[key_t] = counts.get(key_t, 0) + 1

        for bname, b in bones.items():
            if bname == "root" or bname.startswith("eye_") or bname.startswith("clavicle_"):
                continue
            for end in ("head", "tail"):
                p = np.asarray(b[end], dtype=np.float32)
                key = tuple(np.round(p, 4).tolist())
                if key in shifts:
                    p_new = p + shifts[key] / max(counts[key], 1)
                    max_drift = 0.042 if (bname.startswith("clavicle_") or bname.startswith("foot_") or bname == "chest" or bname.startswith("upper_arm_")) else 0.028
                    d_orig = p_new - orig[bname][end]
                    nrm = float(np.linalg.norm(d_orig))
                    if nrm > max_drift:
                        d_orig = d_orig * (max_drift / nrm)
                    p_new = orig[bname][end] + d_orig
                    if bname.startswith("foot_") and end == "tail":
                        p_new[2] = max(float(p_new[2]), 0.022)
                    if bname.startswith("upper_arm_") and end == "head":
                        # Lock shoulder X & Z tightly to true deltoid/glenohumeral landmark (only refine Y depth)
                        p_new[0] = np.clip(float(p_new[0]), float(orig[bname][end][0]) - 0.006, float(orig[bname][end][0]) + 0.006)
                        p_new[2] = np.clip(float(p_new[2]), float(orig[bname][end][2]) - 0.006, float(orig[bname][end][2]) + 0.006)
                    if (bname.startswith("upper_arm_") and end == "tail") or (bname.startswith("forearm_") and end == "head"):
                        p_new[2] = np.clip(float(p_new[2]), float(orig[bname][end][2]) - 0.012, float(orig[bname][end][2]) + 0.012)
                    if not (bname.endswith("_L") or bname.endswith("_R")):
                        p_new[0] = torso_cx
                    elif bname.startswith("clavicle_") and end == "head":
                        p_new[0] = torso_cx
                    b[end] = p_new.tolist()

        # Keep clavicles strictly connected to chest.tail and upper_arm.head
        for s in ("L", "R"):
            bones[f"clavicle_{s}"]["head"] = list(bones["chest"]["tail"])
            bones[f"clavicle_{s}"]["tail"] = list(bones[f"upper_arm_{s}"]["head"])

        for bname in sorted(bones.keys()):
            if bname.endswith("_L") and not bname.startswith("eye_"):
                rname = bname[:-2] + "_R"
                if rname in bones:
                    for end in ("head", "tail"):
                        pL = np.asarray(bones[bname][end], dtype=np.float32)
                        pR = np.asarray(bones[rname][end], dtype=np.float32)
                        ax = 0.5 * (abs(float(pL[0]) - torso_cx) + abs(float(pR[0]) - torso_cx))
                        ay = 0.5 * (float(pL[1]) + float(pR[1]))
                        az = 0.5 * (float(pL[2]) + float(pR[2]))
                        if bname.startswith("clavicle_") and end == "head":
                            ax = 0.0
                        bones[bname][end] = [torso_cx + ax, ay, az]
                        bones[rname][end] = [torso_cx - ax, ay, az]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["work_dir"])
    asset_id, version, prop = cfg["asset_id"], cfg["version"], cfg["proportions"]
    ver_dir = root / "packages" / asset_id / version
    exports_dir = ver_dir / "exports"
    ver_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    head_pol, _ = read_ply(root / "outputs" / "step3_carved_head_polished.ply")
    eye_L_loc, _ = read_ply(root / "outputs" / "step4_eye_L_local.ply")
    eye_R_loc, _ = read_ply(root / "outputs" / "step4_eye_R_local.ply")
    skel_v2 = json.loads((root / "outputs" / "step2_skeleton.json").read_text(encoding="utf-8"))

    eL = eye_L_loc.copy(); pL = skel_v2["bones"]["eye_L"]["head"]
    eL["x"] += pL[0]; eL["y"] += pL[1]; eL["z"] += pL[2]
    eR = eye_R_loc.copy(); pR = skel_v2["bones"]["eye_R"]["head"]
    eR["x"] += pR[0]; eR["y"] += pR[1]; eR["z"] += pR[2]
    master = np.concatenate([head_pol, eL, eR], axis=0)
    write_ply(ver_dir / "master.ply", master)

    xyz = np.column_stack([master["x"], master["y"], master["z"]]).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * np.column_stack([master["f_dc_0"], master["f_dc_1"], master["f_dc_2"]]), 0.0, 1.0).astype(np.float32)
    op = 1.0 / (1.0 + np.exp(-np.clip(master["opacity"], -20, 20)))
    xyz_op = xyz[op > 0.15]

    face_cx = float(skel_v2["bones"]["root"]["head"][0])
    z_top = float(np.percentile(xyz_op[:, 2], 99.5))

    # 1. Data-driven neck & initial arm extraction
    nz0, nz1 = float(prop["neck_z_range"][0]), float(prop["neck_z_range"][1])
    z_neck = detect_neck_z(xyz_op, max(0.55, nz0 - 0.05), min(0.95, nz1 + 0.05))
    torso_mid_y = float(np.median(xyz_op[(xyz_op[:, 2] >= 0.50) & (xyz_op[:, 2] <= z_neck) & (np.abs(xyz_op[:, 0] - face_cx) < 0.09), 1]))

    _, sh_R_raw, _, _, _ = extract_arm_component(xyz, rgb, "R", prop, face_cx, torso_mid_y, z_neck)
    _, sh_L_raw, _, _, _ = extract_arm_component(xyz, rgb, "L", prop, face_cx, torso_mid_y, z_neck)
    sh_y_sym = 0.5 * (float(sh_R_raw[1]) + float(sh_L_raw[1]))
    sh_z_sym = 0.5 * (float(sh_R_raw[2]) + float(sh_L_raw[2]))
    sh_abs_x = 0.5 * (abs(float(sh_R_raw[0]) - face_cx) + abs(float(sh_L_raw[0]) - face_cx))
    arm_R_idx, sh_R, el_R_raw, wr_R_raw, hd_R_raw = extract_arm_component(xyz, rgb, "R", prop, face_cx, torso_mid_y, z_neck, override_sh=np.array([face_cx - sh_abs_x, sh_y_sym, sh_z_sym], dtype=np.float32))
    arm_L_idx, sh_L, el_L_raw, wr_L_raw, hd_L_raw = extract_arm_component(xyz, rgb, "L", prop, face_cx, torso_mid_y, z_neck, override_sh=np.array([face_cx + sh_abs_x, sh_y_sym, sh_z_sym], dtype=np.float32))

    def sym_pair(pL_v, pR_v):
        ax = 0.5 * (abs(float(pL_v[0]) - face_cx) + abs(float(pR_v[0]) - face_cx))
        ay = 0.5 * (float(pL_v[1]) + float(pR_v[1]))
        az = 0.5 * (float(pL_v[2]) + float(pR_v[2]))
        return np.array([face_cx + ax, ay, az], dtype=np.float32), np.array([face_cx - ax, ay, az], dtype=np.float32)

    el_L, el_R = sym_pair(el_L_raw, el_R_raw)
    wr_L, wr_R = sym_pair(wr_L_raw, wr_R_raw)
    hd_L, hd_R = sym_pair(hd_L_raw, hd_R_raw)

    # 2. Data-driven crotch & legs (strictly excluding arm splats!)
    is_arm_all = np.zeros(len(xyz), dtype=bool)
    is_arm_all[arm_R_idx] = True
    is_arm_all[arm_L_idx] = True
    xyz_no_arms = xyz[(op > 0.15) & (~is_arm_all)]
    z_crotch = detect_crotch_z(xyz_no_arms, face_cx, float(prop["legs_max_z"]))
    legs = extract_leg_joints(xyz_no_arms, face_cx, z_crotch)

    # 3. Torso centerline joints (pelvis -> chest -> clav_base -> neck -> head)
    torso_band = xyz_no_arms[(xyz_no_arms[:, 2] >= z_crotch) & (xyz_no_arms[:, 2] <= z_neck)]
    def torso_pt(z_val: float) -> list[float]:
        c = slice_mid(torso_band, z_val, half_dz=0.020)
        return [face_cx, float(c[1]), float(z_val)]

    z_pelvis = z_crotch + 0.020
    z_clav_base = min(sh_z_sym, z_neck - 0.055)
    z_chest_base = 0.5 * (z_pelvis + z_clav_base)
    p_pelvis = torso_pt(z_pelvis)
    p_chest_base = torso_pt(z_chest_base)
    p_clav_base = torso_pt(z_clav_base)
    p_neck = torso_pt(z_neck)

    head_pts = xyz_op[xyz_op[:, 2] >= z_neck]
    z_head_tail = z_neck + 0.62 * (z_top - z_neck)
    p_head_c = slice_mid(head_pts, z_head_tail, half_dz=0.022)
    p_head_tail = [face_cx, float(p_head_c[1]), float(z_head_tail)]

    identity_q = [1.0, 0.0, 0.0, 0.0]
    bones = {
        "root": {"parent": None, "head": [face_cx, p_pelvis[1], 0.0], "tail": [face_cx, p_pelvis[1], 0.08], "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}},
        "spine_lower": {"parent": "root", "head": p_pelvis, "tail": p_chest_base, "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-20, 25], "y": [-15, 15], "z": [-25, 25]}},
        "chest": {"parent": "spine_lower", "head": p_chest_base, "tail": p_clav_base, "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-18, 22], "y": [-15, 15], "z": [-25, 25]}},
        "neck": {"parent": "chest", "head": p_clav_base, "tail": p_neck, "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-20, 20], "y": [-15, 15], "z": [-30, 30]}},
        "head": {"parent": "neck", "head": p_neck, "tail": p_head_tail, "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-25, 25], "y": [-20, 20], "z": [-40, 40]}},
        "eye_L": {"parent": "head", "head": skel_v2["bones"]["eye_L"]["head"], "tail": skel_v2["bones"]["eye_L"]["tail"], "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-15, 12], "y": [0, 0], "z": [-20, 20]}, "meta": skel_v2["bones"]["eye_L"]["meta"]},
        "eye_R": {"parent": "head", "head": skel_v2["bones"]["eye_R"]["head"], "tail": skel_v2["bones"]["eye_R"]["tail"], "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-15, 12], "y": [0, 0], "z": [-20, 20]}, "meta": skel_v2["bones"]["eye_R"]["meta"]},
        "clavicle_R": {"parent": "chest", "head": p_clav_base, "tail": sh_R.tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-15, 15], "y": [-20, 25], "z": [-20, 20]}},
        "upper_arm_R": {"parent": "clavicle_R", "head": sh_R.tolist(), "tail": el_R.tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-45, 90], "y": [-20, 85], "z": [-60, 60]}},
        "forearm_R": {"parent": "upper_arm_R", "head": el_R.tolist(), "tail": wr_R.tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-15, 15], "y": [-10, 125], "z": [-20, 20]}},
        "hand_R": {"parent": "forearm_R", "head": wr_R.tolist(), "tail": hd_R.tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-30, 30], "y": [-25, 25], "z": [-30, 30]}},
        "clavicle_L": {"parent": "chest", "head": p_clav_base, "tail": sh_L.tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-15, 15], "y": [-25, 20], "z": [-20, 20]}},
        "upper_arm_L": {"parent": "clavicle_L", "head": sh_L.tolist(), "tail": el_L.tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-45, 90], "y": [-85, 20], "z": [-60, 60]}},
        "forearm_L": {"parent": "upper_arm_L", "head": el_L.tolist(), "tail": wr_L.tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-15, 15], "y": [-125, 10], "z": [-20, 20]}},
        "hand_L": {"parent": "forearm_L", "head": wr_L.tolist(), "tail": hd_L.tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-30, 30], "y": [-25, 25], "z": [-30, 30]}},
        "thigh_L": {"parent": "spine_lower", "head": legs["hip_L"].tolist(), "tail": legs["knee_L"].tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-85, 35], "y": [-30, 30], "z": [-35, 35]}},
        "shin_L": {"parent": "thigh_L", "head": legs["knee_L"].tolist(), "tail": legs["ankle_L"].tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [0, 125], "y": [-10, 10], "z": [-15, 15]}},
        "foot_L": {"parent": "shin_L", "head": legs["ankle_L"].tolist(), "tail": legs["toe_L"].tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-35, 45], "y": [-15, 15], "z": [-20, 20]}},
        "thigh_R": {"parent": "spine_lower", "head": legs["hip_R"].tolist(), "tail": legs["knee_R"].tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-85, 35], "y": [-30, 30], "z": [-35, 35]}},
        "shin_R": {"parent": "thigh_R", "head": legs["knee_R"].tolist(), "tail": legs["ankle_R"].tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [0, 125], "y": [-10, 10], "z": [-15, 15]}},
        "foot_R": {"parent": "shin_R", "head": legs["ankle_R"].tolist(), "tail": legs["toe_R"].tolist(), "rest_quat_wxyz": identity_q, "limits_deg": {"x": [-35, 45], "y": [-15, 15], "z": [-20, 20]}},
    }

    # 4. Data-driven local cross-section centering pass (preserves connectivity & L/R symmetry)
    refine_joints_to_cross_sections(bones, xyz_op, face_cx, n_iters=10)

    rig_skeleton = {
        "schema_version": "2.0",
        "convention": {
            "world": "Z-up, -Y-forward, +X-character-left, height=1.0, feet Z=0",
            "bone_local_axes": "Unified across all bones: +X=left, -Y=forward, +Z=up (rest_quat_wxyz=[1,0,0,0])",
        },
        "bones": bones,
    }
    (ver_dir / "rig_skeleton.json").write_text(json.dumps(rig_skeleton, indent=2), encoding="utf-8")

    # 5. Procedural cages (rig_cages.obj) per anatomical group
    is_arm_R = np.zeros(len(xyz), dtype=bool); is_arm_R[arm_R_idx] = True
    is_arm_L = np.zeros(len(xyz), dtype=bool); is_arm_L[arm_L_idx] = True
    is_head = (xyz[:, 2] >= z_neck)
    is_torso = (xyz[:, 2] >= z_crotch) & (xyz[:, 2] < z_neck) & (~is_arm_R) & (~is_arm_L)
    is_leg_L = (xyz[:, 2] < z_crotch) & (~is_arm_L) & (~is_arm_R) & (xyz[:, 0] >= face_cx)
    is_leg_R = (xyz[:, 2] < z_crotch) & (~is_arm_L) & (~is_arm_R) & (xyz[:, 0] < face_cx)

    cage_groups = [
        ("cage_head", xyz[is_head], 0.012),
        ("cage_torso", xyz[is_torso], 0.012),
        ("cage_arm_R", xyz[is_arm_R], 0.008),
        ("cage_arm_L", xyz[is_arm_L], 0.008),
        ("cage_leg_L", xyz[is_leg_L], 0.008),
        ("cage_leg_R", xyz[is_leg_R], 0.008),
    ]
    obj_lines = ["# Procedural MVC cages for 3DGS asset v1 (Z-up, -Y-forward)"]
    v_offset = 0
    cage_stats = {}
    for gname, gpts, margin in cage_groups:
        p01 = np.percentile(gpts, 1, axis=0)
        p99 = np.percentile(gpts, 99, axis=0)
        verts, faces = make_box_mesh(p01, p99, margin=margin)
        obj_lines.append(f"o {gname}")
        for v in verts:
            obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for f in faces:
            obj_lines.append(f"f {f[0]+v_offset} {f[1]+v_offset} {f[2]+v_offset} {f[3]+v_offset}")
        v_offset += len(verts)
        cage_stats[gname] = {"splat_count": int(len(gpts)), "margin": margin, "min": (p01 - margin).tolist(), "max": (p99 + margin).tolist()}
    (ver_dir / "rig_cages.obj").write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

    # 6. Data-driven control points
    eye_z_avg = 0.5 * (skel_v2["bones"]["eye_L"]["meta"]["surface_center"][2] + skel_v2["bones"]["eye_R"]["meta"]["surface_center"][2])
    eye_y_avg = 0.5 * (skel_v2["bones"]["eye_L"]["meta"]["surface_center"][1] + skel_v2["bones"]["eye_R"]["meta"]["surface_center"][1])
    control_points = {
        "schema_version": "1.0",
        "points": {
            "crown_top": {"pos": [face_cx, bones["head"]["tail"][1], z_top], "smooth_weight": 0.25, "role": "crown_accessory_preserve"},
            "eye_L_center": {"pos": skel_v2["bones"]["eye_L"]["meta"]["surface_center"], "smooth_weight": 0.0, "role": "face_detail_preserve"},
            "eye_R_center": {"pos": skel_v2["bones"]["eye_R"]["meta"]["surface_center"], "smooth_weight": 0.0, "role": "face_detail_preserve"},
            "mouth_center": {"pos": [face_cx, eye_y_avg, 0.5 * (eye_z_avg + z_neck)], "smooth_weight": 0.45, "role": "local_mouth_smooth"},
            "neck_joint": {"pos": bones["neck"]["tail"], "smooth_weight": 0.60, "role": "neck_blend"},
            "shoulder_R": {"pos": bones["upper_arm_R"]["head"], "smooth_weight": 0.50, "role": "arm_joint_R"},
            "shoulder_L": {"pos": bones["upper_arm_L"]["head"], "smooth_weight": 0.50, "role": "arm_joint_L"},
            "pelvis_center": {"pos": bones["spine_lower"]["head"], "smooth_weight": 0.30, "role": "torso_root_anchor"},
            "feet_ground": {"pos": [face_cx, bones["spine_lower"]["head"][1], 0.0], "smooth_weight": 0.10, "role": "fdm_bed_contact"},
        },
    }
    (ver_dir / "control_points.json").write_text(json.dumps(control_points, indent=2), encoding="utf-8")

    prev_src = root / "qa" / "gate1_4view.png"
    if prev_src.exists():
        shutil.copy2(prev_src, ver_dir / "preview.png")

    nan_count = sum(int(np.isnan(master[k]).sum() + np.isinf(master[k]).sum()) for k in master.dtype.names)
    q = np.column_stack([master["rot_0"], master["rot_1"], master["rot_2"], master["rot_3"]])
    q_err = float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0)))
    ext = np.percentile(xyz, 99.5, axis=0) - np.percentile(xyz, 0.5, axis=0)

    qa_ok = (nan_count == 0) and (q_err < 1e-4) and (0.95 <= ext[2] <= 1.05) and (len(arm_R_idx) > 500) and (len(arm_L_idx) > 500)
    meta = {
        "asset_id": asset_id,
        "version": version,
        "frozen": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_workflow": f"gs-modeling ({int(len(master))} splats) -> normalize -> carve_sockets -> polish -> doll_eyes -> data_driven_full_rig",
        "supported_pipelines": ["animation_spark_3dgs", "fdm_print_remesh_ready"],
        "face_rig_mode": "doll_eye_with_independent_eyelid_shells",
        "master_stats": {
            "num_gaussians": int(len(master)),
            "sh_degree": 0,
            "opacity_space": "logit",
            "scale_space": "log",
            "extent_xyz": [round(float(v), 4) for v in ext],
        },
        "cage_stats": cage_stats,
        "auto_qa": {
            "numeric_gate": "OK (awaiting_verify_gate_and_independent_visual_review)" if qa_ok else "FAIL",
            "nan_or_inf_count": nan_count,
            "max_quat_norm_error": round(q_err, 6),
            "height_normalized_ok": bool(0.95 <= ext[2] <= 1.05),
            "arm_R_connected_component_pts": int(len(arm_R_idx)),
            "arm_L_connected_component_pts": int(len(arm_L_idx)),
            "total_bones": len(bones),
        },
        "files": {
            "master": "master.ply",
            "skeleton": "rig_skeleton.json",
            "cages": "rig_cages.obj",
            "control_points": "control_points.json",
            "preview": "preview.png",
            "preview_rig": "preview_rig.png",
            "exports": [],
        },
    }
    (ver_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Built v1 package ({len(bones)} bones, z_crotch={z_crotch:.3f}, z_neck={z_neck:.3f}, sh_z={bones['upper_arm_L']['head'][2]:.3f})")


if __name__ == "__main__":
    main()
