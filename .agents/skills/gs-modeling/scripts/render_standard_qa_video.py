"""All-in-One Standard Verification Suite Video Generator (Character-Agnostic, GPU SplatSkinning).

Generates and packs ONE unified regression-testing MP4 (12 fps, 12.5s, 640x640) + 12-panel QA contact sheet
using Spark GPU SplatSkinning (`capture.mjs anim`) with ZERO per-frame full-body PLY writes:
  Part 1 (66f, front face close-up):
    1A. BLINK | 1B. EYES ONLY (Head Locked) | 1C. HEAD TURN (Eyes Locked) | 1D. WINK L | 1E. MOUTH & A-I-U-E-O LIP-SYNC
  Part 2 (36f, 30-deg upper-body close):
    2A. ARMS DROP | 2B. HIGH ABDUCTION (~135) | 2C. FLEX & WAVE | 2D. AXIAL ROT
  Part 3 (48f, 35-deg full-body):
    3A. WALK | 3B. RUN | 3C. JUMP
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from config_loader import load_config
from ply_io import read_ply, write_ply
from step4b_eyelid_shell_test import detect_lash_ceiling_dz, build_eyelid_shell
from procedural_mouth import (
    MOUTH_PRESETS,
    build_procedural_mouth,
    carve_mouth_pocket,
    detect_mouth_landmarks,
)
from harmonic_dqs import (
    SH_C0,
    add_forearm_twist_bones,
    axis_angle,
    blender_rigid_to_spark_qp,
    build_fk_hierarchy,
    compute_fk_transforms,
    compute_harmonic_skin_weights,
    deform_dqs,
    euler_xyz,
    fk_to_spark_bones,
    look_at_eye_rot,
    pack_spark_skin_weights,
    quat_mul_wxyz,
    rotmat_to_quat_wxyz,
    smooth01,
    to_spark_ply,
)


def get_font(size: int):
    for p in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def add_banner(img: Image.Image, text: str, font) -> Image.Image:
    out = img.convert("RGB").resize((640, 640), Image.LANCZOS)
    draw = ImageDraw.Draw(out, "RGBA")
    draw.rectangle([0, 0, 640, 36], fill=(16, 18, 24, 210))
    draw.text((12, 7), text, fill=(245, 248, 255, 255), font=font)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--spark-dir", required=True, help="Path to spark_demo directory containing capture.mjs")
    ap.add_argument("--skip-render", action="store_true")
    args = ap.parse_args()

    t0_all = time.perf_counter()
    cfg = load_config(args.config)
    work = Path(cfg["work_dir"])
    asset_id, version = cfg["asset_id"], cfg["version"]
    pkg = work / "packages" / asset_id / version
    spark = Path(args.spark_dir)
    exports = work / "exports"
    qa_dir = work / "qa"
    exports.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    skel_v1 = json.loads((pkg / "rig_skeleton.json").read_text(encoding="utf-8"))
    skel_v2 = json.loads((work / "outputs" / "step2_skeleton.json").read_text(encoding="utf-8"))
    bones = add_forearm_twist_bones(skel_v1["bones"])
    order = build_fk_hierarchy(bones)

    head_pol, _ = read_ply(work / "outputs" / "step3_carved_head_polished.ply")
    eye_L_loc, _ = read_ply(work / "outputs" / "step4_eye_L_local.ply")
    eye_R_loc, _ = read_ply(work / "outputs" / "step4_eye_R_local.ply")

    pL = np.asarray(skel_v2["bones"]["eye_L"]["head"], dtype=np.float32)
    pR = np.asarray(skel_v2["bones"]["eye_R"]["head"], dtype=np.float32)
    surf_L = np.asarray(skel_v2["bones"]["eye_L"]["meta"]["surface_center"], dtype=np.float32)
    surf_R = np.asarray(skel_v2["bones"]["eye_R"]["meta"]["surface_center"], dtype=np.float32)
    eyeball_r = float(skel_v2["bones"]["eye_L"]["meta"]["eyeball_radius"])
    D_eye = float(np.linalg.norm(surf_L - surf_R))
    face_cx = 0.5 * (float(surf_L[0]) + float(surf_R[0]))
    z_eye = 0.5 * (float(pL[2]) + float(pR[2]))
    sy = float(min(surf_L[1], surf_R[1]))

    eL = eye_L_loc.copy(); eL["x"] += pL[0]; eL["y"] += pL[1]; eL["z"] += pL[2]
    eR = eye_R_loc.copy(); eR["x"] += pR[0]; eR["y"] += pR[1]; eR["z"] += pR[2]

    # Detect & carve procedural mouth pocket (character-agnostic, relative luminance)
    mouth_meta = detect_mouth_landmarks(head_pol, face_cx, z_eye, sy, D_eye, eyeball_r)
    head_pol_mouth = carve_mouth_pocket(head_pol, mouth_meta)
    mouth_rest_closed = build_procedural_mouth(mouth_meta, 0.0, head_pol.dtype, width_scale=MOUTH_PRESETS["closed"][1], smile=MOUTH_PRESETS["closed"][2], roundness=MOUTH_PRESETS["closed"][3])

    n_body = len(head_pol_mouth)
    body_xyz = np.column_stack([head_pol_mouth["x"], head_pol_mouth["y"], head_pol_mouth["z"]]).astype(np.float32)
    body_rot = np.column_stack([head_pol_mouth["rot_0"], head_pol_mouth["rot_1"], head_pol_mouth["rot_2"], head_pol_mouth["rot_3"]]).astype(np.float32)
    body_rot /= np.maximum(np.linalg.norm(body_rot, axis=1, keepdims=True), 1e-8)
    body_fdc = np.column_stack([head_pol_mouth["f_dc_0"], head_pol_mouth["f_dc_1"], head_pol_mouth["f_dc_2"]]).astype(np.float32)
    body_rgb = np.clip(0.5 + SH_C0 * body_fdc, 0.0, 1.0)
    body_op = 1.0 / (1.0 + np.exp(-np.clip(head_pol_mouth["opacity"], -20, 20)))

    eL_xyz0 = np.column_stack([eL["x"], eL["y"], eL["z"]]).astype(np.float32)
    eL_rot0 = np.column_stack([eL["rot_0"], eL["rot_1"], eL["rot_2"], eL["rot_3"]]).astype(np.float32)
    eL_rot0 /= np.maximum(np.linalg.norm(eL_rot0, axis=1, keepdims=True), 1e-8)
    eR_xyz0 = np.column_stack([eR["x"], eR["y"], eR["z"]]).astype(np.float32)
    eR_rot0 = np.column_stack([eR["rot_0"], eR["rot_1"], eR["rot_2"], eR["rot_3"]]).astype(np.float32)
    eR_rot0 /= np.maximum(np.linalg.norm(eR_rot0, axis=1, keepdims=True), 1e-8)

    print("[StandardQA-GPU] Computing Harmonic Heat-Diffusion DQS weights (with rigid cranium)...", flush=True)
    skin_bones, W = compute_harmonic_skin_weights(body_xyz, bones, np.ones(n_body, dtype=bool))

    # Exact chin DQS anchor weights for mouth lockstep tracking
    m_ctr = np.asarray(mouth_meta["center"], dtype=np.float32)
    m_dists = np.linalg.norm(body_xyz - m_ctr[None, :], axis=1)
    m_nn = np.argpartition(m_dists, min(32, n_body - 1))[:32]
    w_mouth_anchor = W[m_nn].mean(axis=0, keepdims=True).astype(np.float32)
    w_mouth_anchor /= np.maximum(w_mouth_anchor.sum(axis=1, keepdims=True), 1e-8)

    # Foot subset (~3% lowest splats) for fast ground-snap dz_g
    op_body = body_op > 0.25
    z_foot_cut = float(np.percentile(body_xyz[op_body, 2], 3.0))
    foot_idx = np.where(op_body & (body_xyz[:, 2] <= z_foot_cut))[0]
    foot_xyz, foot_rot, foot_W = body_xyz[foot_idx], body_rot[foot_idx], W[foot_idx]
    foot_mask = np.ones(len(foot_idx), dtype=bool)
    rest_foot_z = float(np.percentile(body_xyz[op_body, 2], 0.2))

    # Auto-detect T-pose vs A-pose from upper_arm_L
    u_upper_L = np.asarray(bones["upper_arm_L"]["tail"], np.float32) - np.asarray(bones["upper_arm_L"]["head"], np.float32)
    u_upper_L /= max(float(np.linalg.norm(u_upper_L)), 1e-8)
    arm_drop_rest_deg = float(math.degrees(math.atan2(-float(u_upper_L[2]), max(float(u_upper_L[0]), 1e-4))))
    print(f"[StandardQA-GPU] Detected rest arm drop angle: {arm_drop_rest_deg:.1f} deg ({'T-pose' if arm_drop_rest_deg < 28 else 'A-pose'})", flush=True)

    u_fore_L = np.asarray(bones["forearm_twist_L"]["tail"], np.float32) - np.asarray(bones["forearm_L"]["head"], np.float32)
    u_fore_L /= max(float(np.linalg.norm(u_fore_L)), 1e-8)
    e_elbow_L = -np.cross(np.array([0.0, -1.0, 0.0], np.float32), u_fore_L)
    e_elbow_L /= max(float(np.linalg.norm(e_elbow_L)), 1e-8)

    u_fore_R = np.asarray(bones["forearm_twist_R"]["tail"], np.float32) - np.asarray(bones["forearm_R"]["head"], np.float32)
    u_fore_R /= max(float(np.linalg.norm(u_fore_R)), 1e-8)
    e_elbow_R = -np.cross(np.array([0.0, -1.0, 0.0], np.float32), u_fore_R)
    e_elbow_R /= max(float(np.linalg.norm(e_elbow_R)), 1e-8)

    # Eyelid parameters (relative to D_eye & skin_lum)
    lum = 0.299 * body_rgb[:, 0] + 0.587 * body_rgb[:, 1] + 0.114 * body_rgb[:, 2]
    skin_rgb = np.asarray(mouth_meta["skin_rgb"], np.float32)
    skin_lum = float(0.299 * skin_rgb[0] + 0.587 * skin_rgb[1] + 0.114 * skin_rgb[2])
    not_eye_L = np.hypot(body_xyz[:, 0] - surf_L[0], body_xyz[:, 2] - surf_L[2]) > 0.35 * D_eye
    not_eye_R = np.hypot(body_xyz[:, 0] - surf_R[0], body_xyz[:, 2] - surf_R[2]) > 0.35 * D_eye
    skin_mask = (
        not_eye_L & not_eye_R & (body_op > 0.30)
        & (lum >= skin_lum * 0.80) & (lum <= skin_lum * 1.15)
        & (np.abs(body_xyz[:, 0] - face_cx) < 1.50 * D_eye)
        & (body_xyz[:, 2] > z_eye - 1.00 * D_eye) & (body_xyz[:, 2] < z_eye + 0.50 * D_eye)
        & (body_xyz[:, 1] < sy + 0.75 * D_eye)
    )
    skin_xyz = body_xyz[skin_mask]
    skin_fdc = body_fdc[skin_mask]
    med_skin_fdc = np.median(skin_fdc, axis=0, keepdims=True).astype(np.float32)
    skin_fdc = (0.35 * skin_fdc + 0.65 * med_skin_fdc).astype(np.float32)
    skin_scale_med = float(np.median(head_pol_mouth["scale_0"][skin_mask]))
    rx_e, rz_e = max(eyeball_r * 1.48, 0.42 * D_eye), max(eyeball_r * 1.28, 0.36 * D_eye)
    dz_cap_L = detect_lash_ceiling_dz(body_xyz, body_rgb, np.ones(n_body, dtype=bool), surf_L, rx_e, rz_e)
    dz_cap_R = detect_lash_ceiling_dz(body_xyz, body_rgb, np.ones(n_body, dtype=bool), surf_R, rx_e, rz_e)
    lash_top_bot = (rz_e * 0.45, -rz_e * 0.58)

    cam_pos_face = np.array([face_cx, -0.76, z_eye], dtype=np.float32)
    drop_delta = max(0.0, 62.0 - arm_drop_rest_deg)
    h_hd_rest = np.asarray(bones["head"]["head"], np.float32)

    # Write GPU base body PLY ONCE + skin weights ONCE + resting eyes/mouth ONCE
    anim_tag = f"qa_suite_{asset_id}_anim"
    anim_dir = spark / "frames" / anim_tag
    anim_dir.mkdir(parents=True, exist_ok=True)
    write_ply(anim_dir / "body_base.ply", to_spark_ply(head_pol_mouth, body_xyz, body_rot))
    (anim_dir / "body_skin.bin").write_bytes(pack_spark_skin_weights(W).tobytes())

    rest_eyes_mouth = np.concatenate([eL, eR, mouth_rest_closed], axis=0)
    rem_xyz = np.column_stack([rest_eyes_mouth["x"], rest_eyes_mouth["y"], rest_eyes_mouth["z"]]).astype(np.float32)
    rem_rot = np.column_stack([rest_eyes_mouth["rot_0"], rest_eyes_mouth["rot_1"], rest_eyes_mouth["rot_2"], rest_eyes_mouth["rot_3"]]).astype(np.float32)
    rem_rot /= np.maximum(np.linalg.norm(rem_rot, axis=1, keepdims=True), 1e-8)
    write_ply(anim_dir / "rest_eyes_mouth.ply", to_spark_ply(rest_eyes_mouth, rem_xyz, rem_rot))

    cam_face = {"width": 640, "height": 640, "background": "#262a36", "target": [0.0, -float(z_eye - 0.015), 0.0], "pitch": 0, "fov": 24, "distance": 0.78, "front": 0, "side": 35}
    cam_sh = {"width": 640, "height": 640, "background": "#262a36", "target": [0.0, -0.74, 0.0], "pitch": 3, "fov": 25, "distance": 1.38, "front": 30, "side": 90}
    cam_loco = {"width": 640, "height": 640, "background": "#262a36", "target": [0.0, -0.56, 0.0], "pitch": 4, "fov": 28, "distance": 2.75, "front": 35, "side": 90}

    anim_frames = []
    t_prep0 = time.perf_counter()

    # --- PART 1: FACE, EYES & MOUTH (66 frames @ 12fps = 5.5s) ---
    vowel_seq = ["closed", "A", "I", "U", "E", "O", "smile", "closed"]
    for fi in range(66):
        head_yaw, bL, bR, locked = 0.0, 0.0, 0.0, False
        m_open, m_width, m_smile, m_round = MOUTH_PRESETS["closed"]
        target = cam_pos_face.copy()
        if fi < 9:
            b = smooth01(fi / 2.5) if fi <= 2 else (1.0 if fi <= 5 else 1.0 - smooth01((fi - 5) / 3.0))
            bL = bR = b
        elif fi < 21:
            u = (fi - 9) / 11.0
            if u < 0.25:
                target[0] = face_cx + 0.36 * math.sin((u / 0.25) * math.pi)
            elif u < 0.50:
                target[0] = face_cx - 0.36 * math.sin(((u - 0.25) / 0.25) * math.pi)
            elif u < 0.75:
                target[2] = z_eye + 0.24 * math.sin(((u - 0.50) / 0.25) * math.pi)
            else:
                target[2] = z_eye - 0.20 * math.sin(((u - 0.75) / 0.25) * math.pi)
        elif fi < 29:
            head_yaw = 18.0 * math.sin(((fi - 21) / 7.0) * 2.0 * math.pi)
            locked = True
        elif fi < 36:
            k = fi - 29
            bL = smooth01(k / 2.0) if k <= 2 else (1.0 if k <= 4 else 1.0 - smooth01((k - 4) / 2.0))
        else:
            k = fi - 36
            if k < 10:
                a = math.sin((k / 9.0) * math.pi)
                p_c, p_a = MOUTH_PRESETS["closed"], MOUTH_PRESETS["A"]
                m_open = p_c[0] * (1.0 - a) + p_a[0] * a
                m_width = p_c[1] * (1.0 - a) + p_a[1] * a
                m_smile = p_c[2] * (1.0 - a) + p_a[2] * a
                m_round = p_c[3] * (1.0 - a) + p_a[3] * a
            else:
                u_v = (k - 10) / 19.0 * (len(vowel_seq) - 1)
                idx0 = min(int(u_v), len(vowel_seq) - 2)
                frac = smooth01(u_v - idx0)
                p0, p1 = MOUTH_PRESETS[vowel_seq[idx0]], MOUTH_PRESETS[vowel_seq[idx0 + 1]]
                m_open = p0[0] * (1.0 - frac) + p1[0] * frac
                m_width = p0[1] * (1.0 - frac) + p1[1] * frac
                m_smile = p0[2] * (1.0 - frac) + p1[2] * frac
                m_round = p0[3] * (1.0 - frac) + p1[3] * frac

        world = compute_fk_transforms(bones, order, {"neck": euler_xyz(0, 0, 0.35 * head_yaw), "head": euler_xyz(0, 0, 0.65 * head_yaw)})
        R_hd, h_hd, _ = world["head"]

        dyn_parts, dyn_xyz_list, dyn_rot_list = [], [], []
        for bv, e_raw, e_xyz, e_rot, c_e, sc, dz_cap in (
            (bL, eL, eL_xyz0, eL_rot0, pL, surf_L, dz_cap_L),
            (bR, eR, eR_xyz0, eR_rot0, pR, surf_R, dz_cap_R),
        ):
            c_w = h_hd + R_hd @ (c_e - h_hd_rest)
            if locked:
                R_tot = look_at_eye_rot(c_w, cam_pos_face)
            else:
                R_loc = look_at_eye_rot(c_e, target)
                R_tot = (R_hd @ R_loc).astype(np.float32)
            xyz_w = c_w[None, :] + (e_xyz - c_e[None, :]) @ R_tot.T
            rot_w = quat_mul_wxyz(np.broadcast_to(rotmat_to_quat_wxyz(R_tot), e_rot.shape), e_rot)
            e_fd = e_raw.copy()
            if bv > 0.15:
                zc = float(sc[2]) + (lash_top_bot[0] * (1.0 - bv) + lash_top_bot[1] * bv)
                occ = np.ones(len(e_fd), np.float32) if bv >= 0.85 else np.clip((xyz_w[:, 2] - (zc - 0.002)) / 0.004, 0.0, 1.0)
                e_fd["opacity"] = e_fd["opacity"] * (1.0 - occ) - 14.0 * occ
            dyn_parts.append(e_fd); dyn_xyz_list.append(xyz_w); dyn_rot_list.append(rot_w)

            if bv > 0.02:
                lid = build_eyelid_shell(c_e, sc, skin_xyz, skin_fdc, skin_scale_med, bv, head_pol.dtype, rx_rz=(rx_e, rz_e), lash_top_bot=lash_top_bot, dz_top_cap=dz_cap)
                l_xyz = np.column_stack([lid["x"], lid["y"], lid["z"]]).astype(np.float32)
                l_rot = np.column_stack([lid["rot_0"], lid["rot_1"], lid["rot_2"], lid["rot_3"]]).astype(np.float32)
                l_xyz_w = h_hd[None, :] + (l_xyz - h_hd_rest[None, :]) @ R_hd.T
                l_rot_w = quat_mul_wxyz(np.broadcast_to(rotmat_to_quat_wxyz(R_hd), l_rot.shape), l_rot)
                dyn_parts.append(lid); dyn_xyz_list.append(l_xyz_w); dyn_rot_list.append(l_rot_w)

        m_sp = build_procedural_mouth(mouth_meta, m_open, head_pol.dtype, width_scale=m_width, smile=m_smile, roundness=m_round)
        m_xyz = np.column_stack([m_sp["x"], m_sp["y"], m_sp["z"]]).astype(np.float32)
        m_rot = np.column_stack([m_sp["rot_0"], m_sp["rot_1"], m_sp["rot_2"], m_sp["rot_3"]]).astype(np.float32)
        m_rot /= np.maximum(np.linalg.norm(m_rot, axis=1, keepdims=True), 1e-8)
        m_W = np.broadcast_to(w_mouth_anchor, (len(m_xyz), W.shape[1]))
        m_xyz_w, m_rot_w = deform_dqs(m_xyz, m_rot, skin_bones, m_W, world, np.ones(len(m_xyz), dtype=bool))
        dyn_parts.append(m_sp); dyn_xyz_list.append(m_xyz_w); dyn_rot_list.append(m_rot_w)

        dyn_sp = to_spark_ply(np.concatenate(dyn_parts, axis=0), np.concatenate(dyn_xyz_list, axis=0), np.concatenate(dyn_rot_list, axis=0))
        rel_dyn = f"frames/{anim_tag}/face_dyn-{fi:04d}.ply"
        write_ply(spark / rel_dyn, dyn_sp)
        anim_frames.append({
            "png": f"face-{fi:04d}.png",
            "camera": cam_face,
            "bones": fk_to_spark_bones(bones, skin_bones, world),
            "extraPly": [rel_dyn],
        })

    # --- PART 2: SHOULDER & ARM ROM (36 frames @ 12fps = 3.0s, ZERO per-frame PLY writes!) ---
    for fi in range(36):
        if fi < 9:
            a = math.sin((fi / 8.0) * math.pi)
            lr = {
                "clavicle_L": euler_xyz(0, 0.18 * drop_delta * a, 0),
                "upper_arm_L": euler_xyz(0, 0.82 * drop_delta * a, 0),
                "forearm_L": axis_angle(e_elbow_L, 14.0 * a),
                "clavicle_R": euler_xyz(0, -0.18 * drop_delta * a, 0),
                "upper_arm_R": euler_xyz(0, -0.82 * drop_delta * a, 0),
            }
        elif fi < 18:
            a = math.sin(((fi - 9) / 8.0) * math.pi)
            lift_needed = 46.0 + max(0.0, arm_drop_rest_deg - 10.0) * 0.5
            lr = {
                "clavicle_L": euler_xyz(0, -14.0 * a, 0),
                "upper_arm_L": euler_xyz(0, -(lift_needed - 14.0) * a, 0) @ axis_angle(u_upper_L, -30.0 * a),
                "forearm_L": axis_angle(e_elbow_L, 10.0 * a),
                "clavicle_R": euler_xyz(0, -0.16 * drop_delta, 0),
                "upper_arm_R": euler_xyz(0, -0.80 * drop_delta, 0),
            }
        elif fi < 27:
            u = (fi - 18) / 8.0
            w = math.sin(u * math.pi)
            sway = math.sin(u * 2.0 * math.pi)
            lr = {
                "clavicle_L": euler_xyz(0, -6.0 * w, -4.0 * w),
                "upper_arm_L": euler_xyz(0, (4.0 - arm_drop_rest_deg * 0.6) * w, -14.0 * w) @ axis_angle(u_upper_L, -82.0 * w),
                "forearm_L": axis_angle(e_elbow_L, 86.0 * w + 14.0 * sway),
                "hand_L": axis_angle(e_elbow_L, 12.0 * sway),
                "clavicle_R": euler_xyz(0, -0.16 * drop_delta, 0),
                "upper_arm_R": euler_xyz(0, -0.80 * drop_delta, 0),
            }
        else:
            u = (fi - 27) / 8.0
            enter = smooth01(min(u / 0.2, 1.0)) * smooth01(min((1.0 - u) / 0.2, 1.0))
            rot_deg = 52.0 * math.sin(((u - 0.1) / 0.8) * 2.0 * math.pi) if 0.1 <= u <= 0.9 else 0.0
            lr = {
                "clavicle_L": euler_xyz(0, -4.0 * enter, 0),
                "upper_arm_L": euler_xyz(0, 6.0 * enter, -10.0 * enter) @ axis_angle(u_upper_L, (-10.0 + rot_deg) * enter),
                "forearm_L": axis_angle(e_elbow_L, 86.0 * enter),
                "clavicle_R": euler_xyz(0, -0.16 * drop_delta, 0),
                "upper_arm_R": euler_xyz(0, -0.80 * drop_delta, 0),
            }
        world = compute_fk_transforms(bones, order, lr)
        R_hd, h_hd, _ = world["head"]
        t_hd = h_hd - (R_hd @ h_hd_rest)
        anim_frames.append({
            "png": f"sh-{fi:04d}.png",
            "camera": cam_sh,
            "bones": fk_to_spark_bones(bones, skin_bones, world),
            "extraPly": [f"frames/{anim_tag}/rest_eyes_mouth.ply"],
            "extraTransforms": [blender_rigid_to_spark_qp(R_hd, t_hd)],
        })

    # --- PART 3: FULL-BODY LOCOMOTION (48 frames @ 12fps = 4.0s, ZERO per-frame PLY writes!) ---
    for fi in range(48):
        air = 0.0
        arm_hang = max(18.0, 66.0 - arm_drop_rest_deg)
        if fi < 16:
            ph = (fi / 8.0) * 2.0 * math.pi
            s, c = math.sin(ph), math.cos(ph)
            lr = {
                "spine_lower": euler_xyz(8.0 + abs(s), -2.5 * s, -3.0 * s),
                "chest": euler_xyz(3.0, 2.5 * s, 6.0 * s),
                "head": euler_xyz(-5.0, 0, -3.0 * s),
                "thigh_L": euler_xyz(-24.0 * s, 0, 0), "shin_L": euler_xyz(6.0 + 36.0 * max(0.0, -c), 0, 0), "foot_L": euler_xyz(-8.0 * s, 0, 0),
                "thigh_R": euler_xyz(24.0 * s, 0, 0), "shin_R": euler_xyz(6.0 + 36.0 * max(0.0, c), 0, 0), "foot_R": euler_xyz(8.0 * s, 0, 0),
                "clavicle_L": euler_xyz(0, 8.0, 0), "upper_arm_L": euler_xyz(18.0 * s, arm_hang - 8.0, 0), "forearm_L": axis_angle(e_elbow_L, 22.0 + 16.0 * max(0.0, -s)),
                "clavicle_R": euler_xyz(0, -8.0, 0), "upper_arm_R": euler_xyz(-18.0 * s, -(arm_hang - 8.0), 0), "forearm_R": axis_angle(e_elbow_R, 22.0 + 16.0 * max(0.0, s)),
            }
        elif fi < 32:
            ph = ((fi - 16) / 8.0) * 2.0 * math.pi
            s, c = math.sin(ph), math.cos(ph)
            air = 0.030 * max(0.0, abs(s) - 0.55) / 0.45
            lr = {
                "spine_lower": euler_xyz(12.5 + 1.5 * abs(s), -3.0 * s, -5.0 * s),
                "chest": euler_xyz(3.5, 3.0 * s, 10.0 * s),
                "head": euler_xyz(-10.0, 0, -5.0 * s),
                "thigh_L": euler_xyz(-38.0 * s - 6.0, 0, 0), "shin_L": euler_xyz(12.0 + 60.0 * max(0.0, -c), 0, 0), "foot_L": euler_xyz(-10.0 * s, 0, 0),
                "thigh_R": euler_xyz(38.0 * s - 6.0, 0, 0), "shin_R": euler_xyz(12.0 + 60.0 * max(0.0, c), 0, 0), "foot_R": euler_xyz(10.0 * s, 0, 0),
                "clavicle_L": euler_xyz(0, 8.0, 0), "upper_arm_L": euler_xyz(34.0 * s - 6.0, arm_hang - 6.0, -6.0), "forearm_L": axis_angle(e_elbow_L, 68.0 + 14.0 * max(0.0, -s)),
                "clavicle_R": euler_xyz(0, -8.0, 0), "upper_arm_R": euler_xyz(-34.0 * s - 6.0, -(arm_hang - 6.0), 6.0), "forearm_R": axis_angle(e_elbow_R, 68.0 + 14.0 * max(0.0, s)),
            }
        else:
            k = fi - 32
            if k <= 4:
                t = smooth01(k / 4.0)
                th, kn, an, sp, ax, ay, el = -36.0 * t, 64.0 * t, -24.0 * t, 15.0 * t, 28.0 * t, arm_hang - 10.0 * t, 32.0 * t
            elif k <= 11:
                u = (k - 4) / 7.0
                arc = math.sin(u * math.pi)
                air = 0.20 * arc
                th, kn, an, sp, ax, ay, el = -30.0 * arc, 54.0 * arc, 20.0 * arc, 6.0 * arc, -42.0 * arc, arm_hang * (1.0 - arc) - 18.0 * arc, 38.0 * arc
            else:
                u = (k - 11) / 4.0
                cush = math.sin(u * math.pi)
                th, kn, an, sp, ax, ay, el = -28.0 * cush, 50.0 * cush, -18.0 * cush, 11.0 * cush, 12.0 * cush, arm_hang - 8.0 * cush, 24.0 * cush
            lr = {
                "spine_lower": euler_xyz(sp, 0, 0), "chest": euler_xyz(sp * 0.25, 0, 0), "head": euler_xyz(-sp * 0.65, 0, 0),
                "thigh_L": euler_xyz(th, 0, 0), "shin_L": euler_xyz(kn, 0, 0), "foot_L": euler_xyz(an, 0, 0),
                "thigh_R": euler_xyz(th, 0, 0), "shin_R": euler_xyz(kn, 0, 0), "foot_R": euler_xyz(an, 0, 0),
                "clavicle_L": euler_xyz(0, 0.15 * ay, 0), "upper_arm_L": euler_xyz(ax, 0.85 * ay, 0), "forearm_L": axis_angle(e_elbow_L, el),
                "clavicle_R": euler_xyz(0, -0.15 * ay, 0), "upper_arm_R": euler_xyz(ax, -0.85 * ay, 0), "forearm_R": axis_angle(e_elbow_R, el),
            }
        w0 = compute_fk_transforms(bones, order, lr)
        foot_def, _ = deform_dqs(foot_xyz, foot_rot, skin_bones, foot_W, w0, foot_mask)
        dz_g = (rest_foot_z - float(np.percentile(foot_def[:, 2], 6.0))) + air
        R_hd, h_hd, _ = w0["head"]
        t_hd = (h_hd + np.array([0.0, 0.0, dz_g], np.float32)) - (R_hd @ h_hd_rest)
        anim_frames.append({
            "png": f"loco-{fi:04d}.png",
            "camera": cam_loco,
            "bones": fk_to_spark_bones(bones, skin_bones, w0, dz_offset=dz_g),
            "extraPly": [f"frames/{anim_tag}/rest_eyes_mouth.ply"],
            "extraTransforms": [blender_rigid_to_spark_qp(R_hd, t_hd)],
        })

    anim_manifest = {
        "basePly": f"frames/{anim_tag}/body_base.ply",
        "skinBin": f"frames/{anim_tag}/body_skin.bin",
        "numBones": len(skin_bones),
        "defaultCamera": cam_face,
        "frames": anim_frames,
    }
    (anim_dir / "anim.json").write_text(json.dumps(anim_manifest), encoding="utf-8")
    print(f"[StandardQA-GPU] Prepared {len(anim_frames)}-frame GPU anim manifest in {time.perf_counter() - t_prep0:.2f}s.", flush=True)

    render_dest = f"renders_{anim_tag}"
    if not args.skip_render:
        t_r0 = time.perf_counter()
        subprocess.run(
            ["node", "capture.mjs", "anim", render_dest, f"frames/{anim_tag}/anim.json"],
            cwd=str(spark),
            check=True,
        )
        t_r = time.perf_counter() - t_r0
        print(f"[StandardQA-GPU] Spark GPU rendered {len(anim_frames)}f in {t_r:.2f}s ({len(anim_frames)/max(t_r,1e-6):.1f} fps)", flush=True)

    rdir = spark / render_dest
    font = get_font(20)
    all_frames = []
    for i in range(66):
        im = Image.open(rdir / f"face-{i:04d}.png")
        lbl = (
            "1A. BLINK" if i < 9 else
            "1B. EYES ONLY (Head Locked)" if i < 21 else
            "1C. HEAD TURN (Eyes Locked)" if i < 29 else
            "1D. WINK L" if i < 36 else
            "1E. MOUTH OPEN/CLOSE" if i < 46 else
            "1F. LIP-SYNC (A-I-U-E-O)"
        )
        all_frames.append(add_banner(im, f"{lbl}  [{i:02d}/65]", font))
    for i in range(36):
        im = Image.open(rdir / f"sh-{i:04d}.png")
        lbl = "2A. ARMS DROP" if i < 9 else ("2B. HIGH ABDUCTION (~135)" if i < 18 else ("2C. PALM-FWD WAVE" if i < 27 else "2D. AXIAL ROTATION"))
        all_frames.append(add_banner(im, f"{lbl}  [{i:02d}/35]", font))
    for i in range(48):
        im = Image.open(rdir / f"loco-{i:04d}.png")
        lbl = "3A. WALK" if i < 16 else ("3B. RUN" if i < 32 else "3C. JUMP")
        all_frames.append(add_banner(im, f"{lbl}  [{i:02d}/47]", font))

    mp4_out = exports / f"{asset_id}_standard_qa_suite.mp4"
    iio.imwrite(mp4_out, np.stack([np.asarray(fr) for fr in all_frames], axis=0), fps=12, codec="libx264")
    print(f"Saved All-in-One GPU QA Suite Video -> {mp4_out} ({len(all_frames)}f @ 12fps = {len(all_frames)/12:.1f}s | total {time.perf_counter()-t0_all:.2f}s)", flush=True)

    kf = [
        (rdir / "face-0004.png", "1A.BLINK"),
        (rdir / "face-0010.png", "1B.EYES LEFT"),
        (rdir / "face-0023.png", "1C.HEAD YAW"),
        (rdir / "face-0040.png", "1E.MOUTH WIDE A"),
        (rdir / "face-0052.png", "1F.LIP-SYNC I"),
        (rdir / "face-0058.png", "1F.LIP-SYNC O"),
        (rdir / "sh-0004.png", "2A.ARMS DROP"),
        (rdir / "sh-0022.png", "2C.WAVE PALM-FWD"),
        (rdir / "loco-0002.png", "3A.WALK"),
        (rdir / "loco-0018.png", "3B.RUN"),
        (rdir / "loco-0035.png", "3C.JUMP CROUCH"),
        (rdir / "loco-0039.png", "3C.JUMP APEX"),
    ]
    sheet = Image.new("RGB", (640 * 4, 640 * 3), (24, 26, 32))
    font_s = get_font(18)
    for idx, (p, lbl) in enumerate(kf):
        r, c = divmod(idx, 4)
        sheet.paste(add_banner(Image.open(p), lbl, font_s), (c * 640, r * 640))
    sheet_out = qa_dir / "qa_standard_suite_sheet.png"
    sheet.save(sheet_out)
    print(f"Saved All-in-One QA Contact Sheet -> {sheet_out}", flush=True)


if __name__ == "__main__":
    main()
