"""Render a rigged gs-modeling v1/ character walking along a 3D multi-layer walk_3d_trajectory.json
inside an assembled 3DGS stage PLY via Spark GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

SCRIPTS_DIR = Path(__file__).resolve().parent
MODELING_SCRIPTS = SCRIPTS_DIR.parent.parent / "gs-modeling" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
if MODELING_SCRIPTS.exists():
    sys.path.insert(0, str(MODELING_SCRIPTS))

from ply_io import read_ply, write_ply
from harmonic_dqs import (
    add_forearm_twist_bones, build_fk_hierarchy, compute_fk_transforms,
    euler_xyz, deform_dqs, compute_harmonic_skin_weights, quat_mul_wxyz,
)

SH_C0 = 0.28209479177387814


def blender_to_spark_data(data: np.ndarray) -> np.ndarray:
    out = data.copy()
    x, y, z = data["x"].copy(), data["y"].copy(), data["z"].copy()
    out["x"], out["y"], out["z"] = x, -z, y
    s = math.sqrt(0.5)
    w0, x0, y0, z0 = data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]
    out["rot_0"] = s * (w0 + x0)
    out["rot_1"] = s * (x0 - w0)
    out["rot_2"] = s * (y0 + z0)
    out["rot_3"] = s * (z0 - y0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a rigged v1/ character walking a 3D stage trajectory via Spark.")
    ap.add_argument("--map-ply", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--char-v1", type=Path, required=True, help="Path to character v1/ dir (master_open.ply + rig_skeleton.json).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--spark-dir", type=Path, required=True)
    ap.add_argument("--char-height-m", type=float, default=1.40)
    ap.add_argument("--bg-hex", default="#2b3836")
    ap.add_argument("--cam-front", type=float, default=-20.0)
    ap.add_argument("--cam-pitch", type=float, default=19.0)
    ap.add_argument("--cam-dist", type=float, default=12.2)
    ap.add_argument("--duration-ms", type=int, default=90)
    ap.add_argument("--look-around", action="store_true", help="Add curious head/neck/chest look-around motion while walking.")
    ap.add_argument("--hand-prop", type=Path, default=None, help="Optional handheld v1_prop/prop.ply (e.g. lantern cage) attached to hand_R.")
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj = json.loads(args.trajectory.read_text(encoding="utf-8"))

    rig = json.loads((args.char_v1 / "rig_skeleton.json").read_text(encoding="utf-8"))
    bones = add_forearm_twist_bones(rig["bones"])
    order = build_fk_hierarchy(bones)
    hb, _ = read_ply(args.char_v1 / "master_open.ply")
    xyz0 = np.column_stack([hb["x"], hb["y"], hb["z"]]).astype(np.float32)
    rot0 = np.column_stack([hb["rot_0"], hb["rot_1"], hb["rot_2"], hb["rot_3"]]).astype(np.float32)
    rot0 /= np.maximum(np.linalg.norm(rot0, axis=1, keepdims=True), 1e-8)
    op_b = 1.0 / (1.0 + np.exp(-np.clip(hb["opacity"], -20.0, 20.0)))
    skin_bones, weights = compute_harmonic_skin_weights(xyz0, bones, np.ones(len(xyz0), dtype=bool))

    tag = f"walk_{args.out_dir.name}"
    tmp_dir = args.spark_dir / "frames" / tag
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    map_d, _ = read_ply(args.map_ply)
    map_rel = f"frames/{tag}/map_static.ply"
    write_ply(args.spark_dir / map_rel, blender_to_spark_data(map_d))

    cage_xyz0 = None
    cage_base = None
    if args.hand_prop and args.hand_prop.exists():
        cage_raw, _ = read_ply(args.hand_prop)
        c_h = max(float(np.ptp(cage_raw["z"])), 1e-4)
        c_sc = 0.215 / c_h
        cage_xyz0 = np.column_stack([cage_raw["x"], cage_raw["y"], cage_raw["z"]]).astype(np.float32) * c_sc
        cage_base = np.zeros(len(cage_raw), dtype=hb.dtype)
        for k in hb.dtype.names:
            if k in cage_raw.dtype.names:
                cage_base[k] = cage_raw[k]
        for si in range(3):
            cage_base[f"scale_{si}"] = cage_raw[f"scale_{si}"] + math.log(c_sc)

    ls_char = math.log(args.char_height_m)
    seq = []
    for fi, t in enumerate(traj):
        u_t = fi / max(len(traj) - 1, 1)
        ph = float(t["gait_phase"])
        sw = math.sin(ph) * float(t["stride_mul"])
        lift_L = max(0.0, math.cos(ph)) * float(t["knee_lift_mul"])
        lift_R = max(0.0, -math.cos(ph)) * float(t["knee_lift_mul"])
        tpitch = float(t["torso_pitch_deg"])
        bob = 0.018 * abs(math.sin(ph))

        look_yaw = (28.0 * math.sin(u_t * 4.5 * math.pi + 0.4)) if args.look_around else 0.0
        look_pitch = (7.0 * math.sin(u_t * 3.0 * math.pi - 0.5) + 3.0) if args.look_around else 0.0
        has_prop = cage_base is not None

        pose = {
            "spine_lower": euler_xyz(tpitch * 0.55, 0.0, 2.5 * math.sin(ph) + look_yaw * 0.10),
            "chest": euler_xyz(tpitch * 0.45, 0.0, -3.5 * math.sin(ph) + look_yaw * 0.16),
            "neck": euler_xyz(-tpitch * 0.4 + look_pitch * 0.35, 0.0, look_yaw * 0.32),
            "head": euler_xyz(-tpitch * 0.3 + look_pitch * 0.65, 2.0 * math.sin(u_t * 4.5 * math.pi) if args.look_around else 0.0, look_yaw * 0.68 if args.look_around else 4.0 * math.sin(fi * 0.14)),
            "clavicle_R": euler_xyz(0.0, 6.0 if has_prop else 4.0, 10.0 if has_prop else 4.0),
            "upper_arm_R": euler_xyz(14.0 - 2.0 * math.sin(ph), 24.0, 74.0) if has_prop else euler_xyz(16.0 * sw, 54.0, 8.0),
            "forearm_R": euler_xyz(10.0, 12.0, 16.0) if has_prop else euler_xyz(10.0 + 6.0 * max(0.0, sw), 10.0, 6.0),
            "hand_R": euler_xyz(-6.0, -8.0, -4.0) if has_prop else euler_xyz(0.0, 0.0, 0.0),
            "clavicle_L": euler_xyz(0.0, 4.0, -4.0),
            "upper_arm_L": euler_xyz(-16.0 * sw, 54.0, -8.0),
            "forearm_L": euler_xyz(10.0 + 6.0 * max(0.0, -sw), 10.0, -6.0),
            "thigh_L": euler_xyz(22.0 * sw + 10.0 * lift_L, 4.0, 0.0),
            "shin_L": euler_xyz(-26.0 * lift_L - 6.0 * max(0.0, -sw), 0.0, 0.0),
            "foot_L": euler_xyz(8.0 * lift_L, 0.0, 0.0),
            "thigh_R": euler_xyz(-22.0 * sw + 10.0 * lift_R, -4.0, 0.0),
            "shin_R": euler_xyz(-26.0 * lift_R - 6.0 * max(0.0, sw), 0.0, 0.0),
            "foot_R": euler_xyz(8.0 * lift_R, 0.0, 0.0),
        }
        fk = compute_fk_transforms(bones, order, pose)
        xyz_p, rot_p = deform_dqs(xyz0, rot0, skin_bones, weights, fk, np.ones(len(xyz0), dtype=bool))
        z_floor = float(np.percentile(xyz_p[op_b > 0.3, 2], 0.5))
        xyz_p[:, 2] -= (z_floor - bob)

        ch_parts = []
        body_f = hb.copy()
        body_f["x"], body_f["y"], body_f["z"] = xyz_p[:, 0], xyz_p[:, 1], xyz_p[:, 2]
        body_f["rot_0"], body_f["rot_1"], body_f["rot_2"], body_f["rot_3"] = rot_p[:, 0], rot_p[:, 1], rot_p[:, 2], rot_p[:, 3]
        ch_parts.append(body_f)

        if has_prop:
            R_h, _, t_h = fk["hand_R"]
            t_h = t_h.copy(); t_h[2] -= (z_floor - bob)
            hand_tip = (np.array(bones["hand_R"]["tail"], dtype=np.float32) @ R_h.T) + t_h
            piv_loc = np.array([0.0, 0.0, 0.215 * 0.90], dtype=np.float32)
            cf = cage_base.copy()
            c_xyz_l = (cage_xyz0 - piv_loc[None, :]) + hand_tip[None, :]
            cf["x"], cf["y"], cf["z"] = c_xyz_l[:, 0], c_xyz_l[:, 1], c_xyz_l[:, 2]
            ch_parts.append(cf)

        ch = np.concatenate(ch_parts, axis=0) if len(ch_parts) > 1 else ch_parts[0]
        yaw_rad = math.radians(float(t["yaw_deg"]))
        R_w = Rotation.from_euler("z", yaw_rad).as_matrix().astype(np.float32)
        q_xyzw = Rotation.from_euler("z", yaw_rad).as_quat()
        q_w = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
        w_pos = np.array([float(t["x"]), float(t["y"]), float(t["z"])], dtype=np.float32)

        c_pos = (np.column_stack([ch["x"], ch["y"], ch["z"]]).astype(np.float32) * args.char_height_m) @ R_w.T + w_pos[None, :]
        c_rot = quat_mul_wxyz(q_w, np.column_stack([ch["rot_0"], ch["rot_1"], ch["rot_2"], ch["rot_3"]]).astype(np.float32))
        c_rot /= np.maximum(np.linalg.norm(c_rot, axis=1, keepdims=True), 1e-8)

        ch["x"], ch["y"], ch["z"] = c_pos[:, 0], c_pos[:, 1], c_pos[:, 2]
        ch["rot_0"], ch["rot_1"], ch["rot_2"], ch["rot_3"] = c_rot[:, 0], c_rot[:, 1], c_rot[:, 2], c_rot[:, 3]
        for si in range(3):
            ch[f"scale_{si}"] = ch[f"scale_{si}"] + ls_char

        rel = f"frames/{tag}/ch_{fi:03d}.ply"
        write_ply(args.spark_dir / rel, blender_to_spark_data(ch))
        seq.append({"blenderFrame": fi, "ply": [rel], "png": f"f_{fi:03d}.png"})

    W, H = 640, 420
    cam = {"width": W, "height": H, "background": args.bg_hex, "target": [0.0, -1.15, 0.25], "distance": args.cam_dist, "pitch": args.cam_pitch, "fov": 35.0, "front": args.cam_front, "side": args.cam_front + 90.0}
    crel = f"frames/{tag}/cam.json"
    mrel = f"frames/{tag}/man.json"
    (args.spark_dir / crel).write_text(json.dumps(cam), encoding="utf-8")
    (args.spark_dir / mrel).write_text(json.dumps({"static": [map_rel], "review": [], "frames": seq}), encoding="utf-8")
    subprocess.run(["node", "capture.mjs", tag, f"{tag}_out", crel, mrel], cwd=str(args.spark_dir), check=True, capture_output=True)

    frs = [Image.open(args.spark_dir / f"{tag}_out" / f"f_{fi:03d}.png").convert("RGB") for fi in range(len(traj))]
    strip = Image.new("RGB", (W * 4, H))
    for k, idx in enumerate(np.linspace(0, len(frs) - 1, 4).round().astype(int)):
        strip.paste(frs[idx], (k * W, 0))
    pal = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    qfrs = [f.resize((540, 354), Image.Resampling.LANCZOS).quantize(palette=pal, dither=Image.Dither.NONE) for f in frs]
    out_gif = args.out_dir / "stage_walkthrough.gif"
    qfrs[0].save(out_gif, save_all=True, append_images=qfrs[1:], duration=args.duration_ms, loop=0, optimize=True)
    print(f"Saved 3DGS stage walkthrough GIF ({out_gif.stat().st_size/1024:.1f} KB) in {time.time()-t0:.1f}s -> {out_gif}", flush=True)


if __name__ == "__main__":
    main()
