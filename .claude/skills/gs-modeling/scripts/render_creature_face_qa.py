"""Creature / Non-Humanoid Face Rigging & Close-Up QA Video Generator (Spark GPU SplatSkinning).

Builds doll eyes, eyelid shells, refined muzzle `jaw` bone, and inner-mouth cavity patch on any
creature/animal 3DGS model, then renders a 4-part face close-up QA video (MP4 + 3x3 contact sheet):
  1. BLINK (both eyelids descending flush with furred cheek surface)
  2. EYE GAZE (independent binocular eyeball tracking: left -> right -> up -> camera)
  3. WINK + HEAD TILT (left wink with head tilt & ear secondary motion)
  4. JAW OPEN / BARK (muzzle opening with mouth seam_cut + inner mouth cavity)
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
SHARED_DIR = SCRIPTS_DIR.parents[2] / "shared"
for p in (str(SCRIPTS_DIR), str(SHARED_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ply_io import read_ply, write_ply
from harmonic_dqs import (
    blender_rigid_to_spark_qp,
    build_fk_hierarchy,
    compute_harmonic_skin_weights,
    euler_xyz,
    fk_to_spark_bones,
    pack_spark_skin_weights,
    to_spark_ply,
)
from secondary_motion import SecondaryMotionSolver
from creature_face_rig import (
    build_creature_eyeball,
    build_creature_eyelid_shell,
    build_creature_mouth_patch,
    carve_creature_sockets,
    fit_creature_face,
)


def get_font(size: int):
    for p in (
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def add_banner(img: Image.Image, text: str, font, size: int = 512) -> Image.Image:
    out = img.convert("RGB").resize((size, size), Image.LANCZOS)
    draw = ImageDraw.Draw(out, "RGBA")
    draw.rectangle([0, 0, size, 32], fill=(16, 18, 24, 215))
    draw.text((10, 6), text, fill=(245, 248, 255, 255), font=font)
    return out


def _ply_to_spark(d: np.ndarray) -> np.ndarray:
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
    rot = np.column_stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]]).astype(np.float32)
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8)
    return to_spark_ply(d, xyz, rot)


def main():
    ap = argparse.ArgumentParser(description="Build creature face rig (eyes/eyelids/jaw) & render close-up QA video.")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--spark-dir", required=True)
    ap.add_argument("--out-skeleton", required=True, help="Output updated rig_skeleton.json with eye_L, eye_R, jaw")
    ap.add_argument("--out-mp4", required=True)
    ap.add_argument("--out-sheet", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=18)
    args = ap.parse_args()

    t0 = time.perf_counter()
    data, _ = read_ply(Path(args.ply))
    skel = json.loads(Path(args.skeleton).read_text(encoding="utf-8"))
    bones = skel["bones"]

    # 1. Fit creature eyes, cheek quads, iris/lash palette, and muzzle jaw
    face_meta = fit_creature_face(data, bones)
    skel["face_meta"] = face_meta
    Path(args.out_skeleton).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_skeleton).write_text(json.dumps(skel, indent=2), encoding="utf-8")
    print(
        f"[CreatureFace] Fitted eyes D_eye={face_meta['d_eye']:.4f}, "
        f"yaw_L={face_meta['eye_L']['cheek_yaw_deg']:+.1f}deg, yaw_R={face_meta['eye_R']['cheek_yaw_deg']:+.1f}deg, "
        f"fur_rgb={[round(v,2) for v in face_meta['fur_rgb']]}"
    )

    # 2. Carve eye sockets + generate independent eyeballs
    body_carved = carve_creature_sockets(data, face_meta)
    eye_L_loc = build_creature_eyeball(face_meta["eye_L"], face_meta["iris_palette"], data.dtype)
    eye_R_loc = build_creature_eyeball(face_meta["eye_R"], face_meta["iris_palette"], data.dtype)

    # Place eyeballs into world rest coordinates (so head/eye rigid transform applies cleanly)
    eL_rest = eye_L_loc.copy()
    pL = np.asarray(bones["eye_L"]["head"], dtype=np.float32)
    eL_rest["x"] += pL[0]; eL_rest["y"] += pL[1]; eL_rest["z"] += pL[2]
    eR_rest = eye_R_loc.copy()
    pR = np.asarray(bones["eye_R"]["head"], dtype=np.float32)
    eR_rest["x"] += pR[0]; eR_rest["y"] += pR[1]; eR_rest["z"] += pR[2]

    # 3. Compute Harmonic skin weights on carved body (including refined jaw + seam_cut)
    xyz = np.column_stack([body_carved["x"], body_carved["y"], body_carved["z"]]).astype(np.float32)
    from harmonic_dqs import SH_C0 as _SH_C0
    body_rgb = np.clip(0.5 + _SH_C0 * np.column_stack([body_carved["f_dc_0"], body_carved["f_dc_1"], body_carved["f_dc_2"]]), 0.0, 1.0).astype(np.float32)
    skin_bones, W = compute_harmonic_skin_weights(
        xyz, bones, np.ones(len(xyz), dtype=bool), seam_cuts=skel.get("seam_cuts")
    )
    jaw_i = skin_bones.index("jaw") if "jaw" in skin_bones else -1
    if jaw_i >= 0:
        dom = np.argmax(W, axis=1)
        print(f"[CreatureFace] Refined jaw dominant splats: {int((dom == jaw_i).sum())} (influenced >0.2: {int((W[:, jaw_i] > 0.2).sum())})")

    packed = pack_spark_skin_weights(W)
    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    spark = Path(args.spark_dir).resolve()
    anim_tag = f"faceqa_{out_mp4.stem}"
    anim_dir = spark / "frames" / anim_tag
    anim_dir.mkdir(parents=True, exist_ok=True)

    write_ply(anim_dir / "body_base.ply", _ply_to_spark(body_carved))
    packed.tofile(anim_dir / "body_skin.bin")
    write_ply(anim_dir / "eye_L.ply", _ply_to_spark(eL_rest))
    write_ply(anim_dir / "eye_R.ply", _ply_to_spark(eR_rest))

    # Pre-bake quantized kNN-fur-matched eyelid shells & animated mouth patches (0.0 = hidden!)
    blink_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    lid_files: dict[tuple[str, int], str] = {}
    for side in ("eye_L", "eye_R"):
        for bi, bv in enumerate(blink_levels):
            if bv <= 0.02:
                continue
            lid_ply = build_creature_eyelid_shell(
                face_meta[side], face_meta["fur_rgb"], face_meta["lash_rgb"], bv, data.dtype,
                body_xyz=xyz, body_rgb=body_rgb,
            )
            if len(lid_ply) > 0:
                fname = f"lid_{side}_{bi}.ply"
                write_ply(anim_dir / fname, _ply_to_spark(lid_ply))
                lid_files[(side, bi)] = f"frames/{anim_tag}/{fname}"

    mouth_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    mouth_files: dict[int, str] = {}
    for mi, mv in enumerate(mouth_levels):
        if mv <= 0.04:
            continue
        mply = build_creature_mouth_patch(face_meta, mv, data.dtype, body_xyz=xyz)
        if len(mply) > 0:
            fname = f"mouth_{mi}.ply"
            write_ply(anim_dir / fname, _ply_to_spark(mply))
            mouth_files[mi] = f"frames/{anim_tag}/{fname}"

    # 4. Build 96-frame face showcase choreography (4 parts x 24 frames)
    order = build_fk_hierarchy(bones)
    choreo = []  # (label, yaw_cam, pitch_cam, rots, blink_L, blink_R, eye_yaw, eye_pitch, mouth_open)

    # Part 1: BLINK (24f, front-3/4 close-up)
    for i in range(24):
        t = i / 24.0
        b = 0.0
        for c in (0.28, 0.72):
            dist = abs(t - c)
            if dist < 0.14:
                b = max(b, math.cos((dist / 0.14) * math.pi * 0.5) ** 1.4)
        rots = {"head": euler_xyz(rx_deg=3.0 * math.sin(2 * math.pi * t), rz_deg=5.0 * math.sin(2 * math.pi * t))}
        choreo.append(("1. BLINK", -12.0, 6.0, rots, b, b, 0.0, 0.0, 0.0))

    # Part 2: EYE GAZE (24f, eyes look left -> right -> up -> center)
    for i in range(24):
        t = i / 24.0
        ey = 18.0 * math.sin(2 * math.pi * t)
        ep = 10.0 * math.sin(4 * math.pi * t)
        rots = {"head": euler_xyz(rz_deg=4.0 * math.sin(2 * math.pi * t))}
        choreo.append(("2. EYE GAZE", -8.0, 5.0, rots, 0.0, 0.0, ey, ep, 0.0))

    # Part 3: WINK L + HEAD TILT (24f)
    for i in range(24):
        t = i / 24.0
        w = math.sin(math.pi * t) ** 1.5
        rots = {
            "head": euler_xyz(rx_deg=4.0 * w, ry_deg=11.0 * w, rz_deg=-8.0 * w),
            "neck": euler_xyz(ry_deg=5.0 * w),
        }
        choreo.append(("3. WINK & HEAD TILT", -15.0, 6.0, rots, w, 0.0, -6.0 * w, 0.0, 0.0))

    # Part 4: JAW OPEN / BARK (24f, animated procedural mouth + subtle synchronized jaw drop)
    for i in range(24):
        t = i / 24.0
        j_open = max(0.0, math.sin(2 * math.pi * t * 2.0)) ** 0.85
        rots = {
            "head": euler_xyz(rx_deg=-7.0 * j_open, rz_deg=5.0 * math.sin(2 * math.pi * t)),
            "jaw": euler_xyz(rx_deg=+10.0 * j_open),
        }
        choreo.append(("4. MOUTH & JAW", -24.0 + 16.0 * math.sin(math.pi * t), 5.0, rots, 0.0, 0.0, 0.0, -4.0 * j_open, j_open))

    # Run secondary motion (ears & ribbons react softly to head tilts & barks!)
    solver = SecondaryMotionSolver(bones, order)
    rots_in = [c[3] for c in choreo]
    _, world_seq = solver.simulate_sequence(
        rots_in, [None] * len(choreo), dt=1.0 / float(args.fps), loop_warmup=True
    )

    # Close-up camera tightly framed on the creature's face
    face_ctr_b = np.array([face_meta["face_cx"], face_meta["eye_y"] - 0.01, face_meta["eye_z"] - 0.015], dtype=np.float32)
    target_spark = [float(face_ctr_b[0]), float(-face_ctr_b[2]), float(face_ctr_b[1])]
    distance = max(face_meta["d_eye"] * 7.2, 1.05)

    default_cam = {
        "width": args.size,
        "height": args.size,
        "fov": 22,
        "background": "#f2f4f8",
        "target": target_spark,
        "distance": distance,
        "pitch": 6.0,
        "front": -12.0,
    }

    h_head_r = np.asarray(bones["head"]["head"], dtype=np.float32)
    gL_hat = np.asarray(face_meta["eye_L"].get("gaze_dir", [0.0, -1.0, 0.0]), dtype=np.float32)
    gR_hat = np.asarray(face_meta["eye_R"].get("gaze_dir", [0.0, -1.0, 0.0]), dtype=np.float32)
    anim_frames = []
    for fi, ((lbl, yaw, pitch, _, bL, bR, ey, ep, m_open), world) in enumerate(zip(choreo, world_seq)):
        bf = fk_to_spark_bones(bones, skin_bones, world)
        R_hd, h_hd, _ = world["head"]
        t_hd = h_hd - (R_hd @ h_head_r)
        qp_hd = blender_rigid_to_spark_qp(R_hd, t_hd)

        # Retract closed eyeball slightly inward along -gaze_dir during blink (same trick as humanoid pipeline!)
        retract_L = -gL_hat * (0.012 * max(0.0, (bL - 0.35) / 0.65))
        retract_R = -gR_hat * (0.012 * max(0.0, (bR - 0.35) / 0.65))

        R_gaze = euler_xyz(rx_deg=ep, rz_deg=ey)
        R_eL = (R_hd @ R_gaze).astype(np.float32)
        h_eL_w = h_hd + R_hd @ (pL + retract_L - h_head_r)
        t_eL = h_eL_w - (R_eL @ pL)
        qp_eL = blender_rigid_to_spark_qp(R_eL, t_eL)

        R_eR = (R_hd @ R_gaze).astype(np.float32)
        h_eR_w = h_hd + R_hd @ (pR + retract_R - h_head_r)
        t_eR = h_eR_w - (R_eR @ pR)
        qp_eR = blender_rigid_to_spark_qp(R_eR, t_eR)

        extra_plys = [
            f"frames/{anim_tag}/eye_L.ply",
            f"frames/{anim_tag}/eye_R.ply",
        ]
        extra_xforms = [qp_eL, qp_eR]

        # Attach quantized solid eyelid shells when blinking
        for side, bv in (("eye_L", bL), ("eye_R", bR)):
            bi = int(np.argmin([abs(bv - lvl) for lvl in blink_levels]))
            if (side, bi) in lid_files:
                extra_plys.append(lid_files[(side, bi)])
                extra_xforms.append(qp_hd)

        # Attach animated mouth patch ONLY when mouth is open (ZERO bleed-through when closed!)
        mi = int(np.argmin([abs(m_open - lvl) for lvl in mouth_levels]))
        if mi in mouth_files:
            extra_plys.append(mouth_files[mi])
            extra_xforms.append(qp_hd)

        cam_f = dict(default_cam)
        cam_f["pitch"] = float(pitch)
        cam_f["front"] = float(yaw)
        anim_frames.append({
            "png": f"f-{fi:04d}.png",
            "yaw": float(yaw),
            "camera": cam_f,
            "bones": bf,
            "extraPly": extra_plys,
            "extraTransforms": extra_xforms,
        })

    anim_json = anim_dir / "anim.json"
    anim_json.write_text(json.dumps({
        "basePly": f"frames/{anim_tag}/body_base.ply",
        "skinBin": f"frames/{anim_tag}/body_skin.bin",
        "numBones": len(skin_bones),
        "defaultCamera": default_cam,
        "frames": anim_frames,
    }), encoding="utf-8")

    render_dest = f"renders_{anim_tag}"
    print(f"[CreatureFaceQA] Rendering {len(anim_frames)} close-up frames via Spark GPU...", flush=True)
    subprocess.run(["node", "capture.mjs", "anim", render_dest, f"frames/{anim_tag}/anim.json"], cwd=str(spark), check=True)

    render_dir = spark / render_dest
    font = get_font(18)
    video_frames = []
    for fi, (lbl, *_rest) in enumerate(choreo):
        im = Image.open(render_dir / f"f-{fi:04d}.png")
        video_frames.append(np.asarray(add_banner(im, f"{lbl}  [{fi+1:02d}/{len(choreo)}]", font, size=args.size)))

    iio.imwrite(out_mp4, video_frames, fps=args.fps, codec="libx264")
    print(f"[CreatureFaceQA] Saved Face QA MP4 -> {out_mp4} ({time.perf_counter() - t0:.1f}s)")

    if args.out_sheet:
        # Pick representative peak frames across all 4 parts (blink closed, gaze L/R, wink, jaw open)
        pick = [0, 7, 17, 30, 42, 60, 72, 78, 90]
        pick = [min(p, len(video_frames) - 1) for p in pick]
        tile = 320
        sheet = Image.new("RGB", (tile * 3, tile * 3), (240, 242, 246))
        for k, fi in enumerate(pick):
            im = Image.fromarray(video_frames[fi]).resize((tile, tile), Image.LANCZOS)
            sheet.paste(im, ((k % 3) * tile, (k // 3) * tile))
        out_sheet = Path(args.out_sheet)
        out_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_sheet)
        print(f"[CreatureFaceQA] Saved 3x3 Face QA contact sheet -> {out_sheet}")


if __name__ == "__main__":
    main()
