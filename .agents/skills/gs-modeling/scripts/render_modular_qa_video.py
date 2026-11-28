"""Archetype-Agnostic GPU SplatSkinning QA Video & Contact Sheet Generator (Schema v2.1).

Automatically adapts its test choreography to whatever bones exist in `rig_skeleton.json`:
  - Quadruped gait (4-leg trot + spine undulation + head look)
  - Humanoid walk / torso sway (for v2.0 & v2.1 humanoids)
  - Wing flapping & folding (if `wing_L_*` / `wing_R_*` exist)
  - Muzzle / Jaw open-close roar/speak test (if `jaw` exists)
  - SecondaryMotionSolver physics for all bones with `dynamics` (ears, tail, ribbons, cape, hair, wingtips)
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
    add_forearm_twist_bones,
    build_fk_hierarchy,
    compute_harmonic_skin_weights,
    euler_xyz,
    fk_to_spark_bones,
    pack_spark_skin_weights,
    to_spark_ply,
)
from secondary_motion import SecondaryMotionSolver


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


def build_choreography(bones: dict, archetype: str, fps: int = 12):
    """Build a list of (label, yaw_deg, pitch_deg, local_rots, root_shift, wind_vec) frames."""
    has_wings = any(b.startswith(("wing_L_", "wing_R_")) for b in bones)
    has_jaw = "jaw" in bones
    has_quad_legs = "front_upper_L" in bones and "hind_upper_L" in bones
    has_human_legs = "thigh_L" in bones and "thigh_R" in bones

    frames = []

    # 1. Idle + Wind + Muzzle Jaw Open/Close (24 frames)
    for i in range(24):
        t = i / 24.0
        rots = {}
        if "head" in bones:
            rots["head"] = euler_xyz(rx_deg=6.0 * math.sin(2 * math.pi * t), rz_deg=14.0 * math.sin(2 * math.pi * t))
        if has_jaw:
            open_wave = 0.5 - 0.5 * math.cos(2 * math.pi * t * 1.5)
            rots["jaw"] = euler_xyz(rx_deg=-32.0 * open_wave)
        if has_wings:
            flap = 8.0 * math.sin(2 * math.pi * t)
            for s, sgn in (("L", +1.0), ("R", -1.0)):
                if f"wing_{s}_1" in bones:
                    rots[f"wing_{s}_1"] = euler_xyz(ry_deg=sgn * flap)
        wind = np.array([0.4 * math.sin(2 * math.pi * t), 1.0, 0.0], dtype=np.float32)
        yaw = -35.0 if archetype == "quadruped" else 25.0
        lbl = "1. JAW & WIND IDLE" if has_jaw else "1. WIND & SECONDARY MOTION"
        frames.append((lbl, yaw, 10.0, rots, np.zeros(3, dtype=np.float32), wind))

    # 2. Wing Flap Showcase (anatomical bone-driven flap) or Torso/Cape Sway (24 frames)
    for i in range(24):
        t = i / 24.0
        rots = {}
        phase = 2 * math.pi * t * 2.0
        if has_wings:
            # Anatomical bird/creature wing stroke:
            # - wing_1 (humerus/shoulder): clean up/down stroke around Y + slight forward sweep on downstroke
            # - wing_2 (elbow/forearm): folds slightly on upstroke, extends straight on downstroke
            # - wing_3 (wrist/primaries): crisp feathering pitch (X) on upstroke recovery
            upstroke = max(0.0, math.sin(phase))
            downstroke = max(0.0, -math.sin(phase))
            for s, sgn in (("L", +1.0), ("R", -1.0)):
                if f"wing_{s}_1" in bones:
                    rots[f"wing_{s}_1"] = euler_xyz(
                        rx_deg=6.0 * downstroke,
                        ry_deg=sgn * (28.0 * math.sin(phase)),
                        rz_deg=sgn * (-8.0 * downstroke + 6.0 * upstroke),
                    )
                if f"wing_{s}_2" in bones:
                    rots[f"wing_{s}_2"] = euler_xyz(
                        ry_deg=sgn * (-14.0 * upstroke + 6.0 * downstroke),
                        rz_deg=sgn * (14.0 * upstroke),
                    )
                if f"wing_{s}_3" in bones:
                    rots[f"wing_{s}_3"] = euler_xyz(
                        rx_deg=-16.0 * upstroke,
                        ry_deg=sgn * (-8.0 * upstroke),
                    )
            if "chest" in bones:
                rots["chest"] = euler_xyz(rx_deg=4.0 * math.cos(phase))
            shift = np.array([0.0, 0.0, 0.025 * (0.5 - 0.5 * math.cos(phase))], dtype=np.float32)
            yaw = -55.0 + 25.0 * math.sin(2 * math.pi * t)
            lbl = "2. ARTICULATED WING FLAP"
        else:
            if "spine_lower" in bones:
                rots["spine_lower"] = euler_xyz(rz_deg=14.0 * math.sin(phase), ry_deg=4.0 * math.cos(phase))
            if "chest" in bones:
                rots["chest"] = euler_xyz(rz_deg=10.0 * math.sin(phase))
            if "head" in bones:
                rots["head"] = euler_xyz(rz_deg=-10.0 * math.sin(phase))
            shift = np.array([0.0, 0.0, 0.015 * abs(math.sin(phase))], dtype=np.float32)
            yaw = 145.0  # Rear 3/4 view to showcase cape/hair sway
            lbl = "2. SWAY & CLOTH DYNAMICS"
        wind = np.array([0.5 * math.cos(phase), 1.4, 0.0], dtype=np.float32)
        frames.append((lbl, yaw, 12.0, rots, shift, wind))

    # 3. Locomotion (Quadruped Trot or Humanoid Walk) (24 frames)
    for i in range(24):
        t = i / 24.0
        phase = 2 * math.pi * t * 2.0
        rots = {}
        if has_quad_legs:
            # Diagonal trot gait: (FL + HR) vs (FR + HL)
            for s, ph in (("L", phase), ("R", phase + math.pi)):
                rots[f"front_upper_{s}"] = euler_xyz(rx_deg=18.0 * math.sin(ph))
                rots[f"front_lower_{s}"] = euler_xyz(rx_deg=-24.0 * max(0.0, -math.cos(ph)))
                rots[f"hind_upper_{s}"] = euler_xyz(rx_deg=16.0 * math.sin(ph + math.pi))
                rots[f"hind_lower_{s}"] = euler_xyz(rx_deg=22.0 * max(0.0, math.cos(ph + math.pi)))
            if "spine_mid" in bones:
                rots["spine_mid"] = euler_xyz(rx_deg=5.0 * math.sin(2 * phase), rz_deg=6.0 * math.sin(phase))
            if "neck" in bones:
                rots["neck"] = euler_xyz(rx_deg=-7.0 * math.sin(2 * phase))
            if has_wings:
                for s, sgn in (("L", +1.0), ("R", -1.0)):
                    if f"wing_{s}_1" in bones:
                        rots[f"wing_{s}_1"] = euler_xyz(ry_deg=sgn * (14.0 * math.sin(phase)))
            shift = np.array([0.0, 0.0, 0.022 * abs(math.sin(phase))], dtype=np.float32)
            yaw = -60.0
            lbl = "3. QUADRUPED TROT + SECONDARY"
        elif has_human_legs:
            for s, ph in (("L", phase), ("R", phase + math.pi)):
                rots[f"thigh_{s}"] = euler_xyz(rx_deg=24.0 * math.sin(ph))
                rots[f"shin_{s}"] = euler_xyz(rx_deg=36.0 * max(0.0, -math.cos(ph)))
                if f"upper_arm_{s}" in bones:
                    rots[f"upper_arm_{s}"] = euler_xyz(rx_deg=-18.0 * math.sin(ph))
            if "spine_lower" in bones:
                rots["spine_lower"] = euler_xyz(rz_deg=8.0 * math.sin(phase))
            shift = np.array([0.0, 0.0, 0.018 * abs(math.sin(phase))], dtype=np.float32)
            yaw = 135.0
            lbl = "3. WALK + CAPE/HAIR DYNAMICS"
        else:
            shift = np.zeros(3, dtype=np.float32)
            yaw = -40.0
            lbl = "3. TURNTABLE"
        wind = np.array([0.0, 1.8, 0.0], dtype=np.float32)
        frames.append((lbl, yaw, 10.0, rots, shift, wind))

    return frames


def build_wing_showcase(bones: dict, fps: int = 12):
    """Dedicated 3-angle articulated wing-flapping showcase (rear-3/4, front-3/4, side)."""
    frames = []
    views = [
        ("WING FLAP [REAR 3/4]", -135.0, 14.0),
        ("WING FLAP [FRONT 3/4]", -35.0, 10.0),
        ("WING FLAP [SIDE PROFILE]", -85.0, 8.0),
    ]
    for lbl, yaw, pitch in views:
        for i in range(24):
            phase = 2.0 * math.pi * (i / 24.0) * 1.0  # 1 full slow flap per 24 frames (2 sec)
            upstroke = max(0.0, math.sin(phase))
            downstroke = max(0.0, -math.sin(phase))
            rots = {}
            for s, sgn in (("L", +1.0), ("R", -1.0)):
                if f"wing_{s}_1" in bones:
                    rots[f"wing_{s}_1"] = euler_xyz(
                        rx_deg=8.0 * downstroke,
                        ry_deg=sgn * (34.0 * math.sin(phase)),
                        rz_deg=sgn * (-10.0 * downstroke + 8.0 * upstroke),
                    )
                if f"wing_{s}_2" in bones:
                    rots[f"wing_{s}_2"] = euler_xyz(
                        ry_deg=sgn * (-18.0 * upstroke + 8.0 * downstroke),
                        rz_deg=sgn * (18.0 * upstroke),
                    )
                if f"wing_{s}_3" in bones:
                    rots[f"wing_{s}_3"] = euler_xyz(
                        rx_deg=-18.0 * upstroke,
                        ry_deg=sgn * (-10.0 * upstroke),
                    )
            if "chest" in bones:
                rots["chest"] = euler_xyz(rx_deg=5.0 * math.cos(phase))
            if "neck" in bones:
                rots["neck"] = euler_xyz(rx_deg=-4.0 * math.cos(phase))
            shift = np.array([0.0, 0.0, 0.03 * (0.5 - 0.5 * math.cos(phase))], dtype=np.float32)
            wind = np.array([0.2, 0.8, -0.4 * downstroke], dtype=np.float32)
            frames.append((lbl, yaw, pitch, rots, shift, wind))
    return frames


def build_animal_locomotion(bones: dict, fps: int = 12):
    """Dedicated 3-part animal motion test: 1. WALK -> 2. RUN (Gallop) -> 3. FLY (Winged Flight)."""
    has_wings = any(b.startswith(("wing_L_", "wing_R_")) for b in bones)
    has_quad = "front_upper_L" in bones and "hind_upper_L" in bones
    frames = []

    # 1. WALK (24 frames, gentle 4-beat quadruped walk, wings softly tucked)
    for i in range(24):
        t = i / 24.0
        ph = 2.0 * math.pi * t * 2.0
        rots = {}
        if has_quad:
            # 4-beat lateral walk phases: HL(0), FL(pi/2), HR(pi), FR(3pi/2)
            for b_u, b_l, b_p, p_off, knee_sgn in (
                ("hind_upper_L", "hind_lower_L", "hind_paw_L", 0.0, +1.0),
                ("front_upper_L", "front_lower_L", "front_paw_L", 0.5 * math.pi, -1.0),
                ("hind_upper_R", "hind_lower_R", "hind_paw_R", math.pi, +1.0),
                ("front_upper_R", "front_lower_R", "front_paw_R", 1.5 * math.pi, -1.0),
            ):
                p = ph + p_off
                rots[b_u] = euler_xyz(rx_deg=14.0 * math.sin(p))
                rots[b_l] = euler_xyz(rx_deg=knee_sgn * 18.0 * max(0.0, math.cos(p)))
                rots[b_p] = euler_xyz(rx_deg=-8.0 * math.sin(p))
        if "spine_mid" in bones:
            rots["spine_mid"] = euler_xyz(rx_deg=2.5 * math.sin(2 * ph), rz_deg=5.0 * math.sin(ph))
        if "neck" in bones:
            rots["neck"] = euler_xyz(rx_deg=-3.0 * math.sin(2 * ph), rz_deg=-4.0 * math.sin(ph))
        if "head" in bones:
            rots["head"] = euler_xyz(rx_deg=2.5 * math.sin(2 * ph))
        if has_wings:
            for s, sgn in (("L", +1.0), ("R", -1.0)):
                if f"wing_{s}_1" in bones:
                    rots[f"wing_{s}_1"] = euler_xyz(ry_deg=sgn * (-8.0 + 3.0 * math.sin(ph)), rz_deg=sgn * 10.0)
                if f"wing_{s}_2" in bones:
                    rots[f"wing_{s}_2"] = euler_xyz(rz_deg=sgn * 8.0)
        shift = np.array([0.0, 0.0, 0.012 * abs(math.sin(ph))], dtype=np.float32)
        wind = np.array([0.0, 0.7, 0.0], dtype=np.float32)
        # Smooth camera orbit from front-3/4 (-25) to side-3/4 (-80) so all 4 legs are clearly visible
        yaw_w = -25.0 - 55.0 * (0.5 - 0.5 * math.cos(math.pi * t))
        frames.append(("1. WALK (QUADRUPED)", yaw_w, 10.0, rots, shift, wind))

    # 2. RUN / GALLOP (24 frames, energetic rotary gallop with distinct L/R lead-leg timing & camera orbit)
    for i in range(24):
        t = i / 24.0
        ph = 2.0 * math.pi * t * 2.0
        rots = {}
        if has_quad:
            # Distinct rotary gallop phase offsets so L & R front/hind legs clearly separate in motion
            for s, lag in (("L", 0.0), ("R", 1.15)):
                pf = ph + lag
                phind = ph + math.pi + lag
                rots[f"front_upper_{s}"] = euler_xyz(rx_deg=26.0 * math.sin(pf) - 4.0)
                rots[f"front_lower_{s}"] = euler_xyz(rx_deg=-34.0 * max(0.0, -math.cos(pf)))
                rots[f"front_paw_{s}"] = euler_xyz(rx_deg=14.0 * math.sin(pf))
                rots[f"hind_upper_{s}"] = euler_xyz(rx_deg=24.0 * math.sin(phind) + 4.0)
                rots[f"hind_lower_{s}"] = euler_xyz(rx_deg=32.0 * max(0.0, math.cos(phind)))
                rots[f"hind_paw_{s}"] = euler_xyz(rx_deg=-14.0 * math.sin(phind))
        if "pelvis" in bones:
            rots["pelvis"] = euler_xyz(rx_deg=8.0 * math.cos(ph))
        if "spine_mid" in bones:
            rots["spine_mid"] = euler_xyz(rx_deg=-9.0 * math.cos(ph))
        if "chest" in bones:
            rots["chest"] = euler_xyz(rx_deg=-5.0 * math.cos(ph))
        if "neck" in bones:
            rots["neck"] = euler_xyz(rx_deg=9.0 + 4.0 * math.cos(ph))
        if "head" in bones:
            rots["head"] = euler_xyz(rx_deg=-7.0 - 3.0 * math.cos(ph))
        if has_wings:
            for s, sgn in (("L", +1.0), ("R", -1.0)):
                if f"wing_{s}_1" in bones:
                    rots[f"wing_{s}_1"] = euler_xyz(ry_deg=sgn * (6.0 * math.sin(ph)), rz_deg=sgn * 16.0)
                if f"wing_{s}_2" in bones:
                    rots[f"wing_{s}_2"] = euler_xyz(rz_deg=sgn * 12.0)
        air = 0.050 * max(0.0, math.sin(ph))
        shift = np.array([0.0, 0.0, air], dtype=np.float32)
        wind = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        # Orbit camera from side (-80) around front-3/4 (-20) during run so L/R front legs are unmistakably distinct
        yaw_r = -80.0 + 60.0 * (0.5 - 0.5 * math.cos(math.pi * t))
        frames.append(("2. RUN (GALLOP)", yaw_r, 11.0, rots, shift, wind))

    # 3. FLY (36 frames, airborne soaring + full articulated bilateral wingbeats + tucked flight legs)
    for i in range(36):
        t = i / 36.0
        ph = 2.0 * math.pi * t * 2.0
        upstroke = max(0.0, math.sin(ph))
        downstroke = max(0.0, -math.sin(ph))
        rots = {}
        if has_wings:
            for s, sgn in (("L", +1.0), ("R", -1.0)):
                if f"wing_{s}_1" in bones:
                    rots[f"wing_{s}_1"] = euler_xyz(
                        rx_deg=10.0 * downstroke,
                        ry_deg=sgn * (36.0 * math.sin(ph)),
                        rz_deg=sgn * (-10.0 * downstroke + 8.0 * upstroke),
                    )
                if f"wing_{s}_2" in bones:
                    rots[f"wing_{s}_2"] = euler_xyz(
                        ry_deg=sgn * (-18.0 * upstroke + 8.0 * downstroke),
                        rz_deg=sgn * (18.0 * upstroke),
                    )
                if f"wing_{s}_3" in bones:
                    rots[f"wing_{s}_3"] = euler_xyz(
                        rx_deg=-18.0 * upstroke,
                        ry_deg=sgn * (-10.0 * upstroke),
                    )
        if has_quad:
            # Aerodynamic flight leg pose: front legs tucked, hind legs trailed gracefully backward
            for s in ("L", "R"):
                rots[f"front_upper_{s}"] = euler_xyz(rx_deg=24.0 + 4.0 * math.cos(ph))
                rots[f"front_lower_{s}"] = euler_xyz(rx_deg=-42.0)
                rots[f"front_paw_{s}"] = euler_xyz(rx_deg=18.0)
                rots[f"hind_upper_{s}"] = euler_xyz(rx_deg=32.0 + 5.0 * math.cos(ph))
                rots[f"hind_lower_{s}"] = euler_xyz(rx_deg=18.0)
                rots[f"hind_paw_{s}"] = euler_xyz(rx_deg=22.0)
        if "pelvis" in bones:
            rots["pelvis"] = euler_xyz(rx_deg=-8.0 + 4.0 * math.cos(ph))
        if "chest" in bones:
            rots["chest"] = euler_xyz(rx_deg=6.0 * math.cos(ph))
        if "neck" in bones:
            rots["neck"] = euler_xyz(rx_deg=10.0 - 4.0 * math.cos(ph))
        if "head" in bones:
            rots["head"] = euler_xyz(rx_deg=-6.0)
        # Hovering lift synchronized with wingbeat downstroke
        hover_z = 0.14 + 0.045 * (0.5 - 0.5 * math.cos(ph))
        shift = np.array([0.0, 0.0, hover_z], dtype=np.float32)
        wind = np.array([0.0, 2.0, -0.5 * downstroke], dtype=np.float32)
        # Gently orbit camera from front-3/4 to rear-3/4 during flight so both wings are clearly visible!
        yaw = -35.0 - 95.0 * (0.5 - 0.5 * math.cos(math.pi * t))
        frames.append(("3. FLY (WINGED FLIGHT)", yaw, 12.0, rots, shift, wind))

    return frames


def main():
    ap = argparse.ArgumentParser(description="Render modular v2.1 skeleton QA video via Spark GPU SplatSkinning.")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--spark-dir", required=True)
    ap.add_argument("--out-mp4", required=True)
    ap.add_argument("--out-sheet", default=None)
    ap.add_argument("--mode", choices=("standard", "wing_showcase", "animal_locomotion"), default="standard")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    t0 = time.perf_counter()
    skel = json.loads(Path(args.skeleton).read_text(encoding="utf-8"))
    archetype = str(skel.get("archetype", "humanoid"))
    bones = add_forearm_twist_bones(skel["bones"]) if archetype == "humanoid" else skel["bones"]
    order = build_fk_hierarchy(bones)

    data, _ = read_ply(Path(args.ply))
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    rot = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float32)
    rot /= np.maximum(np.linalg.norm(rot, axis=1, keepdims=True), 1e-8)

    skin_bones, W = compute_harmonic_skin_weights(
        xyz, bones, np.ones(len(xyz), dtype=bool), seam_cuts=skel.get("seam_cuts")
    )
    packed = pack_spark_skin_weights(W)

    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    spark = Path(args.spark_dir).resolve()
    anim_tag = f"modqa_{out_mp4.stem}"
    anim_dir = spark / "frames" / anim_tag
    anim_dir.mkdir(parents=True, exist_ok=True)

    # Write rest PLY in Spark coordinates & packed skinning binary under spark/frames/<anim_tag>
    rest_ply = anim_dir / "body_base.ply"
    skin_bin = anim_dir / "body_skin.bin"
    write_ply(rest_ply, to_spark_ply(data, xyz, rot))
    packed.tofile(skin_bin)

    # Simulate choreography + secondary motion
    if args.mode == "wing_showcase":
        choreo = build_wing_showcase(bones, fps=args.fps)
    elif args.mode == "animal_locomotion":
        choreo = build_animal_locomotion(bones, fps=args.fps)
    else:
        choreo = build_choreography(bones, archetype, fps=args.fps)
    solver = SecondaryMotionSolver(bones, order)
    rots_in = [c[3] for c in choreo]
    shifts_in = [c[4] for c in choreo]
    winds_in = [c[5] for c in choreo]
    _, world_seq = solver.simulate_sequence(
        rots_in,
        shifts_in,
        dt=1.0 / float(args.fps),
        wind_fn=lambda fi, _t: winds_in[fi],
        loop_warmup=True,
    )

    # Compute camera framing from model bounds (in Spark Y-up coords: [x, -z, y])
    op = 1.0 / (1.0 + np.exp(-np.clip(data["opacity"], -20, 20)))
    xyz_op = xyz[op > 0.2]
    p02 = np.percentile(xyz_op, 2, axis=0)
    p98 = np.percentile(xyz_op, 98, axis=0)
    ctr_b = 0.5 * (p02 + p98)
    span = float(np.linalg.norm(p98 - p02))
    target_spark = [float(ctr_b[0]), float(-ctr_b[2]), float(ctr_b[1])]
    distance = max(span * 2.35, 2.2)

    default_cam = {
        "width": args.size,
        "height": args.size,
        "fov": 22,
        "background": "#f2f4f8",
        "target": target_spark,
        "distance": distance,
        "pitch": 10.0,
        "front": 0.0,
    }
    anim_frames = []
    for fi, ((lbl, yaw, pitch, _, _, _), world) in enumerate(zip(choreo, world_seq)):
        bf = fk_to_spark_bones(bones, skin_bones, world)
        cam_f = dict(default_cam)
        cam_f["pitch"] = float(pitch)
        cam_f["front"] = float(yaw)
        anim_frames.append({
            "png": f"f-{fi:04d}.png",
            "yaw": float(yaw),
            "camera": cam_f,
            "bones": bf,
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
    cmd = ["node", "capture.mjs", "anim", render_dest, f"frames/{anim_tag}/anim.json"]
    print(f"[ModularQA] Rendering {len(anim_frames)} frames via Spark GPU ({len(skin_bones)} skin bones, {len(solver.dynamic_bones)} dynamic)...", flush=True)
    subprocess.run(cmd, cwd=str(spark), check=True)

    render_dir = spark / render_dest
    font = get_font(18)
    video_frames = []
    for fi, (lbl, *_rest) in enumerate(choreo):
        im = Image.open(render_dir / f"f-{fi:04d}.png")
        video_frames.append(np.asarray(add_banner(im, f"{lbl}  [{fi+1:02d}/{len(choreo)}]", font, size=args.size)))

    iio.imwrite(out_mp4, video_frames, fps=args.fps, codec="libx264")
    print(f"[ModularQA] Saved QA MP4 -> {out_mp4} ({time.perf_counter() - t0:.1f}s)")

    if args.out_sheet:
        idxs = np.linspace(0, len(video_frames) - 1, 9, dtype=int)
        tile = 320
        sheet = Image.new("RGB", (tile * 3, tile * 3), (240, 242, 246))
        for k, fi in enumerate(idxs):
            im = Image.fromarray(video_frames[fi]).resize((tile, tile), Image.LANCZOS)
            sheet.paste(im, ((k % 3) * tile, (k // 3) * tile))
        out_sheet = Path(args.out_sheet)
        out_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_sheet)
        print(f"[ModularQA] Saved 3x3 QA contact sheet -> {out_sheet}")


if __name__ == "__main__":
    main()
