"""Procedural Locomotion & Biomechanical Flight Engine for 3DGS Skeletons (Schema v2.1).

Provides reusable, character-agnostic motion synthesis:
  1. Analytical 2-Bone IK with Pole Vector (`solve_two_bone_ik`):
     Exact law-of-cosines solver compatible with unified rest_quat=[1,0,0,0] skeletons.
     Guarantees ground foot-planting even while the torso/pelvis bobs and flexes.
  2. Duty-Factor Quadruped Gait Generator (`quadruped_gait_frame`):
     Explicit Stance (planted push-back) vs Swing (arcing forward step) timing with
     WALK (beta=0.70), TROT (beta=0.50), and ROTARY GALLOP (beta=0.36) presets,
     plus CoM weight-shift, spinal flexion, and head spatial stabilization.
  3. Biomechanical Avian Wingbeat Generator (`avian_wingbeat_frame`):
     Asymmetric Downstroke Ratio (DSR) time-warping, 2nd-harmonic power/recovery wave,
     root-to-tip phase lag across wing_1/2/3, upstroke folding, pitch feathering, and
     synchronized body lift recoil.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

from harmonic_dqs import build_fk_hierarchy, compute_fk_transforms, euler_xyz


GAIT_PRESETS: dict[str, dict[str, Any]] = {
    "walk": {
        "duty_factor": 0.66,
        "phases": {"hind_L": 0.00, "front_L": 0.25, "hind_R": 0.50, "front_R": 0.75},
        "stride_frac": 0.36,      # fraction of leg length
        "step_height_frac": 0.18,
        "body_bob_frac": 0.036,
        "lateral_sway_frac": 0.030,
        "spine_pitch_deg": 3.0,
        "spine_yaw_deg": 6.5,
        "body_roll_deg": 3.5,
        "suspension_frac": 0.0,
        "neck_forward_deg": 8.0,
        "tail_lift_deg": 18.0,
        "tail_sway_deg": 12.0,
        "ear_sweep_deg": 4.0,
    },
    "trot": {
        "duty_factor": 0.50,
        "phases": {"front_L": 0.00, "hind_R": 0.00, "front_R": 0.50, "hind_L": 0.50},
        "stride_frac": 0.48,
        "step_height_frac": 0.24,
        "body_bob_frac": 0.055,
        "lateral_sway_frac": 0.015,
        "spine_pitch_deg": 4.5,
        "spine_yaw_deg": 4.5,
        "body_roll_deg": 2.2,
        "suspension_frac": 0.025,
        "neck_forward_deg": 14.0,
        "tail_lift_deg": 24.0,
        "tail_sway_deg": 10.0,
        "ear_sweep_deg": 8.0,
    },
    "gallop": {
        "duty_factor": 0.36,
        "phases": {"hind_L": 0.00, "hind_R": 0.18, "front_L": 0.52, "front_R": 0.72},
        "stride_frac": 0.64,
        "step_height_frac": 0.30,
        "body_bob_frac": 0.068,
        "lateral_sway_frac": 0.012,
        "spine_pitch_deg": 12.5,
        "spine_yaw_deg": 3.0,
        "body_roll_deg": 2.5,
        "suspension_frac": 0.13,
        "neck_forward_deg": 22.0,
        "tail_lift_deg": 28.0,
        "tail_sway_deg": 6.0,
        "ear_sweep_deg": 14.0,
    },
}


def _swing_rot(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Shortest-arc 3x3 rotation matrix mapping v_from direction -> v_to direction."""
    a = np.asarray(v_from, dtype=np.float32)
    b = np.asarray(v_to, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-7 or nb < 1e-7:
        return np.eye(3, dtype=np.float32)
    a = a / na
    b = b / nb
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return np.eye(3, dtype=np.float32)
    axis = np.cross(a, b)
    n_ax = float(np.linalg.norm(axis))
    if n_ax < 1e-6:
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(a, ortho)
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        angle = math.pi
    else:
        axis /= n_ax
        angle = math.acos(dot)
    return Rotation.from_rotvec(axis * angle).as_matrix().astype(np.float32)


def solve_two_bone_ik(
    p_root_w: np.ndarray,
    R_parent_w: np.ndarray,
    v1_rest: np.ndarray,
    v2_rest: np.ndarray,
    p_target_w: np.ndarray,
    pole_dir_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical 2-Bone IK returning (R_local_upper, R_local_lower) in unified bone-axis convention.

    Satisfies:
      p_mid_w = p_root_w + (R_parent_w @ R_local_upper) @ v1_rest
      p_end_w = p_mid_w  + (R_parent_w @ R_local_upper @ R_local_lower) @ v2_rest == p_target_w (within reach)
    """
    L1 = max(float(np.linalg.norm(v1_rest)), 1e-4)
    L2 = max(float(np.linalg.norm(v2_rest)), 1e-4)

    D = np.asarray(p_target_w, dtype=np.float32) - np.asarray(p_root_w, dtype=np.float32)
    d_raw = float(np.linalg.norm(D))
    if d_raw < 1e-6:
        return np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)

    u = D / d_raw
    d_min = abs(L1 - L2) + 1e-3
    d_max = (L1 + L2) * 0.995
    d = float(np.clip(d_raw, d_min, d_max))

    # Build bend plane orthogonal unit vector e_perp aligned with pole_dir_w
    pole = np.asarray(pole_dir_w, dtype=np.float32)
    e_perp = pole - float(np.dot(pole, u)) * u
    n_ep = float(np.linalg.norm(e_perp))
    if n_ep < 1e-5:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(u[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        e_perp = fallback - float(np.dot(fallback, u)) * u
        n_ep = max(float(np.linalg.norm(e_perp)), 1e-6)
    e_perp = e_perp / n_ep

    cos_a = float(np.clip((L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d), -1.0, 1.0))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))

    d1_w = L1 * (cos_a * u + sin_a * e_perp)
    p_mid_w = np.asarray(p_root_w, dtype=np.float32) + d1_w
    p_eff_target = np.asarray(p_root_w, dtype=np.float32) + d * u
    d2_w = p_eff_target - p_mid_w

    R_loc_1 = _swing_rot(v1_rest, R_parent_w.T @ d1_w)
    R_w_1 = (R_parent_w @ R_loc_1).astype(np.float32)
    R_loc_2 = _swing_rot(v2_rest, R_w_1.T @ d2_w)
    return R_loc_1, R_loc_2


def sample_foot_trajectory(
    phase: float,
    duty_factor: float,
    stride: float,
    step_height: float,
) -> tuple[float, float, float, bool]:
    """Evaluate Stance/Swing foot offset (dy_forward_neg, dz_up, paw_pitch_deg, is_stance).

    In Blender -Y-forward convention:
      - Negative dy = forward reach (-Y)
      - Positive dy = backward push-off (+Y)
    """
    phi = float(phase) % 1.0
    beta = float(np.clip(duty_factor, 0.15, 0.85))
    if phi < beta:
        # Stance phase: foot planted flat on ground, sweeping backward linearly (-0.5*stride -> +0.5*stride)
        u = phi / beta
        dy = -0.5 * stride + u * stride
        dz = 0.0
        # Subtle toe-off push in the last 25% of stance
        push = max(0.0, (u - 0.72) / 0.28)
        paw_pitch = 16.0 * (push * push)
        return dy, dz, paw_pitch, True
    else:
        # Swing phase: smooth cubic ease back forward with parabolic/sine lift arc
        u = (phi - beta) / (1.0 - beta)
        s = u * u * (3.0 - 2.0 * u)
        dy = 0.5 * stride - s * stride
        dz = step_height * (math.sin(math.pi * u) ** 1.15)
        # Relaxed downward paw flex mid-swing, flattening before touchdown
        paw_pitch = -20.0 * math.sin(math.pi * u)
        return dy, dz, paw_pitch, False


def quadruped_gait_frame(
    bones: dict,
    order: list[str],
    cycle_phase: float,
    gait: str = "walk",
    wing_tuck: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate one biomechanically grounded quadruped locomotion frame via 2-Bone IK.

    Returns (local_rots, root_shift).
    """
    cfg = GAIT_PRESETS.get(gait, GAIT_PRESETS["walk"])
    t = float(cycle_phase) % 1.0
    ph_rad = 2.0 * math.pi * t
    beta = float(cfg["duty_factor"])

    # Measure average leg length from rest skeleton so stride/bob auto-scale to any creature size
    leg_lens = []
    for prefix in ("front", "hind"):
        for s in ("L", "R"):
            u_b, l_b = f"{prefix}_upper_{s}", f"{prefix}_lower_{s}"
            if u_b in bones and l_b in bones:
                l1 = float(np.linalg.norm(np.asarray(bones[u_b]["tail"]) - np.asarray(bones[u_b]["head"])))
                l2 = float(np.linalg.norm(np.asarray(bones[l_b]["tail"]) - np.asarray(bones[l_b]["head"])))
                leg_lens.append(l1 + l2)
    avg_leg_len = float(np.mean(leg_lens)) if leg_lens else 0.42

    stride = avg_leg_len * float(cfg["stride_frac"])
    step_h = avg_leg_len * float(cfg["step_height_frac"])
    bob_amp = avg_leg_len * float(cfg["body_bob_frac"])
    sway_amp = avg_leg_len * float(cfg["lateral_sway_frac"])
    susp_amp = avg_leg_len * float(cfg["suspension_frac"])

    # 1. Spine / Pelvis / Chest CoM motion + Head Stabilization
    rots: dict[str, np.ndarray] = {}
    if gait == "gallop":
        # Bounding spinal flexion (pelvis & chest curl inward during gather, extend during airborne leap)
        flex = math.cos(ph_rad)
        pelvis_pitch = float(cfg["spine_pitch_deg"]) * flex
        spine_pitch = -float(cfg["spine_pitch_deg"]) * 1.15 * flex
        chest_pitch = -float(cfg["spine_pitch_deg"]) * 0.55 * flex
        spine_yaw = float(cfg["spine_yaw_deg"]) * math.sin(ph_rad)
        body_roll = float(cfg["body_roll_deg"]) * math.sin(ph_rad)
        z_bob = -bob_amp * 0.45 * flex + susp_amp * max(0.0, -math.sin(ph_rad - 0.35))
        x_sway = sway_amp * math.sin(ph_rad)
    else:
        # Walk / Trot: 2x vertical bob per stride, 1x lateral weight shift & yaw
        pelvis_pitch = float(cfg["spine_pitch_deg"]) * 0.5 * math.sin(2.0 * ph_rad)
        spine_pitch = -float(cfg["spine_pitch_deg"]) * math.sin(2.0 * ph_rad)
        chest_pitch = float(cfg["spine_pitch_deg"]) * 0.5 * math.sin(2.0 * ph_rad)
        spine_yaw = float(cfg["spine_yaw_deg"]) * math.sin(ph_rad)
        body_roll = float(cfg["body_roll_deg"]) * math.cos(ph_rad)
        z_bob = -bob_amp * (0.5 - 0.5 * math.cos(2.0 * ph_rad)) + susp_amp * max(0.0, math.sin(2.0 * ph_rad))
        x_sway = sway_amp * math.sin(ph_rad)

    if "pelvis" in bones:
        rots["pelvis"] = euler_xyz(rx_deg=pelvis_pitch, ry_deg=body_roll, rz_deg=-0.45 * spine_yaw)
    if "spine_mid" in bones:
        rots["spine_mid"] = euler_xyz(rx_deg=spine_pitch, ry_deg=-0.4 * body_roll, rz_deg=spine_yaw)
    if "chest" in bones:
        rots["chest"] = euler_xyz(rx_deg=chest_pitch, ry_deg=-0.5 * body_roll, rz_deg=0.45 * spine_yaw)

    # Head spatial stabilization + canine/fox forward neck reach (neck lowers forward, head stays level on horizon)
    cum_pitch = pelvis_pitch + spine_pitch + chest_pitch
    cum_yaw = 0.55 * spine_yaw
    neck_fwd = float(cfg.get("neck_forward_deg", 10.0))
    if "neck" in bones:
        rots["neck"] = euler_xyz(rx_deg=neck_fwd - 0.50 * cum_pitch, rz_deg=-0.45 * cum_yaw)
    if "head" in bones:
        rots["head"] = euler_xyz(rx_deg=-0.82 * neck_fwd - 0.35 * cum_pitch, rz_deg=-0.35 * cum_yaw)

    # Canine/fox ear sweep-back during brisk trot/gallop
    ear_sweep = float(cfg.get("ear_sweep_deg", 0.0))
    if ear_sweep > 0.0:
        for s, sgn in (("L", +1.0), ("R", -1.0)):
            if f"ear_{s}_1" in bones:
                rots[f"ear_{s}_1"] = euler_xyz(rx_deg=-ear_sweep, ry_deg=sgn * (ear_sweep * 0.35))

    # Active canine/fox tail counterbalance: lifts tail clear of hind legs + graceful phase-lagged wag!
    # In right-handed Z-up (+X left, -Y forward): for a +Y-pointing tail, z' = +y*sin(rx), so Rx(+) lifts UP!
    tail_lift = float(cfg.get("tail_lift_deg", 18.0))
    tail_sway = float(cfg.get("tail_sway_deg", 10.0))
    if "tail_1" in bones:
        tail_pump = 5.5 * math.cos(ph_rad) if gait == "gallop" else 2.5 * math.sin(2.0 * ph_rad)
        rots["tail_1"] = euler_xyz(rx_deg=+(tail_lift + tail_pump), rz_deg=tail_sway * math.sin(ph_rad - 0.5))
    if "tail_2" in bones:
        rots["tail_2"] = euler_xyz(rx_deg=+0.35 * tail_lift, rz_deg=tail_sway * 0.8 * math.sin(ph_rad - 1.1))
    if "tail_3" in bones:
        rots["tail_3"] = euler_xyz(rx_deg=-0.15 * tail_lift, rz_deg=tail_sway * 0.6 * math.sin(ph_rad - 1.7))

    root_shift = np.array([x_sway, 0.0, z_bob], dtype=np.float32)

    # 2. Compute torso world transforms first so 2-Bone IK solves exact world foot placement!
    world_torso = compute_fk_transforms(bones, order, rots, root_shift=root_shift)

    # 3. Solve Analytical 2-Bone IK for all 4 legs
    leg_defs = [
        ("front_L", "front_upper_L", "front_lower_L", "front_paw_L", "chest", np.array([0.0, +1.0, 0.0], np.float32)),
        ("front_R", "front_upper_R", "front_lower_R", "front_paw_R", "chest", np.array([0.0, +1.0, 0.0], np.float32)),
        ("hind_L",  "hind_upper_L",  "hind_lower_L",  "hind_paw_L",  "pelvis", np.array([0.0, -1.0, 0.0], np.float32)),
        ("hind_R",  "hind_upper_R",  "hind_lower_R",  "hind_paw_R",  "pelvis", np.array([0.0, -1.0, 0.0], np.float32)),
    ]

    for key, b_up, b_lo, b_paw, b_par, pole_dir in leg_defs:
        if b_up not in bones or b_lo not in bones or b_par not in world_torso:
            continue
        p_off = float(cfg["phases"][key])
        dy, dz, paw_pitch, _is_stance = sample_foot_trajectory(t + p_off, beta, stride, step_h)

        # Rest ankle position in world space (ground-anchored baseline)
        h_up_r = np.asarray(bones[b_up]["head"], dtype=np.float32)
        t_up_r = np.asarray(bones[b_up]["tail"], dtype=np.float32)
        h_lo_r = np.asarray(bones[b_lo]["head"], dtype=np.float32)
        t_lo_r = np.asarray(bones[b_lo]["tail"], dtype=np.float32)
        v1_rest = t_up_r - h_up_r
        v2_rest = t_lo_r - h_lo_r

        # Current shoulder/hip world position under moving torso
        R_par_w, h_par_w, _ = world_torso[b_par]
        h_par_r = np.asarray(bones[b_par]["head"], dtype=np.float32)
        p_root_w = h_par_w + R_par_w @ (h_up_r - h_par_r)

        # Target ankle world position: anchored at rest ground Z + swing lift dz (foot stays planted on ground!)
        p_target_w = np.array([
            float(t_lo_r[0]),
            float(t_lo_r[1]) + dy,
            float(t_lo_r[2]) + dz + (susp_amp * max(0.0, -math.sin(ph_rad - 0.35)) if gait == "gallop" else 0.0),
        ], dtype=np.float32)

        R_up, R_lo = solve_two_bone_ik(p_root_w, R_par_w, v1_rest, v2_rest, p_target_w, pole_dir)
        rots[b_up] = R_up
        rots[b_lo] = R_lo
        if b_paw in bones:
            rots[b_paw] = euler_xyz(rx_deg=paw_pitch)

    # 4. Optional calm folded wings while walking/running
    if wing_tuck and any(b.startswith(("wing_L_", "wing_R_")) for b in bones):
        tuck_z = 16.0 if gait == "gallop" else 10.0
        flutter = 4.0 * math.sin(ph_rad)
        for s, sgn in (("L", +1.0), ("R", -1.0)):
            if f"wing_{s}_1" in bones:
                rots[f"wing_{s}_1"] = euler_xyz(ry_deg=sgn * (-6.0 + flutter), rz_deg=sgn * tuck_z)
            if f"wing_{s}_2" in bones:
                rots[f"wing_{s}_2"] = euler_xyz(rz_deg=sgn * (tuck_z * 0.75))

    return rots, root_shift


def _warp_avian_phase(t_cycle: float, dsr: float = 0.44) -> float:
    """Time-warp normalized cycle t in [0, 1) according to Downstroke Ratio (DSR).

    Returns warped angle theta in [0, 2*pi) where [0, pi) is Power Downstroke and [pi, 2*pi) is Recovery Upstroke.
    """
    tc = float(t_cycle) % 1.0
    dsr = float(np.clip(dsr, 0.25, 0.75))
    if tc < dsr:
        pn = (tc / dsr) * 0.5
    else:
        pn = 0.5 + ((tc - dsr) / (1.0 - dsr)) * 0.5
    return pn * 2.0 * math.pi


def avian_wingbeat_frame(
    bones: dict,
    cycle_phase: float,
    dsr: float = 0.44,
    amplitude_deg: float = 34.0,
    hover_base_z: float = 0.14,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate one biomechanically grounded avian flight frame with stroke asymmetry & phase lag."""
    t = float(cycle_phase) % 1.0
    theta0 = _warp_avian_phase(t, dsr=dsr)
    rots: dict[str, np.ndarray] = {}

    # Multi-segment wing articulation with root-to-tip phase lag
    for s, sgn in (("L", +1.0), ("R", -1.0)):
        # Segment 1: Humerus (shoulder) - primary power downstroke + slight forward sweep
        th1 = _warp_avian_phase(t, dsr=dsr)
        flap1 = amplitude_deg * (math.cos(th1) + 0.18 * math.cos(2.0 * th1 - 0.45))
        # Positive sin(th1) corresponds to active downstroke velocity
        down_vel = max(0.0, math.sin(th1))
        up_vel = max(0.0, -math.sin(th1))
        if f"wing_{s}_1" in bones:
            rots[f"wing_{s}_1"] = euler_xyz(
                rx_deg=10.0 * down_vel - 6.0 * up_vel,
                ry_deg=sgn * (-flap1),
                rz_deg=sgn * (-10.0 * down_vel + 8.0 * up_vel),
            )

        # Segment 2: Radius/Ulna (elbow) - trails by 0.09 cycle, folds inward on upstroke recovery
        th2 = _warp_avian_phase(t - 0.09, dsr=dsr)
        up2 = max(0.0, -math.sin(th2))
        down2 = max(0.0, math.sin(th2))
        if f"wing_{s}_2" in bones:
            rots[f"wing_{s}_2"] = euler_xyz(
                ry_deg=sgn * (-20.0 * up2 + 10.0 * down2),
                rz_deg=sgn * (22.0 * up2),
            )

        # Segment 3: Carpometacarpus / Primaries (wrist) - trails by 0.18 cycle, feathers pitch to slice air
        th3 = _warp_avian_phase(t - 0.18, dsr=dsr)
        up3 = max(0.0, -math.sin(th3))
        down3 = max(0.0, math.sin(th3))
        if f"wing_{s}_3" in bones:
            rots[f"wing_{s}_3"] = euler_xyz(
                rx_deg=-22.0 * up3 + 6.0 * down3,
                ry_deg=sgn * (-14.0 * up3 + 8.0 * down3),
            )

    # Synchronized body recoil: rises sharply during power downstroke, dips slightly on recovery
    recoil = -math.cos(theta0 - 0.45)
    hover_z = hover_base_z + 0.045 * (0.5 + 0.5 * recoil)
    body_pitch = 5.5 * math.sin(theta0)

    if "pelvis" in bones:
        rots["pelvis"] = euler_xyz(rx_deg=-6.0 + 0.5 * body_pitch)
    if "spine_mid" in bones:
        rots["spine_mid"] = euler_xyz(rx_deg=0.4 * body_pitch)
    if "chest" in bones:
        rots["chest"] = euler_xyz(rx_deg=0.6 * body_pitch)
    # Head stabilization & streamlined forward canine gaze during flight
    if "neck" in bones:
        rots["neck"] = euler_xyz(rx_deg=16.0 - 0.7 * body_pitch)
    if "head" in bones:
        rots["head"] = euler_xyz(rx_deg=-13.0 - 0.3 * body_pitch)

    # Lift & trail tail cleanly above tucked hind legs during flight (Rx(+) lifts +Y tail UP!)
    if "tail_1" in bones:
        rots["tail_1"] = euler_xyz(rx_deg=+(26.0 + 4.0 * recoil), rz_deg=4.0 * math.sin(theta0))
    if "tail_2" in bones:
        rots["tail_2"] = euler_xyz(rx_deg=+8.0, rz_deg=4.0 * math.sin(theta0 - 0.6))
    for s, sgn in (("L", +1.0), ("R", -1.0)):
        if f"ear_{s}_1" in bones:
            rots[f"ear_{s}_1"] = euler_xyz(rx_deg=-12.0, ry_deg=sgn * 4.0)

    # Streamlined flight leg tuck (gentle breathing sway synced to wingbeat; hind legs stay below lifted tail)
    if "front_upper_L" in bones and "hind_upper_L" in bones:
        leg_sway = 4.0 * recoil
        for s in ("L", "R"):
            rots[f"front_upper_{s}"] = euler_xyz(rx_deg=24.0 + leg_sway)
            rots[f"front_lower_{s}"] = euler_xyz(rx_deg=-44.0)
            rots[f"front_paw_{s}"] = euler_xyz(rx_deg=18.0)
            rots[f"hind_upper_{s}"] = euler_xyz(rx_deg=22.0 + leg_sway)
            rots[f"hind_lower_{s}"] = euler_xyz(rx_deg=14.0)
            rots[f"hind_paw_{s}"] = euler_xyz(rx_deg=18.0)

    root_shift = np.array([0.0, 0.0, hover_z], dtype=np.float32)
    return rots, root_shift
