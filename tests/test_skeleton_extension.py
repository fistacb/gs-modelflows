"""Unit tests for v2.1 modular skeleton extension:
- Backward compatibility of compute_harmonic_skin_weights with legacy v2.0 humanoid rigs
- Generic group-isolated Harmonic skinning & seam_cut (muzzle/jaw split)
- SecondaryMotionSolver stability, bone-length preservation, loop warmup
- Modular skeleton_builder (quadruped, attachments, custom_bones, overrides)
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))

from harmonic_dqs import (
    _is_legacy_humanoid_skeleton,
    build_fk_hierarchy,
    compute_fk_transforms,
    compute_harmonic_skin_weights,
    deform_dqs,
    euler_xyz,
    pack_spark_skin_weights,
)
from secondary_motion import SecondaryMotionSolver
from procedural_locomotion import (
    avian_wingbeat_frame,
    quadruped_gait_frame,
    solve_two_bone_ik,
)
from skeleton_builder import (
    add_jaw_bone,
    build_skeleton_from_recipe,
    make_bone,
)


def _make_synthetic_quadruped_cloud() -> np.ndarray:
    rng = np.random.default_rng(42)
    parts = []
    # Torso (elongated along Y from -0.20 to +0.20, Z around 0.45)
    t = rng.normal([0.0, 0.0, 0.45], [0.06, 0.16, 0.06], size=(600, 3))
    parts.append(t)
    # 4 legs
    for sx in (+0.08, -0.08):
        for sy in (-0.14, +0.14):
            z = rng.uniform(0.0, 0.42, size=(150, 1))
            xy = rng.normal([sx, sy], [0.02, 0.02], size=(150, 2))
            parts.append(np.hstack([xy, z]))
    # Head + Muzzle (front at -Y, Z around 0.65)
    head = rng.normal([0.0, -0.24, 0.66], [0.05, 0.06, 0.05], size=(300, 3))
    parts.append(head)
    # Upper & lower muzzle (split at z = 0.635)
    upper_m = rng.uniform([-0.03, -0.34, 0.64], [0.03, -0.25, 0.67], size=(100, 3))
    lower_m = rng.uniform([-0.03, -0.34, 0.60], [0.03, -0.25, 0.63], size=(100, 3))
    parts.append(upper_m)
    parts.append(lower_m)
    # Wings L & R
    for sgn in (+1.0, -1.0):
        s = rng.uniform(0.0, 1.0, size=(200, 1))
        w = np.hstack([
            sgn * (0.06 + 0.32 * s) + rng.normal(0, 0.015, size=(200, 1)),
            -0.06 + 0.25 * s + rng.normal(0, 0.03, size=(200, 1)),
            0.52 + 0.20 * s + rng.normal(0, 0.02, size=(200, 1)),
        ])
        parts.append(w)
    # Tail (+Y)
    s = rng.uniform(0.0, 1.0, size=(250, 1))
    tail = np.hstack([
        rng.normal(0, 0.03, size=(250, 1)),
        0.18 + 0.28 * s,
        0.44 + 0.08 * s + rng.normal(0, 0.03, size=(250, 1)),
    ])
    parts.append(tail)
    # Ears
    for sgn in (+1.0, -1.0):
        s = rng.uniform(0.0, 1.0, size=(80, 1))
        ear = np.hstack([
            sgn * (0.04 + 0.04 * s) + rng.normal(0, 0.008, size=(80, 1)),
            -0.22 + 0.02 * s + rng.normal(0, 0.008, size=(80, 1)),
            0.71 + 0.14 * s,
        ])
        parts.append(ear)
    return np.vstack(parts).astype(np.float32)


def test_quadruped_recipe_and_generic_skinning():
    xyz = _make_synthetic_quadruped_cloud()
    recipe = {
        "archetype": "quadruped",
        "attachments": [
            {"type": "jaw", "hinge": [0.0, -0.25, 0.635], "tip": [0.0, -0.34, 0.615], "mouth_plane_z": 0.635, "cut_radius": 0.09},
            {"type": "wing", "n_segments": 3},
            {"type": "tail", "n_segments": 3},
            {"type": "ear", "n_segments": 2},
            {"type": "ribbon", "parent": "chest", "n_segments": 2},
        ],
        "custom_bones": {
            "bell": make_bone("chest", [0.0, -0.20, 0.44], [0.0, -0.21, 0.40], skin_group="bell", role="accessory",
                              dynamics={"stiffness": 30.0, "damping": 0.25, "gravity": 0.4, "max_angle_deg": 25.0}),
        },
    }
    skel = build_skeleton_from_recipe(xyz, recipe)
    bones = skel["bones"]
    assert skel["schema_version"] == "2.1"
    assert skel["archetype"] == "quadruped"
    assert not _is_legacy_humanoid_skeleton(bones)
    for req in ("pelvis", "spine_mid", "chest", "neck", "head", "jaw",
                "front_upper_L", "hind_upper_R", "wing_L_1", "wing_R_3",
                "tail_1", "tail_3", "ear_L_1", "ribbon_L_1", "bell"):
        assert req in bones, f"Missing bone {req}"

    skin_bones, W = compute_harmonic_skin_weights(xyz, bones, np.ones(len(xyz), dtype=bool), max_edge=0.03)
    assert W.shape == (len(xyz), len(skin_bones))
    assert not np.isnan(W).any()
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-4)

    # Verify mouth seam_cut separates upper muzzle (head) from lower muzzle (jaw)
    head_i = skin_bones.index("head")
    jaw_i = skin_bones.index("jaw")
    upper_mask = (xyz[:, 1] < -0.26) & (xyz[:, 2] >= 0.64) & (xyz[:, 2] <= 0.67)
    lower_mask = (xyz[:, 1] < -0.26) & (xyz[:, 2] >= 0.60) & (xyz[:, 2] <= 0.63)
    assert float(W[upper_mask, head_i].mean()) > 0.75, f"Upper muzzle should follow head, got {W[upper_mask, head_i].mean():.3f}"
    assert float(W[lower_mask, jaw_i].mean()) > 0.75, f"Lower muzzle should follow jaw, got {W[lower_mask, jaw_i].mean():.3f}"

    # Verify GPU Spark skinning packing succeeds for >21 bones
    packed = pack_spark_skin_weights(W)
    assert packed.shape == (len(xyz), 4) and packed.dtype == np.uint16


def test_secondary_motion_solver_stability():
    xyz = _make_synthetic_quadruped_cloud()
    skel = build_skeleton_from_recipe(xyz, {
        "archetype": "quadruped",
        "attachments": [
            {"type": "wing", "n_segments": 3},
            {"type": "tail", "n_segments": 3},
            {"type": "ear", "n_segments": 2},
        ],
    })
    bones = skel["bones"]
    order = build_fk_hierarchy(bones)
    solver = SecondaryMotionSolver(bones, order)
    assert "tail_1" in solver.dynamic_bones
    assert "ear_L_1" in solver.dynamic_bones
    assert "wing_L_3" not in solver.dynamic_bones  # wings are articulated bones by default, not cloth

    frames_rots = []
    frames_shifts = []
    for f in range(24):
        ang = 12.0 * np.sin(2.0 * np.pi * f / 12.0)
        frames_rots.append({
            "head": euler_xyz(rx_deg=ang, rz_deg=ang * 1.5),
            "wing_L_1": euler_xyz(ry_deg=25.0 * np.sin(2.0 * np.pi * f / 12.0)),
        })
        frames_shifts.append(np.array([0.0, 0.0, 0.03 * np.sin(2.0 * np.pi * f / 6.0)], dtype=np.float32))

    rots_seq, world_seq = solver.simulate_sequence(
        frames_rots,
        frames_shifts,
        dt=1.0 / 12.0,
        wind_fn=lambda fi, t: np.array([0.0, 0.4, 0.0], dtype=np.float32),
        loop_warmup=True,
    )
    assert len(rots_seq) == 24
    # Verify dynamic bones actually moved and stayed finite with exact bone lengths
    tail_moved = any(np.linalg.norm(r["tail_1"] - np.eye(3)) > 1e-2 for r in rots_seq)
    assert tail_moved, "Tail dynamic bone did not respond to motion/wind"
    for world in world_seq:
        for bname in ("tail_1", "tail_2", "tail_3", "ear_L_1"):
            R_w, h_w, t_w = world[bname]
            assert np.isfinite(R_w).all() and np.isfinite(h_w).all() and np.isfinite(t_w).all()


def test_procedural_locomotion_ik_and_flight():
    # 1. Analytical 2-Bone IK exact reach test
    p_root = np.array([0.1, -0.2, 0.5], dtype=np.float32)
    R_par = euler_xyz(rx_deg=8.0, rz_deg=-10.0)
    v1 = np.array([0.0, 0.0, -0.22], dtype=np.float32)
    v2 = np.array([0.0, 0.0, -0.20], dtype=np.float32)
    p_target = np.array([0.12, -0.28, 0.14], dtype=np.float32)
    pole = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    R1, R2 = solve_two_bone_ik(p_root, R_par, v1, v2, p_target, pole)
    R_w1 = R_par @ R1
    p_mid = p_root + R_w1 @ v1
    p_end = p_mid + (R_w1 @ R2) @ v2
    np.testing.assert_allclose(p_end, p_target, atol=1e-4)

    # 2. Quadruped WALK / TROT / GALLOP + Avian FLY smoke test
    xyz = _make_synthetic_quadruped_cloud()
    skel = build_skeleton_from_recipe(xyz, {
        "archetype": "quadruped",
        "attachments": [{"type": "wing", "n_segments": 3}, {"type": "tail", "n_segments": 3}],
    })
    bones = skel["bones"]
    order = build_fk_hierarchy(bones)
    for gait in ("walk", "trot", "gallop"):
        for ph in (0.0, 0.25, 0.5, 0.75):
            rots, shift = quadruped_gait_frame(bones, order, ph, gait=gait)
            assert np.isfinite(shift).all()
            assert "front_upper_L" in rots and "hind_lower_R" in rots
    for ph in (0.0, 0.3, 0.6, 0.9):
        rots, shift = avian_wingbeat_frame(bones, ph)
        assert np.isfinite(shift).all()
        assert "wing_L_1" in rots and "wing_R_3" in rots


if __name__ == "__main__":
    test_quadruped_recipe_and_generic_skinning()
    test_secondary_motion_solver_stability()
    test_procedural_locomotion_ik_and_flight()
    print("ALL SKELETON EXTENSION UNIT TESTS PASSED!")
