"""Lightweight Secondary Motion (Spring-Damper / Inertia) Physics Solver for 3DGS Skeletons.

Animates dynamic bone chains (hair, cape, skirt, ribbon, tail, ears, wingtips, accessories)
driven by parent world motion, gravity, and optional wind, without ever stretching bone lengths.

Per-bone optional schema inside `rig_skeleton.json` (`bones[name]["dynamics"]`):
    {
        "stiffness": 18.0,      # spring restore strength (higher = stiffer / faster snap)
        "damping": 0.25,        # velocity damping ratio (0..1)
        "gravity": 0.25,        # downward gravity pull in world -Z
        "wind_scale": 1.0,      # multiplier for external wind vector
        "inertia": 1.0,         # how strongly parent motion lags the tip
        "max_angle_deg": 35.0   # maximum swing deflection from rest direction
    }
"""
from __future__ import annotations

import math
from typing import Callable
import numpy as np
from scipy.spatial.transform import Rotation

from harmonic_dqs import build_fk_hierarchy, compute_fk_transforms


DEFAULT_DYNAMICS = {
    "stiffness": 22.0,
    "damping": 0.28,
    "gravity": 0.18,
    "wind_scale": 1.0,
    "inertia": 1.0,
    "max_angle_deg": 32.0,
}


def _swing_rotation_matrix(u_from: np.ndarray, u_to: np.ndarray, max_angle_deg: float = 45.0) -> np.ndarray:
    """Shortest-arc rotation matrix mapping unit vector u_from -> u_to, clamped to max_angle_deg."""
    a = u_from / max(float(np.linalg.norm(u_from)), 1e-8)
    b = u_to / max(float(np.linalg.norm(u_to)), 1e-8)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = math.acos(dot)
    max_rad = math.radians(max_angle_deg)
    if angle < 1e-6:
        return np.eye(3, dtype=np.float32)
    axis = np.cross(a, b)
    n_ax = float(np.linalg.norm(axis))
    if n_ax < 1e-6:
        # Antipodal fallback
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(a, ortho)
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
    else:
        axis /= n_ax
    clamped = min(angle, max_rad)
    return Rotation.from_rotvec(axis * clamped).as_matrix().astype(np.float32)


def _clamp_rotation_to_limits(R: np.ndarray, limits_deg: dict | None) -> np.ndarray:
    """Optionally clamp a local rotation matrix to per-axis Euler XYZ limits_deg."""
    if not limits_deg:
        return R
    ex, ey, ez = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
    if "x" in limits_deg:
        ex = float(np.clip(ex, limits_deg["x"][0], limits_deg["x"][1]))
    if "y" in limits_deg:
        ey = float(np.clip(ey, limits_deg["y"][0], limits_deg["y"][1]))
    if "z" in limits_deg:
        ez = float(np.clip(ez, limits_deg["z"][0], limits_deg["z"][1]))
    return Rotation.from_euler("xyz", [ex, ey, ez], degrees=True).as_matrix().astype(np.float32)


class SecondaryMotionSolver:
    """Hierarchical tip-position spring-damper solver for secondary-motion bone chains."""

    def __init__(
        self,
        bones: dict,
        order: list[str] | None = None,
        dynamic_bones: list[str] | None = None,
        substeps: int = 4,
    ):
        self.bones = bones
        self.order = order if order is not None else build_fk_hierarchy(bones)
        self.substeps = max(1, int(substeps))

        if dynamic_bones is not None:
            dyn_set = set(dynamic_bones)
            self.dynamic_bones = [b for b in self.order if b in dyn_set and b in bones]
        else:
            self.dynamic_bones = [
                b for b in self.order
                if isinstance(bones[b].get("dynamics"), dict) and bones[b]["dynamics"].get("enabled", True)
            ]

        self._tip_pos: dict[str, np.ndarray] = {}
        self._tip_vel: dict[str, np.ndarray] = {}
        self._rest_vec: dict[str, np.ndarray] = {}
        self._bone_len: dict[str, float] = {}
        self._params: dict[str, dict] = {}

        for bname in self.dynamic_bones:
            b = bones[bname]
            h = np.asarray(b["head"], dtype=np.float32)
            t = np.asarray(b["tail"], dtype=np.float32)
            v = t - h
            blen = max(float(np.linalg.norm(v)), 1e-4)
            self._rest_vec[bname] = v
            self._bone_len[bname] = blen
            p = dict(DEFAULT_DYNAMICS)
            if isinstance(b.get("dynamics"), dict):
                p.update(b["dynamics"])
            self._params[bname] = p

        self.reset()

    def reset(
        self,
        local_rots: dict[str, np.ndarray] | None = None,
        root_shift: np.ndarray | None = None,
    ) -> None:
        """Initialize tip positions to the rest (or given initial) pose with zero velocity."""
        world = compute_fk_transforms(self.bones, self.order, local_rots or {}, root_shift=root_shift)
        for bname in self.dynamic_bones:
            R_w, h_w, _ = world[bname]
            self._tip_pos[bname] = (h_w + R_w @ self._rest_vec[bname]).astype(np.float32)
            self._tip_vel[bname] = np.zeros(3, dtype=np.float32)

    def step(
        self,
        local_rots: dict[str, np.ndarray],
        root_shift: np.ndarray | None = None,
        dt: float = 1.0 / 12.0,
        wind: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Advance secondary motion by dt seconds and return (updated_local_rots, updated_world)."""
        out_rots = {k: v.copy() for k, v in local_rots.items()}
        if not self.dynamic_bones:
            return out_rots, compute_fk_transforms(self.bones, self.order, out_rots, root_shift=root_shift)

        wind_vec = np.asarray(wind, dtype=np.float32) if wind is not None else np.zeros(3, dtype=np.float32)
        sub_dt = float(dt) / float(self.substeps)

        # Compute FK incrementally in topological order so parent dynamic bones drive their children
        world: dict = {}
        for bname in self.order:
            b = self.bones[bname]
            h_rest = np.asarray(b["head"], dtype=np.float32)
            parent = b.get("parent")
            R_loc_base = out_rots.get(bname, np.eye(3, dtype=np.float32))

            if parent is None or parent not in world:
                R_p = np.eye(3, dtype=np.float32)
                h_w = h_rest + (root_shift if root_shift is not None else np.zeros(3, dtype=np.float32))
            else:
                R_p, h_p, _ = world[parent]
                h_p_rest = np.asarray(self.bones[parent]["head"], dtype=np.float32)
                h_w = h_p + R_p @ (h_rest - h_p_rest)

            if bname in self._params:
                p = self._params[bname]
                blen = self._bone_len[bname]
                rest_v = self._rest_vec[bname]
                # Target tip in world space (including any keyframed base rotation on this bone)
                R_base_w = (R_p @ R_loc_base).astype(np.float32)
                target_tip = h_w + R_base_w @ rest_v

                pos = self._tip_pos[bname].copy()
                vel = self._tip_vel[bname].copy()

                stiff = float(p["stiffness"])
                damp = float(p["damping"])
                grav = float(p["gravity"])
                w_scale = float(p["wind_scale"])
                inertia = float(np.clip(p.get("inertia", 1.0), 0.0, 2.0))
                max_ang = float(p["max_angle_deg"])

                # Blend towards target when inertia < 1.0
                if inertia < 1.0:
                    pos = (1.0 - inertia) * target_tip + inertia * pos

                ext_accel = np.array([0.0, 0.0, -grav], dtype=np.float32) + wind_vec * w_scale
                # Critical-damping scaled coefficient
                c_damp = 2.0 * math.sqrt(max(stiff, 0.1)) * float(np.clip(damp, 0.05, 1.5))

                for _ in range(self.substeps):
                    acc = stiff * (target_tip - pos) + ext_accel - c_damp * vel
                    vel = vel + acc * sub_dt
                    pos = pos + vel * sub_dt
                    # Project onto rigid-length sphere around head_w
                    dv = pos - h_w
                    d_norm = float(np.linalg.norm(dv))
                    if d_norm > 1e-6:
                        pos_proj = h_w + dv * (blen / d_norm)
                        # Remove radial velocity component along bone axis
                        u_dir = dv / d_norm
                        vel = vel - float(np.dot(vel, u_dir)) * u_dir
                        pos = pos_proj

                self._tip_pos[bname] = pos
                self._tip_vel[bname] = vel

                # Convert world tip direction back into bone's base-rotated local space
                dir_base_local = R_base_w.T @ (pos - h_w)
                R_dyn = _swing_rotation_matrix(rest_v, dir_base_local, max_angle_deg=max_ang)
                R_loc = (R_loc_base @ R_dyn).astype(np.float32)
                if b.get("limits_deg"):
                    R_loc = _clamp_rotation_to_limits(R_loc, b["limits_deg"])
                out_rots[bname] = R_loc

                # Re-sync projected tip with clamped rotation so velocity doesn't fight limits
                R_w = (R_p @ R_loc).astype(np.float32)
                self._tip_pos[bname] = (h_w + R_w @ rest_v).astype(np.float32)
            else:
                R_w = (R_p @ R_loc_base).astype(np.float32)

            t_w = (h_w - R_w @ h_rest).astype(np.float32)
            world[bname] = (R_w, h_w, t_w)

        return out_rots, world

    def simulate_sequence(
        self,
        frames_local_rots: list[dict[str, np.ndarray]],
        frames_root_shifts: list[np.ndarray | None] | None = None,
        dt: float = 1.0 / 12.0,
        wind_fn: Callable[[int, float], np.ndarray] | None = None,
        loop_warmup: bool = False,
    ) -> tuple[list[dict[str, np.ndarray]], list[dict]]:
        """Simulate secondary motion over an entire animation sequence."""
        n = len(frames_local_rots)
        shifts = frames_root_shifts if frames_root_shifts is not None else [None] * n
        if n == 0:
            return [], []

        self.reset(frames_local_rots[0], shifts[0])
        if loop_warmup and n > 1:
            for fi in range(n):
                w = wind_fn(fi, fi * dt) if wind_fn else None
                self.step(frames_local_rots[fi], shifts[fi], dt=dt, wind=w)

        out_rots_seq: list[dict[str, np.ndarray]] = []
        out_world_seq: list[dict] = []
        for fi in range(n):
            w = wind_fn(fi, fi * dt) if wind_fn else None
            rots_f, world_f = self.step(frames_local_rots[fi], shifts[fi], dt=dt, wind=w)
            out_rots_seq.append(rots_f)
            out_world_seq.append(world_f)
        return out_rots_seq, out_world_seq
