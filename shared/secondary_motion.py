"""Lightweight Secondary Motion (Spring-Damper / Inertia) Physics Solver for 3DGS Skeletons.

Animates dynamic bone chains (hair, cape, skirt, ribbon, tail, ears, wingtips, accessories)
driven by parent world motion and subtle wind, using bone-length-normalized angular dynamics
so short and long bones behave with physically consistent, material-aware stiffness.

Material presets (`DYNAMICS_PRESETS`):
  - ear      : stiff cartilage, stands upright, subtle tip lag only (3-8 deg)
  - wing_tip : feather flex on flap rebound (10 deg)
  - tail     : heavy plush mass, heavily damped gentle sway (5-14 deg)
  - cape     : medium-weight cloth, anchored root, soft hem flutter (6-14 deg)
  - hair     : supple strands, smooth damped settle (7-15 deg)
  - ribbon   : light fabric strip, liveliest response (10-18 deg)

Per-bone optional schema inside `rig_skeleton.json` (`bones[name]["dynamics"]`):
    {
        "preset": "tail",       # optional preset name merged as defaults
        "stiffness": 30.0,      # angular spring stiffness (omega_n^2, 1/s^2)
        "damping": 0.70,        # damping ratio zeta (0.5..0.9 = smooth, no wild bounce)
        "gravity": 0.0,         # extra gravity pull (0.0 = rest pose is already gravity-balanced!)
        "wind_scale": 0.35,     # susceptibility to external wind
        "inertia": 0.55,        # 0=rigid follow parent, 1=full inertial lag
        "max_angle_deg": 14.0   # hard angular deflection cap from rest/keyframed direction
    }
"""
from __future__ import annotations

import math
from typing import Callable
import numpy as np
from scipy.spatial.transform import Rotation

from harmonic_dqs import build_fk_hierarchy, compute_fk_transforms


DYNAMICS_PRESETS: dict[str, dict[str, float]] = {
    "ear": {
        "stiffness": 70.0,
        "damping": 0.75,
        "gravity": 0.0,
        "wind_scale": 0.12,
        "inertia": 0.28,
        "max_angle_deg": 7.0,
    },
    "wing_tip": {
        "stiffness": 52.0,
        "damping": 0.70,
        "gravity": 0.0,
        "wind_scale": 0.22,
        "inertia": 0.42,
        "max_angle_deg": 11.0,
    },
    "tail": {
        "stiffness": 28.0,
        "damping": 0.76,
        "gravity": 0.0,
        "wind_scale": 0.28,
        "inertia": 0.52,
        "max_angle_deg": 11.0,
    },
    "cape": {
        "stiffness": 34.0,
        "damping": 0.70,
        "gravity": 0.0,
        "wind_scale": 0.42,
        "inertia": 0.50,
        "max_angle_deg": 12.0,
    },
    "hair": {
        "stiffness": 36.0,
        "damping": 0.68,
        "gravity": 0.0,
        "wind_scale": 0.45,
        "inertia": 0.55,
        "max_angle_deg": 13.0,
    },
    "ribbon": {
        "stiffness": 24.0,
        "damping": 0.58,
        "gravity": 0.0,
        "wind_scale": 0.75,
        "inertia": 0.68,
        "max_angle_deg": 16.0,
    },
    "accessory": {
        "stiffness": 42.0,
        "damping": 0.68,
        "gravity": 0.0,
        "wind_scale": 0.25,
        "inertia": 0.50,
        "max_angle_deg": 12.0,
    },
}

DEFAULT_DYNAMICS: dict[str, float] = {
    "stiffness": 36.0,
    "damping": 0.70,
    "gravity": 0.0,
    "wind_scale": 0.35,
    "inertia": 0.50,
    "max_angle_deg": 12.0,
}


def _swing_rotation_matrix(u_from: np.ndarray, u_to: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
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
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = np.cross(a, ortho)
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
    else:
        axis /= n_ax
    clamped = min(angle, max_rad)
    return Rotation.from_rotvec(axis * clamped).as_matrix().astype(np.float32)


def _clamp_rotation_to_limits(R: np.ndarray, limits_deg: dict | None) -> np.ndarray:
    """Clamp a local rotation matrix to per-axis Euler XYZ limits_deg."""
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
    """Hierarchical bone-length-normalized spring-damper solver for secondary-motion chains."""

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
            raw_dyn = b.get("dynamics") if isinstance(b.get("dynamics"), dict) else {}
            preset_name = raw_dyn.get("preset") or b.get("role")
            if preset_name in DYNAMICS_PRESETS:
                p.update(DYNAMICS_PRESETS[str(preset_name)])
            p.update(raw_dyn)
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
                R_base_w = (R_p @ R_loc_base).astype(np.float32)
                target_tip = h_w + R_base_w @ rest_v

                pos = self._tip_pos[bname].copy()
                vel = self._tip_vel[bname].copy()

                stiff = float(p["stiffness"])
                damp = float(p["damping"])
                grav = float(p.get("gravity", 0.0))
                w_scale = float(p.get("wind_scale", 0.35))
                inertia = float(np.clip(p.get("inertia", 0.50), 0.0, 1.0))
                max_ang = float(p.get("max_angle_deg", 12.0))

                # Blend previous world tip with target_tip according to inertia (0=rigid, 1=full lag)
                pos = (1.0 - inertia) * target_tip + inertia * pos
                vel = inertia * vel

                # Scale external accelerations by bone length so angular deflection is scale-invariant!
                # At equilibrium, tan(theta) ~= |ext_accel| / stiffness.
                ext_accel = (np.array([0.0, 0.0, -grav], dtype=np.float32) + wind_vec * w_scale) * blen
                c_damp = 2.0 * math.sqrt(max(stiff, 0.1)) * float(np.clip(damp, 0.15, 1.5))

                for _ in range(self.substeps):
                    acc = stiff * (target_tip - pos) + ext_accel - c_damp * vel
                    vel = vel + acc * sub_dt
                    pos = pos + vel * sub_dt
                    dv = pos - h_w
                    d_norm = float(np.linalg.norm(dv))
                    if d_norm > 1e-6:
                        u_dir = dv / d_norm
                        pos = h_w + u_dir * blen
                        vel = vel - float(np.dot(vel, u_dir)) * u_dir

                self._tip_vel[bname] = vel

                dir_base_local = R_base_w.T @ (pos - h_w)
                R_dyn = _swing_rotation_matrix(rest_v, dir_base_local, max_angle_deg=max_ang)
                R_loc = (R_loc_base @ R_dyn).astype(np.float32)
                if b.get("limits_deg"):
                    R_loc = _clamp_rotation_to_limits(R_loc, b["limits_deg"])
                out_rots[bname] = R_loc

                R_w = (R_p @ R_loc).astype(np.float32)
                self._tip_pos[bname] = (h_w + R_w @ rest_v).astype(np.float32)
            else:
                R_w = (R_p @ R_loc_base).astype(np.float32)

            t_w = (h_w - R_w @ h_rest).astype(np.float32)
            world[bname] = (R_w, h_w, t_w)

        # Lateral cloth coupling: bones sharing the same skin_group & role=="cloth" at the same depth
        # gently blend their swing rotations so a multi-strand cape/skirt behaves as ONE continuous sheet!
        cloth_buckets: dict[tuple[str, int], list[str]] = {}
        for bname in self.dynamic_bones:
            b = self.bones[bname]
            if str(b.get("role", "")).lower() == "cloth" and b.get("skin_group"):
                # Compute depth inside dynamic chain
                depth = 0
                cur = b.get("parent")
                while cur in self._params:
                    depth += 1
                    cur = self.bones[cur].get("parent")
                cloth_buckets.setdefault((str(b["skin_group"]), depth), []).append(bname)

        coupled_any = False
        for bucket in cloth_buckets.values():
            if len(bucket) < 2:
                continue
            rvecs = [Rotation.from_matrix(out_rots[bn]).as_rotvec() for bn in bucket]
            avg_rv = np.mean(rvecs, axis=0)
            for bn, rv in zip(bucket, rvecs):
                # Blend 50% toward sheet average -> keeps gentle wave while preventing tears
                blended_rv = 0.50 * rv + 0.50 * avg_rv
                out_rots[bn] = Rotation.from_rotvec(blended_rv).as_matrix().astype(np.float32)
                coupled_any = True

        if coupled_any:
            world = compute_fk_transforms(self.bones, self.order, out_rots, root_shift=root_shift)
            for bname in self.dynamic_bones:
                R_w, h_w, _ = world[bname]
                self._tip_pos[bname] = (h_w + R_w @ self._rest_vec[bname]).astype(np.float32)

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
