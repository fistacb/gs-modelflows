"""Modular Skeleton Builder (Schema v2.1) for 3DGS Assets.

Supports a hybrid workflow (Auto-Detection + Recipe/Overrides):
  - Base archetypes:
      * 'humanoid' (wraps existing v2.0 humanoid rig_skeleton.json)
      * 'quadruped' (data-driven 4-legged animal fitter: pelvis/spine/chest/neck/head + 4 legs)
      * 'custom' (arbitrary user-defined hierarchy)
  - Composable attachments (attachable to ANY archetype):
      * 'jaw' (muzzle lower-jaw open/close bone with automatic mouth-seam kNN cut)
      * 'wing' (2-4 segment L/R wing chains with wingtip secondary motion)
      * 'tail' (2-5 segment tail chain with spring-damper dynamics)
      * 'ear' (1-3 segment L/R ear chains with spring-damper dynamics)
      * 'cape' / 'skirt' (back/side cloth hem chains for humanoids)
      * 'hair' (ponytail / twintails / side-lock chains)
      * 'ribbon' / 'accessory' (general dangling/spring chains)
  - Custom bones & per-bone overrides for fine-tuning.

Coordinate convention: Blender Z-up, -Y forward, +X character-left, rest_quat_wxyz=[1,0,0,0].
"""
from __future__ import annotations

import copy
from typing import Any
import numpy as np
from scipy.spatial import cKDTree

IDENTITY_Q = [1.0, 0.0, 0.0, 0.0]


def make_bone(
    parent: str | None,
    head: list[float] | np.ndarray,
    tail: list[float] | np.ndarray,
    *,
    limits_deg: dict | None = None,
    skin_group: str | None = None,
    role: str | None = None,
    side_lock: str | None = None,
    rigid: bool = False,
    dynamics: dict | None = None,
    pin_radius: float | None = None,
    influence_radius_scale: float | None = None,
    region_box: dict | None = None,
    seam_cut: dict | None = None,
    non_deforming: bool = False,
    meta: dict | None = None,
) -> dict[str, Any]:
    """Construct a schema v2.1 bone dictionary (fully backward-compatible with v2.0)."""
    b: dict[str, Any] = {
        "parent": parent,
        "head": [float(v) for v in head],
        "tail": [float(v) for v in tail],
        "rest_quat_wxyz": list(IDENTITY_Q),
    }
    if limits_deg is not None:
        b["limits_deg"] = copy.deepcopy(limits_deg)
    if skin_group is not None:
        b["skin_group"] = str(skin_group)
    if role is not None:
        b["role"] = str(role)
    if side_lock is not None:
        b["side_lock"] = str(side_lock)
    if rigid:
        b["rigid"] = True
    if dynamics is not None:
        b["dynamics"] = copy.deepcopy(dynamics)
    if pin_radius is not None:
        b["pin_radius"] = float(pin_radius)
    if influence_radius_scale is not None:
        b["influence_radius_scale"] = float(influence_radius_scale)
    if region_box is not None:
        b["region_box"] = copy.deepcopy(region_box)
    if seam_cut is not None:
        b["seam_cut"] = copy.deepcopy(seam_cut)
    if non_deforming:
        b["non_deforming"] = True
    if meta is not None:
        b["meta"] = copy.deepcopy(meta)
    return b


def _snap_point_to_cloud(xyz: np.ndarray, p: np.ndarray, radius: float = 0.045) -> np.ndarray:
    """Snap a joint point gently toward the median of nearby 3DGS points (if sufficient support exists)."""
    if len(xyz) == 0:
        return p.astype(np.float32)
    d = np.linalg.norm(xyz - p[None, :], axis=1)
    m = d <= radius
    if int(m.sum()) >= 12:
        pts = xyz[m]
        p10 = np.percentile(pts, 10, axis=0)
        p90 = np.percentile(pts, 90, axis=0)
        return (0.5 * (p10 + p90)).astype(np.float32)
    return p.astype(np.float32)


def add_chain(
    bones: dict,
    prefix: str,
    parent: str,
    points: list[list[float]] | list[np.ndarray],
    *,
    skin_group: str | None = None,
    role: str | None = None,
    side_lock: str | None = None,
    limits_deg: dict | None = None,
    dynamics: dict | None = None,
    dynamics_tip_boost: float = 1.0,
    pin_radius: float | None = None,
    influence_radius_scale: float | None = None,
    region_box: dict | None = None,
    xyz_snap: np.ndarray | None = None,
) -> list[str]:
    """Add a connected bone chain `prefix_1 -> prefix_2 -> ...` along 3D polyline `points`."""
    pts = [np.asarray(p, dtype=np.float32) for p in points]
    if len(pts) < 2:
        raise ValueError(f"Chain '{prefix}' needs at least 2 points (got {len(pts)})")
    if xyz_snap is not None and len(xyz_snap) > 0:
        # Keep root anchor fixed; snap interior & tip points to local cloud centroid
        pts = [pts[0]] + [_snap_point_to_cloud(xyz_snap, p) for p in pts[1:]]

    created: list[str] = []
    cur_parent = parent
    n_seg = len(pts) - 1
    for i in range(n_seg):
        bname = f"{prefix}_{i + 1}" if n_seg > 1 else prefix
        dyn_i = None
        if dynamics is not None:
            dyn_i = copy.deepcopy(dynamics)
            if n_seg > 1:
                frac = i / max(n_seg - 1, 1)
                # Root segments stay firm & anchored; distal tips flex progressively more
                root_scale = 0.45 + 0.55 * frac
                dyn_i["stiffness"] = float(dyn_i.get("stiffness", 36.0)) * (1.35 - 0.50 * frac)
                dyn_i["inertia"] = float(dyn_i.get("inertia", 0.50)) * root_scale
                dyn_i["max_angle_deg"] = float(dyn_i.get("max_angle_deg", 12.0)) * (root_scale * (1.0 + (dynamics_tip_boost - 1.0) * frac))
        bones[bname] = make_bone(
            cur_parent,
            pts[i],
            pts[i + 1],
            limits_deg=limits_deg,
            skin_group=skin_group,
            role=role,
            side_lock=side_lock,
            dynamics=dyn_i,
            pin_radius=pin_radius,
            influence_radius_scale=influence_radius_scale,
            region_box=region_box,
        )
        created.append(bname)
        cur_parent = bname
    return created


def _slice_center(pts: np.ndarray, axis: int, val: float, half_w: float = 0.025) -> np.ndarray:
    m = np.abs(pts[:, axis] - val) <= half_w
    spts = pts[m] if int(m.sum()) >= 12 else pts[np.argsort(np.abs(pts[:, axis] - val))[:max(12, min(40, len(pts)))]]
    p10 = np.percentile(spts, 10, axis=0)
    p90 = np.percentile(spts, 90, axis=0)
    c = 0.5 * (p10 + p90)
    c[axis] = val
    return c.astype(np.float32)


def _trace_protrusion_chain(
    pts: np.ndarray,
    anchor: np.ndarray,
    tip: np.ndarray,
    n_segments: int,
    radius: float = 0.06,
) -> list[np.ndarray]:
    """Trace an N-segment curved centerline from anchor to tip through a 3D point cluster."""
    v = tip - anchor
    blen = max(float(np.linalg.norm(v)), 1e-4)
    u = v / blen
    out = [anchor.astype(np.float32)]
    if len(pts) < 15:
        for i in range(1, n_segments + 1):
            out.append((anchor + (i / n_segments) * v).astype(np.float32))
        return out

    rel = pts - anchor[None, :]
    s = (rel @ u) / blen
    perp = np.linalg.norm(rel - (s * blen)[:, None] * u[None, :], axis=1)
    for i in range(1, n_segments + 1):
        si = i / float(n_segments)
        band = (np.abs(s - si) <= (0.55 / n_segments)) & (perp <= max(radius, blen * 0.45))
        if int(band.sum()) >= 10:
            sub = pts[band]
            p15 = np.percentile(sub, 15, axis=0)
            p85 = np.percentile(sub, 85, axis=0)
            pt = 0.5 * (p15 + p85)
            out.append(pt.astype(np.float32))
        else:
            out.append((anchor + si * v).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# Base Archetype Fitters
# ---------------------------------------------------------------------------

def fit_quadruped_skeleton(
    xyz: np.ndarray,
    mid_x: float | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Data-driven quadruped base skeleton fitter (Z-up, -Y forward, +X character-left).

    Automatically locates:
      - 4 legs (front_L/R, hind_L/R) from lower-half (X, Y) clusters
      - Horizontal spine centerline (pelvis -> spine_mid -> chest)
      - Neck & head (chest -> neck -> head)
    """
    cfg = config or {}
    pts = np.asarray(xyz, dtype=np.float32)
    z_min = float(np.percentile(pts[:, 2], 0.5))
    z_max = float(np.percentile(pts[:, 2], 99.5))
    height = max(z_max - z_min, 0.1)

    if mid_x is None:
        # Estimate symmetry midline from lower body / legs
        low = pts[pts[:, 2] <= z_min + 0.55 * height]
        mid_x = float(np.median(low[:, 0])) if len(low) >= 50 else float(np.median(pts[:, 0]))

    # 1. Detect 4 legs in the lower 35% of the model (excluding wide wings/high tails)
    z_paw_top = z_min + float(cfg.get("paw_band_frac", 0.14)) * height
    z_belly = z_min + float(cfg.get("belly_frac", 0.36)) * height
    leg_pts = pts[(pts[:, 2] >= z_min) & (pts[:, 2] <= z_belly)]
    paw_pts = pts[(pts[:, 2] >= z_min) & (pts[:, 2] <= z_paw_top)]
    if len(paw_pts) < 40:
        paw_pts = leg_pts

    # Front (-Y) vs Hind (+Y) split at median Y of paw points
    y_leg_split = float(cfg.get("y_leg_split", np.median(paw_pts[:, 1])))
    y_front_paws = float(np.percentile(paw_pts[paw_pts[:, 1] <= y_leg_split, 1], 35)) if np.any(paw_pts[:, 1] <= y_leg_split) else (y_leg_split - 0.15 * height)
    y_hind_paws = float(np.percentile(paw_pts[paw_pts[:, 1] > y_leg_split, 1], 65)) if np.any(paw_pts[:, 1] > y_leg_split) else (y_leg_split + 0.15 * height)

    # Symmetric X half-width for front and hind legs
    def _leg_half_x(mask_y: np.ndarray, default_hw: float) -> float:
        sub = paw_pts[mask_y]
        if len(sub) < 20:
            return default_hw
        dx = np.abs(sub[:, 0] - mid_x)
        return float(np.clip(np.percentile(dx, 65), 0.03 * height, 0.22 * height))

    fl_hw = _leg_half_x(paw_pts[:, 1] <= y_leg_split, 0.075 * height)
    hl_hw = _leg_half_x(paw_pts[:, 1] > y_leg_split, 0.085 * height)

    # Torso band above belly
    z_torso_lo = z_belly
    z_torso_hi = z_min + float(cfg.get("torso_top_frac", 0.64)) * height
    core_x_lim = max(fl_hw, hl_hw) * 1.9 + 0.03 * height
    torso_pts = pts[
        (pts[:, 2] >= z_torso_lo) & (pts[:, 2] <= z_torso_hi)
        & (np.abs(pts[:, 0] - mid_x) <= core_x_lim)
        & (pts[:, 1] >= y_front_paws - 0.18 * height)
        & (pts[:, 1] <= y_hind_paws + 0.18 * height)
    ]
    if len(torso_pts) < 60:
        torso_pts = pts[(pts[:, 2] >= z_torso_lo) & (pts[:, 2] <= z_torso_hi)]

    def _torso_y_center(y_val: float) -> np.ndarray:
        c = _slice_center(torso_pts, axis=1, val=y_val, half_w=0.04 * height)
        c[0] = mid_x
        return c

    p_pelvis = _torso_y_center(y_hind_paws)
    p_chest = _torso_y_center(y_front_paws)
    p_spine_mid = 0.5 * (p_pelvis + p_chest)
    p_spine_mid[2] = float(_torso_y_center(float(p_spine_mid[1]))[2])

    # 2. Neck & Head (frontmost upper mass ahead of chest)
    head_cand = pts[
        (pts[:, 1] <= p_chest[1] + 0.04 * height)
        & (pts[:, 2] >= p_chest[2])
        & (np.abs(pts[:, 0] - mid_x) <= core_x_lim * 1.1)
    ]
    if len(head_cand) < 40:
        head_cand = pts[pts[:, 2] >= p_chest[2]]

    z_head_mid = float(np.percentile(head_cand[:, 2], 55))
    y_head_mid = float(np.percentile(head_cand[:, 1], 45))
    y_snout = float(np.percentile(head_cand[:, 1], 4))
    p_neck_base = np.array([mid_x, float(p_chest[1] - 0.03 * height), float(p_chest[2] + 0.04 * height)], dtype=np.float32)
    p_head_base = np.array([mid_x, float(0.55 * y_head_mid + 0.45 * p_neck_base[1]), float(0.65 * z_head_mid + 0.35 * p_neck_base[2])], dtype=np.float32)
    p_head_tip = np.array([mid_x, float(0.35 * y_head_mid + 0.65 * y_snout), float(z_head_mid)], dtype=np.float32)

    # 3. Fit each of the 4 legs (hip/shoulder -> knee/elbow -> ankle/wrist -> paw)
    # Use actual 2D paw clusters per front/hind half so slightly oblique or asymmetric stances
    # (common in 3/4-view 3DGS generations) still place every leg bone inside real leg splats!
    def _split_lr_paws(sub_pts: np.ndarray, default_y: float, default_hw: float) -> tuple[tuple[float, float], tuple[float, float]]:
        if len(sub_pts) < 20:
            return (mid_x + default_hw, default_y), (mid_x - default_hw, default_y)
        x_med = float(np.median(sub_pts[:, 0]))
        left_p = sub_pts[sub_pts[:, 0] >= x_med]
        right_p = sub_pts[sub_pts[:, 0] < x_med]
        lx = float(np.median(left_p[:, 0])) if len(left_p) >= 8 else (x_med + default_hw)
        ly = float(np.median(left_p[:, 1])) if len(left_p) >= 8 else default_y
        rx = float(np.median(right_p[:, 0])) if len(right_p) >= 8 else (x_med - default_hw)
        ry = float(np.median(right_p[:, 1])) if len(right_p) >= 8 else default_y
        return (lx, ly), (rx, ry)

    # Exclude low-hanging rear tail splats when locating hind paws (keep Y <= y_hind_paws + 0.16*height)
    front_paws = paw_pts[paw_pts[:, 1] <= y_leg_split]
    hind_paws = paw_pts[(paw_pts[:, 1] > y_leg_split) & (paw_pts[:, 1] <= y_hind_paws + 0.16 * height)]
    if len(hind_paws) < 20:
        hind_paws = paw_pts[paw_pts[:, 1] > y_leg_split]

    (fl_x, fl_y), (fr_x, fr_y) = _split_lr_paws(front_paws, y_front_paws, fl_hw)
    (hl_x, hl_y), (hr_x, hr_y) = _split_lr_paws(hind_paws, y_hind_paws, hl_hw)

    def _fit_single_leg(x_nom: float, y_nom: float, is_front: bool, hip_z: float) -> list[np.ndarray]:
        z_top_leg = hip_z
        z_mid_leg = z_min + 0.50 * (z_top_leg - z_min)
        z_ank_leg = z_min + 0.14 * (z_top_leg - z_min)
        z_paw_leg = z_min + 0.03 * height

        m_leg = (
            (np.abs(pts[:, 0] - x_nom) <= 0.095 * height)
            & (np.abs(pts[:, 1] - y_nom) <= 0.12 * height)
            & (pts[:, 2] <= z_top_leg + 0.03 * height)
        )
        lpts = pts[m_leg]
        if len(lpts) >= 35:
            p0 = _slice_center(lpts, 2, z_top_leg, 0.028 * height)
            p1 = _slice_center(lpts, 2, z_mid_leg, 0.028 * height)
            p2 = _slice_center(lpts, 2, z_ank_leg, 0.022 * height)
            p3 = _slice_center(lpts, 2, z_paw_leg, 0.022 * height)
            p3[1] -= 0.022 * height
        else:
            p0 = np.array([x_nom, y_nom, z_top_leg], dtype=np.float32)
            p1 = np.array([x_nom, y_nom + (0.02 if is_front else -0.02) * height, z_mid_leg], dtype=np.float32)
            p2 = np.array([x_nom, y_nom, z_ank_leg], dtype=np.float32)
            p3 = np.array([x_nom, y_nom - 0.032 * height, z_paw_leg], dtype=np.float32)
        return [p0, p1, p2, p3]

    fl_pts = _fit_single_leg(fl_x, fl_y, True, float(p_chest[2]))
    fr_pts = _fit_single_leg(fr_x, fr_y, True, float(p_chest[2]))
    hl_pts = _fit_single_leg(hl_x, hl_y, False, float(p_pelvis[2]))
    hr_pts = _fit_single_leg(hr_x, hr_y, False, float(p_pelvis[2]))

    # Optional symmetric mirror when explicitly requested (default False preserves true oblique stance)
    if bool(cfg.get("symmetrize_legs", False)):
        def _sym_leg(l_tuple, r_tuple):
            out_l, out_r = [], []
            for a, b in zip(l_tuple, r_tuple):
                ax = 0.5 * (abs(float(a[0]) - mid_x) + abs(float(b[0]) - mid_x))
                ay = 0.5 * (float(a[1]) + float(b[1]))
                az = 0.5 * (float(a[2]) + float(b[2]))
                out_l.append(np.array([mid_x + ax, ay, az], dtype=np.float32))
                out_r.append(np.array([mid_x - ax, ay, az], dtype=np.float32))
            return out_l, out_r
        fl_pts, fr_pts = _sym_leg(fl_pts, fr_pts)
        hl_pts, hr_pts = _sym_leg(hl_pts, hr_pts)

    bones: dict[str, Any] = {
        "root": make_bone(None, [mid_x, float(p_spine_mid[1]), z_min], [mid_x, float(p_spine_mid[1]), z_min + 0.08 * height],
                          limits_deg={"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}, role="root"),
        "pelvis": make_bone("root", p_pelvis, p_spine_mid, skin_group="torso", role="pelvis",
                            limits_deg={"x": [-25, 25], "y": [-18, 18], "z": [-25, 25]}),
        "spine_mid": make_bone("pelvis", p_spine_mid, p_chest, skin_group="torso", role="spine",
                               limits_deg={"x": [-22, 22], "y": [-18, 18], "z": [-25, 25]}),
        "chest": make_bone("spine_mid", p_chest, p_neck_base, skin_group="torso", role="chest",
                           limits_deg={"x": [-22, 22], "y": [-18, 18], "z": [-25, 25]}),
        "neck": make_bone("chest", p_neck_base, p_head_base, skin_group="torso", role="neck",
                          limits_deg={"x": [-35, 35], "y": [-25, 25], "z": [-40, 40]}),
        "head": make_bone("neck", p_head_base, p_head_tip, skin_group="head", role="head", rigid=True,
                          influence_radius_scale=1.45,
                          limits_deg={"x": [-35, 35], "y": [-25, 25], "z": [-45, 45]}),
    }

    for side, pts_f, pts_h in (("L", fl_pts, hl_pts), ("R", fr_pts, hr_pts)):
        # Only enforce side_lock if the leg is actually on that side of mid_x
        sgn = +1.0 if side == "L" else -1.0
        fl_lock = side if (pts_f[2][0] - mid_x) * sgn > 0.01 else "none"
        hl_lock = side if (pts_h[2][0] - mid_x) * sgn > 0.01 else "none"
        # Front leg
        bones[f"front_upper_{side}"] = make_bone(
            "chest", pts_f[0], pts_f[1], skin_group=f"front_leg_{side}", role="limb", side_lock=fl_lock,
            limits_deg={"x": [-65, 65], "y": [-25, 25], "z": [-25, 25]},
        )
        bones[f"front_lower_{side}"] = make_bone(
            f"front_upper_{side}", pts_f[1], pts_f[2], skin_group=f"front_leg_{side}", role="limb", side_lock=fl_lock,
            limits_deg={"x": [-95, 25], "y": [-15, 15], "z": [-15, 15]},
        )
        bones[f"front_paw_{side}"] = make_bone(
            f"front_lower_{side}", pts_f[2], pts_f[3], skin_group=f"front_leg_{side}", role="limb", side_lock=fl_lock,
            limits_deg={"x": [-45, 45], "y": [-15, 15], "z": [-20, 20]},
        )
        # Hind leg
        bones[f"hind_upper_{side}"] = make_bone(
            "pelvis", pts_h[0], pts_h[1], skin_group=f"hind_leg_{side}", role="limb", side_lock=hl_lock,
            limits_deg={"x": [-65, 65], "y": [-25, 25], "z": [-25, 25]},
        )
        bones[f"hind_lower_{side}"] = make_bone(
            f"hind_upper_{side}", pts_h[1], pts_h[2], skin_group=f"hind_leg_{side}", role="limb", side_lock=hl_lock,
            limits_deg={"x": [-25, 95], "y": [-15, 15], "z": [-15, 15]},
        )
        bones[f"hind_paw_{side}"] = make_bone(
            f"hind_lower_{side}", pts_h[2], pts_h[3], skin_group=f"hind_leg_{side}", role="limb", side_lock=hl_lock,
            limits_deg={"x": [-45, 45], "y": [-15, 15], "z": [-20, 20]},
        )

    return bones


# ---------------------------------------------------------------------------
# Composable Attachment Builders (work on humanoid, quadruped, or custom)
# ---------------------------------------------------------------------------

def add_jaw_bone(
    bones: dict,
    xyz: np.ndarray,
    *,
    parent: str = "head",
    hinge: list[float] | np.ndarray | None = None,
    tip: list[float] | np.ndarray | None = None,
    mouth_plane_z: float | None = None,
    cut_radius: float | None = None,
    max_open_deg: float = 35.0,
) -> str:
    """Attach a lower-jaw / muzzle opening bone (`jaw`) with an automatic mouth-seam kNN cut."""
    hb = bones[parent]
    h_head = np.asarray(hb["head"], dtype=np.float32)
    t_head = np.asarray(hb["tail"], dtype=np.float32)
    head_len = max(float(np.linalg.norm(t_head - h_head)), 0.06)

    if hinge is not None and tip is not None:
        p_hinge = np.asarray(hinge, dtype=np.float32)
        p_tip = np.asarray(tip, dtype=np.float32)
    else:
        # Estimate snout tip as frontmost point near head height
        m_muzzle = (
            (np.abs(xyz[:, 0] - h_head[0]) <= head_len * 0.55)
            & (xyz[:, 2] >= min(h_head[2], t_head[2]) - head_len * 0.45)
            & (xyz[:, 2] <= max(h_head[2], t_head[2]) + head_len * 0.25)
            & (xyz[:, 1] <= t_head[1] + 0.02)
        )
        if int(m_muzzle.sum()) >= 20:
            mpts = xyz[m_muzzle]
            y_front = float(np.percentile(mpts[:, 1], 5))
            z_low = float(np.percentile(mpts[mpts[:, 1] <= y_front + 0.04, 2], 25))
            p_tip = np.array([float(h_head[0]), y_front + 0.008, z_low], dtype=np.float32)
            p_hinge = np.array([float(h_head[0]), float(0.55 * h_head[1] + 0.45 * y_front), z_low + 0.015], dtype=np.float32)
        else:
            p_hinge = np.array([float(h_head[0]), float(0.5 * (h_head[1] + t_head[1])), float(t_head[2] - 0.025)], dtype=np.float32)
            p_tip = np.array([float(h_head[0]), float(t_head[1] - 0.03), float(t_head[2] - 0.032)], dtype=np.float32)

    jaw_len = max(float(np.linalg.norm(p_tip - p_hinge)), 0.03)
    cut_z = float(mouth_plane_z) if mouth_plane_z is not None else float(0.5 * (p_hinge[2] + p_tip[2]) + 0.008)
    cut_origin = [float(p_tip[0]), float(0.35 * p_hinge[1] + 0.65 * p_tip[1]), cut_z]
    cut_r = float(cut_radius) if cut_radius is not None else max(jaw_len * 0.85, 0.035)

    seam_cut = {
        "origin": cut_origin,
        "normal": [0.0, 0.0, 1.0],  # +Z = upper jaw (head), -Z = lower jaw (jaw)
        "radius": cut_r,
        "keep_side": -1.0,
    }
    jaw_box = {
        "min": [float(p_tip[0]) - cut_r, float(p_tip[1]) - 0.04, float(min(p_hinge[2], p_tip[2])) - 0.045],
        "max": [float(p_tip[0]) + cut_r, float(p_hinge[1]) + 0.015, cut_z + 0.002],
    }
    bones["jaw"] = make_bone(
        parent,
        p_hinge,
        p_tip,
        skin_group="jaw",
        role="jaw",
        pin_radius=max(jaw_len * 0.40, 0.018),
        influence_radius_scale=1.15,
        region_box=jaw_box,
        limits_deg={"x": [-abs(max_open_deg), 4.0], "y": [-3.0, 3.0], "z": [-5.0, 5.0]},
        seam_cut=seam_cut,
    )
    return "jaw"


def add_wing_bones(
    bones: dict,
    xyz: np.ndarray,
    *,
    parent: str = "chest",
    n_segments: int = 3,
    side: str = "both",
    root_L: list[float] | None = None,
    tip_L: list[float] | None = None,
    dynamics: dict | None = None,
) -> list[str]:
    """Attach L/R wing chains (`wing_L_1..N`, `wing_R_1..N`) to chest/back with tip secondary motion."""
    cb = bones[parent]
    p_chest = np.asarray(cb["head"], dtype=np.float32)
    mid_x = float(p_chest[0])
    pts = np.asarray(xyz, dtype=np.float32)

    dyn_tip = dynamics if dynamics is not None else {
        "preset": "wing_tip",
        "stiffness": 52.0,
        "damping": 0.70,
        "gravity": 0.0,
        "wind_scale": 0.22,
        "inertia": 0.42,
        "max_angle_deg": 11.0,
    }

    created: list[str] = []
    sides = ("L", "R") if side == "both" else (side.upper(),)

    # Auto-detect symmetric wing span if not explicitly provided
    if root_L is None or tip_L is None:
        # Wings protrude laterally (|x - mid_x| large) and rearward (+Y) above mid-chest
        w_cand = pts[
            (np.abs(pts[:, 0] - mid_x) > 0.06)
            & (pts[:, 2] >= p_chest[2] - 0.04)
            & (pts[:, 1] >= p_chest[1] - 0.06)
        ]
        if len(w_cand) >= 30:
            # Find farthest lateral/rearward wingtip
            dist_from_chest = np.linalg.norm(w_cand - p_chest[None, :], axis=1)
            idx_far = np.argsort(dist_from_chest)[-max(15, len(w_cand) // 25):]
            tip_raw = np.median(w_cand[idx_far], axis=0)
            tip_abs_x = max(abs(float(tip_raw[0]) - mid_x), 0.16)
            p_tip_L = np.array([mid_x + tip_abs_x, float(tip_raw[1]), float(tip_raw[2])], dtype=np.float32)
        else:
            p_tip_L = np.array([mid_x + 0.32, float(p_chest[1] + 0.20), float(p_chest[2] + 0.16)], dtype=np.float32)
        p_root_L = np.array([mid_x + 0.035, float(p_chest[1] + 0.03), float(p_chest[2] + 0.04)], dtype=np.float32)
    else:
        p_root_L = np.asarray(root_L, dtype=np.float32)
        p_tip_L = np.asarray(tip_L, dtype=np.float32)

    for s in sides:
        sgn = +1.0 if s == "L" else -1.0
        r_s = np.array([mid_x + sgn * abs(float(p_root_L[0]) - mid_x), float(p_root_L[1]), float(p_root_L[2])], dtype=np.float32)
        t_s = np.array([mid_x + sgn * abs(float(p_tip_L[0]) - mid_x), float(p_tip_L[1]), float(p_tip_L[2])], dtype=np.float32)
        w_pts = pts[((pts[:, 0] - mid_x) * sgn > 0.03) & (pts[:, 2] >= p_chest[2] - 0.05)]
        chain_pts = _trace_protrusion_chain(w_pts, r_s, t_s, n_segments=n_segments, radius=0.09)

        # Build wing chain as articulated skeletal bones (humerus -> radius/ulna -> carpometacarpus).
        # Wings have rigid bones inside, so do NOT apply cloth-like dynamics by default.
        cur_parent = parent
        for i in range(n_segments):
            bname = f"wing_{s}_{i + 1}"
            is_tip = (i == n_segments - 1) and (n_segments >= 2)
            bones[bname] = make_bone(
                cur_parent,
                chain_pts[i],
                chain_pts[i + 1],
                skin_group=f"wing_{s}",
                role="wing",
                side_lock=s,
                influence_radius_scale=1.25,
                limits_deg={"x": [-45, 45], "y": [-70, 70], "z": [-45, 45]},
                dynamics=dynamics if (dynamics is not None and is_tip) else None,
            )
            created.append(bname)
            cur_parent = bname
    return created


def add_tail_bones(
    bones: dict,
    xyz: np.ndarray,
    *,
    parent: str = "pelvis",
    n_segments: int = 3,
    root_pt: list[float] | None = None,
    tip_pt: list[float] | None = None,
    dynamics: dict | None = None,
) -> list[str]:
    """Attach a multi-segment tail chain (`tail_1..N`) with spring-damper dynamics."""
    if parent not in bones:
        parent = "spine_lower" if "spine_lower" in bones else next(iter(bones))
    pb = bones[parent]
    p_pelvis = np.asarray(pb["head"], dtype=np.float32)
    mid_x = float(p_pelvis[0])
    pts = np.asarray(xyz, dtype=np.float32)

    if root_pt is not None and tip_pt is not None:
        r_p = np.asarray(root_pt, dtype=np.float32)
        t_p = np.asarray(tip_pt, dtype=np.float32)
    else:
        t_cand = pts[(pts[:, 1] > p_pelvis[1] + 0.02) & (np.abs(pts[:, 0] - mid_x) <= 0.25)]
        r_p = np.array([mid_x, float(p_pelvis[1] + 0.03), float(p_pelvis[2] + 0.01)], dtype=np.float32)
        if len(t_cand) >= 25:
            y_max_idx = np.argsort(t_cand[:, 1])[-max(15, len(t_cand) // 20):]
            t_raw = np.median(t_cand[y_max_idx], axis=0)
            t_p = np.array([mid_x, float(t_raw[1]), float(t_raw[2])], dtype=np.float32)
        else:
            t_p = np.array([mid_x, float(p_pelvis[1] + 0.28), float(p_pelvis[2] + 0.04)], dtype=np.float32)

    t_pts = pts[pts[:, 1] >= r_p[1] - 0.02]
    chain_pts = _trace_protrusion_chain(t_pts, r_p, t_p, n_segments=n_segments, radius=0.11)
    dyn = dynamics if dynamics is not None else {
        "preset": "tail",
        "stiffness": 28.0,
        "damping": 0.76,
        "gravity": 0.0,
        "wind_scale": 0.28,
        "inertia": 0.52,
        "max_angle_deg": 11.0,
    }
    return add_chain(
        bones,
        "tail",
        parent,
        chain_pts,
        skin_group="tail",
        role="tail",
        influence_radius_scale=1.35,
        limits_deg={"x": [-22, 22], "y": [-18, 18], "z": [-28, 28]},
        dynamics=dyn,
        dynamics_tip_boost=1.25,
    )


def add_ear_bones(
    bones: dict,
    xyz: np.ndarray,
    *,
    parent: str = "head",
    n_segments: int = 2,
    root_L: list[float] | None = None,
    tip_L: list[float] | None = None,
    dynamics: dict | None = None,
) -> list[str]:
    """Attach L/R ear chains (`ear_L_1..N`, `ear_R_1..N`) to the head with springy dynamics."""
    hb = bones[parent]
    h_head = np.asarray(hb["head"], dtype=np.float32)
    t_head = np.asarray(hb["tail"], dtype=np.float32)
    mid_x = float(h_head[0])
    pts = np.asarray(xyz, dtype=np.float32)

    if root_L is not None and tip_L is not None:
        r_L = np.asarray(root_L, dtype=np.float32)
        t_L = np.asarray(tip_L, dtype=np.float32)
    else:
        head_z_ref = max(float(h_head[2]), float(t_head[2]))
        e_cand = pts[(pts[:, 2] >= head_z_ref + 0.02) & (pts[:, 1] <= float(h_head[1]) + 0.14)]
        if len(e_cand) >= 20:
            z_top_idx = np.argsort(e_cand[:, 2])[-max(12, len(e_cand) // 15):]
            top_pts = e_cand[z_top_idx]
            ear_x = max(float(np.percentile(np.abs(top_pts[:, 0] - mid_x), 70)), 0.035)
            ear_y = float(np.median(top_pts[:, 1]))
            ear_z = float(np.percentile(top_pts[:, 2], 85))
            r_L = np.array([mid_x + ear_x * 0.55, ear_y, head_z_ref + 0.02], dtype=np.float32)
            t_L = np.array([mid_x + ear_x, ear_y, ear_z], dtype=np.float32)
        else:
            r_L = np.array([mid_x + 0.04, float(h_head[1]), head_z_ref + 0.03], dtype=np.float32)
            t_L = np.array([mid_x + 0.08, float(h_head[1] + 0.02), head_z_ref + 0.15], dtype=np.float32)

    dyn = dynamics if dynamics is not None else {
        "preset": "ear",
        "stiffness": 70.0,
        "damping": 0.75,
        "gravity": 0.0,
        "wind_scale": 0.12,
        "inertia": 0.28,
        "max_angle_deg": 7.0,
    }
    created: list[str] = []
    for s, sgn in (("L", +1.0), ("R", -1.0)):
        r_s = np.array([mid_x + sgn * abs(float(r_L[0]) - mid_x), float(r_L[1]), float(r_L[2])], dtype=np.float32)
        t_s = np.array([mid_x + sgn * abs(float(t_L[0]) - mid_x), float(t_L[1]), float(t_L[2])], dtype=np.float32)
        chain_pts = _trace_protrusion_chain(
            pts[((pts[:, 0] - mid_x) * sgn > 0.01) & (pts[:, 2] >= r_s[2] - 0.01)],
            r_s, t_s, n_segments=n_segments, radius=0.045,
        )
        # Strictly confine ear influence above the ear base Z so eyes/face/cranium stay 100% rigid on `head`!
        ear_z_floor = float(r_s[2]) - 0.005
        ear_box = {
            "min": [mid_x - 0.01 if s == "L" else -10.0, -10.0, ear_z_floor],
            "max": [10.0 if s == "L" else mid_x + 0.01, 10.0, 10.0],
        }
        created.extend(add_chain(
            bones,
            f"ear_{s}",
            parent,
            chain_pts,
            skin_group=f"ear_{s}",
            role="ear",
            side_lock=s,
            pin_radius=0.024,
            influence_radius_scale=0.85,
            region_box=ear_box,
            limits_deg={"x": [-10, 10], "y": [-8, 8], "z": [-8, 8]},
            dynamics=dyn,
            dynamics_tip_boost=1.2,
        ))
    return created


def add_cape_or_skirt_bones(
    bones: dict,
    xyz: np.ndarray,
    *,
    parent: str = "chest",
    n_segments: int = 2,
    names: tuple[str, ...] = ("cape_back", "cape_L", "cape_R"),
    dynamics: dict | None = None,
) -> list[str]:
    """Attach back/side cape or skirt secondary-motion bone chains to a humanoid torso."""
    if parent not in bones:
        parent = "spine_lower" if "spine_lower" in bones else next(iter(bones))
    pb = bones[parent]
    p_base = np.asarray(pb["head"], dtype=np.float32)
    mid_x = float(p_base[0])
    pts = np.asarray(xyz, dtype=np.float32)

    z_hip = float(bones["thigh_L"]["head"][2]) if "thigh_L" in bones else float(p_base[2] - 0.14)
    z_hem = max(z_hip - 0.16, 0.18)

    dyn = dynamics if dynamics is not None else {
        "preset": "cape",
        "stiffness": 34.0,
        "damping": 0.70,
        "gravity": 0.0,
        "wind_scale": 0.42,
        "inertia": 0.50,
        "max_angle_deg": 12.0,
    }
    created: list[str] = []
    for name in names:
        if name.endswith("_L"):
            dx, side_lock = +0.085, "L"
            cand = pts[(pts[:, 0] > mid_x + 0.03) & (pts[:, 1] > p_base[1] + 0.01) & (pts[:, 2] >= z_hem) & (pts[:, 2] <= p_base[2] + 0.05)]
        elif name.endswith("_R"):
            dx, side_lock = -0.085, "R"
            cand = pts[(pts[:, 0] < mid_x - 0.03) & (pts[:, 1] > p_base[1] + 0.01) & (pts[:, 2] >= z_hem) & (pts[:, 2] <= p_base[2] + 0.05)]
        else:
            dx, side_lock = 0.0, None
            cand = pts[(np.abs(pts[:, 0] - mid_x) <= 0.08) & (pts[:, 1] > p_base[1] + 0.02) & (pts[:, 2] >= z_hem) & (pts[:, 2] <= p_base[2] + 0.05)]

        if len(cand) >= 20:
            y_top = float(np.percentile(cand[cand[:, 2] >= np.percentile(cand[:, 2], 75), 1], 75))
            y_bot = float(np.percentile(cand[cand[:, 2] <= np.percentile(cand[:, 2], 25), 1], 80))
            z_top_c = float(np.percentile(cand[:, 2], 85))
            z_bot_c = float(np.percentile(cand[:, 2], 10))
            x_c = float(np.median(cand[:, 0]))
            r_p = np.array([x_c, y_top, z_top_c], dtype=np.float32)
            t_p = np.array([x_c, y_bot, z_bot_c], dtype=np.float32)
        else:
            r_p = np.array([mid_x + dx, float(p_base[1] + 0.055), float(p_base[2])], dtype=np.float32)
            t_p = np.array([mid_x + dx * 1.15, float(p_base[1] + 0.095), z_hem], dtype=np.float32)

        chain_pts = _trace_protrusion_chain(cand, r_p, t_p, n_segments=n_segments, radius=0.055)
        # Restrict cape bones strictly to the rear outer shell (+Y behind torso/skirt) so inner dress never gets pulled
        cape_y_min = float(p_base[1]) + 0.032
        cape_box = {
            "min": [-10.0, cape_y_min, z_hem - 0.06],
            "max": [10.0, 10.0, float(p_base[2]) + 0.06],
        }
        created.extend(add_chain(
            bones,
            name,
            parent,
            chain_pts,
            skin_group=name,
            role="cloth",
            side_lock=side_lock,
            pin_radius=0.026,
            influence_radius_scale=0.85,
            region_box=cape_box,
            limits_deg={"x": [0, 18], "y": [-10, 10], "z": [-10, 10]},
            dynamics=dyn,
            dynamics_tip_boost=1.25,
        ))
    return created


def add_ribbon_or_hair_bones(
    bones: dict,
    xyz: np.ndarray,
    *,
    prefix: str = "ribbon",
    parent: str = "chest",
    role: str = "ribbon",
    side: str = "both",
    n_segments: int = 2,
    root_L: list[float] | None = None,
    tip_L: list[float] | None = None,
    dynamics: dict | None = None,
) -> list[str]:
    """Attach dangling ribbon / hair-lock / accessory chains (`<prefix>_L_1..N`, `<prefix>_R_1..N` or `<prefix>_1..N`)."""
    pb = bones[parent]
    p_base = np.asarray(pb["head"], dtype=np.float32)
    mid_x = float(p_base[0])
    pts = np.asarray(xyz, dtype=np.float32)

    dyn = dynamics if dynamics is not None else (
        {
            "preset": "hair",
            "stiffness": 36.0,
            "damping": 0.68,
            "gravity": 0.0,
            "wind_scale": 0.45,
            "inertia": 0.55,
            "max_angle_deg": 13.0,
        }
        if role == "hair"
        else {
            "preset": "ribbon",
            "stiffness": 24.0,
            "damping": 0.58,
            "gravity": 0.0,
            "wind_scale": 0.75,
            "inertia": 0.68,
            "max_angle_deg": 16.0,
        }
    )

    if root_L is None or tip_L is None:
        # Default placement: slightly forward/below chest (for ribbons) or rearward/below head (for hair)
        if role == "hair":
            r_L = np.array([mid_x + 0.045, float(p_base[1] + 0.05), float(p_base[2] + 0.02)], dtype=np.float32)
            t_L = np.array([mid_x + 0.075, float(p_base[1] + 0.09), float(p_base[2] - 0.16)], dtype=np.float32)
        else:
            r_L = np.array([mid_x + 0.035, float(p_base[1] - 0.055), float(p_base[2] + 0.02)], dtype=np.float32)
            t_L = np.array([mid_x + 0.085, float(p_base[1] - 0.095), float(p_base[2] - 0.14)], dtype=np.float32)
    else:
        r_L = np.asarray(root_L, dtype=np.float32)
        t_L = np.asarray(tip_L, dtype=np.float32)

    created: list[str] = []
    if side == "center":
        chain_pts = _trace_protrusion_chain(pts, r_L, t_L, n_segments=n_segments, radius=0.04)
        return add_chain(
            bones, prefix, parent, chain_pts, skin_group=prefix, role=role,
            pin_radius=0.024, limits_deg={"x": [-45, 45], "y": [-40, 40], "z": [-40, 40]},
            dynamics=dyn, dynamics_tip_boost=1.4,
        )

    sides = ("L", "R") if side == "both" else (side.upper(),)
    for s in sides:
        sgn = +1.0 if s == "L" else -1.0
        r_s = np.array([mid_x + sgn * abs(float(r_L[0]) - mid_x), float(r_L[1]), float(r_L[2])], dtype=np.float32)
        t_s = np.array([mid_x + sgn * abs(float(t_L[0]) - mid_x), float(t_L[1]), float(t_L[2])], dtype=np.float32)
        chain_pts = _trace_protrusion_chain(pts, r_s, t_s, n_segments=n_segments, radius=0.04)
        created.extend(add_chain(
            bones,
            f"{prefix}_{s}",
            parent,
            chain_pts,
            skin_group=f"{prefix}_{s}",
            role=role,
            side_lock=s,
            pin_radius=0.024,
            limits_deg={"x": [-45, 45], "y": [-40, 40], "z": [-40, 40]},
            dynamics=dyn,
            dynamics_tip_boost=1.4,
        ))
    return created


# ---------------------------------------------------------------------------
# Hybrid Recipe Compiler (Auto-Detect + Custom Bones + Overrides)
# ---------------------------------------------------------------------------

def build_skeleton_from_recipe(
    xyz: np.ndarray,
    recipe: dict[str, Any],
    base_bones: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete schema_version 2.1 `rig_skeleton` dict from a modular recipe.

    Recipe keys:
      - archetype: 'humanoid' | 'quadruped' | 'custom'
      - quadruped_config: optional dict passed to fit_quadruped_skeleton
      - attachments: list of attachment specs, each with {'type': 'jaw'|'wing'|'tail'|'ear'|'cape'|'ribbon'|'hair', ...}
      - custom_bones: dict of arbitrary bones merged directly into `bones`
      - overrides: dict of {bone_name: {field: value}} applied after auto-fit
      - seam_cuts: optional list of top-level kNN seam_cut plane dicts
    """
    archetype = str(recipe.get("archetype", "humanoid")).lower()
    if base_bones is not None:
        bones = copy.deepcopy(base_bones)
    elif archetype == "quadruped":
        bones = fit_quadruped_skeleton(xyz, config=recipe.get("quadruped_config"))
    elif archetype == "custom":
        bones = {}
    else:
        raise ValueError(f"Archetype '{archetype}' requires base_bones (e.g. from existing rig_skeleton.json)")

    # Apply attachments
    for att in recipe.get("attachments", []):
        atype = str(att.get("type", "")).lower()
        kw = {k: v for k, v in att.items() if k != "type"}
        if atype == "jaw":
            add_jaw_bone(bones, xyz, **kw)
        elif atype == "wing":
            add_wing_bones(bones, xyz, **kw)
        elif atype == "tail":
            add_tail_bones(bones, xyz, **kw)
        elif atype == "ear":
            add_ear_bones(bones, xyz, **kw)
        elif atype in ("cape", "skirt"):
            add_cape_or_skirt_bones(bones, xyz, **kw)
        elif atype in ("ribbon", "hair", "accessory"):
            kw.setdefault("role", atype)
            kw.setdefault("prefix", atype)
            add_ribbon_or_hair_bones(bones, xyz, **kw)
        else:
            raise ValueError(f"Unknown attachment type: {atype}")

    # Merge arbitrary custom bones
    for bname, bspec in recipe.get("custom_bones", {}).items():
        if "head" in bspec and "tail" in bspec:
            b = copy.deepcopy(bspec)
            b.setdefault("rest_quat_wxyz", list(IDENTITY_Q))
            bones[bname] = b

    # Apply fine-tuning overrides
    for bname, patch in recipe.get("overrides", {}).items():
        if bname in bones:
            bones[bname].update(copy.deepcopy(patch))

    skel: dict[str, Any] = {
        "schema_version": "2.1",
        "archetype": archetype,
        "convention": {
            "world": "Z-up, -Y-forward, +X-character-left, height=1.0, feet Z=0",
            "bone_local_axes": "Unified across all bones: +X=left, -Y=forward, +Z=up (rest_quat_wxyz=[1,0,0,0])",
        },
        "bones": bones,
    }
    if recipe.get("seam_cuts"):
        skel["seam_cuts"] = copy.deepcopy(recipe["seam_cuts"])
    if recipe.get("attachments"):
        skel["attachments"] = copy.deepcopy(recipe["attachments"])
    return skel
