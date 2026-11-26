"""Shared splat operations and numeric checks (NumPy/SciPy only).

Coordinate convention (same as step1_normalize / render_ply_blender defaults):
  up = +Z, front = -Y (the character faces -Y), height normalized to ~1.0.

Contents:
  - transform_gaussians: apply a smooth point transform to positions AND covariances,
    refusing transforms that fold the surface (Jacobian determinant check).
  - trunk_lean_deg / sole_tilt_deg: side-view (90 deg) posture numbers.
  - oversize_splats: splats that are huge relative to local point spacing.
  - protected_displacement: how far a region that must not change has moved.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -20, 20)))


def xyz_of(data) -> np.ndarray:
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)


def opaque_xyz(data, min_opacity: float = 0.15) -> np.ndarray:
    return xyz_of(data)[sigmoid(data["opacity"]) > min_opacity]


# ---------------------------------------------------------------------------
# Deformation
# ---------------------------------------------------------------------------
def transform_gaussians(data, transform, min_det: float = 0.13, eps: float = 1e-5):
    """Apply ``transform`` ((N,3)->(N,3)) to positions and update rotation/scale.

    The local Jacobian is estimated numerically for every moved splat. The
    covariance basis R*S is multiplied by J and re-factored with SVD, so stretched
    areas get stretched splats instead of gaps or stripes.

    Raises ValueError when det(J) < min_det anywhere (the transform folds or
    collapses the surface). Fix the control points instead of lowering amplitude.

    Returns (new_data, report) where report has moved count and det(J) stats.
    """
    xyz = xyz_of(data)
    new_xyz = np.asarray(transform(xyz), dtype=np.float64)
    changed = np.any(np.abs(new_xyz - xyz) > 1e-8, axis=1)
    report = {"moved": int(changed.sum()), "det_min": 1.0, "det_p01": 1.0, "det_max": 1.0}
    output = data.copy()
    if not changed.any():
        return output, report
    base = xyz[changed]
    steps = np.eye(3) * eps
    jac = np.stack([(transform(base + s) - transform(base - s)) / (2 * eps) for s in steps], axis=2)
    det = np.linalg.det(jac)
    report.update(det_min=float(det.min()), det_p01=float(np.percentile(det, 1)), det_max=float(det.max()))
    if det.min() < min_det:
        raise ValueError(f"deformation folds the surface: det(J) min={det.min():.3f} < {min_det}")
    for key, values in zip("xyz", new_xyz.T):
        output[key] = values
    q_keys = [f"rot_{i}" for i in range(4)]
    s_keys = [f"scale_{i}" for i in range(3)]
    if all(k in data.dtype.names for k in q_keys + s_keys):
        quat = np.column_stack([data[k][changed] for k in q_keys]).astype(np.float64)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12
        scale = np.exp(np.column_stack([data[k][changed] for k in s_keys]).astype(np.float64))
        basis = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix() * scale[:, None, :]
        orient, sizes, _ = np.linalg.svd(jac @ basis)
        orient[np.linalg.det(orient) < 0, :, 2] *= -1
        q_new = Rotation.from_matrix(orient).as_quat()[:, [3, 0, 1, 2]]
        for key, values in zip(q_keys + s_keys, np.column_stack([q_new, np.log(np.maximum(sizes, 1e-9))]).T):
            output[key][changed] = values
    return output, report


# ---------------------------------------------------------------------------
# Posture (side view)
# ---------------------------------------------------------------------------
def _height_frame(xyz):
    z0, z1 = np.percentile(xyz[:, 2], [0.5, 99.5])
    h = max(float(z1 - z0), 1e-6)
    return z0, h, (xyz[:, 2] - z0) / h


def trunk_lean_deg(xyz: np.ndarray, lo: float = 0.10, hi: float = 0.85, band: float = 0.05) -> float:
    """Forward/back lean of the body seen from the side, in degrees.

    Fits a line to the median depth (Y) of horizontal slices between ``lo`` and
    ``hi`` of the height. Positive = top is further +Y = leaning BACK
    (character faces -Y); negative = leaning forward.

    Medians are robust to a single cape or ponytail but large back volumes still
    bias the number; record the reason in the config when a design leans on purpose.
    """
    _, h, t = _height_frame(xyz)
    zs, med = [], []
    for a in np.arange(lo, hi + 1e-9, band):
        m = (t >= a) & (t < a + band)
        if m.sum() >= 30:
            zs.append((a + band / 2) * h)
            med.append(np.median(xyz[m, 1]))
    if len(zs) < 3:
        return float("nan")
    slope = np.polyfit(zs, med, 1)[0]
    return float(np.degrees(np.arctan(slope)))


def sole_tilt_deg(xyz: np.ndarray, frac: float = 0.02) -> float:
    """Front-back tilt of the lowest ``frac`` of points (shoe soles), in degrees."""
    _, _, t = _height_frame(xyz)
    sole = xyz[t < frac]
    if len(sole) < 50 or np.ptp(sole[:, 1]) < 1e-6:
        return float("nan")
    return float(np.degrees(np.arctan(np.polyfit(sole[:, 1], sole[:, 2], 1)[0])))


# ---------------------------------------------------------------------------
# Cleanliness
# ---------------------------------------------------------------------------
def oversize_splats(data, ratio: float = 6.0, min_opacity: float = 0.15, k: int = 6) -> dict:
    """Count opaque splats whose largest axis exceeds ``ratio`` x local spacing.

    These show up as pale bands / colour bleed across part borders (chin, neck,
    collar, hair edge). Returns count, ratio and indices into ``data``.
    """
    names = data.dtype.names
    if not all(f"scale_{i}" in names for i in range(3)):
        return {"count": 0, "ratio": 0.0, "indices": []}
    op = sigmoid(data["opacity"]) > min_opacity
    idx = np.nonzero(op)[0]
    if len(idx) < k + 1:
        return {"count": 0, "ratio": 0.0, "indices": []}
    xyz = xyz_of(data)[idx]
    dist, _ = cKDTree(xyz).query(xyz, k=k)
    spacing = np.maximum(dist[:, -1], 1e-6)
    smax = np.exp(np.max(np.column_stack([data[f"scale_{i}"][idx] for i in range(3)]), axis=1))
    bad = idx[smax > ratio * spacing]
    return {"count": int(len(bad)), "ratio": round(len(bad) / len(idx), 5), "indices": bad.tolist()}


def boxes_mask(xyz: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """Union of axis-aligned boxes: [{"name":..., "min":[x,y,z], "max":[x,y,z]}]."""
    mask = np.zeros(len(xyz), dtype=bool)
    for b in boxes:
        lo, hi = np.asarray(b["min"], float), np.asarray(b["max"], float)
        mask |= np.all((xyz >= lo) & (xyz <= hi), axis=1)
    return mask


def protected_displacement(ref, cand, boxes: list[dict]) -> dict:
    """How much the protected region (boxes, in ref coordinates) moved.

    Same splat count -> per-index displacement. Different count -> distance
    from each protected ref point to its nearest candidate point (a removed or
    moved protected splat shows up as a large distance).
    """
    rx, cx = xyz_of(ref), xyz_of(cand)
    mask = boxes_mask(rx, boxes)
    if not mask.any():
        return {"mode": "none", "protected": 0, "max": 0.0, "p99": 0.0}
    if len(ref) == len(cand):
        d = np.linalg.norm(cx[mask] - rx[mask], axis=1)
        mode = "index"
    else:
        d, _ = cKDTree(cx).query(rx[mask], k=1)
        mode = "nearest"
    return {"mode": mode, "protected": int(mask.sum()), "max": float(d.max()), "p99": float(np.percentile(d, 99))}


def changed_bbox(ref, cand, tol: float = 1e-6):
    """Bounding box of splats that moved (same count only). Used to pick views."""
    if len(ref) != len(cand):
        return None
    rx, cx = xyz_of(ref), xyz_of(cand)
    moved = np.linalg.norm(cx - rx, axis=1) > tol
    if not moved.any():
        return {"moved": 0}
    pts = rx[moved]
    return {"moved": int(moved.sum()), "min": pts.min(0).round(4).tolist(), "max": pts.max(0).round(4).tolist()}
