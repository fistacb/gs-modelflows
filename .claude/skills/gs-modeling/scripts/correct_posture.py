#!/usr/bin/env python3
"""Remove a forward/back lean seen from the side (90 deg) with a linear depth shear.

Only the tilt is removed; the body's own S-curve (chest, skirt, face in front of
the chest) is kept. Rows are moved along Y (depth) only:

    y' = y - k * (clamp(z, z_feet, z_neck) - z_feet)

so the soles do not move, the trunk is straightened, and everything above the neck
moves as one rigid block (the face is not bent). The shear has det J = 1; the
rotation of the Gaussians is updated by transform_gaussians.

The lean k is measured from the data: for horizontal slices between --fit-lo and
--fit-hi of the height, the depth centre of the torso core (a central X band). Points
behind a gap (hanging ribbons, a detached cape edge, a hood tail) are ignored, so
large back volumes do not pull the line. ``trunk_lean_deg`` (the verify_gate number,
slice medians) is reported before and after as well.

    python scripts/correct_posture.py --ply 02_gen/chosen.ply --out 03_body/work/posture.ply
    python scripts/correct_posture.py --ply in.ply --out out.ply --strength 0.8 --report qa/posture.json
    python scripts/correct_posture.py --ply in.ply --out out.ply --lean-deg 12   # use a measured value instead

Check the 90 deg render before/after (render_views.py --only full:90 --ref <before>).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from ply_io import read_ply, write_ply  # noqa: E402
from splat_ops import _height_frame, opaque_xyz, sole_tilt_deg, transform_gaussians, trunk_lean_deg, xyz_of  # noqa: E402


def body_depth_centre(ys: np.ndarray, gap: float) -> float:
    """Centre between the front and back surface of the part that starts at the front."""
    ys = np.sort(ys)
    breaks = np.nonzero(np.diff(ys) > gap)[0]
    if len(breaks):
        ys = ys[: breaks[0] + 1]  # faces -Y: the body is the front-most run of points
    lo, hi = np.percentile(ys, [10, 90])
    return float((lo + hi) / 2)


def core_profile(xyz: np.ndarray, lo: float = 0.10, hi: float = 0.75, step: float = 0.05,
                 core_width: float = 0.06, gap: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Heights (z) and torso-core depth centres (y) of horizontal slices."""
    z0, h, t = _height_frame(xyz)
    cx = float(np.median(xyz[:, 0]))
    zs, ys = [], []
    for a in np.arange(lo, hi + 1e-9, step):
        m = (t >= a) & (t < a + step)
        core = m & (np.abs(xyz[:, 0] - cx) < core_width * h)
        pick = core if core.sum() >= 30 else m
        if pick.sum() >= 10:
            zs.append(z0 + (a + step / 2) * h)
            ys.append(body_depth_centre(xyz[pick, 1], gap * h))
    return np.array(zs), np.array(ys)


def core_lean_deg(xyz: np.ndarray, **kw) -> float:
    zs, ys = core_profile(xyz, **kw)
    if len(zs) < 3:
        return float("nan")
    return float(np.degrees(np.arctan(np.polyfit(zs, ys, 1)[0])))


def make_shear(z_feet: float, z_neck: float, slope: float):
    def shear(p: np.ndarray) -> np.ndarray:
        out = p.copy()
        out[:, 1] = p[:, 1] - slope * (np.clip(p[:, 2], z_feet, z_neck) - z_feet)
        return out
    return shear


def correct(data, strength: float = 1.0, feet: float = 0.04, neck: float = 0.78,
            lean_deg: float | None = None, fit_lo: float = 0.10, fit_hi: float = 0.75):
    xyz = opaque_xyz(data)
    z0, h, _ = _height_frame(xyz)
    measured = core_lean_deg(xyz, lo=fit_lo, hi=fit_hi)
    lean = measured if lean_deg is None else float(lean_deg)
    if not math.isfinite(lean):
        raise ValueError("could not measure the lean (too few points in the slices)")
    slope = math.tan(math.radians(lean)) * strength
    z_feet, z_neck = z0 + feet * h, z0 + neck * h
    out, rep = transform_gaussians(data, make_shear(z_feet, z_neck, slope))
    after = opaque_xyz(out)
    before_all, after_all = xyz_of(data), xyz_of(out)
    fixed = before_all[:, 2] <= z_feet
    report = {
        "lean_used_deg": round(lean, 2),
        "strength": strength,
        "feet_height": feet,
        "neck_height": neck,
        "head_shift": round(-slope * (z_neck - z_feet), 5),
        "core_lean_deg": {"before": round(measured, 2), "after": round(core_lean_deg(after, lo=fit_lo, hi=fit_hi), 2)},
        "trunk_lean_deg": {"before": round(trunk_lean_deg(xyz), 2), "after": round(trunk_lean_deg(after), 2)},
        "sole_tilt_deg": {"before": round(sole_tilt_deg(xyz), 2), "after": round(sole_tilt_deg(after), 2)},
        "feet_points": int(fixed.sum()),
        "feet_max_move": float(np.abs(after_all[fixed] - before_all[fixed]).max()) if fixed.any() else 0.0,
        "jacobian": rep,
    }
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0, help="share of the measured lean to remove")
    ap.add_argument("--lean-deg", type=float, default=None, help="lean to remove instead of the measured one (+ = back)")
    ap.add_argument("--feet", type=float, default=0.04, help="height fraction below which nothing moves")
    ap.add_argument("--neck", type=float, default=0.78, help="height fraction above which the block moves rigidly")
    ap.add_argument("--fit-lo", type=float, default=0.10)
    ap.add_argument("--fit-hi", type=float, default=0.75)
    ap.add_argument("--report", default=None, help="JSON report path (default: <out>.posture.json)")
    args = ap.parse_args()
    data, _ = read_ply(args.ply)
    out, rep = correct(data, args.strength, args.feet, args.neck, args.lean_deg, args.fit_lo, args.fit_hi)
    write_ply(args.out, out)
    rp = Path(args.report or f"{args.out}.posture.json")
    rp.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(f"core lean {rep['core_lean_deg']['before']:+.1f} -> {rep['core_lean_deg']['after']:+.1f} deg, "
          f"trunk_lean {rep['trunk_lean_deg']['before']:+.1f} -> {rep['trunk_lean_deg']['after']:+.1f}, "
          f"sole {rep['sole_tilt_deg']['before']:+.1f} -> {rep['sole_tilt_deg']['after']:+.1f}, "
          f"feet moved {rep['feet_max_move']:.1e}, head shift {rep['head_shift']:+.3f} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
