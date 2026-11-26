#!/usr/bin/env python3
"""Move body rows up/down so landmark heights match the design (front view).

Give pairs "current height -> target height" (height fractions 0..1 of the model, or
absolute Z with --absolute). A monotone PCHIP curve through the pairs maps Z; the
soles (the lowest pair, default 0 -> 0) and the top stay where you put them, and
the rows in between stretch or squash smoothly. Only Z moves; transform_gaussians
updates the Gaussians and stops on a fold.

Typical pairs come from measure_landmarks.py (design rows) and the front render of the
model (current rows):

    {"pairs": [[0.0, 0.0], [0.30, 0.33], [0.52, 0.55], [0.80, 0.80], [1.0, 1.0]]}

    python scripts/align_rows.py --ply in.ply --out out.ply --pairs qa/rows.json
    python scripts/align_rows.py --ply in.ply --out out.ply --pair 0.30:0.33 --pair 0.52:0.55

0 -> 0 and 1 -> 1 are added when missing (soles and top fixed). The report lists the
reached height of each pair.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from ply_io import read_ply, write_ply  # noqa: E402
from splat_ops import _height_frame, opaque_xyz, transform_gaussians, xyz_of  # noqa: E402


def build_map(pairs: list[tuple[float, float]], z0: float, h: float, absolute: bool = False):
    pts = [((s - z0) / h, (d - z0) / h) if absolute else (s, d) for s, d in pairs]
    srcs = {round(s, 6) for s, _ in pts}
    if 0.0 not in srcs:
        pts.append((0.0, 0.0))
    if 1.0 not in srcs:
        pts.append((1.0, 1.0))
    pts.sort()
    src = np.array([p[0] for p in pts])
    dst = np.array([p[1] for p in pts])
    if np.any(np.diff(src) <= 0):
        raise ValueError("source heights must be distinct")
    if np.any(np.diff(dst) <= 0):
        raise ValueError("target heights must keep the same order (otherwise rows fold)")
    curve = PchipInterpolator(src, dst, extrapolate=False)
    lo_shift, hi_shift = dst[0] - src[0], dst[-1] - src[-1]

    def remap(p: np.ndarray) -> np.ndarray:
        t = (p[:, 2] - z0) / h
        out_t = curve(np.clip(t, src[0], src[-1]))
        out_t = np.where(t < src[0], t + lo_shift, out_t)
        out_t = np.where(t > src[-1], t + hi_shift, out_t)
        out = p.copy()
        out[:, 2] = z0 + out_t * h
        return out

    return remap, list(zip(src.tolist(), dst.tolist()))


def align(data, pairs, absolute: bool = False):
    xyz = opaque_xyz(data)
    z0, h, _ = _height_frame(xyz)
    remap, used = build_map(pairs, z0, h, absolute)
    out, rep = transform_gaussians(data, remap)
    reached = remap(np.array([[0.0, 0.0, z0 + s * h] for s, _ in used]))[:, 2]
    report = {
        "frame": {"z0": z0, "height": h},
        "pairs": [{"src": s, "target": d, "reached": round(float((r - z0) / h), 5)} for (s, d), r in zip(used, reached)],
        "max_row_move": float(np.abs(xyz_of(out)[:, 2] - xyz_of(data)[:, 2]).max()),
        "jacobian": rep,
    }
    return out, report


def parse_pairs(args) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    if args.pairs:
        pairs += [tuple(map(float, p)) for p in json.loads(Path(args.pairs).read_text(encoding="utf-8"))["pairs"]]
    for p in args.pair or []:
        s, _, d = p.partition(":")
        pairs.append((float(s), float(d)))
    if not pairs:
        raise SystemExit("give --pairs <json> or --pair src:dst")
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=None, help='JSON {"pairs": [[src, dst], ...]}')
    ap.add_argument("--pair", action="append", help="src:dst (repeatable)")
    ap.add_argument("--absolute", action="store_true", help="pairs are Z values, not height fractions")
    ap.add_argument("--report", default=None, help="JSON report path (default: <out>.rows.json)")
    args = ap.parse_args()
    data, _ = read_ply(args.ply)
    out, rep = align(data, parse_pairs(args), args.absolute)
    write_ply(args.out, out)
    Path(args.report or f"{args.out}.rows.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    worst = max(abs(p["reached"] - p["target"]) for p in rep["pairs"])
    print(f"{len(rep['pairs'])} rows aligned, worst miss {worst:.4f}, max move {rep['max_row_move']:.4f}, "
          f"det min {rep['jacobian']['det_min']:.3f} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
