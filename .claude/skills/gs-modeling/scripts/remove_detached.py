#!/usr/bin/env python3
"""Remove floating blobs that are not connected to the body.

Opaque Gaussians are put into a voxel grid (default 0.018 of the height ~1 model) and
grouped into 26-connected components. The largest component (the body) is kept, and
so is every component with at least --keep-fraction of the largest one's points
(separate parts such as a held prop). Everything else is removed, together with the
faint (non-opaque) Gaussians that lie in removed voxels or outside every kept voxel.

--box limits the removal to boxes in the protect-box format of splat_ops.boxes_mask
([{"name", "min": [x, y, z], "max": [x, y, z]}]): only blobs whose points are all
inside a box are removed. Use it when a small detached part belongs to the design.

    python scripts/remove_detached.py --ply in.ply --out out.ply
    python scripts/remove_detached.py --ply in.ply --out out.ply --voxel 0.012 --keep-fraction 0.05
    python scripts/remove_detached.py --ply in.ply --out out.ply --box qa/cleanup_boxes.json

Check the removed blobs in the report (centre and size) before trusting the result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from ply_io import read_ply, write_ply  # noqa: E402
from splat_ops import boxes_mask, sigmoid, xyz_of  # noqa: E402


def components(xyz: np.ndarray, voxel: float):
    """Component label per point (0 = none) and the point count of each label."""
    lo = xyz.min(0)
    idx = np.floor((xyz - lo) / voxel).astype(np.int64)
    grid = np.zeros(idx.max(0) + 1, dtype=bool)
    grid[tuple(idx.T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    point_label = labels[tuple(idx.T)]
    counts = np.bincount(point_label, minlength=n + 1)
    return point_label, counts, labels, lo


def remove_detached(data, voxel: float = 0.018, keep_fraction: float = 0.02, min_opacity: float = 0.15,
                    boxes: list[dict] | None = None):
    xyz_all = xyz_of(data)
    opaque = sigmoid(data["opacity"].astype(np.float64)) >= min_opacity
    xyz = xyz_all[opaque]
    point_label, counts, labels, lo = components(xyz, voxel)
    counts[0] = 0
    biggest = int(counts.max())
    keep_label = counts >= keep_fraction * biggest
    keep_label[0] = False
    if boxes:
        inside = boxes_mask(xyz, boxes)
        n = len(counts)
        all_inside = np.bincount(point_label, weights=inside, minlength=n) >= counts  # every point in a box
        keep_label |= ~all_inside
        keep_label[0] = False
    keep_opaque = keep_label[point_label]

    # faint Gaussians: keep only those in (or next to) a kept voxel
    kept_grid = ndimage.binary_dilation(keep_label[labels], structure=np.ones((3, 3, 3), dtype=bool))
    idx = np.floor((xyz_all[~opaque] - lo) / voxel).astype(np.int64)
    valid = np.all((idx >= 0) & (idx < np.array(kept_grid.shape)), axis=1)
    keep_faint = np.zeros(len(idx), dtype=bool)
    keep_faint[valid] = kept_grid[tuple(idx[valid].T)]

    keep = np.zeros(len(data), dtype=bool)
    keep[np.nonzero(opaque)[0]] = keep_opaque
    keep[np.nonzero(~opaque)[0]] = keep_faint

    removed = []
    for lab in np.nonzero(~keep_label & (counts > 0))[0]:
        p = xyz[point_label == lab]
        removed.append({"points": int(len(p)), "centre": np.round(p.mean(0), 4).tolist(),
                        "size": np.round(p.max(0) - p.min(0), 4).tolist()})
    removed.sort(key=lambda r: -r["points"])
    report = {
        "voxel": voxel,
        "keep_fraction": keep_fraction,
        "components": int((counts > 0).sum()),
        "kept_components": int(keep_label.sum()),
        "largest_points": biggest,
        "removed_points": int((~keep).sum()),
        "removed_opaque": int((~keep_opaque).sum()),
        "removed_faint": int((~keep_faint).sum()),
        "removed_fraction": float((~keep).mean()),
        "removed_blobs": removed[:50],
    }
    return data[keep], report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.018)
    ap.add_argument("--keep-fraction", type=float, default=0.02,
                    help="keep components with at least this share of the largest one's points")
    ap.add_argument("--min-opacity", type=float, default=0.15)
    ap.add_argument("--box", default=None, help="JSON list of boxes; only blobs fully inside a box are removed")
    ap.add_argument("--report", default=None, help="JSON report path (default: <out>.detached.json)")
    args = ap.parse_args()
    data, _ = read_ply(args.ply)
    boxes = json.loads(Path(args.box).read_text(encoding="utf-8")) if args.box else None
    out, rep = remove_detached(data, args.voxel, args.keep_fraction, args.min_opacity, boxes)
    write_ply(args.out, out)
    Path(args.report or f"{args.out}.detached.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"{rep['components']} components, kept {rep['kept_components']}, removed {rep['removed_points']} points "
          f"({rep['removed_fraction']:.2%}, {len(rep['removed_blobs'])} blobs) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
