"""Build a modular schema v2.1 rig_skeleton.json from a normalized 3DGS PLY and a JSON recipe.

Supports:
  - Extending an existing v2.0 humanoid rig (`--base-skeleton`) with cape/hair/wing/tail/ribbon/jaw bones
  - Building a quadruped or custom rig from scratch (`--recipe` with `archetype: "quadruped" | "custom"`)
  - Computing and validating Harmonic skin weights (`--verify-weights`)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
SHARED_DIR = SCRIPTS_DIR.parents[2] / "shared"
for p in (str(SCRIPTS_DIR), str(SHARED_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ply_io import read_ply
from harmonic_dqs import add_forearm_twist_bones, compute_harmonic_skin_weights
from skeleton_builder import build_skeleton_from_recipe


def main():
    ap = argparse.ArgumentParser(description="Build modular v2.1 rig_skeleton.json from PLY + recipe JSON.")
    ap.add_argument("--ply", required=True, help="Normalized 3DGS .ply (Z-up, -Y forward, +X left)")
    ap.add_argument("--recipe", required=True, help="Modular rig recipe JSON")
    ap.add_argument("--out", required=True, help="Output rig_skeleton.json path")
    ap.add_argument("--base-skeleton", default=None, help="Optional existing rig_skeleton.json to extend")
    ap.add_argument("--verify-weights", action="store_true", help="Compute Harmonic weights and print per-bone stats")
    args = ap.parse_args()

    d, _ = read_ply(Path(args.ply))
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float32)
    op = 1.0 / (1.0 + np.exp(-np.clip(d["opacity"], -20, 20)))
    xyz_op = xyz[op > 0.15]

    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    base_bones = None
    if args.base_skeleton:
        base_skel = json.loads(Path(args.base_skeleton).read_text(encoding="utf-8"))
        base_bones = base_skel["bones"]

    skel = build_skeleton_from_recipe(xyz_op, recipe, base_bones=base_bones)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(skel, indent=2), encoding="utf-8")
    print(f"Saved modular v2.1 skeleton ({len(skel['bones'])} bones, archetype={skel['archetype']}) -> {out_path}")

    if args.verify_weights:
        bones_eval = add_forearm_twist_bones(skel["bones"]) if skel["archetype"] == "humanoid" else skel["bones"]
        sb, W = compute_harmonic_skin_weights(
            xyz,
            bones_eval,
            np.ones(len(xyz), dtype=bool),
            seam_cuts=skel.get("seam_cuts"),
        )
        print(f"Harmonic weights verified: {len(sb)} skin bones, NaN={bool(np.isnan(W).any())}")
        dom = np.argmax(W, axis=1)
        for bi, bname in enumerate(sb):
            cnt = int((dom == bi).sum())
            print(f"  {bname:20s}: {cnt:6d} dominant splats")


if __name__ == "__main__":
    main()
