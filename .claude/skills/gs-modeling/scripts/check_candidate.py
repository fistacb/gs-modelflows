"""Numeric-first check for one trial candidate PLY. Renders only when numbers pass.

One call per trial: numbers -> (optional) render -> one short summary line.
The full report goes to JSON on disk so it does not have to enter the
conversation. Exit code 0 = numbers passed, 1 = failed (no render was made).

Usage:
  python check_candidate.py --cand work/try3.ply --ref 02_gen/chosen.ply \
      --protect qa/protect_boxes.json --out qa/views/stage3/try3 [--render]

  --protect: JSON list of boxes in ref coordinates that must not change, e.g.
      [{"name": "soles", "min": [-1, -1, -0.01], "max": [1, 1, 0.03]}]

Checks (each threshold has a CLI flag):
  NaN/Inf, splat count, normalized height, 3D floaters (same as verify_gate),
  side-view trunk lean and sole tilt, oversized splats, protected-region
  displacement, and the bbox of moved splats (to choose which view to render).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from ply_io import read_ply  # noqa: E402
from splat_ops import (  # noqa: E402
    changed_bbox, opaque_xyz, oversize_splats, protected_displacement,
    sigmoid, sole_tilt_deg, trunk_lean_deg, xyz_of,
)
from verify_gate import check_3d_floaters, write_json_nobom  # noqa: E402


def evaluate(cand, ref=None, boxes=None, *, max_lean=8.0, max_sole=3.0, max_floater=0.005,
             max_oversize=0.002, max_protect=0.002, min_count=10000) -> tuple[dict, list[str]]:
    fails: list[str] = []
    xyz = xyz_of(cand)
    nan_inf = int(sum(int((~np.isfinite(cand[f])).sum()) for f in cand.dtype.names))
    z0, z1 = np.percentile(xyz[:, 2], [0.5, 99.5])
    height = float(z1 - z0)
    op_xyz = opaque_xyz(cand)
    floaters = check_3d_floaters(op_xyz.astype(np.float32), max_floater)
    lean, sole = trunk_lean_deg(op_xyz), sole_tilt_deg(op_xyz)
    over = oversize_splats(cand)
    m = {
        "count": int(len(cand)), "nan_inf": nan_inf, "height": round(height, 4),
        "floater_ratio": floaters["isolated_ratio"], "lean_deg": round(lean, 2), "sole_deg": round(sole, 2),
        "oversize_count": over["count"], "oversize_ratio": over["ratio"],
    }
    if nan_inf:
        fails.append(f"nan_inf={nan_inf}")
    if len(cand) < min_count:
        fails.append(f"count={len(cand)}<{min_count}")
    if not 0.75 <= height <= 1.25:
        fails.append(f"height={height:.3f}")
    if floaters["floater_flag"]:
        fails.append(f"floaters={floaters['isolated_ratio']*100:.2f}%")
    if np.isfinite(lean) and abs(lean) > max_lean:
        fails.append(f"lean={lean:+.1f}deg")
    if np.isfinite(sole) and abs(sole) > max_sole:
        fails.append(f"sole={sole:+.1f}deg")
    if over["ratio"] > max_oversize:
        fails.append(f"oversize={over['count']}")
    if ref is not None:
        if boxes:
            pd = protected_displacement(ref, cand, boxes)
            m["protect"] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in pd.items()}
            if pd["max"] > max_protect:
                fails.append(f"protected_moved={pd['max']:.4f}")
        m["changed"] = changed_bbox(ref, cand)
        m["count_delta"] = int(len(cand) - len(ref))
    return m, fails


def find_blender(explicit: str | None) -> str | None:
    for c in (explicit, os.environ.get("GS_BLENDER_BIN"), shutil.which("blender")):
        if c and Path(c).exists():
            return str(c)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--protect", default=None, help="JSON file with boxes that must not change")
    ap.add_argument("--out", required=True, help="output prefix (report: <out>.json, render: <out>.png)")
    ap.add_argument("--render", action="store_true", help="render with render_ply_blender.py only if numbers pass")
    ap.add_argument("--res", type=int, default=384, help="panel resolution for the tuning render")
    ap.add_argument("--face-center", default=None)
    ap.add_argument("--blender", default=None)
    ap.add_argument("--max-lean-deg", type=float, default=8.0)
    ap.add_argument("--max-sole-tilt-deg", type=float, default=3.0)
    ap.add_argument("--max-floater-ratio", type=float, default=0.005)
    ap.add_argument("--max-oversize-ratio", type=float, default=0.002)
    ap.add_argument("--max-protect-move", type=float, default=0.002)
    args = ap.parse_args()

    cand, _ = read_ply(args.cand)
    ref = read_ply(args.ref)[0] if args.ref else None
    boxes = json.loads(Path(args.protect).read_text(encoding="utf-8")) if args.protect else None
    metrics, fails = evaluate(
        cand, ref, boxes, max_lean=args.max_lean_deg, max_sole=args.max_sole_tilt_deg,
        max_floater=args.max_floater_ratio, max_oversize=args.max_oversize_ratio,
        max_protect=args.max_protect_move,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render = None
    if not fails and args.render:
        blender = find_blender(args.blender)
        if blender is None:
            fails.append("blender_not_found")
        else:
            render = str(out.with_suffix(".png"))
            cmd = [blender, "-b", "--python", str(SCRIPTS / "render_ply_blender.py"), "--",
                   "--ply", args.cand, "--out", render, "--res", str(args.res)]
            if args.face_center:
                cmd += ["--face-center", args.face_center]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not Path(render).exists():
                fails.append("render_failed")
                (out.parent / (out.name + "_render.log")).write_text(r.stdout[-4000:] + r.stderr[-4000:], encoding="utf-8")
                render = None
    report = {"cand": args.cand, "ref": args.ref, "metrics": metrics, "failures": fails,
              "result": "PASS_NUMERIC" if not fails else "FAIL", "render": render}
    write_json_nobom(out.with_suffix(".json"), report)
    keys = ("count", "height", "lean_deg", "sole_deg", "floater_ratio", "oversize_count")
    line = " ".join(f"{k}={metrics[k]}" for k in keys)
    if "protect" in metrics:
        line += f" protect_max={metrics['protect']['max']}"
    print(f"{report['result']} {line}" + (f" | {'; '.join(fails)}" if fails else "") + (f" | render={render}" if render else ""))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
