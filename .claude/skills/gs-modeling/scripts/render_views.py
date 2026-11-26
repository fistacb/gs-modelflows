#!/usr/bin/env python3
"""Fixed-camera Spark renders of a Z-up character PLY (full body and bust sets).

The camera, scale and framing are computed from the data so the same PLY always
gives the same frames, and before/after renders of two candidates line up.

    python scripts/render_views.py --ply 03_body/body.ply --out qa/views/stage3/body
    python scripts/render_views.py --ply try3.ply --out qa/views/stage3/try3 --only full:90
    python scripts/render_views.py --ply try3.ply --ref 02_gen/chosen.ply --out ... --only bust:0,60

Output: <out>_<set>_<angle>.png (angle like p035 / m035 / 000) and <out>_views.json
with the camera and the scale of the front views (units per pixel).

Sets (quality_checklist.md section 1):
    full: 0 / +35 / -35 / 90 / 180      bust: 0 / +35 / -35 / +60 / 90

Renderer: renderer/spark_demo (capture.mjs, capture.js, capture.html; found via GS_SPARK_DIR
or a renderer/spark_demo folder above this script, which install_skills.py copies) copied into a
temporary folder with the extended-covariance options turned on (covSplats, extSplats,
sortRadial) so stretched splats are not shown as stripes. The three/spark vendor files
come from --vendor or the SPARK_VENDOR environment variable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from ply_io import read_ply, write_ply  # noqa: E402
from splat_ops import opaque_xyz  # noqa: E402
from config_loader import resolve_spark_dir  # noqa: E402

RENDERER = resolve_spark_dir()  # GS_SPARK_DIR, then renderer/spark_demo above this script
DEFAULT_VENDOR = "C:/Users/chiro/Documents/Codex/2026-11-19/blender-three-js-spark-mp4-ply/vendor"
SETS = {"full": [0, 35, -35, 90, 180], "bust": [0, 35, -35, 60, 90]}
FOV = 3.0

JS_PATCHES = [
    ("sortRadial: false, autoUpdate: false",
     "sortRadial: true, autoUpdate: false, accumExtSplats: true, covSplats: true"),
    ("url: path, lod: false", "url: path, lod: false, extSplats: true, covSplats: true"),
]
REVIEW_LOOP = """    for (const angle of [0, 45, 90, 135, 180, 225, 270, 315]) {
      await capture([baseline], options.front + angle, `angle-${String(angle).padStart(3, '0')}.png`);
    }"""
SHOT_LOOP = """    const base = options;
    const shots = base.shots ?? [0, 45, 90, 135, 180, 225, 270, 315].map(a => ({ name: `angle-${String(a).padStart(3, '0')}`, angle: a }));
    let framed = null;
    for (const shot of shots) {
      options = { ...base, ...shot };
      const key = JSON.stringify([options.target, options.distance]);
      if (key !== framed) {
        // the first draw after moving the target still uses the old splat transform: draw once and discard
        await capture([baseline], base.front + shot.angle, '_warmup.png');
        framed = key;
      }
      await capture([baseline], base.front + shot.angle, `${shot.name}.png`);
    }
    options = base;"""


def angle_tag(a: int) -> str:
    return f"{'m' if a < 0 else 'p' if a > 0 else ''}{abs(a):03d}"


def parse_only(text: str | None) -> dict[str, list[int]]:
    if not text:
        return {k: list(v) for k, v in SETS.items()}
    out: dict[str, list[int]] = {}
    for part in text.split(";"):
        name, _, angles = part.partition(":")
        name = name.strip()
        if name not in SETS:
            raise SystemExit(f"unknown set {name!r} (use full or bust)")
        out[name] = [int(a) for a in angles.split(",")] if angles.strip() else list(SETS[name])
    return out


def framing(data, bust_center: float = 0.805, bust_span: float = 0.63) -> dict:
    """Camera targets from the opaque points (Z-up, height ~1, faces -Y)."""
    xyz = opaque_xyz(data)
    lo, hi = xyz.min(0), xyz.max(0)
    h = float(hi[2] - lo[2])
    cx = float(np.median(xyz[:, 0]))
    extent = max(h, float(hi[0] - lo[0]))
    full_dist = extent * 1.04 / (2 * math.tan(math.radians(FOV / 2)))
    bz = float(lo[2] + bust_center * h)
    face = xyz[(np.abs(xyz[:, 0] - cx) < 0.035 * h) & (np.abs(xyz[:, 2] - (bz - 0.015 * h)) < 0.035 * h)]
    by = float(np.quantile(face[:, 1], 0.30)) if len(face) > 50 else float(np.median(xyz[:, 1]))
    bust_dist = bust_span * h / (2 * math.tan(math.radians(FOV / 2)))
    return {
        "height": h,
        "full": {"target": [cx, -float((lo[2] + hi[2]) / 2), float(np.median(xyz[:, 1]))], "distance": full_dist},
        "bust": {"target": [cx, -bz, by], "distance": bust_dist},
    }


def to_y_up(data):
    """Rotate +90 deg about X (Z-up, -Y front -> Y-up) for the Spark loader."""
    out = data.copy()
    out["y"], out["z"] = -data["z"], data["y"]
    if "rot_0" in data.dtype.names:
        s = math.sqrt(0.5)
        w, x, y, z = (data[f"rot_{i}"].astype(np.float64) for i in range(4))
        out["rot_0"] = s * (w - x)
        out["rot_1"] = s * (x + w)
        out["rot_2"] = s * (y - z)
        out["rot_3"] = s * (z + y)
    return out


def build_renderer(dst: Path, vendor: Path) -> None:
    for name in ("capture.html", "capture.js", "capture.mjs", "package.json"):
        shutil.copy2(RENDERER / name, dst / name)
    shutil.copytree(vendor, dst / "vendor")
    js = (dst / "capture.js").read_text(encoding="utf-8")
    for old, new in JS_PATCHES:
        if old not in js:
            raise SystemExit(f"capture.js changed; patch target not found: {old}")
        js = js.replace(old, new)
    (dst / "capture.js").write_text(js, encoding="utf-8")
    mjs = (dst / "capture.mjs").read_text(encoding="utf-8")
    if REVIEW_LOOP not in mjs:
        raise SystemExit("capture.mjs changed; review loop not found")
    (dst / "capture.mjs").write_text(mjs.replace(REVIEW_LOOP, SHOT_LOOP), encoding="utf-8")


def render(ply: str, out: str, only: dict[str, list[int]], *, size: int = 720, ref: str | None = None,
           vendor: str | None = None, node: str = "node", background: str = "#252028",
           bust_center: float = 0.805, bust_span: float = 0.63) -> dict:
    data, _ = read_ply(ply)
    frame = framing(read_ply(ref)[0] if ref else data, bust_center, bust_span)
    vendor_dir = Path(vendor or os.environ.get("SPARK_VENDOR") or DEFAULT_VENDOR)
    if not (vendor_dir / "spark.module.min.js").exists():
        raise SystemExit(f"Spark vendor files not found in {vendor_dir} (use --vendor or SPARK_VENDOR)")
    shots = []
    for name, angles in only.items():
        cam = frame[name]
        for a in angles:
            shots.append({"name": f"{name}_{angle_tag(a)}", "angle": a,
                          "target": cam["target"], "distance": cam["distance"]})
    camera = {"width": size, "height": size, "background": background, "pitch": 0, "fov": FOV,
              "front": 0, "side": 90, "target": frame["full"]["target"],
              "distance": frame["full"]["distance"], "shots": shots}
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spark_views_") as tmp:
        root = Path(tmp)
        build_renderer(root, vendor_dir)
        (root / "assets").mkdir()
        write_ply(root / "assets" / "view.ply", to_y_up(data))
        (root / "camera.json").write_text(json.dumps(camera), encoding="utf-8")  # no BOM
        r = subprocess.run([node, "capture.mjs", "review", "view", "camera.json"], cwd=root,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"capture.mjs failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        files = []
        for s in shots:
            src = root / "evidence" / "view" / f"{s['name']}.png"
            dst = Path(f"{out}_{s['name']}.png")
            shutil.copy2(src, dst)
            files.append(str(dst))
    tan = math.tan(math.radians(FOV / 2))
    report = {"ply": ply, "ref_framing": ref, "size": size, "files": files, "framing": frame,
              "units_per_px": {k: 2 * frame[k]["distance"] * tan / size for k in ("full", "bust")}}
    Path(f"{out}_views.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--only", default=None,
                    help="limit sets/angles, e.g. 'full:90' or 'full:0,90;bust:60' (default: both full sets)")
    ap.add_argument("--ref", default=None, help="take the framing from this PLY (before/after with the same camera)")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--vendor", default=None)
    ap.add_argument("--node", default="node")
    ap.add_argument("--bust-center", type=float, default=0.805, help="bust centre as a fraction of height")
    ap.add_argument("--bust-span", type=float, default=0.63, help="bust frame height as a fraction of height")
    args = ap.parse_args()
    rep = render(args.ply, args.out, parse_only(args.only), size=args.size, ref=args.ref, vendor=args.vendor,
                 node=args.node, bust_center=args.bust_center, bust_span=args.bust_span)
    print(f"rendered {len(rep['files'])} views -> {args.out}_*.png "
          f"(full {rep['units_per_px']['full']:.5f} u/px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
