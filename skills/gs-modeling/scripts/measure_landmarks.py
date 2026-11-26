#!/usr/bin/env python3
"""Measure the silhouette and named landmark heights of a front image (design or render).

The silhouette comes from the alpha channel (front_rgba.png) or, for an opaque render,
from the difference to the background colour (taken from the corners). Heights are
height fractions measured from the soles like the PLY frame: 0 = lowest opaque row,
1 = highest.

Landmarks are rows you read on the image (eyes, chin, shoulders, waist, hem, ankles,
hair tips): give them as --mark name=y or name=x,y in pixels of that image. Eye marks
with x (eye_l / eye_r) also give the eye distance D_eye in height units.

    python scripts/measure_landmarks.py --img 01_input/front_rgba.png --out 01_input/landmarks.json \
        --mark eye_l=402,286 --mark eye_r=470,286 --mark chin=330 --mark waist=520 --mark hem=760
    # the same rows on the model's front render, then the pairs for align_rows.py:
    python scripts/measure_landmarks.py --img qa/views/stage3/try1_full_000.png --out qa/try1_rows.json \
        --mark chin=318 --mark waist=505 --mark hem=742 --pairs-with 01_input/landmarks.json --pairs-out qa/rows.json

Output JSON: image size, bbox, height_px, marks {name: {y_px, x_px?, t}}, D_eye, and a
width profile (every 2 % of the height: left, right, width, as fractions of the height,
centred on the median column) for outline comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def silhouette(img: np.ndarray, alpha_min: int = 24, bg_tol: float = 18.0) -> np.ndarray:
    if img.shape[2] == 4 and img[..., 3].min() < 250:
        return img[..., 3] >= alpha_min
    rgb = img[..., :3].astype(np.float32)
    corners = np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
                              rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    return np.linalg.norm(rgb - bg, axis=2) > bg_tol


def parse_mark(text: str) -> tuple[str, float | None, float]:
    name, _, val = text.partition("=")
    parts = [float(v) for v in val.split(",")]
    if len(parts) == 1:
        return name.strip(), None, parts[0]
    if len(parts) == 2:
        return name.strip(), parts[0], parts[1]
    raise SystemExit(f"bad --mark {text!r} (use name=y or name=x,y)")


def measure(img: np.ndarray, marks: list[tuple[str, float | None, float]], step: float = 0.02, **kw) -> dict:
    mask = silhouette(img, **kw)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty silhouette")
    top, bottom = int(ys.min()), int(ys.max())
    h = float(bottom - top)
    cx = float(np.median(xs))
    t_of = lambda y: round((bottom - y) / h, 5)  # noqa: E731
    out_marks = {}
    for name, x, y in marks:
        m = {"y_px": y, "t": t_of(y)}
        if x is not None:
            m["x_px"] = x
            m["x"] = round((x - cx) / h, 5)
        out_marks[name] = m
    d_eye = None
    if "eye_l" in out_marks and "eye_r" in out_marks and "x_px" in out_marks["eye_l"] and "x_px" in out_marks["eye_r"]:
        d_eye = round(abs(out_marks["eye_l"]["x_px"] - out_marks["eye_r"]["x_px"]) / h, 5)
    profile = []
    for t in np.arange(0.0, 1.0 + 1e-9, step):
        row = int(round(bottom - t * h))
        cols = np.nonzero(mask[row])[0]
        if len(cols):
            profile.append({"t": round(float(t), 3), "left": round((cols.min() - cx) / h, 4),
                            "right": round((cols.max() - cx) / h, 4), "width": round((cols.max() - cols.min()) / h, 4)})
    return {
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "bbox": [int(xs.min()), top, int(xs.max()), bottom],
        "height_px": h,
        "centre_x_px": cx,
        "marks": out_marks,
        "D_eye": d_eye,
        "profile": profile,
    }


def pairs_with(current: dict, design: dict) -> dict:
    """align_rows pairs: current (render) height -> design height, for marks both files have."""
    names = sorted(set(current["marks"]) & set(design["marks"]), key=lambda n: current["marks"][n]["t"])
    return {"pairs": [[current["marks"][n]["t"], design["marks"][n]["t"]] for n in names], "names": names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mark", action="append", default=[], help="name=y or name=x,y in pixels (repeatable)")
    ap.add_argument("--step", type=float, default=0.02, help="width-profile spacing (height fraction)")
    ap.add_argument("--alpha-min", type=int, default=24)
    ap.add_argument("--bg-tol", type=float, default=18.0, help="colour distance to the background for opaque renders")
    ap.add_argument("--pairs-with", default=None, help="design landmarks.json; writes align_rows pairs")
    ap.add_argument("--pairs-out", default=None, help="pairs JSON path (default: <out>.pairs.json)")
    args = ap.parse_args()
    img = np.array(Image.open(args.img).convert("RGBA"))
    rep = measure(img, [parse_mark(m) for m in args.mark], args.step, alpha_min=args.alpha_min, bg_tol=args.bg_tol)
    rep["source"] = args.img
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    marks = ", ".join(f"{k} {v['t']:.3f}" for k, v in rep["marks"].items())
    print(f"height {rep['height_px']:.0f} px, D_eye {rep['D_eye']}, marks: {marks or '-'} -> {args.out}")
    if args.pairs_with:
        pr = pairs_with(rep, json.loads(Path(args.pairs_with).read_text(encoding="utf-8")))
        pp = Path(args.pairs_out or f"{args.out}.pairs.json")
        pp.write_text(json.dumps(pr, indent=2), encoding="utf-8")
        diffs = ", ".join(f"{n} {d - s:+.3f}" for n, (s, d) in zip(pr["names"], pr["pairs"]))
        print(f"pairs (design - current): {diffs or '-'} -> {pp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
