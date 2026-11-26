"""Labeled, downscaled comparison sheet for decisions (before/after, seed choice).

The main agent should look at ONE small sheet at a decision point instead of
several full-size renders. Default long edge is 1024 px.

Usage:
  python make_sheet.py --out qa/views/stage3/compare.png --long-edge 1024 \
      before.png:"before 90deg" after.png:"after 90deg" [--cols 2]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

BG = (32, 35, 42)
BANNER = 22


def parse_item(item: str) -> tuple[str, str]:
    # "path:label" (Windows drive letters like C:\ are kept in the path)
    head, sep, tail = item.rpartition(":")
    if sep and head and not (len(head) == 1 and head.isalpha()) and Path(head).suffix:
        return head, tail
    return item, Path(item).stem


def build_sheet(items: list[tuple[str, str]], cols: int, long_edge: int) -> Image.Image:
    n = len(items)
    cols = max(1, min(cols or n, n))
    rows = math.ceil(n / cols)
    ims = [Image.open(p).convert("RGB") for p, _ in items]
    cw = max(im.width for im in ims)
    ch = max(im.height for im in ims)
    full_w, full_h = cols * cw, rows * (ch + BANNER)
    s = min(1.0, long_edge / max(full_w, full_h))
    cw2, ch2 = max(1, int(cw * s)), max(1, int(ch * s))
    sheet = Image.new("RGB", (cols * cw2, rows * (ch2 + BANNER)), BG)
    d = ImageDraw.Draw(sheet)
    for i, (im, (_, label)) in enumerate(zip(ims, items)):
        r, c = divmod(i, cols)
        x, y = c * cw2, r * (ch2 + BANNER)
        im.thumbnail((cw2, ch2), Image.LANCZOS)
        sheet.paste(im, (x + (cw2 - im.width) // 2, y + BANNER + (ch2 - im.height) // 2))
        d.text((x + 6, y + 5), label[:60], fill=(235, 235, 235))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("items", nargs="+", help="image[:label]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=0, help="0 = one row")
    ap.add_argument("--long-edge", type=int, default=1024)
    args = ap.parse_args()
    sheet = build_sheet([parse_item(i) for i in args.items], args.cols, args.long_edge)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"sheet={args.out} size={sheet.width}x{sheet.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
