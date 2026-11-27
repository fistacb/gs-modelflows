"""Compile a gs-stage Theme JSON directly from a reference Theme Template sheet or any moodboard image.

Modes:
  --mode template : Samples the structured 2x5 palette grid from a Theme Template sheet
                    (top-right swatch box: deep shadow, meadow, sunlit moss, highlight cream, stone gray,
                     mist/sky blue, flower accent A, flower accent B, warm earth, dark mortar).
  --mode kmeans   : Extracts a balanced 10-role palette from any general reference illustration or screenshot
                    via fast NumPy K-means clustering sorted by luminance & saturation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def mix(a: list[float], b: list[float], t: float) -> list[float]:
    return [round(a[i] * (1.0 - t) + b[i] * t, 3) for i in range(3)]


def to_hex(rgb: list[float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(np.clip(c * 255.0, 0, 255)) for c in rgb))


def extract_palette_template(img_path: Path) -> dict[str, list[float]]:
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    arr = np.asarray(im, dtype=np.float32) / 255.0

    def sample(xf: float, yf: float) -> list[float]:
        px, py = int(W * xf), int(H * yf)
        col = arr[max(0, py - 4) : min(H, py + 5), max(0, px - 4) : min(W, px + 5)].mean(axis=(0, 1))
        return [round(float(v), 3) for v in col]

    return {
        "deep_forest":     sample(0.850, 0.180),
        "meadow_green":    sample(0.880, 0.180),
        "sunlit_moss":     sample(0.909, 0.180),
        "sunbeam_cream":   sample(0.938, 0.180),
        "stone_gray":      sample(0.968, 0.180),
        "mist_blue":       sample(0.850, 0.245),
        "flower_lavender": sample(0.880, 0.245),
        "flower_pink":     sample(0.909, 0.245),
        "warm_earth":      sample(0.938, 0.245),
        "deep_mortar":     sample(0.968, 0.245),
    }


def extract_palette_kmeans(img_path: Path, k: int = 10, seed: int = 42) -> dict[str, list[float]]:
    im = Image.open(img_path).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
    pts = (np.asarray(im, dtype=np.float32) / 255.0).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    centers = pts[rng.choice(len(pts), size=k, replace=False)].copy()
    for _ in range(18):
        d2 = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        lbl = np.argmin(d2, axis=1)
        for ci in range(k):
            m = lbl == ci
            if m.any():
                centers[ci] = pts[m].mean(axis=0)

    lum = 0.299 * centers[:, 0] + 0.587 * centers[:, 1] + 0.114 * centers[:, 2]
    sat = centers.max(axis=1) - centers.min(axis=1)
    by_lum = np.argsort(lum)
    by_sat = np.argsort(sat)[::-1]
    greenish = np.argsort(centers[:, 1] - 0.5 * (centers[:, 0] + centers[:, 2]))[::-1]
    bluish = np.argsort(centers[:, 2] - 0.5 * (centers[:, 0] + centers[:, 1]))[::-1]
    neutral = np.argsort(sat + np.abs(lum - 0.5))

    def c(idx: int) -> list[float]:
        return [round(float(v), 3) for v in centers[idx]]

    return {
        "deep_forest":     c(int(by_lum[0])),
        "deep_mortar":     c(int(by_lum[min(1, k - 1)])),
        "meadow_green":    c(int(greenish[0])),
        "sunlit_moss":     c(int(greenish[min(1, k - 1)])),
        "sunbeam_cream":   c(int(by_lum[-1])),
        "stone_gray":      c(int(neutral[0])),
        "mist_blue":       c(int(bluish[0])),
        "flower_lavender": c(int(by_sat[0])),
        "flower_pink":     c(int(by_sat[min(1, k - 1)])),
        "warm_earth":      c(int(by_lum[k // 2])),
    }


def build_theme_json(name: str, title: str, source_image: Path, pal: dict[str, list[float]], sunbeams: int = 5, flowers: int = 1100) -> dict:
    bg_rgb = mix(pal["deep_forest"], [0.06, 0.14, 0.14], 0.45)
    return {
        "name": name,
        "title": title,
        "source_image": str(source_image),
        "palette": pal,
        "background_hex": to_hex(bg_rgb),
        "lighting": {
            "ambient": mix(pal["mist_blue"], [0.68, 0.74, 0.72], 0.55),
            "sun_dir": [-0.45, -0.42, 0.78],
            "sun_color": [round(c * 0.48, 3) for c in pal["sunbeam_cream"]],
            "point_light_strength": 0.40,
            "light_colors": {
                "soft_ivory": pal["sunbeam_cream"],
                "soft_aqua":  mix(pal["mist_blue"], [0.45, 0.88, 0.95], 0.55),
                "soft_mint":  mix(pal["sunlit_moss"], [0.85, 0.96, 0.70], 0.45),
            },
        },
        "prop_grade": {
            "tint": [1.02, 1.04, 0.96],
            "saturation": 1.06,
            "brightness": 1.03,
        },
        "surfaces": {
            "forest_floor": {"base_rgb": mix(pal["deep_forest"], pal["meadow_green"], 0.55), "alt_rgb": pal["meadow_green"]},
            "stone_path":   {"base_rgb": mix(pal["stone_gray"], pal["sunbeam_cream"], 0.28), "alt_rgb": pal["stone_gray"]},
            "ruin_stone":   {"base_rgb": mix(pal["stone_gray"], pal["sunbeam_cream"], 0.34), "alt_rgb": mix(pal["stone_gray"], pal["warm_earth"], 0.18)},
            "ruin_wall":    {"base_rgb": mix(pal["stone_gray"], pal["sunbeam_cream"], 0.22), "alt_rgb": mix(pal["stone_gray"], pal["deep_mortar"], 0.25)},
            "mortar":       {"base_rgb": pal["deep_mortar"]},
            "stairs":       {"base_rgb": mix(pal["stone_gray"], pal["sunbeam_cream"], 0.30), "alt_rgb": pal["stone_gray"]},
            "moss":         {"base_rgb": pal["meadow_green"], "alt_rgb": pal["sunlit_moss"]},
        },
        "overhang_moss": {
            "rgb_a": pal["meadow_green"],
            "rgb_b": pal["sunlit_moss"],
            "rgb_shadow": pal["deep_forest"],
        },
        "flowers": {
            "colors": [
                pal["flower_lavender"],
                pal["flower_pink"],
                pal["sunbeam_cream"],
                [0.98, 0.98, 0.95],
            ],
            "count": flowers,
        },
        "atmosphere": {
            "mist_count": 75,
            "mist_rgb": mix(pal["mist_blue"], [0.80, 0.88, 0.92], 0.45),
            "mist_opacity": 0.042,
            "mist_radius": 0.48,
            "sparkle_count": 90,
            "sparkle_rgb": pal["sunbeam_cream"],
            "sunbeam_count": sunbeams,
            "sunbeam_rgb": pal["sunbeam_cream"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile a gs-stage theme.json from a reference Theme Template or moodboard image.")
    ap.add_argument("--image", type=Path, required=True, help="Reference Theme Template or moodboard image.")
    ap.add_argument("--out", type=Path, required=True, help="Output theme.json path.")
    ap.add_argument("--name", default="compiled_theme", help="Theme identifier.")
    ap.add_argument("--title", default=None, help="Human-readable theme title.")
    ap.add_argument("--mode", choices=("template", "kmeans"), default="template",
                    help="template: sample structured 2x5 palette box; kmeans: auto-extract from any image.")
    ap.add_argument("--sunbeams", type=int, default=5, help="Number of volumetric diagonal sunbeam shafts (0 to disable).")
    ap.add_argument("--flowers", type=int, default=1100, help="Number of scattered wildflower splats (0 to disable).")
    args = ap.parse_args()

    pal = extract_palette_template(args.image) if args.mode == "template" else extract_palette_kmeans(args.image)
    theme = build_theme_json(args.name, args.title or args.name, args.image, pal, sunbeams=args.sunbeams, flowers=args.flowers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(theme, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Compiled theme ({args.mode}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
