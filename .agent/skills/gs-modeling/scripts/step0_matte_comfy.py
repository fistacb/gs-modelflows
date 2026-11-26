#!/usr/bin/env python3
"""Step 0a (ML matte): cut the character out with BiRefNet through the ComfyUI API.

Use this instead of step0_prep_image.py when the colour-key cut fails: a background
close to the character's colours, gradients or shadows on the background, white
clothing on a white background, fine hair.

Graph: LoadImage -> LoadBackgroundRemovalModel (birefnet.safetensors) -> RemoveBackground
-> MaskToImage -> SaveImage. The soft mask is kept as alpha; only the largest connected
part of alpha >= --alpha-min is kept (stray specks dropped), then the character is
centred on a 1024 canvas with 8 % padding like step0_prep_image.py.

    python scripts/step0_matte_comfy.py --src 01_input/front_raw.png --dst 01_input/front_rgba.png
    python scripts/step0_matte_comfy.py --src raw.png --dst rgba.png --api http://127.0.0.1:8188 --model birefnet.safetensors

ComfyUI must be running. Check the result for gaps that should be transparent (between
arm and body); enclosed gaps are sometimes left filled by the model.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from step0_generate_ply import http_json, upload_image_to_comfyui  # noqa: E402


def matte_graph(image_name: str, model: str, prefix: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": model}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},
        "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": prefix}},
    }


def run_matte(api: str, src: Path, model: str, timeout: float = 240.0) -> np.ndarray:
    name = upload_image_to_comfyui(api, src)
    job = http_json(f"{api}/prompt", {"prompt": matte_graph(name, model, "gs_matte/mask")})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        result = http_json(f"{api}/history/{job}").get(job)
        if result is None:
            time.sleep(1.5)
            continue
        if result["status"]["status_str"] != "success":
            raise RuntimeError(f"background removal failed: {json.dumps(result['status'])[:800]}")
        img = result["outputs"]["5"]["images"][0]
        q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
        with urllib.request.urlopen(f"{api}/view?{q}", timeout=60) as resp:
            return np.array(Image.open(io.BytesIO(resp.read())).convert("L"))
    raise TimeoutError(f"ComfyUI job {job} did not finish in {timeout:.0f} s")


def clean_alpha(mask: np.ndarray, alpha_min: int = 24) -> np.ndarray:
    """Keep the soft mask only on the largest connected part of alpha >= alpha_min."""
    solid = (mask >= alpha_min).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n <= 1:
        raise ValueError("the matte is empty")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = cv2.dilate((labels == largest).astype(np.uint8), np.ones((3, 3), np.uint8))  # keep the soft edge
    return np.where(keep > 0, mask, 0).astype(np.uint8)


def to_canvas(rgb: np.ndarray, alpha: np.ndarray, size: int = 1024, pad: float = 0.08) -> Image.Image:
    ys, xs = np.nonzero(alpha > 16)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tight = np.dstack([rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1]])
    p = int(size * pad)
    scale = (size - 2 * p) / max(tight.shape[:2])
    nw, nh = int(round(tight.shape[1] * scale)), int(round(tight.shape[0] * scale))
    resized = Image.fromarray(tight, mode="RGBA").resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--model", default="birefnet.safetensors")
    ap.add_argument("--alpha-min", type=int, default=24)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    rgb = np.array(Image.open(src).convert("RGB"))
    mask = run_matte(args.api.rstrip("/"), src, args.model)
    if mask.shape != rgb.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR))
    alpha = clean_alpha(mask, args.alpha_min)
    canvas = to_canvas(rgb, alpha, args.size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    preview = dst.parent / f"{dst.stem}_preview.png"
    Image.alpha_composite(Image.new("RGBA", canvas.size, (40, 44, 52, 255)), canvas).convert("RGB").save(preview)
    dropped = int(((mask >= args.alpha_min) & (alpha == 0)).sum())
    print(f"matte {mask.shape[1]}x{mask.shape[0]}, dropped {dropped} px of specks -> {dst} (preview {preview.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
