#!/usr/bin/env python3
"""Environment diagnostics for gs-modelflows (designed for both humans and coding agents)."""
import importlib
import json
import os
import platform
import shutil
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def find_chrome() -> str | None:
    if os.environ.get("SPARK_CHROME"):
        return os.environ["SPARK_CHROME"]
    candidates = [
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return shutil.which("google-chrome") or shutil.which("chromium")


def check_comfyui(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/system_stats", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def main() -> None:
    spark_dir = Path(os.environ.get("GS_SPARK_DIR") or (REPO_ROOT / "renderer" / "spark_demo"))
    vendor_files = ["three.module.js", "three.core.js", "Pass.js", "spark.module.min.js"]
    vendor_ok = all((spark_dir / "vendor" / f).exists() for f in vendor_files)

    comfy_url = os.environ.get("GS_COMFYUI_URL", "http://127.0.0.1:8188")
    blender_bin = os.environ.get("GS_BLENDER_BIN") or shutil.which("blender")

    torch_ok = check_module("torch")
    cuda_ok = False
    if torch_ok:
        import torch

        cuda_ok = bool(torch.cuda.is_available())

    report = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_modules": {
            "numpy": check_module("numpy"),
            "scipy": check_module("scipy"),
            "PIL": check_module("PIL"),
            "cv2": check_module("cv2"),
            "imageio": check_module("imageio"),
            "skimage": check_module("skimage"),
            "torch": torch_ok,
            "torch_cuda": cuda_ok,
        },
        "tools": {
            "node": shutil.which("node"),
            "npm": shutil.which("npm"),
            "chrome": find_chrome(),
            "blender": blender_bin,
        },
        "spark_renderer": {
            "dir": str(spark_dir),
            "capture_mjs": (spark_dir / "capture.mjs").exists(),
            "vendor_synced": vendor_ok,
        },
        "comfyui": {
            "url": comfy_url,
            "dir": os.environ.get("GS_COMFYUI_DIR"),
            "reachable": check_comfyui(comfy_url),
        },
        "flow_readiness": {
            "gs-to-sprite": bool(check_module("numpy") and check_module("PIL") and shutil.which("node") and find_chrome() and vendor_ok),
            "gs-to-mesh": bool(check_module("numpy") and check_module("scipy") and check_module("skimage") and torch_ok and blender_bin),
            "gs-modeling": bool(check_module("numpy") and check_module("scipy") and check_module("PIL") and check_module("cv2") and shutil.which("node") and find_chrome() and vendor_ok),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
