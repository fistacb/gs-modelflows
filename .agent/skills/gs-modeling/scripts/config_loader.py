"""Shared configuration loader for gs-modeling (supports environment variable overrides)."""
import copy
import json
import os
from pathlib import Path


def _find_template_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "resources" / "character_config.template.json",
        here.parent.parent / "skills" / "gs-modeling" / "resources" / "character_config.template.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


TEMPLATE_PATH = _find_template_path()


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_spark_dir(work_dir: str | Path | None = None) -> Path:
    """Resolve Spark GPU renderer directory via GS_SPARK_DIR -> work_dir/spark_demo -> repo renderer/spark_demo."""
    env_dir = os.environ.get("GS_SPARK_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "capture.mjs").exists():
            return p
    if work_dir is not None:
        w = Path(work_dir) / "spark_demo"
        if (w / "capture.mjs").exists():
            return w
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "renderer" / "spark_demo"
        if (cand / "capture.mjs").exists():
            return cand
    return (Path(work_dir) / "spark_demo") if work_dir is not None else Path("renderer/spark_demo")


def load_config(config_path: str | Path | None = None) -> dict:
    base = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    if config_path:
        p = Path(config_path)
        if p.exists():
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            base = deep_merge(base, user_cfg)
    # Apply environment variable overrides
    if os.environ.get("GS_COMFYUI_URL"):
        base.setdefault("comfyui", {})["api_url"] = os.environ["GS_COMFYUI_URL"].rstrip("/")
    return base
