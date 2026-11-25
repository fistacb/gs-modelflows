"""Shared configuration loader for 3dgs-hybrid-doll-pipeline (eliminates character-specific hardcodes)."""
import copy
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "resources" / "character_config.template.json"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str | Path | None = None) -> dict:
    base = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    if config_path:
        p = Path(config_path)
        if p.exists():
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            base = deep_merge(base, user_cfg)
    return base
