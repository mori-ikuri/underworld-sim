"""Configuration loading.

All tunable parameters live in a JSON file (default: ``config/default.json``
at the repository root). Nothing behavioural is hard-coded in the engine.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.json"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load the default config, optionally deep-merged with an override file."""
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        config = json.load(fh)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            override = json.load(fh)
        config = _deep_merge(config, override)
    return config
