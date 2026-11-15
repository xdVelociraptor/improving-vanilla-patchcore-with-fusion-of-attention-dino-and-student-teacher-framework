"""YAML configuration loading with `extends` inheritance and stable hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve(path: str | Path, relative_to: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if relative_to is not None and (relative_to / p).exists():
        return relative_to / p
    if (CONFIG_DIR / p).exists():
        return CONFIG_DIR / p
    return PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file, recursively applying `extends: [a.yaml, b.yaml]`."""
    path = _resolve(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = deep_merge(merged, load_yaml(_resolve(parent, path.parent)))
    return deep_merge(merged, data)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def get(cfg: dict, dotted: str, default: Any = KeyError) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if default is KeyError:
                raise ConfigError(f"missing config key: {dotted}")
            return default
        cur = cur[part]
    return cur


def project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
