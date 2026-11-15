"""Content-addressed feature cache that refuses incompatible entries.

The key covers everything that changes the stored tensor: image content hash and ID,
transform signature, augmentation realization (None for deterministic preprocessing),
backbone identity + weight checksum, layer selection, feature-pipeline parameters,
precision and a code version tag. The full key dict is stored next to the tensor and
compared on read, so a hash collision or a stale file raises instead of being used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import stable_hash, canonical_json

CACHE_CODE_VERSION = "cache-v1"


class IncompatibleCacheError(RuntimeError):
    pass


class FeatureCache:
    def __init__(self, root: Path | str, namespace: str):
        self.root = Path(root) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(*, image_sha256: str, image_id: str, transform: dict, augmentation: dict | None,
                 backbone: str, weights_sha256: str, layers, params: dict, precision: str) -> dict:
        return {
            "image_sha256": image_sha256, "image_id": image_id, "transform": transform,
            "augmentation": augmentation, "backbone": backbone, "weights_sha256": weights_sha256,
            "layers": list(layers), "params": params, "precision": precision, "code_version": CACHE_CODE_VERSION,
        }

    def _paths(self, key: dict) -> tuple[Path, Path]:
        h = stable_hash(key, n=32)
        d = self.root / h[:2]
        return d / f"{h}.npy", d / f"{h}.json"

    def get(self, key: dict) -> np.ndarray | None:
        arr_p, meta_p = self._paths(key)
        if not arr_p.exists():
            return None
        if not meta_p.exists():
            raise IncompatibleCacheError(f"cache entry {arr_p} has no metadata; delete it")
        stored = json.loads(meta_p.read_text(encoding="utf-8"))
        if canonical_json(stored) != canonical_json(key):
            raise IncompatibleCacheError(f"cache entry {arr_p} was written for a different key")
        return np.load(arr_p, allow_pickle=False)

    def put(self, key: dict, arr: np.ndarray) -> None:
        arr_p, meta_p = self._paths(key)
        arr_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = arr_p.with_suffix(".tmp.npy")
        np.save(tmp, arr, allow_pickle=False)
        tmp.replace(arr_p)
        meta_p.write_text(canonical_json(key), encoding="utf-8")
