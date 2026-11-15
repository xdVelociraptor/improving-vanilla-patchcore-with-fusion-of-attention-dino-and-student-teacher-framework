"""Branch implementations. Each branch exposes fit / load / score and never reads labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class BranchContext:
    dataset: str
    category: str
    data_root: Path
    work_dir: Path
    device: torch.device
    seed: int
    manifest_sha256: str
    num_workers: int = 0


@dataclass
class BranchOutput:
    rel_ids: list[str]
    scores: dict[str, np.ndarray]          # named image-score aggregations (raw, higher = more anomalous)
    primary: str                           # name of the pre-registered aggregation for this variant
    maps: np.ndarray | None = None         # (N, h, w) float32 grid over the network input; NaN = padding
    map_kind: str = "none"                 # e.g. "patchcore_patch_scores", "st_discrepancy", "cosine_recon"
    geometry: list[dict] = field(default_factory=list)
    status: list[str] = field(default_factory=list)   # "ok" or "failed:<reason>" per image
    extra: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"score__{k}": np.asarray(v, dtype=np.float64) for k, v in self.scores.items()}
        if self.maps is not None:
            arrays["maps"] = self.maps.astype(np.float32)
        for k, v in self.extra.items():
            if isinstance(v, np.ndarray):
                arrays[f"extra__{k}"] = v
        meta = {"rel_ids": self.rel_ids, "primary": self.primary, "map_kind": self.map_kind,
                "geometry": self.geometry, "status": self.status,
                "extra": {k: v for k, v in self.extra.items() if not isinstance(v, np.ndarray)}}
        import json
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "BranchOutput":
        import json
        z = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode())
        scores = {k[len("score__"):]: z[k] for k in z.files if k.startswith("score__")}
        extra = dict(meta.get("extra", {}))
        extra.update({k[len("extra__"):]: z[k] for k in z.files if k.startswith("extra__")})
        return BranchOutput(rel_ids=meta["rel_ids"], scores=scores, primary=meta["primary"],
                            maps=z["maps"] if "maps" in z.files else None, map_kind=meta["map_kind"],
                            geometry=meta["geometry"], status=meta["status"], extra=extra)
