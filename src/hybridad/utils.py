"""Seeding, device selection, frozen-module guard and small I/O helpers."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(pref)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA but torch.cuda.is_available() is False")
    return dev


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def module_fingerprint(module: nn.Module) -> str:
    """SHA-256 over every parameter and buffer (name, dtype, shape, bytes)."""
    h = hashlib.sha256()
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        if t.numel():
            h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class FrozenModule(nn.Module):
    """Wraps a pretrained network so it can never leave eval mode or receive gradients.

    `train(True)` on any ancestor is ignored for the wrapped module; parameters have
    requires_grad=False; forward runs under torch.no_grad(). `fingerprint` captures
    parameters AND buffers so tests can assert nothing changed.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()
        self.fingerprint_at_load = module_fingerprint(self.module)

    def train(self, mode: bool = True):  # noqa: D401 - torch API
        super().train(False)
        self.module.eval()
        return self

    def forward(self, *args, **kwargs):
        assert not self.module.training, "frozen module entered training mode"
        with torch.no_grad():
            return self.module(*args, **kwargs)

    def assert_unchanged(self) -> None:
        now = module_fingerprint(self.module)
        if now != self.fingerprint_at_load:
            raise RuntimeError("frozen module parameters or buffers changed")


def assert_finite(t: torch.Tensor | np.ndarray, what: str) -> None:
    ok = bool(torch.isfinite(t).all()) if isinstance(t, torch.Tensor) else bool(np.isfinite(t).all())
    if not ok:
        raise FloatingPointError(f"non-finite values in {what}")
