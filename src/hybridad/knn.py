"""Exact k-nearest-neighbour search with an explicit distance convention.

The PatchCore author code uses ``faiss.IndexFlatL2``, whose returned distances are
**squared** Euclidean. Both backends here return squared Euclidean distances so the
protected branch reproduces that convention; `sqrt` is available as a named variant
(it is monotone, so max-aggregated image rankings are unchanged).

Backends:
  faiss  - IndexFlatL2 (exact), CPU; used when the faiss package imports.
  torch  - chunked exact search on CPU or GPU: ||a||^2 + ||b||^2 - 2 a.b in float32,
           clamped at 0, with an optional float64 re-computation of the selected neighbours
           so near-duplicate points do not suffer catastrophic cancellation.
"""
from __future__ import annotations

import numpy as np
import torch

try:  # pragma: no cover - availability depends on the environment
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:  # noqa: BLE001
    faiss = None
    HAVE_FAISS = False


class KNNIndex:
    def __init__(self, backend: str = "auto", device: str | torch.device = "cpu",
                 query_chunk: int = 4096, bank_chunk: int = 65536, refine_fp64: bool = True):
        if backend == "auto":
            backend = "faiss" if HAVE_FAISS else "torch"
        if backend == "faiss" and not HAVE_FAISS:
            raise RuntimeError("faiss backend requested but faiss is not importable; use backend='torch'")
        self.backend = backend
        self.device = torch.device(device)
        self.query_chunk = query_chunk
        self.bank_chunk = bank_chunk
        self.refine_fp64 = refine_fp64
        self.bank: np.ndarray | None = None
        self._index = None
        self._bank_t: torch.Tensor | None = None

    def fit(self, bank: np.ndarray) -> "KNNIndex":
        bank = np.ascontiguousarray(bank, dtype=np.float32)
        if bank.ndim != 2 or len(bank) == 0:
            raise ValueError(f"memory bank must be a non-empty 2-D array, got {bank.shape}")
        if not np.isfinite(bank).all():
            raise FloatingPointError("memory bank contains non-finite values")
        self.bank = bank
        if self.backend == "faiss":
            self._index = faiss.IndexFlatL2(bank.shape[1])
            self._index.add(bank)
        else:
            self._bank_t = torch.from_numpy(bank).to(self.device)
            self._bank_sq = (self._bank_t.double() ** 2).sum(1).float()
        return self

    def search(self, query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Returns (squared distances [N,k], indices [N,k]) sorted ascending."""
        if self.bank is None:
            raise RuntimeError("index not fitted")
        query = np.ascontiguousarray(query, dtype=np.float32)
        if not np.isfinite(query).all():
            raise FloatingPointError("query features contain non-finite values")
        if self.backend == "faiss":
            d, i = self._index.search(query, k)
            return np.maximum(d, 0.0), i
        return self._torch_search(query, k)

    @torch.no_grad()
    def _torch_search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        out_d, out_i = [], []
        B = self._bank_t
        for qs in range(0, len(query), self.query_chunk):
            q = torch.from_numpy(query[qs:qs + self.query_chunk]).to(self.device)
            q_sq = (q.double() ** 2).sum(1, keepdim=True).float()
            best_d = torch.full((len(q), k), float("inf"), device=self.device)
            best_i = torch.full((len(q), k), -1, dtype=torch.long, device=self.device)
            for bs in range(0, len(B), self.bank_chunk):
                b = B[bs:bs + self.bank_chunk]
                d = (q_sq + self._bank_sq[bs:bs + self.bank_chunk][None] - 2.0 * (q @ b.T)).clamp_min_(0)
                kk = min(k, d.shape[1])
                dv, di = torch.topk(d, kk, dim=1, largest=False)
                cat_d = torch.cat([best_d, dv], 1)
                cat_i = torch.cat([best_i, di + bs], 1)
                sel = torch.topk(cat_d, k, dim=1, largest=False)
                best_d = sel.values
                best_i = torch.gather(cat_i, 1, sel.indices)
            if self.refine_fp64:
                nb = B[best_i.clamp_min(0)].double()
                best_d = ((q.double()[:, None, :] - nb) ** 2).sum(-1).float()
                order = torch.argsort(best_d, dim=1)
                best_d = torch.gather(best_d, 1, order)
                best_i = torch.gather(best_i, 1, order)
            out_d.append(best_d.cpu().numpy())
            out_i.append(best_i.cpu().numpy())
        return np.concatenate(out_d), np.concatenate(out_i)


def brute_force_sq(query: np.ndarray, bank: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """float64 reference for tests."""
    d = ((query.astype(np.float64)[:, None, :] - bank.astype(np.float64)[None]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, idx, 1), idx
