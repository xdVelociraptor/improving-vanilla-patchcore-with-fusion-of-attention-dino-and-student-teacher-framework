"""patchcore_hr: separately named higher-resolution PatchCore variant (EXPLORATORY X3).

Same feature pipeline, distance convention and scoring rule as the protected branch (it subclasses it); only the
input resolution (512-px letterbox -> 64x64 layer2 grid) and coreset fraction (1 %, the author's IM320 setting)
differ. Fitting features (~6 GB per LOCO category) are streamed to a disk memmap and the coreset projection is
computed in chunks, so neither RAM nor the 8 GB GPU has to hold them at once. The protected branch is untouched.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader
from tqdm import tqdm

from hybridad.branches import BranchOutput
from hybridad.branches.patchcore import PatchCoreBranch, ApproximateGreedyCoreset, _ids_sha
from hybridad.data import ImageDataset, grid_valid_fraction
from hybridad.utils import write_json, sha256_file


class ChunkedCoreset(ApproximateGreedyCoreset):
    """Same algorithm and RNG call order as the author sampler; projection done in chunks."""

    def run_memmap(self, feats: np.ndarray, chunk: int = 65536) -> np.ndarray:
        mapper = torch.nn.Linear(feats.shape[1], self.project_dim, bias=False).to(self.device)
        with torch.no_grad():
            reduced = torch.cat([mapper(torch.from_numpy(np.ascontiguousarray(feats[i:i + chunk])).to(self.device))
                                 for i in range(0, len(feats), chunk)])
        n_start = int(np.clip(self.number_of_starting_points, None, len(reduced)))
        start_points = np.random.choice(len(reduced), n_start, replace=False).tolist()
        with torch.no_grad():
            approx = self._batchwise_distances(reduced, reduced[start_points])
            anchor = torch.mean(approx, axis=-1).reshape(-1, 1)
            idx = []
            for _ in tqdm(range(int(len(reduced) * self.percentage)), desc="coreset-hr", leave=False, mininterval=20):
                sel = torch.argmax(anchor).item()
                idx.append(sel)
                d = self._batchwise_distances(reduced, reduced[sel:sel + 1])
                anchor = torch.minimum(anchor, d)
        return np.array(idx)


class PatchCoreHRBranch(PatchCoreBranch):
    name = "patchcore_hr"

    def fit(self, records, out_dir) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "patchcore_meta.json").exists() and (out_dir / "memory_bank.npy").exists():
            return self.load(out_dir)
        ds = ImageDataset(records, self.ctx.data_root, self.spec)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        hw = self._grid_hw()
        n_max = len(records) * hw[0] * hw[1]
        tmp = self.ctx.work_dir / "tmp" / f"pchr_{self.ctx.category}.npy"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        buf = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n_max, self.target_dim))
        n = 0
        for batch in tqdm(loader, desc=f"patchcore_hr features {self.ctx.category}", leave=False, mininterval=20):
            f, _ = self._embed_batch(batch["image"])
            f = f.cpu().numpy()
            vf = grid_valid_fraction(batch["valid"], hw).reshape(len(f), -1).numpy()
            for j in range(len(f)):
                keep = f[j][vf[j] >= self.min_valid]
                buf[n:n + len(keep)] = keep
                n += len(keep)
        feats = buf[:n]
        if not np.isfinite(np.asarray(feats[:: max(1, n // 10000)])).all():
            raise FloatingPointError("non-finite patchcore_hr features")
        seed = int(self.cfg.get("coreset_seed", 0))
        torch.manual_seed(seed)
        np.random.seed(seed)
        idx = ChunkedCoreset(float(self.cfg["coreset_percentage"]), self.ctx.device,
                             int(self.cfg.get("coreset_starting_points", 10)),
                             int(self.cfg.get("coreset_project_dim", 128))).run_memmap(feats)
        bank = np.asarray(feats[idx], dtype=np.float32)
        np.save(out_dir / "memory_bank.npy", bank)
        del buf, feats
        tmp.unlink(missing_ok=True)
        write_json(out_dir / "patchcore_meta.json", {
            "feature_signature": self.feature_signature(), "n_fit_images": len(records), "n_fit_patches": int(n),
            "n_memory": int(len(bank)), "grid_hw": list(hw), "coreset_percentage": float(self.cfg["coreset_percentage"]),
            "coreset_seed": seed, "fit_rel_ids_sha": _ids_sha([r.rel_id for r in records]),
            "manifest_sha256": self.ctx.manifest_sha256, "memory_bank_sha256": sha256_file(out_dir / "memory_bank.npy"),
            "distance": "squared_euclidean", "image_aggregation": "max over valid patches",
            "min_valid_fraction": self.min_valid, "variant": "EXPLORATORY patchcore_hr (X3)"})
        return self.load(out_dir)

    def score(self, records) -> BranchOutput:
        """Streaming version of the parent's scoring (identical rule: max over valid patches of k-NN sq. distance)."""
        ds = ImageDataset(records, self.ctx.data_root, self.spec)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        hw = self._grid_hw()
        n = len(records)
        maps = np.full((n, *hw), np.nan, np.float32)
        s_max = np.full(n, np.nan)
        status = [""] * n
        for batch in tqdm(loader, desc=f"patchcore_hr score {self.ctx.category}", leave=False, mininterval=20):
            f, _ = self._embed_batch(batch["image"])
            f = f.cpu().numpy()
            vf = grid_valid_fraction(batch["valid"], hw).reshape(len(f), -1).numpy()
            for j, i in enumerate(batch["index"].tolist()):
                d, _ = self.index.search(f[j], self.k)
                patch = d.mean(-1)
                v = vf[j] >= self.min_valid
                if not v.any():
                    status[i] = "failed:no_valid_patches"
                    continue
                g = patch.reshape(hw).astype(np.float32)
                g[~v.reshape(hw)] = np.nan
                maps[i] = g
                s_max[i] = float(patch[v].max())
                status[i] = "ok"
        return BranchOutput(rel_ids=[r.rel_id for r in records], scores={"max": s_max}, primary="max", maps=maps,
                            map_kind="patchcore_patch_sq_distance", status=status, extra={"grid_hw": list(hw)})
