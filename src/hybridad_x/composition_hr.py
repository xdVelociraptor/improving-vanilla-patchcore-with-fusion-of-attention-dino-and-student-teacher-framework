"""composition_hr: the Part-Occupancy Composition Memory at a higher token resolution (EXPLORATORY X3).

Differences from `hybridad.branches.composition` (which it subclasses):
* intended for an 896-px letterbox (64x64 DINOv2 token grid), so small parts span more than one token;
* memory-safe fitting: tokens are streamed from the fp16 cache instead of held in RAM;
* count evidence gated per visual cluster on NORMAL fit images: cluster k contributes its connected-component
  count only if that count equals its modal value (>= 1) in at least `count_stability` of the fit images.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from hybridad.branches.composition import CompositionBranch
from hybridad.data import ImageDataset, grid_valid_fraction
from hybridad.utils import write_json, read_json, sha256_file


class _LazyTokens:
    def __init__(self, branch, records, ds):
        self.b, self.records, self.ds = branch, records, ds

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.b._token_one(self.records[i], self.ds, i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class CompositionHRBranch(CompositionBranch):
    name = "composition_hr"

    def _token_one(self, r, ds, i):
        import torch
        hit = self.cache.get(self._key(r)) if self.cache is not None else None
        if hit is None:
            item = ds[i]
            with torch.no_grad():
                out = self.encoder(item["image"][None].to(self.ctx.device), [self.layer], final_norm_for=(self.layer,))
            t = out[self.layer][0, self.encoder.num_special:, :]
            assert t.shape[0] == self.hw[0] * self.hw[1]
            hit = t.float().cpu().numpy().astype(np.float16)
            if self.cache is not None:
                self.cache.put(self._key(r), hit)
        return hit.astype(np.float32)

    def _tokens(self, records):
        ds = ImageDataset(records, self.ctx.data_root, self.spec)
        valid = []
        for i in range(len(records)):
            v = ds[i]["valid"]
            valid.append((grid_valid_fraction(v[None], self.hw)[0, 0] >= self.min_valid).reshape(-1).numpy())
        return _LazyTokens(self, records, ds), valid

    def _group_dist(self, g, X, Y):
        if g == "counts":
            m = self.state.get("count_mask")
            if m is None or not np.any(m):
                return 0.0
            return float(np.abs(X["counts"][m] - Y["counts"][m]).sum())
        return super()._group_dist(g, X, Y)

    def fit(self, records, out_dir) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "composition_state.npz"
        if path.exists():
            return self.load(out_dir)
        toks, valid = self._tokens(records)
        rng = np.random.default_rng([self.ctx.seed, 7])
        n_total = int(self.cfg.get("fit_token_subsample", 100000))
        per_img = max(1, n_total // len(records))
        sub = []
        for t, v in tqdm(zip(toks, valid), total=len(records), desc=f"composition_hr tokens {self.ctx.category}",
                         leave=False, mininterval=20):
            tv = t[v]
            sub.append(tv[rng.choice(len(tv), min(per_img, len(tv)), replace=False)])
        sub = np.concatenate(sub)
        pca = PCA(n_components=int(self.cfg["pca_dim"]), random_state=self.ctx.seed).fit(sub)
        self.state = {"pca_mean": pca.mean_.astype(np.float32), "pca_components": pca.components_.astype(np.float32)}
        zs = self._project(sub)
        km = KMeans(n_clusters=int(self.cfg["n_clusters"]), n_init=int(self.cfg.get("kmeans_restarts", 4)),
                    random_state=self.ctx.seed).fit(zs)
        self.state["centers"] = (km.cluster_centers_ / (np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-8)
                                 ).astype(np.float32)
        self.state["count_mask"] = np.ones(int(self.cfg["n_clusters"]), bool)   # all counts computed during describe
        descs, fails = [], []
        for t, v, r in zip(toks, valid, records):
            d, st, _ = self.describe(t, v)
            (descs.append(d) if d is not None else fails.append((r.rel_id, st)))
        # count gating on NORMAL fit images
        count_info = {}
        if "counts" in self.groups:
            C = np.stack([d["counts"] for d in descs])
            mode = stats.mode(C, axis=0, keepdims=False).mode
            agree = (C == mode[None]).mean(0)
            mask = (agree >= float(self.cfg.get("count_stability", 0.9))) & (mode >= 1)
            self.state["count_mask"] = mask
            count_info = {"modal_count": mode.tolist(), "agreement": agree.round(3).tolist(),
                          "clusters_with_count_evidence": np.where(mask)[0].tolist()}
        D = self._dist_matrix(descs, descs)
        for g in D:
            np.fill_diagonal(D[g], np.nan)
        app_fallback = float(np.nanmax(D["appearance"])) if "appearance" in D else 0.0
        if "appearance" in D:
            D["appearance"] = np.where(np.isnan(D["appearance"]) & ~np.eye(len(descs), dtype=bool), app_fallback,
                                       D["appearance"])
        scales = {}
        for g in D:
            nn = np.nanmin(D[g], axis=1)
            s = float(np.median(nn))
            scales[g] = s if s > 1e-12 else float(np.nanmean(D[g]) + 1e-12)
        if "counts" in D and not np.any(self.state["count_mask"]):
            scales["counts"] = 1.0   # counts contribute 0 for every pair
        self.state.update({"scales": scales, "app_fallback": app_fallback})
        self.memory = descs
        np.savez_compressed(path, pca_mean=self.state["pca_mean"], pca_components=self.state["pca_components"],
                            centers=self.state["centers"], count_mask=self.state["count_mask"],
                            **{f"mem__{k}": np.stack([d[k] for d in descs]) for k in descs[0]})
        mass = np.stack([d["mass"] for d in descs])
        write_json(out_dir / "composition_meta.json", {
            "signature": self._signature(records), "groups": self.groups, "scales": scales,
            "appearance_fallback": app_fallback, "n_memory": len(descs), "fit_failures": fails,
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "cluster_mean_mass_on_fit": mass.mean(0).tolist(), "count_gating": count_info,
            "state_sha256": sha256_file(path), "manifest_sha256": self.ctx.manifest_sha256,
            "variant": "EXPLORATORY composition_hr (X3)",
            "terminology": "visual clusters / occupancy descriptors (semantic part consistency not verified)"})
        return self.load(out_dir)

    def load(self, out_dir) -> dict:
        meta = super().load(out_dir)
        z = np.load(out_dir / "composition_state.npz")
        self.state["count_mask"] = z["count_mask"]
        return meta
