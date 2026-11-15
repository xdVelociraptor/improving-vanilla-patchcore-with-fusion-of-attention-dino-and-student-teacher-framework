"""Composition / layout evidence: Part-Occupancy Composition Memory (POCM).

This is this project's own engineering design (inspired by, NOT a reproduction of, CSAD's
patch-histogram idea and PSAD's part-level reasoning). Novelty is NOT claimed. Its components
are "visual clusters" of DINOv2 features; they have not been verified to be semantic parts, and
the outputs are called occupancy descriptors, not component counts.

Pipeline (all fitting uses normal FIT images only):
 1. Frozen DINOv2 ViT-B/14-reg4 final-layer normalized patch tokens (special tokens dropped) on
    a 32x32 grid; cells with < 50 % real pixels (padding) are excluded everywhere.
 2. PCA (fit on a seeded subsample of fit-image tokens) to `pca_dim`, then L2 normalization.
 3. K visual clusters by k-means on the normalized projections (seeded, several restarts);
    soft assignment a_pk = softmax_k(cos(f_p, c_k) / tau).
 4. Per image, separate descriptor groups (mass is never folded into appearance):
      mass       m_k  = sum_p a_pk / n_valid          (fraction of VALID image area)
      appearance mu_k = sum_p a_pk f_p / sum_p a_pk    (only where the cluster is present)
      layout     h_kg = cluster-k occupancy of cell g of a GxG grid laid over the valid region
                        (enabled only for categories where position is meaningful)
      adjacency  A_kl = co-occurrence of clusters k,l across 4-neighbour valid cell pairs
                        (translation/rotation invariant relational layout)
      counts     optional connected-component counts per cluster (DISABLED by default: occupancy
                 is not instance count unless instance separation is inspected)
 5. Group distances (Hellinger-style L1 on sqrt of proportions; 1 - cosine for appearance,
    mass-weighted over clusters present in both images; if no cluster is present in both, the
    appearance distance takes a pre-defined fallback = the maximum leave-one-out value on fit
    images). Each group is scaled by the median leave-one-out nearest-neighbour distance among
    fit images, the scaled groups are summed, and the image score is the mean joint distance to
    the k nearest FIT images (multi-mode memory: every normal assembly is kept, none averaged).
 6. No anomaly map is produced: this branch gives image-level evidence only. Hard cluster maps
    are stored for inspection as `extra.cluster_map`, not as localization.
Failures (too few valid cells, non-finite descriptors) are recorded per image and handled by the
pre-registered missing-score policy (flag as anomalous), never as a confident normal score.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from ..backbones import DinoEncoder
from ..cache import FeatureCache
from ..data import ImageDataset, TransformSpec, grid_valid_fraction
from ..provenance import WEIGHTS
from ..utils import write_json, read_json, sha256_file
from . import BranchContext, BranchOutput

GROUPS = ("mass", "appearance", "layout", "adjacency", "counts")


class CompositionBranch:
    name = "composition"

    def __init__(self, cfg: dict, ctx: BranchContext):
        self.cfg = cfg
        self.ctx = ctx
        self.spec = TransformSpec.from_cfg(cfg["transform"][ctx.dataset])
        side = self.spec.crop or self.spec.size
        self.hw = (side // 14, side // 14)
        self.layer = int(cfg.get("feature_layer", 11))
        self.min_valid = float(cfg.get("min_valid_fraction", 0.5))
        self.groups = [g for g in GROUPS if cfg["groups"].get(g, False)]
        if not self.groups:
            raise ValueError("composition branch needs at least one descriptor group")
        self.encoder = DinoEncoder(cfg.get("pos_interp", "dinomaly")).to(ctx.device)
        self.cache = FeatureCache(ctx.work_dir / "cache", "dino_tokens") if cfg.get("cache", True) else None
        self.state: dict | None = None

    # ------------------------------------------------------------------ features
    def _key(self, r) -> dict:
        return FeatureCache.make_key(
            image_sha256=r.pixel_sha256, image_id=r.rel_id, transform=self.spec.signature(), augmentation=None,
            backbone=f"dinov2_vitb14_reg4/{self.encoder.pos_interp}", weights_sha256=WEIGHTS["dinov2_vitb14_reg4"]["sha256"],
            layers=[self.layer], params={"final_norm": True, "tokens": "patch_only"}, precision="fp32->fp16 storage")

    @torch.no_grad()
    def _tokens(self, records) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per image: (h*w, 768) float32 patch tokens and (h*w,) valid flags."""
        ds = ImageDataset(records, self.ctx.data_root, self.spec)
        toks, valid = [], []
        for i in tqdm(range(len(ds)), desc=f"composition tokens {self.ctx.category}", leave=False, mininterval=10):
            r = records[i]
            item = ds[i]
            v = (grid_valid_fraction(item["valid"][None], self.hw)[0, 0] >= self.min_valid).reshape(-1).numpy()
            hit = self.cache.get(self._key(r)) if self.cache is not None else None
            if hit is None:
                out = self.encoder(item["image"][None].to(self.ctx.device), [self.layer], final_norm_for=(self.layer,))
                t = out[self.layer][0, self.encoder.num_special:, :]
                assert t.shape[0] == self.hw[0] * self.hw[1]
                hit = t.float().cpu().numpy().astype(np.float16)
                if self.cache is not None:
                    self.cache.put(self._key(r), hit)
            toks.append(hit.astype(np.float32))
            valid.append(v)
        return toks, valid

    # ------------------------------------------------------------------ descriptors
    def _project(self, t: np.ndarray) -> np.ndarray:
        z = (t - self.state["pca_mean"]) @ self.state["pca_components"].T
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

    def _assign(self, z: np.ndarray) -> np.ndarray:
        logits = (z @ self.state["centers"].T) / float(self.cfg["tau"])
        logits -= logits.max(1, keepdims=True)
        a = np.exp(logits)
        return a / a.sum(1, keepdims=True)

    def _layout_cells(self, v: np.ndarray) -> np.ndarray:
        """Cell index (0..G*G-1) of each grid position, over the bounding box of the valid region."""
        G = int(self.cfg.get("layout_grid", 4))
        h, w = self.hw
        vv = v.reshape(h, w)
        rows, cols = np.where(vv)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        rr = np.clip(((np.arange(h) - r0) * G) // max(1, r1 - r0), 0, G - 1)
        cc = np.clip(((np.arange(w) - c0) * G) // max(1, c1 - c0), 0, G - 1)
        return (rr[:, None] * G + cc[None, :]).reshape(-1)

    def describe(self, tokens: np.ndarray, valid: np.ndarray) -> tuple[dict | None, str, np.ndarray | None]:
        K = self.state["centers"].shape[0]
        n_valid = int(valid.sum())
        if n_valid < int(self.cfg.get("min_valid_cells", 16)):
            return None, "failed:too_few_valid_cells", None
        z = self._project(tokens[valid])
        if not np.isfinite(z).all():
            return None, "failed:nonfinite_features", None
        a = self._assign(z)
        d: dict = {}
        mass = a.sum(0)
        d["mass"] = mass / n_valid
        present = mass >= float(self.cfg.get("min_presence_cells", 2.0))
        app = (a.T @ z) / np.maximum(mass[:, None], 1e-8)
        app /= np.linalg.norm(app, axis=1, keepdims=True) + 1e-8
        app[~present] = 0.0
        d["appearance"] = app
        d["present"] = present.astype(np.float32)
        full = np.zeros((valid.size, K), np.float32)
        full[valid] = a
        h, w = self.hw
        if "layout" in self.groups:
            G = int(self.cfg.get("layout_grid", 4))
            cell = self._layout_cells(valid)
            lay = np.zeros((K, G * G), np.float64)
            cnt = np.bincount(cell[valid], minlength=G * G).astype(np.float64)
            for g in range(G * G):
                sel = valid & (cell == g)
                if sel.any():
                    lay[:, g] = full[sel].sum(0)
            d["layout"] = (lay / np.maximum(cnt[None], 1.0)).reshape(-1)
        if "adjacency" in self.groups:
            A = np.zeros((K, K), np.float64)
            F3 = full.reshape(h, w, K)
            V2 = valid.reshape(h, w)
            n_pairs = 0
            for dy, dx in ((0, 1), (1, 0)):
                p = F3[: h - dy, : w - dx].reshape(-1, K)
                q = F3[dy:, dx:].reshape(-1, K)
                ok = (V2[: h - dy, : w - dx] & V2[dy:, dx:]).reshape(-1)
                A += p[ok].T @ q[ok]
                n_pairs += int(ok.sum())
            A = (A + A.T) / 2.0 / max(n_pairs, 1)
            d["adjacency"] = A[np.triu_indices(K)]
        hard = np.full(valid.size, -1, np.int16)
        hard[valid] = a.argmax(1)
        if "counts" in self.groups:
            mc = int(self.cfg.get("count_min_cells", 3))
            counts = np.zeros(K)
            H2 = hard.reshape(h, w)
            for k in range(K):
                lab, n = ndimage.label(H2 == k)
                if n:
                    sizes = np.bincount(lab.reshape(-1))[1:]
                    counts[k] = int((sizes >= mc).sum())
            d["counts"] = counts
        if not all(np.isfinite(np.asarray(v)).all() for v in d.values()):
            return None, "failed:nonfinite_descriptor", None
        return d, "ok", hard.reshape(h, w)

    def _group_dist(self, g: str, X: dict, Y: dict) -> float:
        if g == "mass":
            return float(np.abs(np.sqrt(X["mass"]) - np.sqrt(Y["mass"])).sum())
        if g == "appearance":
            both = (X["present"] > 0) & (Y["present"] > 0)
            if not both.any():
                return float("nan")   # replaced by the pre-defined fallback
            wts = (X["mass"] + Y["mass"])[both]
            cos = (X["appearance"][both] * Y["appearance"][both]).sum(1)
            return float((wts * (1 - cos)).sum() / wts.sum())
        if g in ("layout", "adjacency"):
            return float(np.abs(np.sqrt(np.maximum(X[g], 0)) - np.sqrt(np.maximum(Y[g], 0))).sum())
        if g == "counts":
            return float(np.abs(X["counts"] - Y["counts"]).sum())
        raise KeyError(g)

    def _dist_matrix(self, Q: list[dict], M: list[dict]) -> dict[str, np.ndarray]:
        out = {}
        for g in self.groups:
            D = np.empty((len(Q), len(M)))
            for i, q in enumerate(Q):
                for j, m in enumerate(M):
                    D[i, j] = self._group_dist(g, q, m)
            out[g] = D
        return out

    # ------------------------------------------------------------------ fit
    def _signature(self, records) -> dict:
        keys = ("feature_layer", "pca_dim", "n_clusters", "tau", "kmeans_restarts", "fit_token_subsample", "groups",
                "layout_grid", "min_presence_cells", "min_valid_cells", "knn", "count_min_cells", "pos_interp")
        return {"cfg": {k: self.cfg.get(k) for k in keys}, "transform": self.spec.signature(), "seed": self.ctx.seed,
                "fit_ids": hashlib.sha256("\n".join(sorted(r.rel_id for r in records)).encode()).hexdigest()}

    def fit(self, records, out_dir) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "composition_state.npz"
        if path.exists():
            return self.load(out_dir)
        toks, valid = self._tokens(records)
        rng = np.random.default_rng([self.ctx.seed, 7])
        pool = np.concatenate([t[v] for t, v in zip(toks, valid)])
        n_sub = min(len(pool), int(self.cfg.get("fit_token_subsample", 100000)))
        sub = pool[rng.choice(len(pool), n_sub, replace=False)]
        pca = PCA(n_components=int(self.cfg["pca_dim"]), random_state=self.ctx.seed).fit(sub)
        self.state = {"pca_mean": pca.mean_.astype(np.float32), "pca_components": pca.components_.astype(np.float32)}
        zs = self._project(sub)
        km = KMeans(n_clusters=int(self.cfg["n_clusters"]), n_init=int(self.cfg.get("kmeans_restarts", 4)),
                    random_state=self.ctx.seed).fit(zs)
        centers = km.cluster_centers_ / (np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-8)
        self.state["centers"] = centers.astype(np.float32)
        descs, fails = [], []
        for t, v, r in zip(toks, valid, records):
            d, st, _ = self.describe(t, v)
            if d is None:
                fails.append((r.rel_id, st))
            else:
                descs.append(d)
        if len(descs) < int(self.cfg["knn"]) + 2:
            raise RuntimeError(f"only {len(descs)} usable fitting descriptors")
        D = self._dist_matrix(descs, descs)
        for g in D:
            np.fill_diagonal(D[g], np.nan)   # leave-one-out: never match an image to itself
        app_fallback = float(np.nanmax(D["appearance"])) if "appearance" in D else 0.0
        if "appearance" in D:
            D["appearance"] = np.where(np.isnan(D["appearance"]) & ~np.eye(len(descs), dtype=bool), app_fallback,
                                       D["appearance"])
        scales = {}
        for g in D:
            nn = np.nanmin(D[g], axis=1)
            s = float(np.median(nn))
            scales[g] = s if s > 1e-12 else float(np.nanmean(D[g]) + 1e-12)
        self.state.update({"scales": scales, "app_fallback": app_fallback})
        self.memory = descs
        mass = np.stack([d["mass"] for d in descs])
        np.savez_compressed(path, pca_mean=self.state["pca_mean"], pca_components=self.state["pca_components"],
                            centers=self.state["centers"], **{f"mem__{k}": np.stack([d[k] for d in descs])
                                                              for k in descs[0]})
        meta = {"signature": self._signature(records), "groups": self.groups, "scales": scales,
                "appearance_fallback": app_fallback, "n_memory": len(descs), "fit_failures": fails,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "cluster_mean_mass_on_fit": mass.mean(0).tolist(),
                "clusters_never_present": [int(k) for k in np.where(mass.max(0) < 1e-3)[0]],
                "state_sha256": sha256_file(path), "manifest_sha256": self.ctx.manifest_sha256,
                "terminology": "visual clusters / occupancy descriptors (semantic part consistency not verified)"}
        write_json(out_dir / "composition_meta.json", meta)
        return self.load(out_dir)

    def load(self, out_dir) -> dict:
        meta = read_json(out_dir / "composition_meta.json")
        path = out_dir / "composition_state.npz"
        if sha256_file(path) != meta["state_sha256"]:
            raise RuntimeError(f"{path} changed after fitting")
        z = np.load(path)
        self.state = {"pca_mean": z["pca_mean"], "pca_components": z["pca_components"], "centers": z["centers"],
                      "scales": meta["scales"], "app_fallback": meta["appearance_fallback"]}
        keys = [k[5:] for k in z.files if k.startswith("mem__")]
        n = z[f"mem__{keys[0]}"].shape[0]
        self.memory = [{k: z[f"mem__{k}"][i] for k in keys} for i in range(n)]
        return meta

    # ------------------------------------------------------------------ score
    def score(self, records) -> BranchOutput:
        if self.state is None:
            raise RuntimeError("composition branch not fitted/loaded")
        toks, valid = self._tokens(records)
        n, k = len(records), int(self.cfg["knn"])
        joint = np.full(n, np.nan)
        per_group = {g: np.full(n, np.nan) for g in self.groups}
        status, cmaps = [], np.full((n, *self.hw), -1, np.int16)
        for i, (t, v) in enumerate(zip(toks, valid)):
            d, st, hard = self.describe(t, v)
            status.append(st)
            if d is None:
                continue
            cmaps[i] = hard
            D = self._dist_matrix([d], self.memory)
            if "appearance" in D:
                D["appearance"] = np.nan_to_num(D["appearance"], nan=self.state["app_fallback"])
            tot = sum(D[g][0] / self.state["scales"][g] for g in self.groups)
            nn = np.argsort(tot, kind="stable")[:k]
            joint[i] = float(tot[nn].mean())
            for g in self.groups:
                per_group[g][i] = float((D[g][0][nn] / self.state["scales"][g]).mean())
        scores = {"joint_knn": joint}
        scores.update({f"group__{g}": v for g, v in per_group.items()})
        return BranchOutput(rel_ids=[r.rel_id for r in records], scores=scores, primary="joint_knn", maps=None,
                            map_kind="none (image-level evidence only)", status=status,
                            extra={"cluster_map": cmaps})
