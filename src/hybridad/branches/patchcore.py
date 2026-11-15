"""Protected PatchCore branch, reference-compatible with amazon-science/patchcore-inspection
(commit fcaa92f). Verified against the vendored author code in tests/test_patchcore_reference.py.

Reference behaviour reproduced exactly:
  * WideResNet-50-2 (torchvision IMAGENET1K_V1), features from layer2 and layer3.
  * 3x3 neighbourhood aggregation: Unfold(k=3, stride=1, pad=1) per layer; layer3 patches are
    bilinearly resized to the layer2 grid; each layer's 9*C vector is adaptive-avg-pooled to
    `pretrain_embed_dimension` (1024); the two layers are stacked and adaptive-avg-pooled to
    `target_embed_dimension` (1024).
  * Approximate greedy coreset (10 random starting points, 128-d random linear projection),
    percentage 0.1 for the IM224 baseline.
  * Nearest-neighbour scoring with faiss IndexFlatL2 => SQUARED Euclidean distance, k=1,
    patch score = mean over k neighbours, image score = MAX over patches. No neighbourhood
    re-weighting (the paper's Eq. 7 is NOT in the author code); it is available only as the
    separately named score `paper_reweighted_b{b}`.
  * Segmentation map: bilinear upsampling of the patch grid to the input size + Gaussian filter
    with sigma = 4 input pixels (scipy.ndimage defaults).

Project-level additions that leave reference scores unchanged on unpadded inputs:
  * valid-region handling for letterboxed inputs (LOCO adaptation): patches whose cell is less
    than `min_valid_fraction` real image are excluded from the memory and from the max.
  * the author run script min-max normalizes scores over the TEST set; that is not used here.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..backbones import WRNFeatures
from ..cache import FeatureCache
from ..data import ImageDataset, TransformSpec, grid_valid_fraction
from ..knn import KNNIndex
from ..provenance import WEIGHTS
from ..utils import write_json, read_json, sha256_file, assert_finite
from . import BranchContext, BranchOutput


class PatchMaker:
    def __init__(self, patchsize: int, stride: int = 1):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded = unfolder(features)
        n_patches = []
        for s in features.shape[-2:]:
            n_patches.append(int((s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1))
        unfolded = unfolded.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        return unfolded.permute(0, 4, 1, 2, 3), n_patches


def embed_reference(feature_maps: list[torch.Tensor], patch_maker: PatchMaker, pretrain_dim: int,
                    target_dim: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Op-for-op port of PatchCore._embed (+ Preprocessing/MeanMapper + Aggregator).

    Returns (B * h * w, target_dim) features in row-major patch order and the patch grid (h, w).
    """
    features = [patch_maker.patchify(x) for x in feature_maps]
    patch_shapes = [x[1] for x in features]
    features = [x[0] for x in features]
    ref = patch_shapes[0]
    for i in range(1, len(features)):
        f = features[i]
        dims = patch_shapes[i]
        f = f.reshape(f.shape[0], dims[0], dims[1], *f.shape[2:])
        f = f.permute(0, -3, -2, -1, 1, 2)
        base = f.shape
        f = f.reshape(-1, *f.shape[-2:])
        f = F.interpolate(f.unsqueeze(1), size=(ref[0], ref[1]), mode="bilinear", align_corners=False)
        f = f.squeeze(1)
        f = f.reshape(*base[:-2], ref[0], ref[1])
        f = f.permute(0, -2, -1, 1, 2, 3)
        f = f.reshape(len(f), -1, *f.shape[-3:])
        features[i] = f
    features = [x.reshape(-1, *x.shape[-3:]) for x in features]
    # Preprocessing: one MeanMapper per layer
    pooled = [F.adaptive_avg_pool1d(x.reshape(len(x), 1, -1), pretrain_dim).squeeze(1) for x in features]
    stacked = torch.stack(pooled, dim=1)
    # Aggregator
    agg = F.adaptive_avg_pool1d(stacked.reshape(len(stacked), 1, -1), target_dim).reshape(len(stacked), -1)
    return agg, (ref[0], ref[1])


class ApproximateGreedyCoreset:
    """Port of patchcore.sampler.ApproximateGreedyCoresetSampler (same RNG call order)."""

    def __init__(self, percentage: float, device, number_of_starting_points: int = 10, project_dim: int = 128):
        if not 0 < percentage < 1:
            raise ValueError("coreset percentage must be in (0, 1)")
        self.percentage = percentage
        self.device = device
        self.number_of_starting_points = number_of_starting_points
        self.project_dim = project_dim

    @staticmethod
    def _batchwise_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_a = a.unsqueeze(1).bmm(a.unsqueeze(2)).reshape(-1, 1)
        b_b = b.unsqueeze(1).bmm(b.unsqueeze(2)).reshape(1, -1)
        a_b = a.mm(b.T)
        return (-2 * a_b + a_a + b_b).clamp(0, None).sqrt()

    def run(self, features: np.ndarray) -> np.ndarray:
        """Returns the selected row indices. Caller must seed torch and numpy immediately before."""
        feats = torch.from_numpy(features)
        if feats.shape[1] != self.project_dim:
            mapper = torch.nn.Linear(feats.shape[1], self.project_dim, bias=False)
            mapper = mapper.to(self.device)
            with torch.no_grad():
                reduced = mapper(feats.to(self.device))
        else:
            reduced = feats.to(self.device)
        n_start = int(np.clip(self.number_of_starting_points, None, len(reduced)))
        start_points = np.random.choice(len(reduced), n_start, replace=False).tolist()
        with torch.no_grad():
            approx = self._batchwise_distances(reduced, reduced[start_points])
            anchor = torch.mean(approx, axis=-1).reshape(-1, 1)
            idx = []
            n_samples = int(len(reduced) * self.percentage)
            for _ in tqdm(range(n_samples), desc="coreset", leave=False, mininterval=5):
                sel = torch.argmax(anchor).item()
                idx.append(sel)
                d = self._batchwise_distances(reduced, reduced[sel:sel + 1])
                anchor = torch.cat([anchor, d], dim=-1)
                anchor = torch.min(anchor, dim=1).values.reshape(-1, 1)
        return np.array(idx)


class PatchCoreBranch:
    name = "patchcore"

    def __init__(self, cfg: dict, ctx: BranchContext):
        self.cfg = cfg
        self.ctx = ctx
        self.spec = TransformSpec.from_cfg(cfg["transform"][ctx.dataset])
        self.layers = tuple(cfg["layers"])
        self.pretrain_dim = int(cfg["pretrain_embed_dimension"])
        self.target_dim = int(cfg["target_embed_dimension"])
        self.patch_maker = PatchMaker(int(cfg["patchsize"]), int(cfg.get("patchstride", 1)))
        self.k = int(cfg["anomaly_scorer_num_nn"])
        self.min_valid = float(cfg.get("min_valid_fraction", 0.5))
        self.batch_size = int(cfg.get("batch_size", 8))
        self.backbone = WRNFeatures(self.layers).to(ctx.device)
        self.cache = FeatureCache(ctx.work_dir / "cache", "patchcore") if cfg.get("cache", True) else None
        self.index: KNNIndex | None = None
        self.bank: np.ndarray | None = None

    # ------------------------------------------------------------------ features
    def feature_signature(self) -> dict:
        return {"layers": list(self.layers), "pretrain_dim": self.pretrain_dim, "target_dim": self.target_dim,
                "patchsize": self.patch_maker.patchsize, "stride": self.patch_maker.stride,
                "transform": self.spec.signature(), "precision": "fp32",
                "weights_sha256": WEIGHTS["wide_resnet50_2"]["sha256"]}

    @torch.no_grad()
    def _embed_batch(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        fm = self.backbone(images.to(self.ctx.device, torch.float32))
        feats, hw = embed_reference([fm[l] for l in self.layers], self.patch_maker, self.pretrain_dim, self.target_dim)
        return feats.reshape(images.shape[0], hw[0] * hw[1], -1), hw

    def _features(self, records) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
        """Per-image (h*w, D) features and (h*w,) valid fractions, using the cache when enabled."""
        ds = ImageDataset(records, self.ctx.data_root, self.spec)
        feats: list = [None] * len(records)
        valid: list = [None] * len(records)
        hw = None
        todo = []
        for i, r in enumerate(records):
            if self.cache is not None:
                hit = self.cache.get(self._key(r))
                if hit is not None:
                    feats[i] = hit
                    continue
            todo.append(i)
        loader = DataLoader(torch.utils.data.Subset(ds, list(range(len(records)))), batch_size=self.batch_size,
                            shuffle=False, num_workers=self.ctx.num_workers)
        need = set(todo)
        for batch in tqdm(loader, desc=f"patchcore features {self.ctx.category}", leave=False, mininterval=10):
            idxs = batch["index"].tolist()
            vf = None
            if any(i in need for i in idxs):
                f, hw = self._embed_batch(batch["image"])
                f = f.cpu().numpy().astype(np.float32)
                for j, i in enumerate(idxs):
                    if i in need:
                        feats[i] = f[j]
                        if self.cache is not None:
                            self.cache.put(self._key(records[i]), f[j])
            if hw is None:
                hw = self._grid_hw()
            vf = grid_valid_fraction(batch["valid"], hw).reshape(len(idxs), -1).numpy()
            for j, i in enumerate(idxs):
                valid[i] = vf[j]
        return feats, valid, hw

    def _grid_hw(self) -> tuple[int, int]:
        s = self.spec.crop or self.spec.size
        with torch.no_grad():
            fm = self.backbone(torch.zeros(1, 3, s, s, device=self.ctx.device))
        h, w = fm[self.layers[0]].shape[-2:]
        return int(h), int(w)

    def _key(self, r) -> dict:
        return FeatureCache.make_key(
            image_sha256=r.pixel_sha256, image_id=r.rel_id, transform=self.spec.signature(), augmentation=None,
            backbone="wide_resnet50_2", weights_sha256=WEIGHTS["wide_resnet50_2"]["sha256"], layers=self.layers,
            params={"pretrain_dim": self.pretrain_dim, "target_dim": self.target_dim,
                    "patchsize": self.patch_maker.patchsize, "stride": self.patch_maker.stride},
            precision="fp32")

    # ------------------------------------------------------------------ fit
    def fit(self, records, out_dir) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "patchcore_meta.json"
        bank_path = out_dir / "memory_bank.npy"
        if meta_path.exists() and bank_path.exists():
            return self.load(out_dir)
        feats, valid, hw = self._features(records)
        keep = [f[v >= self.min_valid] for f, v in zip(feats, valid)]
        all_feats = np.concatenate(keep, axis=0).astype(np.float32)
        assert_finite(all_feats, "patchcore fitting features")
        pct = float(self.cfg["coreset_percentage"])
        seed = int(self.cfg.get("coreset_seed", self.ctx.seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        if pct >= 1.0:
            idx = np.arange(len(all_feats))
        else:
            idx = ApproximateGreedyCoreset(pct, self.ctx.device,
                                           int(self.cfg.get("coreset_starting_points", 10)),
                                           int(self.cfg.get("coreset_project_dim", 128))).run(all_feats)
        bank = all_feats[idx]
        np.save(bank_path, bank)
        meta = {
            "feature_signature": self.feature_signature(),
            "n_fit_images": len(records), "n_fit_patches": int(len(all_feats)), "n_memory": int(len(bank)),
            "grid_hw": list(hw), "coreset_percentage": pct, "coreset_seed": seed,
            "fit_rel_ids_sha": _ids_sha([r.rel_id for r in records]),
            "manifest_sha256": self.ctx.manifest_sha256,
            "memory_bank_sha256": sha256_file(bank_path),
            "distance": "squared_euclidean (faiss IndexFlatL2 convention)",
            "image_aggregation": "max over valid patches",
            "min_valid_fraction": self.min_valid,
        }
        write_json(meta_path, meta)
        return self.load(out_dir)

    def load(self, out_dir) -> dict:
        meta = read_json(out_dir / "patchcore_meta.json")
        if meta["feature_signature"] != self.feature_signature():
            raise RuntimeError(f"PatchCore memory at {out_dir} was built with a different feature signature; refusing")
        bank_path = out_dir / "memory_bank.npy"
        if sha256_file(bank_path) != meta["memory_bank_sha256"]:
            raise RuntimeError(f"PatchCore memory bank {bank_path} changed after fitting")
        self.bank = np.load(bank_path)
        self.index = KNNIndex(self.cfg.get("knn_backend", "auto"), device=self.ctx.device).fit(self.bank)
        self.grid_hw = tuple(meta["grid_hw"])
        return meta

    # ------------------------------------------------------------------ score
    def score(self, records) -> BranchOutput:
        if self.index is None:
            raise RuntimeError("PatchCore branch is not fitted/loaded")
        feats, valid, hw = self._features(records)
        h, w = hw
        n = len(records)
        maps = np.full((n, h, w), np.nan, np.float32)
        s_max = np.full(n, np.nan)
        b = int(self.cfg.get("reweight_b", 9))
        s_rw = np.full(n, np.nan)
        status = []
        for i in range(n):
            d, nn_idx = self.index.search(feats[i], self.k)
            patch = d.mean(-1)
            v = valid[i] >= self.min_valid
            if not v.any():
                status.append("failed:no_valid_patches")
                continue
            grid = patch.reshape(h, w).astype(np.float32)
            grid[~v.reshape(h, w)] = np.nan
            maps[i] = grid
            vals = np.where(v, patch, -np.inf)
            j = int(np.argmax(vals))
            s_max[i] = float(patch[j])
            s_rw[i] = self._paper_reweight(feats[i][j], nn_idx[j, 0], b)
            status.append("ok")
        return BranchOutput(
            rel_ids=[r.rel_id for r in records],
            scores={"max": s_max, f"paper_reweighted_b{b}": s_rw},
            primary="max", maps=maps, map_kind="patchcore_patch_sq_distance",
            status=status, extra={"grid_hw": list(hw)})

    def _paper_reweight(self, query: np.ndarray, m_star: int, b: int) -> float:
        """PatchCore paper Eq. 7 (separately named variant; absent from the author code).

        s = (1 - exp(||q - m*||) / sum_{m in N_b(m*)} exp(||q - m||)) * s*, with s* = ||q - m*||
        (Euclidean, not squared), N_b(m*) = b nearest memory items of m* (including m* itself).
        """
        _, nb = self.index.search(self.bank[m_star][None], b)
        dists = np.sqrt(((self.bank[nb[0]] - query[None]) ** 2).sum(-1).astype(np.float64))
        d_star = float(np.sqrt(((self.bank[m_star] - query) ** 2).sum()))
        lse = np.logaddexp.reduce(dists)
        return float((1.0 - np.exp(d_star - lse)) * d_star)


def reference_segmentation(grid: np.ndarray, input_hw: tuple[int, int], sigma: float = 4.0) -> np.ndarray:
    """Author RescaleSegmentor: bilinear to input size, then gaussian_filter(sigma=4).

    NaN (padding) cells are filled with the median of valid cells before smoothing so padding
    cannot bleed into real pixels, and restored to NaN afterwards (by the caller's valid mask).
    """
    g = grid.copy()
    bad = ~np.isfinite(g)
    if bad.all():
        return np.full(input_hw, np.nan, np.float32)
    if bad.any():
        g[bad] = np.nanmedian(g)
    t = torch.from_numpy(g)[None, None].float()
    up = F.interpolate(t, size=input_hw, mode="bilinear", align_corners=False)[0, 0].numpy()
    return ndimage.gaussian_filter(up, sigma=sigma)


def _ids_sha(ids: list[str]) -> str:
    import hashlib
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
