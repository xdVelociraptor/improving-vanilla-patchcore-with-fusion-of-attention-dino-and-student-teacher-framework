"""DINOv2 global-attention reconstruction branch (Dinomaly-style), with named ablations.

Reference: Dinomaly author code (github.com/guojiajeremy/Dinomaly @ 1f252be), files
dinomaly_mvtec_sep.py / dinomaly_mvtec_uni.py, models/uad.py (ViTill), models/vision_transformer.py
(bMlp, Block, LinearAttention2, Attention), utils.py (global_cosine_hm_percent, cal_anomaly_maps,
evaluation_batch, WarmCosineScheduler), optimizers/StableAdamW.py.

Reproduced:
  * frozen DINOv2 ViT-B/14 with 4 registers; target layers [2,3,4,5,6,7,8,9] (zero-based block
    indices); encoder groups [[0,1,2,3],[4,5,6,7]] -> layers 2-5 and 6-9;
  * decoder input = mean of the 8 target-layer outputs over ALL tokens (special tokens included,
    as in ViTill.fuse_feature), through a noisy bottleneck bMlp(768, 3072, 768, dropout 0.2 applied
    to input, hidden and output);
  * 8 decoder Blocks (768-d, 12 heads, mlp_ratio 4, qkv_bias, LayerNorm eps 1e-8) with
    LinearAttention2 (elu+1 kernel, no softmax); no positional embedding in the decoder; decoder
    outputs reversed and grouped [[0..3],[4..7]] -> loose group-wise reconstruction;
  * loss = global cosine (whole feature map of each group flattened per image), hard-mined by
    scaling the gradient of the easiest p fraction of points by 0.1, p warmed linearly 0 -> 0.9
    over 1000 iterations;
  * StableAdamW(lr 2e-3, betas (0.9, 0.999), wd 1e-4, amsgrad, eps 1e-8), WarmCosine 2e-3 -> 2e-4,
    100 warm-up iterations, grad-norm clip 0.1, batch 16, 5000 iterations (per-class script);
  * trunc_normal(std 0.01, [-0.03, 0.03]) init of Linear weights, zero biases, LayerNorm (1, 0);
  * anomaly map = mean over groups of (1 - cosine) between encoder and decoder groups; reference
    image score = mean of the top 1% of the map after upsampling to the input size
    (align_corners=True), resizing to 256 and a 5x5 Gaussian (sigma 4).

Adaptations (named, configurable): full-image preprocessing (MVTec AD: resize to 448, no crop;
LOCO: aspect-preserving letterbox to 448) giving a 32x32 grid; padding excluded from the loss,
pooling and maps (cells with < 50% real pixels); adapted image score = top 1% of VALID grid cells.
Special tokens are never reshaped into the grid; the encoder's attention still sees padding
tokens, which `padding_diagnostic` checks on normal calibration images.
"""
from __future__ import annotations

import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn
from tqdm import tqdm

from ..backbones import DinoEncoder
from ..data import ImageDataset, TransformSpec, grid_valid_fraction, IMAGENET_MEAN, IMAGENET_STD
from ..optim import StableAdamW, WarmCosineScheduler
from ..provenance import WEIGHTS
from ..utils import write_json, read_json, sha256_file
from . import BranchContext, BranchOutput


# ----------------------------------------------------------------------------- decoder parts
class bMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class LinearAttention2(nn.Module):
    """Dinomaly LinearAttention2: phi(x) = elu(x) + 1 kernel attention (reductions in float32)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("...sd,...se->...de", k, v)
        z = 1.0 / torch.einsum("...sd,...d->...s", q, k.sum(dim=-2))
        out = torch.einsum("...de,...sd,...s->...se", kv, q, z)
        out = out.transpose(1, 2).reshape(B, N, C).to(x.dtype)
        return self.proj(out)


class SoftmaxAttention(nn.Module):
    """Standard multi-head softmax attention (Dinomaly `Attention`, attn_drop=0) via fused SDPA."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class IdentityAttention(nn.Module):
    """Matched NO-token-mixing control: same qkv/proj parameters, attention matrix = identity,
    so every token only sees itself (output = proj(v)). Isolates cross-token mixing."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        v = self.qkv(x).reshape(B, N, 3, C)[:, :, 2]
        return self.proj(v)


ATTENTION = {"linear": LinearAttention2, "softmax": SoftmaxAttention, "identity": IdentityAttention}


class Block(nn.Module):
    def __init__(self, dim, num_heads, attn: str, mlp_ratio=4.0, qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-8)
        self.attn = ATTENTION[attn](dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-8)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ReconHead(nn.Module):
    """Trainable part: bottleneck + decoder (+ width projections for the compact head only)."""

    def __init__(self, embed_dim=768, width=768, heads=12, depth=8, attn="linear", bottleneck_drop=0.2,
                 activation_checkpointing=False):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.bottleneck = bMlp(embed_dim, embed_dim * 4, embed_dim, drop=bottleneck_drop)
        self.in_proj = nn.Identity() if width == embed_dim else nn.Linear(embed_dim, width)
        self.out_proj = nn.Identity() if width == embed_dim else nn.Linear(width, embed_dim)
        self.blocks = nn.ModuleList([Block(width, heads, attn) for _ in range(depth)])

    def forward(self, fused: torch.Tensor) -> list[torch.Tensor]:
        x = self.in_proj(self.bottleneck(fused))
        outs = []
        for blk in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                # exact recomputation in backward; blocks contain no dropout, so results are unchanged
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            outs.append(self.out_proj(x))
        return outs[::-1]


def init_trainable(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def _modify_grad(g, inds, factor):
    g = g.clone()
    g[inds.expand_as(g)] *= factor
    return g


def global_cosine_hm_percent(en, de, p, factor, valid=None):
    """Dinomaly global_cosine_hm_percent with an optional (B,1,h,w) validity mask.

    With valid=None (or all ones) it is the reference loss. Masked cells are zeroed in both
    tensors (so they do not enter the global cosine) and excluded from the hard-mining quantile.
    """
    loss = 0.0
    for a, b in zip(en, de):
        a = a.detach()
        if valid is not None:
            a = a * valid
            b = b * valid
        with torch.no_grad():
            point = 1 - F.cosine_similarity(a, b, dim=1).unsqueeze(1)
            pool = point[valid.bool()] if valid is not None else point.reshape(-1)
            k = int(pool.numel() * (1 - p))
            thresh = torch.topk(pool.reshape(-1), k=max(k, 1))[0][-1] if k >= 1 else pool.max() + 1
        loss = loss + torch.mean(1 - F.cosine_similarity(a.reshape(a.shape[0], -1).float(),
                                                         b.reshape(b.shape[0], -1).float(), dim=1))
        if b.requires_grad:
            b.register_hook(partial(_modify_grad, inds=point < thresh, factor=factor))
    return loss / len(en)


# ----------------------------------------------------------------------------- branch
class DinoReconBranch:
    name = "global_attention"

    def __init__(self, cfg: dict, ctx: BranchContext):
        self.cfg = cfg
        self.ctx = ctx
        self.spec = TransformSpec.from_cfg(cfg["transform"][ctx.dataset])
        side = self.spec.crop or self.spec.size
        if side % 14:
            raise ValueError(f"input side {side} not divisible by the 14-pixel patch")
        self.hw = (side // 14, side // 14)
        self.layers = list(cfg["target_layers"])
        self.en_groups = cfg["fuse_layer_encoder"]
        self.de_groups = cfg["fuse_layer_decoder"]
        self.min_valid = float(cfg.get("min_valid_fraction", 0.5))
        mp = cfg.get("matmul_precision", "fp32")
        if mp not in ("fp32", "tf32"):
            raise ValueError("matmul_precision must be fp32 or tf32")
        # Dinomaly pins torch 1.12, whose default is allow_tf32=False for matmuls: fp32 is the faithful setting.
        torch.backends.cuda.matmul.allow_tf32 = mp == "tf32"
        if cfg.get("decoder_precision", "fp32") != "fp32":
            raise NotImplementedError("only an fp32 decoder is implemented")
        self.encoder = DinoEncoder(cfg.get("pos_interp", "dinomaly")).to(ctx.device)
        self.head: ReconHead | None = None

    # ------------------------------------------------------------------ core forward
    def _head(self) -> ReconHead:
        h = self.cfg["head"]
        return ReconHead(768, int(h["width"]), int(h["heads"]), int(h["depth"]), h["attention"],
                         float(h["bottleneck_dropout"]), bool(self.cfg.get("activation_checkpointing", False)))

    def _encode(self, x: torch.Tensor):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.get("encoder_precision") == "bf16"
                            and x.is_cuda):
            outs = self.encoder(x, self.layers)
        en_list = [outs[i].float() for i in self.layers]
        return en_list

    @torch.no_grad()
    def encoder_targets(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Frozen-encoder part only: (fused decoder input over all tokens, encoder group grids)."""
        en_list = self._encode(x)
        fused = torch.stack(en_list, 1).mean(1)
        en = [torch.stack([en_list[i] for i in g], 1).mean(1) for g in self.en_groups]
        return fused, [self.encoder.patch_grid(e, self.hw) for e in en]

    def decode(self, fused: torch.Tensor) -> list[torch.Tensor]:
        de_list = self.head(fused)
        de = [torch.stack([de_list[i] for i in g], 1).mean(1) for g in self.de_groups]
        return [self.encoder.patch_grid(d, self.hw) for d in de]

    def forward_groups(self, x: torch.Tensor):
        fused, en = self.encoder_targets(x)
        return en, self.decode(fused)

    def _cell_valid(self, v: torch.Tensor) -> torch.Tensor:
        return (grid_valid_fraction(v, self.hw) >= self.min_valid).float()

    # ------------------------------------------------------------------ fit
    def _signature(self, records) -> dict:
        import hashlib
        keep = ("target_layers", "fuse_layer_encoder", "fuse_layer_decoder", "head", "iterations", "batch_size",
                "lr", "final_lr", "warmup_iters", "weight_decay", "eps", "grad_clip", "hm_p_final",
                "hm_warmup_iters", "hm_factor", "pos_interp", "encoder_precision", "matmul_precision",
                "decoder_precision", "min_valid_fraction", "mask_loss", "encoder_cache")
        return {"cfg": {k: self.cfg.get(k) for k in keep}, "transform": self.spec.signature(),
                "encoder_sha256": WEIGHTS["dinov2_vitb14_reg4"]["sha256"], "seed": self.ctx.seed,
                "fit_ids": hashlib.sha256("\n".join(sorted(r.rel_id for r in records)).encode()).hexdigest()}

    def _preload(self, records) -> tuple[torch.Tensor, torch.Tensor]:
        spec_raw = TransformSpec(self.spec.kind, self.spec.size, self.spec.crop, "none")
        ds = ImageDataset(records, self.ctx.data_root, spec_raw)
        X = torch.empty((len(ds), 3, *(self.hw[0] * 14, self.hw[1] * 14)), dtype=torch.uint8)
        V = torch.empty((len(ds), 1, self.hw[0] * 14, self.hw[1] * 14), dtype=torch.float32)
        for i in tqdm(range(len(ds)), desc=f"load {self.ctx.category}", leave=False, mininterval=10):
            item = ds[i]
            X[i] = (item["image"] * 255).round().to(torch.uint8)
            V[i] = item["valid"]
        return X, V

    def _encoder_cache(self, records, X: torch.Tensor):
        """Exact cache of the frozen encoder's outputs for the fitting images. No augmentation is used,
        so every iteration would recompute identical values. Stored as float32 .npy memmaps in work_dir,
        keyed by everything that determines them, and shared by all decoder variants."""
        import hashlib
        from numpy.lib.format import open_memmap
        from ..config import stable_hash
        ids = "\n".join(r.rel_id + ":" + r.pixel_sha256 for r in records)
        key = {"fit_ids": hashlib.sha256(ids.encode()).hexdigest(),
               "transform": self.spec.signature(), "encoder_sha256": WEIGHTS["dinov2_vitb14_reg4"]["sha256"],
               "pos_interp": self.cfg.get("pos_interp", "dinomaly"), "layers": self.layers,
               "en_groups": self.en_groups, "encoder_precision": self.cfg.get("encoder_precision", "fp32"),
               "matmul_precision": self.cfg.get("matmul_precision", "fp32"), "version": "enc-cache-v1"}
        d = self.ctx.work_dir / "cache" / "dino_train" / stable_hash(key, n=24)
        meta_p = d / "meta.json"
        names = ["fused"] + [f"en{i}" for i in range(len(self.en_groups))]
        if meta_p.exists():
            if read_json(meta_p)["key"] != key:
                raise RuntimeError(f"encoder cache {d} has a different key; refusing")
            return [np.load(d / f"{n}.npy", mmap_mode="r") for n in names]
        d.mkdir(parents=True, exist_ok=True)
        n = len(X)
        T = self.encoder.num_special + self.hw[0] * self.hw[1]
        arrs = [open_memmap(d / "fused.npy", mode="w+", dtype=np.float32, shape=(n, T, 768))]
        arrs += [open_memmap(d / f"en{i}.npy", mode="w+", dtype=np.float32, shape=(n, 768, *self.hw))
                 for i in range(len(self.en_groups))]
        for s0 in tqdm(range(0, n, 16), desc=f"encoder cache {self.ctx.category}", leave=False, mininterval=10):
            x = self._norm(X[s0:s0 + 16].to(self.ctx.device))
            fused, en = self.encoder_targets(x)
            arrs[0][s0:s0 + len(x)] = fused.cpu().numpy()
            for a, e in zip(arrs[1:], en):
                a[s0:s0 + len(x)] = e.cpu().numpy()
        for a in arrs:
            a.flush()
        del arrs
        write_json(meta_p, {"key": key, "n": n})
        return [np.load(d / f"{nm}.npy", mmap_mode="r") for nm in names]

    def _norm(self, xb: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(IMAGENET_MEAN, device=xb.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=xb.device).view(1, 3, 1, 1)
        return (xb.float() / 255.0 - mean) / std

    def fit(self, records, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / "head_final.pt"
        if final.exists():
            return self.load(out_dir)
        cfg, dev = self.cfg, self.ctx.device
        iters, bs = int(cfg["iterations"]), int(cfg["batch_size"])
        micro = int(cfg.get("micro_batch", bs))
        if bs % micro:
            raise ValueError("batch_size must be a multiple of micro_batch")
        X, V = self._preload(records)
        if len(X) < bs:
            raise ValueError(f"only {len(X)} fitting images for batch size {bs}")
        torch.manual_seed(self.ctx.seed)
        self.head = self._head().to(dev)
        init_trainable(self.head)
        opt = StableAdamW(self.head.parameters(), lr=float(cfg["lr"]), betas=(0.9, 0.999),
                          weight_decay=float(cfg["weight_decay"]), amsgrad=True, eps=float(cfg["eps"]))
        sched = WarmCosineScheduler(opt, base_value=float(cfg["lr"]), final_value=float(cfg["final_lr"]),
                                    total_iters=iters, warmup_iters=int(cfg["warmup_iters"]))
        state_path = out_dir / "train_state.pt"
        start = 0
        if state_path.exists():
            st = torch.load(state_path, map_location=dev, weights_only=False)
            if st["signature"] != self._signature(records):
                raise RuntimeError(f"resume state {state_path} belongs to a different configuration; refusing")
            self.head.load_state_dict(st["head"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            # RNG states must be CPU ByteTensors even though the checkpoint was mapped to the training device
            torch.set_rng_state(st["rng_cpu"].cpu())
            if torch.cuda.is_available() and st.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(st["rng_cuda"].cpu())
            start = st["it"]
        cache = self._encoder_cache(records, X) if cfg.get("encoder_cache", False) else None
        steps_per_epoch = len(X) // bs   # drop_last=True, as in the reference DataLoader
        order_for = lambda e: np.random.default_rng([self.ctx.seed, 3, e]).permutation(len(X))  # noqa: E731
        mask_loss = bool(cfg.get("mask_loss", True))
        self.head.train()
        t0, log = time.time(), []
        def batch_indices(it_):
            e_, j_ = divmod(it_, steps_per_epoch)
            return order_for(e_)[j_ * bs:(j_ + 1) * bs]

        def fetch(it_):   # cached encoder outputs for every micro-batch of iteration it_ (worker thread)
            ix = batch_indices(it_)
            return [(torch.from_numpy(np.ascontiguousarray(cache[0][ix[m0:m0 + micro]])),
                     [torch.from_numpy(np.ascontiguousarray(c[ix[m0:m0 + micro]])) for c in cache[1:]])
                    for m0 in range(0, bs, micro)]

        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1) if cache is not None else None
        nxt = pool.submit(fetch, start) if pool is not None else None
        for it in tqdm(range(start, iters), desc=f"dino_recon {self.ctx.category}", initial=start, total=iters,
                       leave=False, mininterval=30):
            idx = batch_indices(it)
            cached = None
            if pool is not None:
                cached = nxt.result()
                if it + 1 < iters:
                    nxt = pool.submit(fetch, it + 1)
            p = min(float(cfg["hm_p_final"]) * it / int(cfg["hm_warmup_iters"]), float(cfg["hm_p_final"]))
            opt.zero_grad(set_to_none=True)
            loss_total = 0.0
            for k, m0 in enumerate(range(0, bs, micro)):
                sub = idx[m0:m0 + micro]
                valid = self._cell_valid(V[sub].to(dev)) if mask_loss else None
                if valid is not None and bool((valid > 0).all()):
                    valid = None   # unpadded input: exactly the reference loss
                if cached is not None:
                    fused = cached[k][0].to(dev, non_blocking=True)
                    en = [t.to(dev, non_blocking=True) for t in cached[k][1]]
                    de = self.decode(fused)
                else:
                    x = self._norm(X[sub].to(dev, non_blocking=True))
                    en, de = self.forward_groups(x)
                loss = global_cosine_hm_percent(en, de, p=p, factor=float(cfg["hm_factor"]), valid=valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite reconstruction loss at iteration {it}")
                (loss * (len(sub) / bs)).backward()
                loss_total += float(loss.detach()) * len(sub) / bs
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=float(cfg["grad_clip"]))
            opt.step()
            sched.step()
            if (it + 1) % 100 == 0:
                log.append({"it": it + 1, "loss": loss_total, "lr": sched.get_last_lr()[0], "p": p,
                            "elapsed_s": time.time() - t0})
            if (it + 1) % int(cfg.get("checkpoint_every", 1000)) == 0 and it + 1 < iters:
                torch.save({"head": self.head.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "it": it + 1, "signature": self._signature(records), "rng_cpu": torch.get_rng_state(),
                            "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None},
                           state_path)
                _append(out_dir / "train_log.jsonl", log)
                log = []
        if pool is not None:
            pool.shutdown(wait=True)
        _append(out_dir / "train_log.jsonl", log)
        self.encoder.net.assert_unchanged()
        torch.save({"head": self.head.state_dict()}, final)
        meta = {"signature": self._signature(records), "iterations": iters, "resumed_from": start,
                "train_seconds_this_session": time.time() - t0, "head_sha256": sha256_file(final),
                "encoder_fingerprint": self.encoder.net.fingerprint_at_load, "grid_hw": list(self.hw),
                "n_trainable_params": int(sum(p.numel() for p in self.head.parameters())),
                "n_fit_images": len(records), "manifest_sha256": self.ctx.manifest_sha256,
                "checkpoint_selection": "final iteration of a fixed schedule (no test-based selection)"}
        write_json(out_dir / "dino_recon_meta.json", meta)
        if state_path.exists():
            state_path.unlink()
        return self.load(out_dir)

    def load(self, out_dir: Path) -> dict:
        meta = read_json(out_dir / "dino_recon_meta.json")
        final = out_dir / "head_final.pt"
        if sha256_file(final) != meta["head_sha256"]:
            raise RuntimeError(f"{final} changed after training")
        if meta["encoder_fingerprint"] != self.encoder.net.fingerprint_at_load:
            raise RuntimeError("DINOv2 encoder weights differ from those used in training")
        self.head = self._head().to(self.ctx.device)
        self.head.load_state_dict(torch.load(final, map_location=self.ctx.device, weights_only=True)["head"])
        self.head.eval()
        return meta

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def grid_maps(self, x: torch.Tensor) -> torch.Tensor:
        en, de = self.forward_groups(x)
        maps = [1 - F.cosine_similarity(a.float(), b.float(), dim=1) for a, b in zip(en, de)]
        return torch.stack(maps, 1).mean(1)   # (B, h, w)

    def score(self, records, fill_override: str | None = None) -> BranchOutput:
        if self.head is None:
            raise RuntimeError("global_attention branch not fitted/loaded")
        self.head.eval()
        ds = ImageDataset(records, self.ctx.data_root, self.spec)
        n = len(records)
        fracs = [float(f) for f in self.cfg.get("pool_fractions", [0.01, 0.005, 0.05])]
        scores = {f"grid_top{f * 100:g}pct": np.full(n, np.nan) for f in fracs}
        scores["grid_max"] = np.full(n, np.nan)
        scores["ref_top1pct_256_gauss"] = np.full(n, np.nan)
        maps = np.full((n, *self.hw), np.nan, np.float32)
        status = []
        gk = _gaussian_kernel(5, 4.0).to(self.ctx.device)
        bs = int(self.cfg.get("score_batch_size", 8))
        for s0 in range(0, n, bs):
            items = [ds[i] for i in range(s0, min(n, s0 + bs))]
            x = torch.stack([it["image"] for it in items]).to(self.ctx.device)
            if fill_override == "black":
                v = torch.stack([it["valid"] for it in items]).to(self.ctx.device)
                black = self._norm(torch.zeros_like(x, dtype=torch.uint8))
                x = torch.where(v > 0.5, x, black)
            v = torch.stack([it["valid"] for it in items]).to(self.ctx.device)
            gm = self.grid_maps(x)
            cv = self._cell_valid(v)[:, 0] > 0.5
            # reference pooling location: upsample (align_corners=True) -> 256 -> gaussian -> top 1%
            up = F.interpolate(gm[:, None], size=x.shape[-2:], mode="bilinear", align_corners=True)
            up = F.interpolate(up, size=256, mode="bilinear", align_corners=False)
            up = gk(up)
            vpix = F.interpolate(v, size=256, mode="bilinear", align_corners=False) > 0.5
            for j in range(len(items)):
                i = s0 + j
                vals = gm[j][cv[j]]
                if vals.numel() == 0 or not torch.isfinite(vals).all():
                    status.append("failed:no_valid_or_nonfinite")
                    continue
                for f in fracs:
                    k = max(1, int(round(f * vals.numel())))
                    scores[f"grid_top{f * 100:g}pct"][i] = float(torch.topk(vals, k).values.mean())
                scores["grid_max"][i] = float(vals.max())
                pv = up[j, 0][vpix[j, 0]]
                scores["ref_top1pct_256_gauss"][i] = float(torch.topk(pv, max(1, int(pv.numel() * 0.01))).values.mean())
                g = gm[j].cpu().numpy()
                g[~cv[j].cpu().numpy()] = np.nan
                maps[i] = g
                status.append("ok")
        return BranchOutput(rel_ids=[r.rel_id for r in records], scores=scores, primary="grid_top1pct",
                            maps=maps, map_kind="dino_group_mean_1_minus_cosine", status=status)

    def padding_diagnostic(self, records) -> dict:
        """On NORMAL calibration images: does padding content change scores or light up borders?"""
        if self.spec.kind != "letterbox":
            return {"applicable": False, "reason": "no padding in this preprocessing"}
        a = self.score(records)
        b = self.score(records, fill_override="black")
        sa, sb = a.scores["grid_top1pct"], b.scores["grid_top1pct"]
        ok = np.isfinite(sa) & np.isfinite(sb)
        rho = float(spearmanr(sa[ok], sb[ok]).statistic) if ok.sum() > 2 else float("nan")
        # border (valid cells adjacent to padding) vs interior mean map value
        m = a.maps
        valid = np.isfinite(m)
        pad = ~valid
        border = np.zeros_like(valid)
        for axis in (1, 2):
            for sh in (1, -1):
                border |= valid & np.roll(pad, sh, axis=axis)
        return {"applicable": True, "n_images": int(ok.sum()), "fill_default": "imagenet_mean",
                "fill_alternative": "black",
                "spearman_default_vs_black_fill": rho,
                "median_relative_score_change": float(np.median(np.abs(sb[ok] - sa[ok]) / np.abs(sa[ok]))),
                "mean_map_border_cells": float(np.nanmean(np.where(border, m, np.nan))),
                "mean_map_interior_cells": float(np.nanmean(np.where(valid & ~border, m, np.nan))),
                "note": "normal calibration images only; no labels used"}


def _gaussian_kernel(kernel_size=5, sigma=4.0) -> nn.Conv2d:
    coords = torch.arange(kernel_size)
    xg = coords.repeat(kernel_size).view(kernel_size, kernel_size)
    xy = torch.stack([xg, xg.t()], dim=-1).float()
    mean, var = (kernel_size - 1) / 2.0, sigma ** 2.0
    k = (1.0 / (2.0 * math.pi * var)) * torch.exp(-torch.sum((xy - mean) ** 2.0, dim=-1) / (2 * var))
    k = (k / k.sum()).view(1, 1, kernel_size, kernel_size)
    conv = nn.Conv2d(1, 1, kernel_size, groups=1, bias=False, padding=kernel_size // 2)
    conv.weight.data = k
    conv.weight.requires_grad = False
    return conv


def _append(path: Path, rows):
    import json
    if rows:
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
