"""Explicitly local student-teacher branch: EfficientAD-S PDN teacher (33x33 receptive field) and
a PDN student with identical geometry.

ADAPTATION, not a reproduction of complete EfficientAD: only the local student-teacher pathway
is used. The autoencoder and the student's second 384-channel half (which exists only to
imitate the autoencoder) are omitted by design; contextual modelling comes from the DINOv2
branch. Reference implementation followed: anomalib efficient_ad (commit d716bb9), a
third-party reimplementation.

Reproduced from anomalib:
  * teacher channel mean/std computed over normal FITTING images (population std);
  * loss_st = mean of squared distances >= their 0.999 quantile  +  mean(student(penalty_img)^2);
  * penalty images: ImageNette, Resize(2*size) -> RandomGrayscale(0.3) -> CenterCrop(size), batch 1;
  * Adam(lr 1e-4, weight_decay 1e-5), StepLR(step = 0.95 * steps, gamma 0.1), batch size 1,
    70,000 steps (anomalib default max_steps for EfficientAD);
  * anomaly map = channel mean of (normalized teacher - student)^2, padded by 4 and bilinearly
    upsampled to the input size; image score = max of the map (reference aggregation).
Not reproduced (rank-invariant or unused): the map quantile normalization 0.1*(m-qa)/(qb-qa)
is a positive affine transform of the single map and so leaves per-category rankings and our
calibrated scores unchanged; anomalib computes it on its validation loader, which for MVTec
defaults to the test split, and is therefore not used here.

Project additions: letterbox valid-region handling (teacher statistics, loss and pooling use
only output cells whose 33x33 receptive field is at least `min_valid_fraction` real image);
deterministic, resumable sampling of training and penalty images.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..backbones import SmallPDN, load_pdn_teacher, receptive_field, PDN_S_LAYERS
from ..data import ImageDataset, TransformSpec
from ..provenance import WEIGHTS
from ..utils import FrozenModule, write_json, read_json, sha256_file, assert_finite
from . import BranchContext, BranchOutput

RF, STRIDE = receptive_field(PDN_S_LAYERS)  # (33, 4)


def _penalty_files(root: Path) -> list[Path]:
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
    if not files:
        raise FileNotFoundError(f"no penalty images under {root}")
    return files


def _load_penalty(path: Path, size: int, gray: bool) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((2 * size, 2 * size), Image.BILINEAR)
    if gray:
        im = im.convert("L").convert("RGB")
    off = size // 2
    im = im.crop((off, off, off + size, off + size))
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0


def _penalty_crop_rgb(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((2 * size, 2 * size), Image.BILINEAR)
    off = size // 2
    return np.asarray(im.crop((off, off, off + size, off + size)))


def penalty_bank(files: list[Path], size: int, work_dir: Path) -> np.ndarray:
    """Exact preload of the penalty images: Resize(2*size) -> CenterCrop(size) as uint8 RGB, memory-mapped.

    Grayscale conversion is pixel-wise, so applying it after the crop (in `_penalty_from_bank`) gives the
    same pixels as the reference order Resize -> RandomGrayscale -> CenterCrop (tested)."""
    import hashlib
    import json
    from concurrent.futures import ThreadPoolExecutor as TPE
    from numpy.lib.format import open_memmap
    key = hashlib.sha256(("\n".join(str(f) for f in files) + f"|{size}|bank-v1").encode()).hexdigest()[:24]
    d = Path(work_dir) / "cache" / "penalty"
    arr_p, meta_p = d / f"imagenette_{size}_{key}.npy", d / f"imagenette_{size}_{key}.json"
    if meta_p.exists():
        return np.load(arr_p, mmap_mode="r")
    d.mkdir(parents=True, exist_ok=True)
    tmp = arr_p.with_suffix(".building.npy")
    bank = open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(files), size, size, 3))
    with TPE(max_workers=12) as ex:
        for i, a in enumerate(ex.map(lambda f: _penalty_crop_rgb(f, size), files)):
            bank[i] = a
    bank.flush()
    del bank
    tmp.replace(arr_p)
    meta_p.write_text(json.dumps({"n": len(files), "size": size, "key": key}))
    return np.load(arr_p, mmap_mode="r")


def _penalty_from_bank(bank: np.ndarray, i: int, gray: bool) -> torch.Tensor:
    a = np.asarray(bank[i])
    if gray:
        a = np.asarray(Image.fromarray(a).convert("L").convert("RGB"))
    return torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255.0


class LocalSTBranch:
    name = "local_st"

    def __init__(self, cfg: dict, ctx: BranchContext):
        self.cfg = cfg
        self.ctx = ctx
        self.spec = TransformSpec.from_cfg(cfg["transform"][ctx.dataset])
        if self.spec.normalize != "none":
            raise ValueError("PDN normalizes internally; the local_st transform must use normalize: none")
        self.min_valid = float(cfg.get("min_valid_fraction", 0.5))
        self.teacher = FrozenModule(load_pdn_teacher("small")).to(ctx.device)
        self.student: SmallPDN | None = None
        self.mean = self.std = None

    # ------------------------------------------------------------------ helpers
    def _cell_valid(self, valid: torch.Tensor) -> torch.Tensor:
        """Fraction of real pixels inside each output cell's 33x33 receptive field. valid: (B,1,H,W)."""
        return F.avg_pool2d(valid.float(), kernel_size=RF, stride=STRIDE)

    def _load_all(self, records) -> tuple[torch.Tensor, torch.Tensor]:
        ds = ImageDataset(records, self.ctx.data_root, self.spec)
        xs, vs = [], []
        for i in tqdm(range(len(ds)), desc=f"load {self.ctx.category}", leave=False, mininterval=10):
            item = ds[i]
            xs.append(item["image"])
            vs.append(item["valid"])
        return torch.stack(xs), torch.stack(vs)

    @torch.no_grad()
    def _teacher_stats(self, X: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = 0
        s = torch.zeros(384, dtype=torch.float64, device=self.ctx.device)
        s2 = torch.zeros_like(s)
        for i in range(0, len(X), 16):
            y = self.teacher(X[i:i + 16].to(self.ctx.device)).double()
            cv = (self._cell_valid(V[i:i + 16].to(self.ctx.device)) >= self.min_valid).double()
            n += cv.sum().item()
            s += (y * cv).sum(dim=(0, 2, 3))
            s2 += (y * y * cv).sum(dim=(0, 2, 3))
        mean = s / n
        std = torch.sqrt((s2 / n - mean ** 2).clamp_min(0))
        if (std <= 0).any():
            raise FloatingPointError("teacher channel with zero variance on fitting images")
        return mean.float()[None, :, None, None], std.float()[None, :, None, None]

    def _t_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (self.teacher(x) - self.mean) / self.std

    # ------------------------------------------------------------------ fit
    def fit(self, records, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / "student_final.pt"
        if final.exists():
            return self.load(out_dir)
        cfg, dev = self.cfg, self.ctx.device
        steps = int(cfg["train_steps"])
        seed = self.ctx.seed
        X, V = self._load_all(records)
        state_path = out_dir / "train_state.pt"
        torch.manual_seed(seed)
        student = SmallPDN(out_channels=int(cfg.get("student_out_channels", 384))).to(dev)
        opt = torch.optim.Adam(student.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(float(cfg["lr_step_fraction"]) * steps),
                                                gamma=float(cfg["lr_gamma"]))
        start = 0
        if state_path.exists():
            st = torch.load(state_path, map_location=dev, weights_only=False)
            if st["signature"] != self._signature(records):
                raise RuntimeError(f"resume state {state_path} belongs to a different configuration; refusing")
            student.load_state_dict(st["student"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            self.mean, self.std = st["mean"].to(dev), st["std"].to(dev)
            start = st["step"]
        else:
            self.mean, self.std = self._teacher_stats(X, V)
        use_penalty = cfg.get("penalty", "imagenette") != "none"
        pen_files = _penalty_files(cfg["_paths"]["imagenette"]) if use_penalty else []
        size = self.spec.size
        rng_order = lambda epoch: np.random.default_rng([seed, 1, epoch]).permutation(len(X))  # noqa: E731

        def pen_item(step):
            r = np.random.default_rng([seed, 2, step])
            return int(r.integers(len(pen_files))), bool(r.random() < 0.3)

        bank = penalty_bank(pen_files, size, self.ctx.work_dir) if use_penalty else None
        pool = ThreadPoolExecutor(max_workers=2) if use_penalty else None
        prefetch = {}

        def schedule(k):
            for s in range(k, min(k + 16, steps)):
                if s not in prefetch:
                    j, g = pen_item(s)
                    prefetch[s] = pool.submit(_penalty_from_bank, bank, j, g)

        student.train()
        hard_q = float(cfg["hard_quantile"])
        t0 = time.time()
        log = []
        order, epoch = rng_order(start // len(X)), start // len(X)
        for step in tqdm(range(start, steps), desc=f"local_st {self.ctx.category}", initial=start, total=steps,
                         leave=False, mininterval=30):
            e = step // len(X)
            if e != epoch:
                order, epoch = rng_order(e), e
            i = int(order[step % len(X)])
            x = X[i:i + 1].to(dev, non_blocking=True)
            v = V[i:i + 1].to(dev)
            with torch.no_grad():
                t = self._t_norm(x)
            s = student(x)[:, :384]
            d = (t - s) ** 2
            cv = (self._cell_valid(v) >= self.min_valid).expand_as(d)
            d = d[cv]
            if d.numel() > 2 ** 24:
                d = d[torch.randperm(d.numel(), device=dev)[: 2 ** 24]]
            d_hard = torch.quantile(d, hard_q)
            loss_hard = d[d >= d_hard].mean()
            loss = loss_hard
            loss_pen = torch.zeros((), device=dev)
            if use_penalty:
                schedule(step)
                pimg = prefetch.pop(step).result()[None].to(dev)
                loss_pen = (student(pimg)[:, :384] ** 2).mean()
                loss = loss + loss_pen
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite local_st loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % 500 == 0:
                log.append({"step": step + 1, "loss_hard": float(loss_hard), "loss_penalty": float(loss_pen),
                            "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0})
            if (step + 1) % int(cfg.get("checkpoint_every", 5000)) == 0 and step + 1 < steps:
                torch.save({"student": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "mean": self.mean.cpu(), "std": self.std.cpu(), "step": step + 1,
                            "signature": self._signature(records)}, state_path)
                _append_log(out_dir / "train_log.jsonl", log)
                log = []
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
        _append_log(out_dir / "train_log.jsonl", log)
        self.teacher.assert_unchanged()
        torch.save({"student": student.state_dict(), "mean": self.mean.cpu(), "std": self.std.cpu()}, final)
        meta = {"signature": self._signature(records), "steps": steps, "train_seconds_this_session": time.time() - t0,
                "resumed_from_step": start, "student_sha256": sha256_file(final),
                "teacher_fingerprint": self.teacher.fingerprint_at_load,
                "receptive_field": [RF, RF], "stride": STRIDE, "penalty": cfg.get("penalty", "imagenette"),
                "n_fit_images": len(records), "manifest_sha256": self.ctx.manifest_sha256}
        write_json(out_dir / "local_st_meta.json", meta)
        if state_path.exists():
            state_path.unlink()
        return self.load(out_dir)

    def _signature(self, records) -> dict:
        keys = ("pdn", "student_out_channels", "train_steps", "lr", "weight_decay", "lr_step_fraction", "lr_gamma",
                "hard_quantile", "penalty", "min_valid_fraction", "lr_schedule_unit")
        import hashlib
        return {"cfg": {k: self.cfg.get(k) for k in keys}, "transform": self.spec.signature(),
                "teacher_sha256": WEIGHTS["pdn_small_teacher"]["sha256"], "seed": self.ctx.seed,
                "fit_ids": hashlib.sha256("\n".join(sorted(r.rel_id for r in records)).encode()).hexdigest()}

    def load(self, out_dir: Path) -> dict:
        meta = read_json(out_dir / "local_st_meta.json")
        final = out_dir / "student_final.pt"
        if sha256_file(final) != meta["student_sha256"]:
            raise RuntimeError(f"{final} changed after training")
        if meta["teacher_fingerprint"] != self.teacher.fingerprint_at_load:
            raise RuntimeError("teacher weights differ from those used in training")
        st = torch.load(final, map_location=self.ctx.device, weights_only=True)
        self.student = SmallPDN(out_channels=int(self.cfg.get("student_out_channels", 384))).to(self.ctx.device)
        self.student.load_state_dict(st["student"])
        self.student.eval()
        self.mean, self.std = st["mean"].to(self.ctx.device), st["std"].to(self.ctx.device)
        return meta

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def maps(self, x: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (grid map (B,1,64,64) with reference 4-cell zero padding, upsampled map (B,1,H,W))."""
        t = self._t_norm(x)
        s = self.student(x)[:, :384]
        m = ((t - s) ** 2).mean(1, keepdim=True)
        cv = self._cell_valid(v) >= self.min_valid
        m = torch.where(cv, m, torch.zeros_like(m))
        m_pad = F.pad(m, (4, 4, 4, 4))
        up = F.interpolate(m_pad, size=x.shape[-2:], mode="bilinear")
        grid = F.pad(torch.where(cv, m, torch.full_like(m, float("nan"))), (4, 4, 4, 4))
        return grid, up

    def score(self, records) -> BranchOutput:
        if self.student is None:
            raise RuntimeError("local_st branch not fitted/loaded")
        ds = ImageDataset(records, self.ctx.data_root, self.spec)
        n = len(records)
        s_max, s_top = np.full(n, np.nan), np.full(n, np.nan)
        maps, status = None, []
        frac = float(self.cfg.get("tail_fraction", 0.01))
        for i in range(n):
            item = ds[i]
            x = item["image"][None].to(self.ctx.device)
            v = item["valid"][None].to(self.ctx.device)
            grid, up = self.maps(x, v)
            vals = up[0, 0][v[0, 0] > 0.5]
            if vals.numel() == 0 or not torch.isfinite(vals).all():
                status.append("failed:no_valid_or_nonfinite")
                continue
            s_max[i] = float(vals.max())
            k = max(1, int(round(frac * vals.numel())))
            s_top[i] = float(torch.topk(vals, k).values.mean())
            g = grid[0, 0].cpu().numpy()
            if maps is None:
                maps = np.full((n, *g.shape), np.nan, np.float32)
            maps[i] = g
            status.append("ok")
        return BranchOutput(rel_ids=[r.rel_id for r in records],
                            scores={"max": s_max, f"top{frac * 100:g}pct": s_top}, primary="max",
                            maps=maps, map_kind="st_squared_discrepancy_channel_mean_padded4", status=status)


def _append_log(path: Path, rows: list[dict]) -> None:
    import json
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
