"""Image preprocessing with explicit geometry and valid-region tracking.

Every transform returns the network input, a pixel-level valid mask (1 = real image
content, 0 = padding) and a `Geometry` that maps original-image coordinates to input
coordinates, so anomaly maps can be placed back onto the original image without guessing.

Resizing uses PIL BILINEAR, which is what torchvision's Resize does for PIL inputs
(it antialiases when downscaling). The EfficientAD/anomalib reference resizes tensors with
antialias=True instead; the difference is sub-pixel and documented in docs/METHODS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .manifests import Record

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Geometry:
    orig_w: int
    orig_h: int
    in_w: int
    in_h: int
    sx: float   # x_in = x_orig * sx + ox   (pixel-centre convention, see to_input)
    sy: float
    ox: float
    oy: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TransformSpec:
    """Declarative preprocessing. kind in {resize_crop, resize_square_crop, resize_square, letterbox}."""
    kind: str
    size: int                 # resize target (shorter side / square side / letterbox side)
    crop: int | None = None   # centre crop size, if any
    normalize: str = "imagenet"   # imagenet | none (PDN normalizes internally)

    def signature(self) -> dict:
        return {"kind": self.kind, "size": self.size, "crop": self.crop, "normalize": self.normalize,
                "resample": "PIL_BILINEAR", "pad_fill": "imagenet_mean"}

    @staticmethod
    def from_cfg(cfg: dict) -> "TransformSpec":
        return TransformSpec(kind=cfg["kind"], size=int(cfg["size"]),
                             crop=None if cfg.get("crop") in (None, "null") else int(cfg["crop"]),
                             normalize=cfg.get("normalize", "imagenet"))


_FILL = tuple(int(round(255 * m)) for m in IMAGENET_MEAN)


def apply_transform(img: Image.Image, spec: TransformSpec) -> tuple[torch.Tensor, torch.Tensor, Geometry]:
    img = img.convert("RGB")
    w, h = img.size
    if spec.kind == "resize_crop":
        # torchvision Resize(int): shorter side -> size, aspect kept; then CenterCrop(crop)
        if w <= h:
            rw, rh = spec.size, int(spec.size * h / w)
        else:
            rw, rh = int(spec.size * w / h), spec.size
        arr = np.asarray(img.resize((rw, rh), Image.BILINEAR))
        c = spec.crop or spec.size
        top, left = int(round((rh - c) / 2.0)), int(round((rw - c) / 2.0))
        arr = arr[top:top + c, left:left + c]
        valid = np.ones((c, c), np.float32)
        geo = Geometry(w, h, c, c, rw / w, rh / h, -left, -top)
    elif spec.kind == "resize_square_crop":
        # Dinomaly: Resize((size,size)) then CenterCrop(crop)
        s = spec.size
        arr = np.asarray(img.resize((s, s), Image.BILINEAR))
        c = spec.crop or s
        top = left = int(round((s - c) / 2.0))
        arr = arr[top:top + c, left:left + c]
        valid = np.ones((c, c), np.float32)
        geo = Geometry(w, h, c, c, s / w, s / h, -left, -top)
    elif spec.kind == "resize_square":
        s = spec.size
        arr = np.asarray(img.resize((s, s), Image.BILINEAR))
        valid = np.ones((s, s), np.float32)
        geo = Geometry(w, h, s, s, s / w, s / h, 0.0, 0.0)
    elif spec.kind == "letterbox":
        s = spec.size
        scale = s / max(w, h)
        rw, rh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = np.asarray(img.resize((rw, rh), Image.BILINEAR))
        left, top = (s - rw) // 2, (s - rh) // 2
        arr = np.empty((s, s, 3), np.uint8)
        arr[...] = np.array(_FILL, np.uint8)
        arr[top:top + rh, left:left + rw] = resized
        valid = np.zeros((s, s), np.float32)
        valid[top:top + rh, left:left + rw] = 1.0
        geo = Geometry(w, h, s, s, rw / w, rh / h, float(left), float(top))
    else:
        raise ValueError(f"unknown transform kind {spec.kind}")
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
    if spec.normalize == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
    elif spec.normalize != "none":
        raise ValueError(f"unknown normalization {spec.normalize}")
    return x, torch.from_numpy(valid)[None], geo


def grid_valid_fraction(valid: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Fraction of real pixels inside each cell of a regular grid over the input. valid: (B,1,H,W)."""
    return F.adaptive_avg_pool2d(valid.float(), grid_hw)


def map_to_original(amap: np.ndarray, geo: Geometry, out_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Resample a map covering the network input onto original-image pixels.

    `amap` (h, w) spans the whole input tensor (in_h x in_w) with cell centres at
    ((j+0.5)*in_w/w, (i+0.5)*in_h/h). Original pixels that fall outside the input (cropped
    away) are NaN: the branch did not observe them. Bilinear interpolation, edge-clamped.
    """
    H, W = out_hw or (geo.orig_h, geo.orig_w)
    ys = (np.arange(H) + 0.5) * (geo.orig_h / H)
    xs = (np.arange(W) + 0.5) * (geo.orig_w / W)
    xin = xs * geo.sx + geo.ox
    yin = ys * geo.sy + geo.oy
    h, w = amap.shape
    gx = xin / geo.in_w * w - 0.5
    gy = yin / geo.in_h * h - 0.5
    t = torch.from_numpy(np.ascontiguousarray(amap, dtype=np.float32))[None, None]
    gxn = torch.from_numpy((gx + 0.5) / w * 2 - 1).float()
    gyn = torch.from_numpy((gy + 0.5) / h * 2 - 1).float()
    grid = torch.stack(torch.meshgrid(gyn, gxn, indexing="ij")[::-1], dim=-1)[None]
    out = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0].numpy()
    outside_x = (xin < 0) | (xin > geo.in_w)
    outside_y = (yin < 0) | (yin > geo.in_h)
    out[outside_y, :] = np.nan
    out[:, outside_x] = np.nan
    return out


class ImageDataset(Dataset):
    """Returns only what inference may use: image, valid mask, index. Labels are NOT exposed."""

    def __init__(self, records: list[Record], root: Path, spec: TransformSpec):
        self.records = records
        self.root = Path(root)
        self.spec = spec

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        with Image.open(self.root / r.rel_id) as im:
            x, valid, _ = apply_transform(im, self.spec)
        return {"image": x, "valid": valid, "index": i}

    def geometry(self, i: int) -> Geometry:
        r = self.records[i]
        return geometry_for(r.width, r.height, self.spec)


def geometry_for(w: int, h: int, spec: TransformSpec) -> Geometry:
    """Geometry without decoding the image (uses a blank image of the same size)."""
    _, _, geo = apply_transform(Image.new("RGB", (w, h)), spec)
    return geo


def load_mask(root: Path, rec: Record) -> np.ndarray | None:
    """Binary ground-truth mask at original resolution, or None if the record has none.

    MVTec AD: one PNG. LOCO: a directory of PNGs (one per defect region) -> union.
    """
    if not rec.mask_rel:
        return None
    p = Path(root) / rec.mask_rel
    if p.is_dir():
        m = None
        for f in sorted(p.glob("*.png")):
            a = np.asarray(Image.open(f).convert("L")) > 0
            m = a if m is None else (m | a)
        return m
    return np.asarray(Image.open(p).convert("L")) > 0
