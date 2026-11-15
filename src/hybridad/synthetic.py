"""Synthetic smoke fixture in MVTec AD and MVTec LOCO layouts.

CODE-CORRECTNESS ONLY. These images are procedurally drawn blobs; results on them are not
benchmark evidence and are never mixed with real-dataset outputs (separate dataset names
`synthetic_ad` / `synthetic_loco`, separate run directories).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _background(rng, w, h):
    base = rng.normal(128, 12, size=(h, w, 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(base)


def _draw_object(img, rng, cx, cy, r, color):
    d = ImageDraw.Draw(img)
    jitter = rng.integers(-2, 3, size=2)
    d.ellipse([cx - r + jitter[0], cy - r + jitter[1], cx + r + jitter[0], cy + r + jitter[1]], fill=color)


def _scratch(img, rng, w, h):
    d = ImageDraw.Draw(img)
    x0, y0 = rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4)
    x1, y1 = x0 + rng.integers(8, 16), y0 + rng.integers(-6, 7)
    d.line([x0, y0, x1, y1], fill=(20, 20, 20), width=3)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).line([x0, y0, x1, y1], fill=255, width=3)
    return mask


def make_synthetic_ad(root: Path, category: str = "blobs", n_train: int = 40, n_good: int = 10, n_bad: int = 10,
                      size: int = 96, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    c = Path(root) / category
    for sub in ("train/good", "test/good", "test/scratch", "ground_truth/scratch"):
        (c / sub).mkdir(parents=True, exist_ok=True)

    def normal():
        img = _background(rng, size, size)
        _draw_object(img, rng, size // 2, size // 2, size // 5, (200, 60, 60))
        return img
    for i in range(n_train):
        normal().save(c / "train/good" / f"{i:03d}.png")
    for i in range(n_good):
        normal().save(c / "test/good" / f"{i:03d}.png")
    for i in range(n_bad):
        img = normal()
        m = _scratch(img, rng, size, size)
        img.save(c / "test/scratch" / f"{i:03d}.png")
        m.save(c / "ground_truth/scratch" / f"{i:03d}_mask.png")
    return Path(root)


def make_synthetic_loco(root: Path, category: str = "pairs", n_train: int = 40, n_val: int = 12, n_good: int = 10,
                        n_logical: int = 8, n_structural: int = 8, w: int = 128, h: int = 96, seed: int = 1) -> Path:
    """Normal = exactly two blobs (left red, right blue). Logical = a third blob or a missing one."""
    rng = np.random.default_rng(seed)
    c = Path(root) / category
    for sub in ("train/good", "validation/good", "test/good", "test/logical_anomalies", "test/structural_anomalies",
                "ground_truth/logical_anomalies", "ground_truth/structural_anomalies"):
        (c / sub).mkdir(parents=True, exist_ok=True)
    r = h // 6

    def scene(n_left=1, n_right=1):
        img = _background(rng, w, h)
        for k in range(n_left):
            _draw_object(img, rng, w // 4, h // 2 + (k - (n_left - 1) / 2) * 2.2 * r, r, (200, 60, 60))
        for k in range(n_right):
            _draw_object(img, rng, 3 * w // 4, h // 2 + (k - (n_right - 1) / 2) * 2.2 * r, r, (60, 60, 200))
        return img
    for i in range(n_train):
        scene().save(c / "train/good" / f"{i:03d}.png")
    for i in range(n_val):
        scene().save(c / "validation/good" / f"{i:03d}.png")
    for i in range(n_good):
        scene().save(c / "test/good" / f"{i:03d}.png")
    for i in range(n_logical):
        extra = i % 2 == 0
        img = scene(2 if extra else 1, 1 if extra else 0)
        img.save(c / "test/logical_anomalies" / f"{i:03d}.png")
        md = c / "ground_truth/logical_anomalies" / f"{i:03d}"
        md.mkdir(exist_ok=True)
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).rectangle([w // 2, 0, w - 1, h - 1] if not extra else [0, 0, w // 2, h - 1], fill=255)
        m.save(md / "000.png")
    for i in range(n_structural):
        img = scene()
        m = _scratch(img, rng, w, h)
        img.save(c / "test/structural_anomalies" / f"{i:03d}.png")
        md = c / "ground_truth/structural_anomalies" / f"{i:03d}"
        md.mkdir(exist_ok=True)
        m.save(md / "000.png")
    (c / "defects_config.json").write_text(json.dumps([
        {"defect_name": "synthetic_logical", "pixel_value": 255, "saturation_threshold": 1, "relative_saturation": True},
    ]))
    return Path(root)
