"""Immutable split manifests for MVTec AD and MVTec LOCO AD.

Each image gets one row with a relative ID (``<category>/<split dir>/<type>/<file>``),
its role in the protocol, labels (for evaluation only) and two hashes:
``file_sha256`` (bytes on disk) and ``pixel_sha256`` (decoded RGB array), used to detect
exact and re-encoded duplicates across roles.

Roles (column ``split``):
  fit              normal images used to train networks and build memories
  calib_scale      held-out normal images used only for branch score scaling
  calib_threshold  second held-out normal half (role name kept for manifest compatibility); used only
                   for the OR fusion rule's per-branch offsets
  calib_shared     held-out normal images used for BOTH (small categories; disclosed)
  test             official test split, evaluation only
  excluded_duplicate  removed because identical content appears in a later role
"""
from __future__ import annotations

import csv
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import write_json, read_json

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

MVTEC_AD_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
MVTEC_AD_TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
MVTEC_LOCO_CATEGORIES = ("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors")

CALIB_ROLES_SCALE = ("calib_scale", "calib_shared")
CALIB_ROLES_THRESHOLD = ("calib_threshold", "calib_shared")


class ManifestError(RuntimeError):
    pass


def layout_of(dataset: str) -> str:
    """'loco' (train/validation/test with logical/structural types) or 'ad' (train/test)."""
    return "loco" if "loco" in dataset else "ad"


@dataclass
class Record:
    dataset: str
    category: str
    rel_id: str           # relative to the dataset root, forward slashes
    split: str
    source_split: str     # train / validation / test directory in the official release
    label: int            # 0 normal, 1 anomalous (evaluation only)
    anomaly_type: str     # official defect directory name (evaluation only)
    subgroup: str         # good / structural / logical (LOCO) ; good / defect (AD)
    mask_rel: str         # "" if none
    width: int
    height: int
    file_sha256: str
    pixel_sha256: str


FIELDS = [f.name for f in fields(Record)]


def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _hash_image(path: Path) -> tuple[str, str, int, int]:
    raw = path.read_bytes()
    fsha = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:
        arr = np.asarray(im.convert("RGB"))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return fsha, h.hexdigest(), arr.shape[1], arr.shape[0]


def _order_key(salt: str, category: str, rel_id: str) -> str:
    return hashlib.sha256(f"{salt}|{category}|{rel_id}".encode()).hexdigest()


def _assign_calibration(rel_ids: list[str], category: str, protocol: dict, pool_is_all_calib: bool) -> dict[str, str]:
    """Deterministically assign fit / calibration roles from a hash ordering."""
    salt = protocol["split_salt"]
    ordered = sorted(rel_ids, key=lambda r: _order_key(salt, category, r))
    if pool_is_all_calib:
        calib = ordered
        roles = {}
    else:
        n = len(ordered)
        n_cal = max(int(protocol["ad_calibration_min"]), int(round(protocol["ad_calibration_fraction"] * n)))
        if n_cal >= n:
            raise ManifestError(f"{category}: calibration size {n_cal} leaves no fitting images")
        calib = ordered[:n_cal]
        roles = {r: "fit" for r in ordered[n_cal:]}
    if len(calib) >= int(protocol["separate_scale_threshold_min"]):
        half = (len(calib) + 1) // 2
        roles.update({r: "calib_scale" for r in calib[:half]})
        roles.update({r: "calib_threshold" for r in calib[half:]})
    else:
        roles.update({r: "calib_shared" for r in calib})
    return roles


def scan_category(dataset: str, root: Path, category: str, protocol: dict, workers: int = 8) -> list[Record]:
    cdir = root / category
    if not cdir.is_dir():
        raise ManifestError(f"missing category directory: {cdir}")
    items: list[tuple[str, str, int, str, str, str]] = []  # rel_id, source_split, label, type, subgroup, mask
    train_imgs = _list_images(cdir / "train" / "good")
    if not train_imgs:
        raise ManifestError(f"{category}: no training images under {cdir / 'train' / 'good'}")
    for p in train_imgs:
        items.append((f"{category}/train/good/{p.name}", "train", 0, "good", "good", ""))
    layout = layout_of(dataset)
    if layout == "loco":
        val_imgs = _list_images(cdir / "validation" / "good")
        if not val_imgs:
            raise ManifestError(f"{category}: LOCO validation/good is missing")
        for p in val_imgs:
            items.append((f"{category}/validation/good/{p.name}", "validation", 0, "good", "good", ""))
    test_dir = cdir / "test"
    for tdir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        atype = tdir.name
        for p in _list_images(tdir):
            rel = f"{category}/test/{atype}/{p.name}"
            if atype == "good":
                items.append((rel, "test", 0, "good", "good", ""))
                continue
            if layout == "loco":
                sub = {"logical_anomalies": "logical", "structural_anomalies": "structural"}.get(atype)
                if sub is None:
                    raise ManifestError(f"unexpected LOCO test type {atype}")
                mdir = cdir / "ground_truth" / atype / p.stem
                mask = f"{category}/ground_truth/{atype}/{p.stem}" if mdir.is_dir() else ""
            else:
                sub = "defect"
                m = cdir / "ground_truth" / atype / f"{p.stem}_mask.png"
                mask = f"{category}/ground_truth/{atype}/{m.name}" if m.exists() else ""
            items.append((rel, "test", 1, atype, sub, mask))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(lambda it: _hash_image(root / it[0]), items))

    train_ids = [it[0] for it in items if it[1] == "train"]
    val_ids = [it[0] for it in items if it[1] == "validation"]
    roles: dict[str, str] = {}
    if layout == "loco":
        roles.update({r: "fit" for r in train_ids})
        roles.update(_assign_calibration(val_ids, category, protocol, pool_is_all_calib=True))
    else:
        roles.update(_assign_calibration(train_ids, category, protocol, pool_is_all_calib=False))

    records = []
    for (rel, src, label, atype, sub, mask), (fsha, psha, w, h) in zip(items, hashes):
        split = "test" if src == "test" else roles[rel]
        records.append(Record(dataset, category, rel, split, src, label, atype, sub, mask, w, h, fsha, psha))
    _resolve_duplicates(records)
    return records


# Later roles win: a copy in an earlier role is excluded. calib_threshold outranks calib_scale so
# scale/threshold halves are also content-disjoint.
_ROLE_RANK = {"fit": 0, "calib_scale": 1, "calib_threshold": 1.5, "calib_shared": 1.5, "test": 2}


def _resolve_duplicates(records: list[Record]) -> None:
    """Remove the earlier-role copy of any content duplicated across roles.

    Test images are never removed: a fit/calibration image identical to a test image is
    dropped from fit/calibration so memories and thresholds never contain test content.
    Duplicates within the same role are kept but reported by `duplicate_report`.
    """
    by_hash: dict[str, list[Record]] = {}
    for r in records:
        for key in (r.file_sha256, r.pixel_sha256):
            by_hash.setdefault(key, []).append(r)
    for group in by_hash.values():
        if len(group) < 2:
            continue
        top = max(_ROLE_RANK.get(r.split, -1) for r in group)
        for r in group:
            if r.split in _ROLE_RANK and _ROLE_RANK[r.split] < top:
                r.split = "excluded_duplicate"


def duplicate_report(records: list[Record]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for r in records:
        groups.setdefault(r.pixel_sha256, []).append(r.rel_id)
    return [g for g in groups.values() if len(g) > 1]


def manifest_path(manifest_dir: Path, dataset: str, category: str) -> Path:
    return manifest_dir / dataset / f"{category}.csv"


def _to_csv_bytes(records: list[Record]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator="\n")
    w.writeheader()
    for r in sorted(records, key=lambda r: r.rel_id):
        w.writerow(asdict(r))
    return buf.getvalue().encode("utf-8")


def build_manifest(dataset: str, root: Path, category: str, protocol: dict, manifest_dir: Path) -> dict:
    """Create the manifest, or verify that an existing one is byte-identical. Never overwrites."""
    records = scan_category(dataset, root, category, protocol)
    data = _to_csv_bytes(records)
    sha = hashlib.sha256(data).hexdigest()
    path = manifest_path(manifest_dir, dataset, category)
    meta_path = path.with_suffix(".meta.json")
    if path.exists():
        old = path.read_bytes()
        if old != data:
            raise ManifestError(
                f"existing manifest {path} differs from a fresh scan (sha {hashlib.sha256(old).hexdigest()[:12]} "
                f"vs {sha[:12]}). Manifests are immutable: the dataset or protocol changed. "
                "Move the old manifest aside deliberately if a new protocol version is intended."
            )
        return read_json(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    counts: dict[str, int] = {}
    for r in records:
        counts[r.split] = counts.get(r.split, 0) + 1
    meta = {
        "dataset": dataset,
        "category": category,
        "manifest_sha256": sha,
        "protocol": {k: protocol[k] for k in sorted(protocol)},
        "counts": counts,
        "duplicate_groups": duplicate_report(records),
        "calibration_reuse": counts.get("calib_shared", 0) > 0,
    }
    write_json(meta_path, meta)
    return meta


def load_manifest(manifest_dir: Path, dataset: str, category: str) -> tuple[list[Record], str]:
    path = manifest_path(manifest_dir, dataset, category)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}. Run `hybridad manifests` first.")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    meta = read_json(path.with_suffix(".meta.json"))
    if meta["manifest_sha256"] != sha:
        raise ManifestError(f"manifest {path} was modified after creation")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    recs = []
    for row in rows:
        row["label"] = int(row["label"])
        row["width"] = int(row["width"])
        row["height"] = int(row["height"])
        recs.append(Record(**row))
    check_leakage(recs)
    return recs, sha


def select(records: list[Record], roles: tuple[str, ...] | str) -> list[Record]:
    if isinstance(roles, str):
        roles = (roles,)
    return [r for r in records if r.split in roles]


def check_leakage(records: list[Record]) -> None:
    """Assert disjoint identities and contents across fit / calibration / test."""
    role_of_group = {
        "fit": "fit", "calib_scale": "calib", "calib_threshold": "calib", "calib_shared": "calib", "test": "test",
    }
    seen_id: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    for r in records:
        g = role_of_group.get(r.split)
        if g is None:
            continue
        if r.rel_id in seen_id:
            raise ManifestError(f"image id appears twice: {r.rel_id}")
        seen_id[r.rel_id] = g
        for key in (r.file_sha256, r.pixel_sha256):
            other = seen_content.get(key)
            if other is not None and other != g:
                raise ManifestError(f"identical content in roles {other} and {g}: {r.rel_id}")
            seen_content[key] = g
        if g in ("fit", "calib") and r.label != 0:
            raise ManifestError(f"anomalous image in normal-only role {r.split}: {r.rel_id}")
    scale = [r for r in records if r.split in ("calib_scale",)]
    second = [r for r in records if r.split in ("calib_threshold",)]
    if {r.rel_id for r in scale} & {r.rel_id for r in second}:
        raise ManifestError("the two held-out normal calibration halves overlap")
