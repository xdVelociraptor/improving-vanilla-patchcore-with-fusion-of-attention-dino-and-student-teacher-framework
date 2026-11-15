"""Verify the publisher dataset files against the stored split manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(manifest_root: Path, dataset_root: Path) -> tuple[int, int]:
    paths = sorted(manifest_root.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No manifests found under {manifest_root}")
    ok = failures = 0
    for manifest in paths:
        with manifest.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                name = record["rel_id"]
                path = (dataset_root / name).resolve()
                if not path.is_relative_to(dataset_root):
                    raise SystemExit(f"Path escapes the dataset folder in {manifest.name}: {name}")
                try:
                    with path.open("rb") as image:
                        actual = hashlib.file_digest(image, "sha256").hexdigest()
                    if actual == record["file_sha256"]:
                        ok += 1
                        continue
                    detail = "SHA-256 differs"
                except OSError as exc:
                    detail = exc.strerror or type(exc).__name__
                if failures < 10:
                    print(f"{manifest_root.name}: {name}: {detail}")
                failures += 1
    print(f"{manifest_root.name}: {ok:,} matched, {failures:,} failed")
    return ok, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ad-root", type=Path, default=ROOT / "data" / "MVTecAD")
    parser.add_argument("--loco-root", type=Path, default=ROOT / "data" / "MVTec_LOCO")
    parser.add_argument("--dataset", choices=("all", "mvtec_ad", "mvtec_loco"), default="all")
    args = parser.parse_args()
    failures = 0
    for name, root in (("mvtec_ad", args.ad_root), ("mvtec_loco", args.loco_root)):
        if args.dataset not in ("all", name):
            continue
        if not root.is_dir():
            print(f"{name}: dataset folder not found: {root}")
            failures += 1
            continue
        _, error_count = verify(ROOT / "manifests" / name, root.resolve())
        failures += error_count
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
