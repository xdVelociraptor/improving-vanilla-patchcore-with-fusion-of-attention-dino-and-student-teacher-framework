"""Copy the original local research archive into a separate source checkout."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ("data", "external_data", "weights", "work", "runs", ".venv")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def restore(source: Path, target: Path) -> None:
    source, target = source.resolve(strict=True), target.resolve(strict=True)
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError("The original archive and checkout must be separate folders.")
    required = [source / name for name in PARTS]
    missing = [str(p) for p in required if not p.is_dir()]
    if missing:
        raise ValueError("Missing required archive folders: " + ", ".join(missing))
    print("Reading archive:", source, flush=True)
    print("Writing checkout:", target, flush=True)

    to_copy = []
    needed = 0
    existing = 0
    for name in PARTS:
        base = source / name
        for directory, dirs, files in os.walk(base, followlinks=False):
            for child in dirs:
                if (Path(directory) / child).is_symlink():
                    raise ValueError("Archive contains a linked directory: " + str(Path(directory) / child))
            for child in files:
                src = Path(directory) / child
                if src.is_symlink():
                    raise ValueError("Archive contains a linked file: " + str(src))
                rel = src.relative_to(source)
                dst = target / rel
                if not dst.resolve().is_relative_to(target):
                    raise ValueError("Checkout contains a link outside its folder: " + str(dst))
                if dst.exists():
                    if not dst.is_file() or src.stat().st_size != dst.stat().st_size or digest(src) != digest(dst):
                        raise ValueError("Existing checkout file differs; nothing was overwritten: " + str(rel))
                    existing += 1
                else:
                    needed += src.stat().st_size
                    to_copy.append((src, dst))
    free = shutil.disk_usage(target).free
    if free < needed + 2_000_000_000:
        raise OSError(f"Insufficient disk space: need {needed / 1e9:.1f} GB plus 2 GB spare; {free / 1e9:.1f} GB available.")
    print(f"Copying {len(to_copy):,} files ({needed / 1e9:.2f} GB); {existing:,} identical files already present.", flush=True)
    count = 0
    for src, dst in to_copy:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents an unexpected concurrent file from being overwritten.
        with src.open("rb") as inp, dst.open("xb") as out:
            shutil.copyfileobj(inp, out, length=1024 * 1024)
        count += 1
        if count % 5000 == 0:
            print(f"Copied {count:,}/{len(to_copy):,} files", flush=True)
    print("Local archive copied; original files were not modified.", flush=True)


def main() -> int:
    default = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "OneDrive" / "Desktop" / "kutty dinosaur"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=default)
    args = p.parse_args()
    try:
        restore(args.archive, ROOT)
    except (OSError, ValueError) as exc:
        print("Setup stopped:", exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
