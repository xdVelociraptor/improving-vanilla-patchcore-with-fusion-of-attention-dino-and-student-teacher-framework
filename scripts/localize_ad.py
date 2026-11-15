"""MVTec AD pixel metrics (pixel AUROC, AU-PRO@0.3) via the FROZEN hybridad.localization.localize, one category at a
time, only for categories whose branches are all trained. Skips categories already done; logs each test access.
  python scripts/localize_ad.py [--experiment G_full] [--category ...] [--threads 1]"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", default="G_full")
ap.add_argument("--category", nargs="*")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--min-free-gb", type=float, default=6.0,
                help="start a category only when at least this much RAM is available (protects the trainer's page cache)")
a = ap.parse_args()


def available_gb() -> float:
    """Available physical memory on Windows (GlobalMemoryStatusEx.ullAvailPhys), in GB."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def wait_for_ram(min_gb: float) -> None:
    import time
    while available_gb() < min_gb:
        print(f"waiting: {available_gb():.1f} GB available < {min_gb} GB", flush=True)
        time.sleep(120)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = str(a.threads)
import torch  # noqa: E402
torch.set_num_threads(a.threads)
from hybridad.localization import localize  # noqa: E402
from hybridad.pipeline import load_experiment, branch_enabled, score_path, run_dir  # noqa: E402

cfg = load_experiment(ROOT / f"configs/experiments/{a.experiment}.yaml")
ds = "mvtec_ad"
for cat in a.category or cfg["datasets"][ds]["categories"]:
    if (run_dir(cfg, ds, cat, 0) / "localization.json").exists():
        print(cat, "already done", flush=True)
        continue
    slots = [bc for bc in cfg["experiment"]["branch_cfgs"].values() if branch_enabled(bc, cfg, ds, cat)[0]]
    if not all(score_path(cfg, bc, ds, cat, 0, "test").exists() for bc in slots):
        print(cat, "not trained yet; skipped", flush=True)
        continue
    wait_for_ram(a.min_free_gb)
    r = localize(cfg, ds, cat, 0)   # logs its own test-set access with the pre-registration status
    print(f"{cat}: pixel AUROC {r.get('pixel_auroc'):.4f}  AU-PRO@0.3 {r.get('au_pro_0.3'):.4f}  "
          f"unobserved px {r.get('mean_unobserved_pixel_fraction'):.3f}", flush=True)
