"""MVTec LOCO localization in two stages, using exactly the package code of `hybridad localize`.

Stage `maps`  (light): writes the full-resolution fused LOCO anomaly maps as TIFFs, one image at a time
               (hybridad.localization.write_loco_maps). Reads no labels or masks, so it can run while other work runs.
Stage `score` (needs ~3 GB free RAM): runs the official MVTec LOCO evaluator v2.0 on those TIFFs
               (hybridad.localization.run_loco_evaluator, with its compatibility shims) and writes
               runs_rerun/<exp>/mvtec_loco/<cat>/seed0/localization.json; logs the test access.
`hybridad localize` does both stages in one call; this script only lets them run at different times.

  python scripts/localize_stream.py maps  --experiment G_full [--category ...] [--threads 4]
  python scripts/localize_stream.py score --experiment G_full [--category ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["maps", "score"])
    ap.add_argument("--experiment", default="G_full")
    ap.add_argument("--category", nargs="*")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dataset", default="mvtec_loco")
    ap.add_argument("--extra", nargs="*", default=[], help="config overrides (e.g. smoke/override.yaml)")
    ap.add_argument("--data-root", default=None, help="override the dataset root (synthetic checks)")
    a = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    from hybridad.localization import write_loco_maps, run_loco_evaluator, log_localization_access, loco_maps_dir
    from hybridad.pipeline import load_experiment, run_dir
    from hybridad.utils import write_json

    ds = a.dataset
    cfg = load_experiment(ROOT / f"configs/experiments/{a.experiment}.yaml", a.extra)
    if a.data_root:
        cfg["paths"]["datasets"][ds] = a.data_root
    cats = a.category or cfg["datasets"][ds]["categories"]
    base = loco_maps_dir(cfg, 0)

    for cat in cats:
        done = base / cat / "MAPS_COMPLETE.json"
        if a.stage == "maps":
            if done.exists():
                print(cat, "maps already complete", flush=True)
                continue
            info = write_loco_maps(cfg, ds, cat, 0, log=lambda m: print(m, flush=True))
            write_json(done, info)
            print(cat, "maps complete", flush=True)
        else:
            if not done.exists():
                print(cat, "maps not generated yet; run the maps stage first", flush=True)
                continue
            res = {"dataset": ds, "category": cat, **log_localization_access(cfg, ds, cat), **json.loads(done.read_text())}
            res.update(run_loco_evaluator(cfg, ds, cat, 0))
            write_json(run_dir(cfg, ds, cat, 0) / "localization.json", res)
            print(cat, res.get("au_spro_status"), res.get("au_spro_mean_at_0.05"), flush=True)


if __name__ == "__main__":
    main()
