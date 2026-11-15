"""DIAGNOSTIC ONLY (no pre-registered code change; outputs to work_dir/tmp, never used as results).

Re-runs leather's DINO-reconstruction training with the exact frozen code and configuration, and logs CUDA memory
every 20 optimizer steps: allocated (live tensors), reserved (allocator cache), and peak. It records allocation
stack traces, and when allocated memory passes --stop-gb it dumps a snapshot of the live allocations and stops.
  allocated grows  -> tensors are being kept alive (a leak): the snapshot shows the allocation sites.
  only reserved grows -> allocator fragmentation.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--category", default="leather")
ap.add_argument("--stop-gb", type=float, default=6.5)
ap.add_argument("--max-steps", type=int, default=1200)
a = ap.parse_args()

import torch  # noqa: E402
from hybridad.branches import dino_recon  # noqa: E402
from hybridad.manifests import select  # noqa: E402
from hybridad.pipeline import load_experiment, make_branch, manifests_for, work_dir  # noqa: E402

cfg = load_experiment(ROOT / "configs/experiments/C_dino.yaml")
ds = "mvtec_ad"
recs, msha = manifests_for(cfg, ds, a.category, create=False)
out = work_dir(cfg) / "tmp" / f"diag_{a.category}"
if out.exists():
    import shutil
    shutil.rmtree(out)
log_p = ROOT / "logs" / f"diagnose_{a.category}.jsonl"
log_p.write_text("")
torch.cuda.memory._record_memory_history(max_entries=200000)

state = {"n": 0, "t0": time.time()}
real_step = dino_recon.StableAdamW.step


class Stop(Exception):
    pass


def logging_step(self, *args, **kwargs):
    r = real_step(self, *args, **kwargs)
    state["n"] += 1
    if state["n"] % 20 == 0:
        torch.cuda.synchronize()
        rec = {"step": state["n"], "s": round(time.time() - state["t0"], 1),
               "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
               "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
               "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
               "n_alloc_retries": torch.cuda.memory_stats().get("num_alloc_retries", 0)}
        with open(log_p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["allocated_gb"] > a.stop_gb or state["n"] >= a.max_steps:
            snap = torch.cuda.memory._snapshot()
            with open(ROOT / "logs" / f"diagnose_{a.category}_snapshot.pickle", "wb") as f:
                pickle.dump(snap, f)
            raise Stop(f"stopped at step {state['n']}: {rec}")
    return r


dino_recon.StableAdamW.step = logging_step
branch = make_branch(cfg["experiment"]["branch_cfgs"]["global_attention"], cfg, ds, a.category, 0, msha)
try:
    branch.fit(select(recs, "fit"), out)
    print("finished without reaching the stop threshold", flush=True)
except Stop as e:
    print(str(e), flush=True)
except torch.OutOfMemoryError as e:
    snap = torch.cuda.memory._snapshot()
    with open(ROOT / "logs" / f"diagnose_{a.category}_snapshot.pickle", "wb") as f:
        pickle.dump(snap, f)
    print("OOM at step", state["n"], str(e)[:200], flush=True)
