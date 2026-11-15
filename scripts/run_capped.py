"""Runs the frozen hybridad CLI unchanged, under a per-process CUDA memory cap (no code, config or math change).

Why (logs/diagnose_leather.jsonl, 2026-09-28): during DINO's hard-mining warm-up the per-step temporaries change size,
so PyTorch's caching allocator keeps opening new segments. On Windows (WDDM) cudaMalloc then falls back to system
memory instead of failing, so the allocator never frees its cache: reserved grew 4.3 -> 16.8 GB on the 8 GB card by
step 860 while live memory stayed at 1.73 GB (peak 3.93 GB), until system memory ran out. With a cap, an allocation
that would exceed it makes the allocator free cached blocks and retry, which keeps reserved bounded.
The cap only changes when cached memory is returned to the driver; it cannot change the method or its math. (Strict
bitwise identity would additionally require that no cuBLAS/cuDNN/SDPA kernel choice depends on free workspace; that is
unlikely here, since the encoder outputs are cached and the decoder is Linear/attention.)

Usage: python scripts/run_capped.py [--fraction 0.8] [--mem-log logs/x.jsonl] -- prepare --experiment ... (CLI args)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

argv = sys.argv[1:]
split = argv.index("--") if "--" in argv else len(argv)
ap = argparse.ArgumentParser()
ap.add_argument("--fraction", type=float, default=0.8)
ap.add_argument("--mem-log", default=None)
ap.add_argument("--every", type=int, default=100)
a = ap.parse_args(argv[:split])
cli_args = argv[split + 1:]

import torch  # noqa: E402

torch.cuda.set_per_process_memory_fraction(a.fraction)
print(f"CUDA memory cap: {a.fraction} x {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GB", flush=True)

if a.mem_log:
    from hybridad.branches import dino_recon  # noqa: E402

    log_p = ROOT / a.mem_log
    state = {"n": 0, "t0": time.time()}
    real_step = dino_recon.StableAdamW.step

    def logging_step(self, *args, **kwargs):
        r = real_step(self, *args, **kwargs)   # the optimizer step itself is untouched
        state["n"] += 1
        if state["n"] % a.every == 0:
            st = torch.cuda.memory_stats()
            rec = {"step": state["n"], "s": round(time.time() - state["t0"], 1),
                   "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
                   "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
                   "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   "n_alloc_retries": st.get("num_alloc_retries", 0), "n_ooms": st.get("num_ooms", 0)}
            with open(log_p, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return r

    dino_recon.StableAdamW.step = logging_step

from hybridad.cli import main  # noqa: E402

main(cli_args)
