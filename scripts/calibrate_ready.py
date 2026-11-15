"""Calibration lane: calibrate every (experiment, category) whose branch score files all exist.
Never fits or trains anything; repeats until every calibration exists. Reads no labels."""
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hybridad.pipeline import (load_experiment, branch_enabled, score_path, calibrate, run_dir)  # noqa: E402

exps = [load_experiment(p) for p in sorted(glob.glob("configs/experiments/*.yaml"))]
exps = [c for c in exps if not c["experiment"].get("reproduction")]
while True:
    pending = 0
    for cfg in exps:
        for ds in ("mvtec_loco", "mvtec_ad"):
            for cat in cfg["datasets"][ds]["categories"]:
                if (run_dir(cfg, ds, cat, 0) / "calibration.json").exists():
                    continue
                slots = [bc for bc in cfg["experiment"]["branch_cfgs"].values() if branch_enabled(bc, cfg, ds, cat)[0]]
                if not slots:
                    continue   # not applicable (e.g. composition-only on a texture category)
                if all(score_path(cfg, bc, ds, cat, 0, p).exists() for bc in slots for p in ("calib", "test")):
                    try:
                        calibrate(cfg, ds, cat, 0)
                        print(time.strftime("%H:%M:%S"), "calibrated", cfg["experiment"]["name"], ds, cat, flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(time.strftime("%H:%M:%S"), "FAILED", cfg["experiment"]["name"], ds, cat, e, flush=True)
                        pending += 1
                else:
                    pending += 1
    if pending == 0:
        print("CALIBRATION_LANE_DONE", flush=True)
        break
    time.sleep(300)
