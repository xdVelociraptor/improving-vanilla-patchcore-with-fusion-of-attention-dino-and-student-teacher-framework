"""Self-check: confirms this folder runs on its own and reproduces its saved AUROC results.

  .venv\\Scripts\\python.exe check_package.py           full check (a few minutes; re-hashes every dataset image)
  .venv\\Scripts\\python.exe check_package.py --quick   fast check (spot-checks data, reproduces one category per dataset)

Sections: environment, pretrained weights (SHA-256), datasets (manifest verification), ImageNette, trained models and
score files, AUROC reproduction from the bundled score files (written to a scratch folder, compared with runs/ exactly),
and the unit tests. Nothing under runs/ is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
RESULTS = []


def report(section, ok, detail):
    RESULTS.append({"section": section, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {section}: {detail}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the scratch reproduction folder runs_check/")
    a = ap.parse_args()
    t0 = time.time()

    # 1. environment ----------------------------------------------------------------------------------------------
    import torch
    import hybridad
    here = Path(hybridad.__file__).resolve().is_relative_to(ROOT)
    report("environment", here and sys.version_info[:2] == (3, 12),
           f"Python {sys.version.split()[0]}, torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, "
           f"hybridad imported from {'this folder' if here else Path(hybridad.__file__).parent}")
    for mod in ("faiss", "sklearn", "scipy", "pandas", "yaml", "tifffile", "matplotlib", "PIL"):
        __import__(mod)

    # 2. weights ----------------------------------------------------------------------------------------------------
    from hybridad.provenance import verify_weight
    bad = []
    for name in ("wide_resnet50_2", "dinov2_vitb14_reg4", "pdn_small_teacher"):
        try:
            verify_weight(name)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name}: {e}")
    report("pretrained weights", not bad, "3 files, SHA-256 verified" if not bad else "; ".join(bad))

    # 3. datasets ---------------------------------------------------------------------------------------------------
    from hybridad.pipeline import load_experiment, manifests_for, data_root
    cfg = load_experiment(ROOT / "configs/experiments/G_full.yaml")
    rng = random.Random(0)
    n_files, problems = 0, []
    for ds in ("mvtec_ad", "mvtec_loco"):
        root = data_root(cfg, ds)
        for cat in cfg["datasets"][ds]["categories"]:
            try:
                recs, _ = manifests_for(cfg, ds, cat, create=not a.quick)   # full: rebuild + byte-compare the manifest
            except Exception as e:  # noqa: BLE001
                problems.append(f"{ds}/{cat}: {e}")
                continue
            missing = [r.rel_id for r in recs if not (root / r.rel_id).is_file()]
            masks = [r.mask_rel for r in recs if r.mask_rel and not (root / r.mask_rel).exists()]
            problems += [f"{ds}/{cat}: missing {m}" for m in (missing + masks)[:3]]
            for r in rng.sample(recs, min(3, len(recs))):
                if sha256(root / r.rel_id) != r.file_sha256:
                    problems.append(f"{ds}/{cat}: checksum mismatch {r.rel_id}")
            n_files += len(recs)
    report("datasets", not problems, f"{n_files} images in 20 categories present"
           + ("; manifests rebuilt and byte-identical" if not a.quick else "; spot-checked checksums")
           if not problems else "; ".join(problems[:6]))

    pen = ROOT / cfg["paths"]["imagenette"]
    n_pen = sum(1 for p in pen.rglob("*") if p.suffix.lower() in (".jpeg", ".jpg", ".png")) if pen.is_dir() else 0
    report("ImageNette (student-teacher penalty images)", n_pen > 0, f"{n_pen} images in {pen.relative_to(ROOT)}")

    # 4. trained models and score files -----------------------------------------------------------------------------
    from hybridad import pipeline
    pipeline.BRANCH_MODULES["patchcore_hr"] = ("hybridad_x.patchcore_hr", "PatchCoreHRBranch")
    pipeline.BRANCH_MODULES["composition_hr"] = ("hybridad_x.composition_hr", "CompositionHRBranch")
    need, have = 0, 0
    lost = []
    for agg in sorted(list((ROOT / "runs").glob("*/*/seed0/aggregate.json")) +
                      list((ROOT / "runs/exploratory").glob("*/*/seed0/aggregate.json"))):
        info = json.loads(agg.read_text())
        exp, ds = agg.parts[-4], agg.parts[-3]
        path = (ROOT / "configs/experiments" / f"{exp}.yaml")
        path = path if path.exists() else ROOT / "configs/exploratory/experiments" / f"{exp}.yaml"
        if not path.exists() or not info.get("categories_evaluated"):
            continue
        c = load_experiment(path)
        for cat in info["categories_evaluated"]:
            for slot, bc in c["experiment"]["branch_cfgs"].items():
                if not pipeline.branch_enabled(bc, c, ds, cat)[0]:
                    continue
                for part in ("calib", "test"):
                    need += 1
                    p = pipeline.score_path(c, bc, ds, cat, 0, part)
                    have += p.exists()
                    if not p.exists():
                        lost.append(str(p.relative_to(ROOT)))
                if not pipeline.model_dir(c, bc, ds, cat, 0).is_dir():
                    lost.append(f"model {exp}/{ds}/{cat}/{slot}")
    report("trained models and score files", not lost, f"{have}/{need} score files for every saved result"
           if not lost else f"missing: {lost[:4]}")

    # 5. AUROC reproduction -----------------------------------------------------------------------------------------
    import pandas as pd
    from hybridad.runner import prepare, evaluate_experiment
    scratch = ROOT / "runs_check"
    jobs = [("configs/experiments/G_full.yaml", "mvtec_ad"), ("configs/experiments/G_full.yaml", "mvtec_loco"),
            ("configs/exploratory/experiments/X3_mean.yaml", "mvtec_loco")]
    checked, diffs = 0, []
    for rel, ds in jobs:
        c = load_experiment(ROOT / rel)
        name = c["experiment"]["name"]
        saved_dir = ROOT / ("runs/exploratory" if "exploratory" in rel else "runs") / name / ds / "seed0"
        saved = pd.read_csv(saved_dir / "category_metrics.csv").set_index("category")
        cats = list(saved.index)[:1] if a.quick else list(saved.index)
        c["paths"]["runs"] = str(scratch)
        rep = prepare(c, ds, cats, 0, log=lambda *x: None)            # score files exist, so this only re-fits normalizers
        if not all(r["ok"] for r in rep.values()):
            diffs.append(f"{name}/{ds}: prepare failed {rep}")
            continue
        evaluate_experiment(c, ds, cats, 0, exploratory=True, make_plots=False, log=lambda *x: None)
        new = pd.read_csv(scratch / name / ds / "seed0" / "category_metrics.csv").set_index("category")
        for cat in cats:
            for col in [k for k in saved.columns if k.startswith(("auroc", "branch_auroc")) and k != "auroc_note"]:
                x, y = saved.loc[cat, col], new.loc[cat, col]
                checked += 1
                if not ((pd.isna(x) and pd.isna(y)) or abs(float(x) - float(y)) <= 1e-12):
                    diffs.append(f"{name}/{ds}/{cat}/{col}: saved {x} vs recomputed {y}")
    if not a.keep and scratch.exists():
        shutil.rmtree(scratch)
    report("AUROC reproduction from bundled scores", not diffs,
           f"{checked} AUROC values (G_full on both datasets, X3_mean on LOCO) identical to runs/"
           if not diffs else "; ".join(diffs[:4]))

    # 6. unit tests -------------------------------------------------------------------------------------------------
    tests = ["tests/test_metrics_calibration.py", "tests/test_knn.py", "tests/test_localization.py"] if a.quick else ["tests"]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=ROOT,
                       capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["no output"])[-1]
    report("unit tests", r.returncode == 0, last)

    ok = all(x["ok"] for x in RESULTS)
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "self_check.json").write_text(json.dumps(
        {"mode": "quick" if a.quick else "full", "seconds": round(time.time() - t0, 1), "all_passed": ok,
         "results": RESULTS}, indent=1))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.time() - t0:.0f} s "
          f"(details: logs/self_check.json)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
