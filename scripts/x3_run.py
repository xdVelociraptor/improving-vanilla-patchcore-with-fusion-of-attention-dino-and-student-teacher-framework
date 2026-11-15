"""EXPLORATORY X3 runner. Registers the hybridad_x branches at runtime (no change to the frozen pipeline source).
  python scripts/x3_run.py prepare            # fit + score (no labels), calibrate on validation normals
  python scripts/x3_run.py evaluate           # ONE evaluation on the LOCO test split, labelled EXPLORATORY
  python scripts/x3_run.py smoke              # synthetic fixture (code correctness only)"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridad import pipeline  # noqa: E402

pipeline.BRANCH_MODULES["patchcore_hr"] = ("hybridad_x.patchcore_hr", "PatchCoreHRBranch")
pipeline.BRANCH_MODULES["composition_hr"] = ("hybridad_x.composition_hr", "CompositionHRBranch")
from hybridad.runner import prepare, evaluate_experiment  # noqa: E402

stage = sys.argv[1]
exps = ["X3_mean", "X3_max"]
if stage == "smoke":
    from hybridad.synthetic import make_synthetic_loco
    extra = [ROOT / "configs/smoke/override.yaml", ROOT / "configs/exploratory/smoke.yaml"]
    for name in exps:
        cfg = pipeline.load_experiment(ROOT / f"configs/exploratory/experiments/{name}.yaml", extra)
        root = pipeline.work_dir(cfg) / "synthetic"
        if not (root / "synthetic_loco").exists():
            make_synthetic_loco(root / "synthetic_loco")
        cfg["paths"]["datasets"]["synthetic_loco"] = str(root / "synthetic_loco")
        cfg["paths"]["runs"] = "runs_smoke/exploratory"
        rep = prepare(cfg, "synthetic_loco", ["pairs"], 0)
        assert all(r["ok"] for r in rep.values()), rep
        s = evaluate_experiment(cfg, "synthetic_loco", ["pairs"], 0, exploratory=True, make_plots=False)
        print("SMOKE", name, s.get("macro_auroc"), "(synthetic; not evidence)")
    sys.exit(0)
cats = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
for name in exps:
    cfg = pipeline.load_experiment(ROOT / f"configs/exploratory/experiments/{name}.yaml")
    if stage == "prepare":
        rep = prepare(cfg, "mvtec_loco", cats, 0)
        print(name, {c: (r["ok"], r.get("error", "")[:200]) for c, r in rep.items()}, flush=True)
    elif stage == "evaluate":
        s = evaluate_experiment(cfg, "mvtec_loco", cats, 0, exploratory=True, make_plots=True)
        with open(ROOT / "runs_rerun/TEST_ACCESS_LOG.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "experiment": name,
                                "dataset": "mvtec_loco", "categories": cats, "confirmatory": False,
                                "reasons": ["EXPLORATORY X3 (docs/X3.md)"]}) + "\n")
        print(f"{name}: macro AUROC {s['macro_auroc']:.3f} pooled AUROC {s['pooled_auroc_of_fused_scores']:.3f} "
              f"logical {s['macro_auroc_logical']:.3f} structural {s['macro_auroc_structural']:.3f}", flush=True)
