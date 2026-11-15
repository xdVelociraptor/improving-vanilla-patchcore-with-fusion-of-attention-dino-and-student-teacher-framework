"""Command-line entry points.  Run `python -m hybridad.cli <command> -h` for options."""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from .config import load_yaml, project_path, CONFIG_DIR, PROJECT_ROOT


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _categories(cfg: dict, dataset: str, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return list(cfg["datasets"][dataset]["categories"])


def _datasets(arg: str) -> list[str]:
    return ["mvtec_ad", "mvtec_loco"] if arg == "all" else [arg]


def _experiments(patterns: list[str]) -> list[str]:
    paths = []
    for p in patterns:
        hits = sorted(glob.glob(str(project_path(p)))) or sorted(glob.glob(str(CONFIG_DIR / p)))
        if not hits:
            raise SystemExit(f"no experiment config matches {p}")
        paths += hits
    return paths


def cmd_manifests(a):
    from .manifests import build_manifest
    from .pipeline import data_root
    cfg = load_yaml(a.config)
    for ds in _datasets(a.dataset):
        for cat in _categories(cfg, ds, a.category):
            meta = build_manifest(ds, data_root(cfg, ds), cat, cfg["protocol"], project_path(cfg["paths"]["manifests"]))
            _log(f"{ds}/{cat}: {meta['counts']} sha={meta['manifest_sha256'][:12]} "
                 f"dups={len(meta['duplicate_groups'])} calib_reuse={meta['calibration_reuse']}")


def cmd_prepare(a):
    from .pipeline import load_experiment
    from .runner import prepare
    for path in _experiments(a.experiment):
        cfg = load_experiment(path, a.extra)
        for ds in _datasets(a.dataset):
            for seed in a.seeds:
                rep = prepare(cfg, ds, _categories(cfg, ds, a.category), seed, log=_log)
                bad = [c for c, r in rep.items() if not r["ok"]]
                _log(f"{cfg['experiment']['name']} {ds} seed{seed}: prepared {len(rep) - len(bad)}/{len(rep)}"
                     + (f" FAILED: {bad}" if bad else ""))


def cmd_preregister(a):
    from .pipeline import load_experiment
    from .prereg import preregister
    cfgs = [load_experiment(p, a.extra) for p in _experiments(a.experiments)]
    ds = {d: _categories(cfgs[0], d, None) for d in _datasets(a.dataset)}
    rec = preregister(cfgs, ds, a.note)
    _log(f"pre-registered {len(rec['experiments'])} experiments, code hash {rec['code_hash']}")


def cmd_evaluate(a):
    from .pipeline import load_experiment
    from .runner import evaluate_experiment, reproduce
    for path in _experiments(a.experiment):
        cfg = load_experiment(path, a.extra)
        for ds in _datasets(a.dataset):
            if cfg["experiment"].get("datasets") and ds not in cfg["experiment"]["datasets"]:
                continue
            for seed in a.seeds:
                if cfg["experiment"].get("reproduction"):
                    s = reproduce(cfg, ds, _categories(cfg, ds, a.category), seed, a.exploratory, log=_log)
                    _log(f"{cfg['experiment']['name']} {ds}: reproduction macro image AUROC {s['macro_image_auroc']}")
                    continue
                s = evaluate_experiment(cfg, ds, _categories(cfg, ds, a.category), seed, a.exploratory,
                                        not a.no_plots, log=_log)
                _log(f"{cfg['experiment']['name']} {ds} seed{seed}: macro AUROC {s.get('macro_auroc')} "
                     f"pooled AUROC {s.get('pooled_auroc_of_fused_scores')} "
                     f"[{'confirmatory' if s['evaluation_status']['confirmatory'] else 'EXPLORATORY'}]")


def cmd_smoke(a):
    """Synthetic fit-calibrate-evaluate-save-reload cycle. Code correctness only, not benchmark evidence."""
    from .pipeline import load_experiment, work_dir
    from .runner import prepare, evaluate_experiment
    from .synthetic import make_synthetic_ad, make_synthetic_loco
    smoke = CONFIG_DIR / "smoke" / "override.yaml"
    probe = load_experiment(_experiments(a.experiment)[0], [smoke] + (a.extra or []))
    root = work_dir(probe) / "synthetic"
    if not (root / "synthetic_ad").exists():
        make_synthetic_ad(root / "synthetic_ad")
        make_synthetic_loco(root / "synthetic_loco")
    for path in _experiments(a.experiment):
        cfg = load_experiment(path, [smoke] + (a.extra or []))
        cfg["paths"]["datasets"]["synthetic_ad"] = str(root / "synthetic_ad")
        cfg["paths"]["datasets"]["synthetic_loco"] = str(root / "synthetic_loco")
        for ds in ("synthetic_ad", "synthetic_loco"):
            cats = cfg["datasets"][ds]["categories"]
            rep = prepare(cfg, ds, cats, 0, log=_log)
            if not all(r["ok"] for r in rep.values()):
                raise SystemExit(f"smoke prepare failed: {json.dumps(rep, indent=1)[:3000]}")
            s = evaluate_experiment(cfg, ds, cats, 0, exploratory=True, make_plots=True, log=_log)
            _log(f"SMOKE {cfg['experiment']['name']} {ds}: AUROC {s.get('macro_auroc')} "
                 f"(synthetic fixture: NOT benchmark evidence)")


def cmd_localize(a):
    from .localization import localize
    from .pipeline import load_experiment
    for path in _experiments(a.experiment):
        cfg = load_experiment(path, a.extra)
        for ds in _datasets(a.dataset):
            for cat in _categories(cfg, ds, a.category):
                r = localize(cfg, ds, cat, a.seed)
                _log(f"{cfg['experiment']['name']} {ds}/{cat}: pixel AUROC {r.get('pixel_auroc')} "
                     f"AU-PRO {r.get('au_pro_0.3')} AU-sPRO@0.05 {r.get('au_spro_mean_at_0.05')} "
                     f"{r.get('au_spro_status', '')}")


def cmd_report(a):
    from .report import build
    _log(f"wrote {build(a.config, a.out)}")


def cmd_env(a):
    from .provenance import environment_report
    from .utils import write_json
    out = PROJECT_ROOT / "docs" / "environment.json"
    write_json(out, environment_report())
    _log(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hybridad")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, multi_exp=True):
        p.add_argument("--extra", nargs="*", default=[], help="extra YAML overrides (applied last)")

    p = sub.add_parser("manifests", help="build/verify immutable split manifests")
    p.add_argument("--config", default="base.yaml")
    p.add_argument("--dataset", default="all")
    p.add_argument("--category", nargs="*")
    p.set_defaults(fn=cmd_manifests)

    p = sub.add_parser("prepare", help="fit branches, score calibration+test images, calibrate (no labels read)")
    p.add_argument("--experiment", nargs="+", required=True)
    p.add_argument("--dataset", default="all")
    p.add_argument("--category", nargs="*")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    common(p)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("preregister", help="freeze the experiment suite before test evaluation")
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--dataset", default="all")
    p.add_argument("--note", required=True)
    common(p)
    p.set_defaults(fn=cmd_preregister)

    p = sub.add_parser("evaluate", help="final test evaluation (gated by pre-registration)")
    p.add_argument("--experiment", nargs="+", required=True)
    p.add_argument("--dataset", default="all")
    p.add_argument("--category", nargs="*")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--exploratory", action="store_true", help="allow non-pre-registered configs; outputs are labelled")
    p.add_argument("--no-plots", action="store_true")
    common(p)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("smoke", help="synthetic end-to-end cycle (code correctness only)")
    p.add_argument("--experiment", nargs="+", default=["experiments/A_patchcore.yaml"])
    common(p)
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("localize", help="pixel-level localization metrics (after evaluate)")
    p.add_argument("--experiment", nargs="+", required=True)
    p.add_argument("--dataset", default="all")
    p.add_argument("--category", nargs="*")
    p.add_argument("--seed", type=int, default=0)
    common(p)
    p.set_defaults(fn=cmd_localize)

    p = sub.add_parser("report", help="build docs/RESULTS.md from saved outputs")
    p.add_argument("--config", default="base.yaml")
    p.add_argument("--out", default="docs/RESULTS.md")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("env", help="write docs/environment.json")
    p.set_defaults(fn=cmd_env)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
