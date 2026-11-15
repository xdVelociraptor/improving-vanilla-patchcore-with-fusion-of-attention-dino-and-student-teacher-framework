"""High-level runs: fit/score/calibrate for categories, and final (test) evaluation with outputs."""
from __future__ import annotations

import time
import traceback

import pandas as pd

from . import prereg
from .config import project_path
from .evaluate import join_labels, category_metrics, dataset_summary, json_safe
from .pipeline import fit_and_score, calibrate, predict, manifests_for, run_dir
from .provenance import environment_report, WEIGHTS, REFERENCE_CODE, AUX_DATA
from .utils import write_json


def prepare(cfg: dict, dataset: str, categories: list[str], seed: int, log=print) -> dict:
    """Fit + score (calibration and test images) + calibrate. Reads no labels."""
    report = {}
    reproduction = bool(cfg["experiment"].get("reproduction"))
    allowed = cfg["experiment"].get("datasets")
    if allowed and dataset not in allowed:
        log(f"{cfg['experiment']['name']}: not defined for {dataset}; skipped")
        return {}
    for cat in categories:
        t0 = time.time()
        try:
            if reproduction:   # full official training set: no held-out normals, so no calibration
                st = fit_and_score(cfg, dataset, cat, seed, parts=("test",), log=log)
                report[cat] = {"ok": True, "branches": st, "active": list(st),
                               "seconds": time.time() - t0, "note": "reproduction: AUROC only, no held-out calibration"}
                continue
            st = fit_and_score(cfg, dataset, cat, seed, log=log)
            cal = calibrate(cfg, dataset, cat, seed)
            report[cat] = {"ok": True, "branches": st, "active": cal["active_branches"],
                           "seconds": time.time() - t0}
        except Exception as e:  # noqa: BLE001 - recorded, run continues, partial run flagged
            report[cat] = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
            log(f"[{dataset}/{cat}] FAILED: {e}")
    out = project_path(cfg["paths"]["runs"]) / cfg["experiment"]["name"] / dataset / f"seed{seed}"
    write_json(out / "prepare_status.json", report)
    return report


def reproduce(cfg: dict, dataset: str, categories: list[str], seed: int, exploratory: bool = False, log=print) -> dict:
    """Published-protocol reproduction: full official training set, AUROC only."""
    from .metrics import auroc
    statuses = {c: prereg.check(cfg, dataset, c, exploratory) for c in categories}
    overall = {"confirmatory": all(s["confirmatory"] for s in statuses.values()),
               "reasons": sorted({r for s in statuses.values() for r in s["reasons"]})}
    prereg.log_test_access(cfg, dataset, categories, overall)
    from .pipeline import load_scores, _primary
    rows = []
    for cat in categories:
        fit_and_score(cfg, dataset, cat, seed, parts=("test",), log=log)
        records, _ = manifests_for(cfg, dataset, cat, create=False)
        lab = {r.rel_id: r.label for r in records}
        for slot, out in load_scores(cfg, dataset, cat, seed, "test").items():
            bcfg = cfg["experiment"]["branch_cfgs"][slot]
            s = _primary(bcfg, out)
            y = [lab[i] for i in out.rel_ids]
            a, note = auroc(y, s)
            rows.append({"dataset": dataset, "category": cat, "branch": slot, "variant": bcfg["variant"],
                         "score": bcfg.get("primary_score"), "image_auroc": a, "note": note,
                         "n_fit_images": sum(1 for r in records if r.source_split == "train"),
                         "status": "published-protocol reproduction: no held-out calibration; not the paired baseline"})
    out_dir = project_path(cfg["paths"]["runs"]) / cfg["experiment"]["name"] / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "reproduction_metrics.csv", index=False)
    summary = {"macro_image_auroc": float(df["image_auroc"].mean()) if len(df) else None,
               "evaluation_status": overall, "rows": rows}
    write_json(out_dir / "reproduction_summary.json", summary)
    return summary


def evaluate_experiment(cfg: dict, dataset: str, categories: list[str], seed: int, exploratory: bool = False,
                        make_plots: bool = True, log=print) -> dict:
    """Final evaluation on the official test split, gated by pre-registration."""
    statuses = {c: prereg.check(cfg, dataset, c, exploratory) for c in categories}
    overall = {"confirmatory": all(s["confirmatory"] for s in statuses.values()),
               "reasons": sorted({r for s in statuses.values() for r in s["reasons"]})}
    prereg.log_test_access(cfg, dataset, categories, overall)
    preds, cat_metrics, failures = [], [], {}
    for cat in categories:
        try:
            records, _ = manifests_for(cfg, dataset, cat, create=False)
            df = join_labels(predict(cfg, dataset, cat, seed), records)
            preds.append(df)
            cat_metrics.append(category_metrics(df, dataset, cfg["evaluation"]))
        except Exception as e:  # noqa: BLE001
            failures[cat] = f"{type(e).__name__}: {e}"
            log(f"[{dataset}/{cat}] evaluation missing: {e}")
    out = project_path(cfg["paths"]["runs"]) / cfg["experiment"]["name"] / dataset / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    all_preds = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    all_preds["evaluation_status"] = "confirmatory" if overall["confirmatory"] else "EXPLORATORY"
    all_preds.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(cat_metrics).to_csv(out / "category_metrics.csv", index=False)
    expected = cfg["datasets"][dataset]["categories"] if dataset in cfg["datasets"] else categories
    summary = dataset_summary(cat_metrics, all_preds, dataset, expected)
    summary["categories_failed"] = failures
    summary["evaluation_status"] = overall
    summary["experiment"] = cfg["experiment"]["name"]
    summary["seed"] = seed
    summary["fusion"] = cfg["experiment"]["fusion"]
    write_json(out / "aggregate.json", json_safe(summary))
    write_json(out / "environment.json", environment_report())
    write_json(out / "provenance.json", {"weights": WEIGHTS, "reference_code": REFERENCE_CODE, "aux_data": AUX_DATA})
    write_json(out / "config_used.json", {k: v for k, v in cfg.items() if k != "_source"} | {"source": cfg["_source"]})
    if make_plots and preds:
        from .visualize import evaluation_figures
        evaluation_figures(all_preds, cat_metrics, out / "figures", cfg, dataset)
    return summary
