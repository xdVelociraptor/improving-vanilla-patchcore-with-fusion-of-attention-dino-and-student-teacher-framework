"""Evaluation: joins frozen fused scores with manifest labels and computes the reported AUROC metrics.

Nothing here feeds back into fitting, calibration or fusion.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .manifests import Record, layout_of
from .metrics import auroc, bootstrap_ci

BRANCH_SLOTS = ("patchcore", "local_st", "global_attention", "composition")


def join_labels(df: pd.DataFrame, records: list[Record]) -> pd.DataFrame:
    lab = pd.DataFrame([{"rel_id": r.rel_id, "label": r.label, "anomaly_type": r.anomaly_type,
                         "subgroup": r.subgroup} for r in records if r.split == "test"])
    out = df.merge(lab, on="rel_id", how="left", validate="one_to_one")
    if out["label"].isna().any():
        raise ValueError("predictions contain images that are not in the test manifest")
    if len(out) != len(lab):
        raise ValueError(f"expected {len(lab)} test predictions, got {len(out)}")
    return out


def _auc_stat(y, s):
    if y.min() == y.max():
        return None
    return auroc(y, s)[0]


def category_metrics(df: pd.DataFrame, dataset: str, eval_cfg: dict) -> dict:
    y = df["label"].to_numpy().astype(int)
    s = df["fused_score"].to_numpy()
    m: dict = {"dataset": dataset, "category": df["category"].iloc[0], "n_test": len(df),
               "n_anomalous": int(y.sum()), "n_normal": int((1 - y).sum())}
    a, note = auroc(y, s)
    m["auroc"] = a
    m["auroc_note"] = note
    ci = bootstrap_ci(y, s, _auc_stat, eval_cfg["bootstrap_resamples"], eval_cfg["bootstrap_seed"])
    m["auroc_ci_lo"], m["auroc_ci_hi"] = ci["lo"], ci["hi"]
    if layout_of(dataset) == "loco":
        normal = y == 0
        for sub in ("logical", "structural"):
            sel = normal | (df["subgroup"] == sub).to_numpy()
            m[f"auroc_{sub}"] = auroc(y[sel], s[sel])[0]
            m[f"n_{sub}"] = int((df["subgroup"] == sub).to_numpy().sum())
        m["auroc_balanced_logical_structural"] = float(np.mean([m["auroc_logical"], m["auroc_structural"]]))
    for slot in BRANCH_SLOTS:
        col = f"score_{slot}"
        if col in df and df[col].notna().all():
            m[f"branch_auroc__{slot}"] = auroc(y, df[col].to_numpy())[0]
        else:
            m[f"branch_auroc__{slot}"] = float("nan")
    return m


def dataset_summary(cat_metrics: list[dict], preds: pd.DataFrame, dataset: str, expected_categories: list[str]) -> dict:
    got = [m["category"] for m in cat_metrics]
    missing = [c for c in expected_categories if c not in got]
    out: dict = {"dataset": dataset, "categories_evaluated": got, "categories_missing": missing,
                 "partial_run": bool(missing)}
    if not cat_metrics:
        return out

    def macro(k):
        v = [m[k] for m in cat_metrics if k in m and m[k] is not None and np.isfinite(m[k])]
        return float(np.mean(v)) if v else float("nan")
    out["macro_auroc"] = macro("auroc")
    if layout_of(dataset) == "loco":
        for k in ("auroc_logical", "auroc_structural", "auroc_balanced_logical_structural"):
            out[f"macro_{k}"] = macro(k)
    for slot in BRANCH_SLOTS:
        out[f"macro_branch_auroc__{slot}"] = macro(f"branch_auroc__{slot}")
    y = preds["label"].to_numpy().astype(int)
    out["pooled_auroc_of_fused_scores"] = auroc(y, preds["fused_score"].to_numpy())[0]
    out["pooled_note"] = ("pooled AUROC ranks standardized fused scores across categories and is reported only as a "
                          "labelled secondary number; the macro mean over categories is the primary summary")
    return out


def json_safe(o):
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating,)):
        return json_safe(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o
