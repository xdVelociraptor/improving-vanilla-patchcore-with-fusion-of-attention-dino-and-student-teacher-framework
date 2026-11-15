"""Image-level ranking metrics (AUROC) with explicit conventions.

Conventions (recorded in every output):
  * label 1 = anomalous, 0 = normal; higher score = more anomalous.
  * AUROC is computed from continuous scores (sklearn, ties count 1/2) and is undefined
    (NaN + reason) when only one class is present.
  * AUROC confidence intervals: stratified percentile bootstrap (positives and negatives
    resampled separately).

This package reports AUROC-based metrics only.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


class MetricInputError(ValueError):
    pass


def _check(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise MetricInputError(f"labels {y.shape} and scores {s.shape} differ in length")
    if not set(np.unique(y)).issubset({0, 1}):
        raise MetricInputError("labels must be 0/1")
    if not np.isfinite(s).all():
        bad = int((~np.isfinite(s)).sum())
        raise MetricInputError(f"{bad} non-finite scores; resolve them explicitly before computing metrics")
    return y, s


def auroc(labels, scores) -> tuple[float, str]:
    y, s = _check(labels, scores)
    if y.min() == y.max():
        return float("nan"), "undefined: only one class present"
    return float(roc_auc_score(y, s)), ""


def roc_curve_exact(labels, scores) -> dict:
    """Empirical ROC curve (true- vs false-positive rate) over every distinct score, for plotting AUROC."""
    y, s = _check(labels, scores)
    P, N = int(y.sum()), int((1 - y).sum())
    cut = np.unique(s)[::-1]
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    last = np.searchsorted(-s_sorted, -cut, side="right") - 1
    tpr = np.concatenate([[0.0], tp_cum[last] / P if P else np.full(len(cut), np.nan)])
    fpr = np.concatenate([[0.0], fp_cum[last] / N if N else np.full(len(cut), np.nan)])
    return {"tpr": tpr, "fpr": fpr, "P": P, "N": N}


def bootstrap_ci(labels, scores, stat, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Stratified percentile bootstrap (resamples positives and negatives separately)."""
    y, s = _check(labels, scores)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)) if len(pos) else pos,
                              rng.choice(neg, len(neg)) if len(neg) else neg])
        v = stat(y[idx], s[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return {"lo": float("nan"), "hi": float("nan"), "n_valid": len(vals)}
    return {"lo": float(np.quantile(vals, alpha / 2)), "hi": float(np.quantile(vals, 1 - alpha / 2)),
            "n_valid": len(vals), "method": f"stratified percentile bootstrap, {n_boot} resamples, seed {seed}"}
