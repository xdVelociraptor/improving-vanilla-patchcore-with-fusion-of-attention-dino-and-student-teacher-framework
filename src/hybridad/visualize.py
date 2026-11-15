"""Figures generated only from measured, saved outputs: ROC curves with their AUROC values."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import auroc, roc_curve_exact  # noqa: E402

SLOTS = ("patchcore", "local_st", "global_attention", "composition")


def evaluation_figures(preds: pd.DataFrame, cat_metrics: list[dict], out: Path, cfg: dict, dataset: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cats = sorted(preds["category"].unique())
    _grid_plot(preds, cats, out / "roc_curves.png", _roc_ax, "ROC curves: fused score and each branch (AUROC in legend)")


def _grid_plot(preds, cats, path, fn, title):
    n = len(cats)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), squeeze=False)
    for ax, c in zip(axes.ravel(), cats):
        fn(ax, preds[preds["category"] == c])
        ax.set_title(c, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _roc_ax(ax, d):
    y, s = d["label"].to_numpy(), d["fused_score"].to_numpy()
    if y.min() == y.max():
        ax.text(0.5, 0.5, "single class", ha="center")
        return
    r = roc_curve_exact(y, s)
    ax.step(r["fpr"], r["tpr"], where="post", lw=1.2, label=f"fused ({auroc(y, s)[0]:.3f})")
    for slot in SLOTS:
        col = f"score_{slot}"
        if col in d and d[col].notna().all():
            v = d[col].to_numpy()
            rr = roc_curve_exact(y, v)
            ax.step(rr["fpr"], rr["tpr"], where="post", lw=0.7, alpha=0.7, label=f"{slot} ({auroc(y, v)[0]:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.6)
    ax.set_xlabel("false-positive rate", fontsize=8)
    ax.set_ylabel("true-positive rate", fontsize=8)
    ax.legend(fontsize=5, loc="lower right")
