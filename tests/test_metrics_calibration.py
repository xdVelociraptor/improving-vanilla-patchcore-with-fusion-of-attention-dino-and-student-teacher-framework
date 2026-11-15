"""Score direction, undefined AUROC, non-finite scores, bootstrap CI, normalizers, OR offsets, calibration binding."""
import math

import numpy as np
import pytest

from hybridad.calibration import (fit_normalizer, calibrate_normal_only, verify_binding, CalibrationError, standardize,
                                  MISSING_SENTINEL, fuse, or_offset)
from hybridad.metrics import auroc, bootstrap_ci, roc_curve_exact, MetricInputError


def test_score_direction_higher_is_anomalous():
    y = [0, 0, 1, 1]
    assert auroc(y, [0.1, 0.2, 0.8, 0.9])[0] == 1.0
    assert auroc(y, [0.9, 0.8, 0.2, 0.1])[0] == 0.0


def test_ties_count_half():
    assert auroc([0, 1], [0.5, 0.5])[0] == 0.5


def test_single_class_auroc_undefined():
    a, note = auroc([0, 0, 0], [0.1, 0.2, 0.3])
    assert math.isnan(a) and "one class" in note


def test_non_finite_scores_rejected():
    with pytest.raises(MetricInputError):
        auroc([0, 1], [0.1, np.nan])


def test_roc_curve_area_matches_auroc():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(40), np.zeros(60)].astype(int)
    s = rng.normal(size=100) + y
    r = roc_curve_exact(y, s)
    assert np.trapezoid(r["tpr"], r["fpr"]) == pytest.approx(auroc(y, s)[0])


def test_bootstrap_ci_brackets_auroc():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(50), np.zeros(50)].astype(int)
    s = rng.normal(size=100) + 1.5 * y
    ci = bootstrap_ci(y, s, lambda a, b: auroc(a, b)[0], n_boot=500, seed=0)
    assert ci["lo"] <= auroc(y, s)[0] <= ci["hi"]


def test_or_offset_with_ties():
    normal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], float)
    t = or_offset(normal, 0.10)
    assert (normal >= t).sum() == 1
    tied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 10], float)   # the cut falls inside a tie
    assert (tied >= or_offset(tied, 0.10)).sum() == 0


def test_normalizer_and_constant_branch():
    n = fit_normalizer(np.arange(100, dtype=float))
    assert n["median"] == pytest.approx(49.5) and not n["degenerate"]
    c = fit_normalizer(np.ones(20))
    assert c["degenerate"]


def _cal(fusion="max"):
    rng = np.random.default_rng(0)
    scale = {"a": rng.normal(size=30), "b": rng.normal(size=30) * 5 + 3}
    second = {"a": rng.normal(size=30), "b": rng.normal(size=30) * 5 + 3}
    return calibrate_normal_only(scale, second, fusion=fusion, or_alpha=0.1, missing_policy="flag_anomalous",
                                 binding={"x": 1}, shared_calibration=False)


def test_calibration_binding_detects_changes():
    rec = _cal()
    verify_binding(rec, {"x": 1})
    with pytest.raises(CalibrationError):
        verify_binding(rec, {"x": 2})
    rec["fusion"] = "mean"   # edited after the fact
    with pytest.raises(CalibrationError):
        verify_binding(rec, {"x": 1})


def test_or_fusion_has_offsets_other_rules_do_not():
    assert _cal("max")["or_offsets"] is None
    rec = _cal("or")
    assert set(rec["or_offsets"]) == {"a", "b"}


def test_missing_scores_become_alarms_not_passes():
    norm = {"a": {"median": 0.0, "scale": 1.0}}
    z, miss = standardize({"a": np.array([0.5, np.nan])}, norm, "flag_anomalous")
    assert miss["a"].tolist() == [False, True]
    assert z["a"][1] == MISSING_SENTINEL
    with pytest.raises(CalibrationError):
        standardize({"a": np.array([np.nan])}, norm, "error")
    with pytest.raises(CalibrationError):
        fuse({"a": np.array([np.nan])}, "max")
