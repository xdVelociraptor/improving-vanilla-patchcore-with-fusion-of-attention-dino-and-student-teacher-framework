"""Branch score normalization and fusion.

Standardized score:  z = (s - median) / (Q95 - median + eps), fitted on the
calib_scale (or calib_shared) normal images of the category. z is NOT a probability.

Fusion rules: max (pre-registered default), mean, and the OR rule. The OR rule subtracts a
per-branch offset fitted on the second held-out normal half (calib_threshold / calib_shared) from
each standardized score and takes the maximum.

A calibration record binds: active branches, branch variants, score-file hashes,
normalizers, fusion rule and OR offsets. `binding_hash` changes if any of them changes, and
evaluation refuses a record whose binding no longer matches.
"""
from __future__ import annotations

import math

import numpy as np

from .config import stable_hash

MISSING_SENTINEL = 1.0e6  # standardized score assigned to a missing branch score under policy flag_anomalous


class CalibrationError(RuntimeError):
    pass


def fit_normalizer(scale_scores, q: float = 0.95, rel_tol: float = 1e-6) -> dict:
    s = np.asarray(scale_scores, dtype=np.float64)
    if len(s) == 0:
        raise CalibrationError("no scale-calibration scores")
    if not np.isfinite(s).all():
        raise CalibrationError("non-finite scale-calibration scores")
    med = float(np.median(s))
    qv = float(np.quantile(s, q))
    eps = 1e-12 * max(1.0, abs(med))
    spread = qv - med
    tol = rel_tol * max(abs(med), 1e-12)
    method = f"(s - median) / (Q{int(q * 100)} - median + eps)"
    degenerate = False
    if spread <= tol:
        mad = float(np.median(np.abs(s - med))) * 1.4826
        std = float(s.std())
        if mad > tol:
            spread, method = mad, "near-constant Q95: (s - median) / (1.4826*MAD)"
        elif std > tol:
            spread, method = std, "near-constant Q95 and MAD: (s - median) / std"
        else:
            spread, method, degenerate = 1.0, "constant branch: (s - median) / 1  [DEGENERATE]", True
    return {"median": med, f"q{int(q * 100)}": qv, "scale": float(spread + eps), "method": method,
            "degenerate": degenerate, "n": int(len(s)), "quantile": q}


def apply_normalizer(scores, norm: dict) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    return (s - norm["median"]) / norm["scale"]


def or_offset(normal_scores, alpha: float) -> float:
    """OR-rule offset for one branch: with the normal scores sorted descending s_(1) >= ... >= s_(n) and
    k = floor(alpha * n), the smallest float strictly above s_(k+1). Fitted on normal images only."""
    s = np.sort(np.asarray(normal_scores, dtype=np.float64))[::-1]
    if len(s) == 0 or not np.isfinite(s).all():
        raise CalibrationError("OR offset needs finite normal scores")
    k = int(math.floor(alpha * len(s) + 1e-12))
    return -math.inf if k >= len(s) else float(np.nextafter(s[k], np.inf))


def fuse(z: dict[str, np.ndarray], rule: str, or_offsets: dict[str, float] | None = None) -> np.ndarray:
    """z: branch -> standardized scores (same image order). NaN = missing (handled upstream)."""
    if not z:
        raise CalibrationError("no active branches to fuse")
    stack = np.stack([np.asarray(z[b], dtype=np.float64) for b in sorted(z)], axis=0)
    if not np.isfinite(stack).all():
        raise CalibrationError("non-finite standardized scores reached fusion; apply the missing-score policy first")
    if rule == "max":
        return stack.max(0)
    if rule == "mean":
        return stack.mean(0)
    if rule == "or":
        if or_offsets is None:
            raise CalibrationError("OR fusion requires per-branch offsets")
        t = np.array([or_offsets[b] for b in sorted(z)], dtype=np.float64)[:, None]
        return (stack - t).max(0)
    raise CalibrationError(f"unknown fusion rule {rule}")


def apply_missing_policy(raw: dict[str, np.ndarray], policy: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Returns (raw with missing entries marked NaN, missing masks). Policy is applied after normalization."""
    masks = {b: ~np.isfinite(np.asarray(v, dtype=np.float64)) for b, v in raw.items()}
    if policy not in ("flag_anomalous", "error"):
        raise CalibrationError(f"unknown missing-score policy {policy}")
    if policy == "error":
        for b, m in masks.items():
            if m.any():
                raise CalibrationError(f"branch {b} has {int(m.sum())} missing scores")
    return raw, masks


def standardize(raw: dict[str, np.ndarray], normalizers: dict[str, dict], policy: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw, masks = apply_missing_policy(raw, policy)
    z = {}
    for b, v in raw.items():
        zb = apply_normalizer(np.where(masks[b], 0.0, v), normalizers[b])
        zb[masks[b]] = MISSING_SENTINEL
        z[b] = zb
    return z, masks


def calibrate_normal_only(scale_raw: dict[str, np.ndarray], second_raw: dict[str, np.ndarray], *, fusion: str,
                          or_alpha: float, missing_policy: str, binding: dict, shared_calibration: bool) -> dict:
    """Normalizers from the scale half; OR-rule offsets (OR fusion only) from the second normal half."""
    normalizers = {}
    for b in sorted(scale_raw):
        v = np.asarray(scale_raw[b], dtype=np.float64)
        if not np.isfinite(v).all():
            raise CalibrationError(f"branch {b}: non-finite scores on normal scale-calibration images "
                                   f"({int((~np.isfinite(v)).sum())}); a pre-registered fallback cannot be fitted from them")
        normalizers[b] = fit_normalizer(v)
    active = [b for b in sorted(normalizers) if not normalizers[b]["degenerate"]]
    excluded = [b for b in sorted(normalizers) if normalizers[b]["degenerate"]]
    if not active:
        raise CalibrationError("every branch is degenerate on the scale set")
    or_offsets = None
    if fusion == "or":
        z2, _ = standardize({b: second_raw[b] for b in active}, normalizers, missing_policy)
        or_offsets = {b: or_offset(z2[b], or_alpha / len(active)) for b in active}
    rec = {
        "mode": "normal_only",
        "active_branches": active,
        "excluded_degenerate_branches": excluded,
        "normalizers": normalizers,
        "fusion": fusion,
        "or_offsets": or_offsets,
        "missing_policy": missing_policy,
        "shared_scale_calibration": shared_calibration,
        "binding": binding,
    }
    rec["binding_hash"] = binding_hash(rec)
    return rec


def binding_hash(rec: dict) -> str:
    keys = ("mode", "active_branches", "normalizers", "fusion", "or_offsets", "missing_policy", "binding")
    return stable_hash({k: rec.get(k) for k in keys}, n=24)


def verify_binding(rec: dict, binding: dict) -> None:
    if rec.get("binding") != binding:
        diff = {k: (rec.get("binding", {}).get(k), binding.get(k)) for k in set(binding) | set(rec.get("binding", {}))
                if rec.get("binding", {}).get(k) != binding.get(k)}
        raise CalibrationError(f"calibration is stale: binding differs in {sorted(diff)}; recalibrate")
    if binding_hash(rec) != rec.get("binding_hash"):
        raise CalibrationError("calibration record was edited after creation (binding hash mismatch)")
