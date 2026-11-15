"""Pixel-level localization evaluation (secondary task).

* Maps are resampled onto ORIGINAL image pixels through each branch's recorded geometry, so the
  ground-truth mask geometry is preserved. Pixels a branch never observed (e.g. outside a centre
  crop) are NaN; before scoring they are set to that image's minimum observed value ("no
  evidence") and the unobserved fraction is reported.
* Branch maps are standardized with a pixel-level normalizer fitted on the scale-calibration
  normal images, then fused with the experiment's fusion rule. The composition branch has no
  localization map and is excluded (its broadcast image score is NOT localization).
* Maps are built one image at a time: the pixel normalizers keep only their 1-in-7 pixel samples, and
  MVTec LOCO fused maps are written to TIFF as they are produced, so memory does not grow with the
  number of images.
* MVTec AD: pixel AUROC and AU-PRO up to FPR 0.3 (re-implementation of the official
  MVTec AD evaluation algorithm: per-region overlap over connected ground-truth components,
  FPR over all anomaly-free pixels, trapezoidal integration normalized by 0.3).
* MVTec LOCO: AU-sPRO via the official evaluation script v2.0 (third_party/mvtec_loco_ad_evaluation),
  which needs full-resolution TIFF maps; those are written to the work directory. The script runs
  through three compatibility shims (see `run_loco_evaluator`); the vendored code is unmodified.
* Every call logs its test-set access (the evaluation reads ground-truth masks) and records whether the
  configuration and code match the pre-registration.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

from .branches import BranchOutput
from .config import PROJECT_ROOT
from .data import TransformSpec, geometry_for, map_to_original, load_mask
from .manifests import select, CALIB_ROLES_SCALE, layout_of
from .pipeline import score_path, branch_enabled, data_root, manifests_for, run_dir, work_dir
from .utils import write_json

LOCO_EVALUATOR_SHIMS = "os.nice no-op, '/' path separators, de-duplicated initial curve points"


def _map_sources(cfg, dataset, category, seed, part) -> dict:
    """{slot: (branch cfg, transform spec, BranchOutput, rel_id -> row)} for enabled branches that produce maps."""
    out = {}
    for slot, bcfg in cfg["experiment"]["branch_cfgs"].items():
        if not branch_enabled(bcfg, cfg, dataset, category)[0]:
            continue
        o = BranchOutput.load(score_path(cfg, bcfg, dataset, category, seed, part))
        if o.maps is None:
            continue
        out[slot] = (bcfg, TransformSpec.from_cfg(bcfg["transform"][dataset]), o,
                     {r: i for i, r in enumerate(o.rel_ids)})
    return out


def _full_map(src, rec) -> np.ndarray:
    """One branch map on the original pixels of `rec` (PatchCore first gets its reference smoothing)."""
    bcfg, spec, o, idx = src
    m = o.maps[idx[rec.rel_id]]
    if bcfg["branch"] == "patchcore":
        from .branches.patchcore import reference_segmentation
        side = spec.crop or spec.size
        m = reference_segmentation(m, (side, side))
    return map_to_original(m, geometry_for(rec.width, rec.height, spec))


def _branch_full_maps(cfg, dataset, category, seed, part, records):
    """{slot: [full-resolution map per record]} for branches that produce maps (holds every map in memory)."""
    return {s: [_full_map(src, r) for r in records] for s, src in _map_sources(cfg, dataset, category, seed, part).items()}


def _pixel_sample(m: np.ndarray) -> np.ndarray:
    return m[np.isfinite(m)].ravel()[::7]


def _normalizer_from_samples(samples: list[np.ndarray], q=0.95) -> dict:
    vals = np.concatenate(samples)
    med, qv = float(np.median(vals)), float(np.quantile(vals, q))
    return {"median": med, "scale": max(qv - med, 1e-12)}


def _pixel_normalizer(maps: list[np.ndarray], q=0.95) -> dict:
    return _normalizer_from_samples([_pixel_sample(m) for m in maps], q)


def prepare_fusion(cfg, dataset, category, seed):
    """Test records, per-branch pixel normalizers (from calib_scale normals), test map sources and fusion rule."""
    recs, _ = manifests_for(cfg, dataset, category, create=False)
    cal = [r for r in recs if r.split in CALIB_ROLES_SCALE]
    test = select(recs, "test")
    calib_src = _map_sources(cfg, dataset, category, seed, "calib")
    test_src = _map_sources(cfg, dataset, category, seed, "test")
    if not test_src:
        raise RuntimeError("no branch with localization maps in this experiment")
    norms = {s: _normalizer_from_samples([_pixel_sample(_full_map(src, r)) for r in cal]) for s, src in calib_src.items()}
    return test, norms, test_src, cfg["experiment"]["fusion"]


def fuse_one(test_src, norms, rule, rec) -> tuple[np.ndarray, float]:
    """Fused full-resolution map of one test image and its unobserved-pixel fraction."""
    st = np.stack([(_full_map(src, rec) - norms[s]["median"]) / norms[s]["scale"] for s, src in test_src.items()])
    f = np.nanmax(st, 0) if rule in ("max", "or") else np.nanmean(st, 0)
    bad = ~np.isfinite(f)
    if bad.any():
        f[bad] = np.nanmin(f) if (~bad).any() else 0.0
    return f.astype(np.float32), float(bad.mean())


def _info(test_src, norms, rule, unobserved) -> dict:
    return {"branches_with_maps": sorted(test_src), "pixel_normalizers": norms, "fusion": rule,
            "mean_unobserved_pixel_fraction": float(np.mean(unobserved)),
            "excluded": "composition (image-level evidence only)"}


def fused_maps(cfg, dataset, category, seed) -> tuple[list, list[np.ndarray], dict]:
    test, norms, test_src, rule = prepare_fusion(cfg, dataset, category, seed)
    fused, unobserved = [], []
    for r in test:
        f, u = fuse_one(test_src, norms, rule, r)
        fused.append(f)
        unobserved.append(u)
    return test, fused, _info(test_src, norms, rule, unobserved)


def loco_maps_dir(cfg, seed) -> Path:
    return work_dir(cfg) / "loco_maps" / cfg["experiment"]["name"] / f"seed{seed}"


def write_loco_maps(cfg, dataset, category, seed, log=None) -> dict:
    """Writes the fused full-resolution LOCO maps as TIFFs, one image at a time; returns the fusion info."""
    import tifffile
    test, norms, test_src, rule = prepare_fusion(cfg, dataset, category, seed)
    base = loco_maps_dir(cfg, seed)
    unobserved = []
    for i, r in enumerate(test):
        f, u = fuse_one(test_src, norms, rule, r)
        unobserved.append(u)
        _, _, defect, name = r.rel_id.split("/")
        p = base / category / "test" / defect / (Path(name).stem + ".tiff")
        p.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(p, f)
        if log and i % 50 == 0:
            log(f"{category}: {i + 1}/{len(test)} maps")
    return {**_info(test_src, norms, rule, unobserved), "n_maps": len(test)}


def run_loco_evaluator(cfg, dataset, category, seed) -> dict:
    """Runs the official MVTec LOCO evaluator v2.0 on the TIFF maps of one category.

    Compatibility shims (process priority and bookkeeping only; the vendored evaluator is unmodified):
    os.nice is POSIX-only; the evaluator expects '/'-separated relative paths, so os.path.join emits '/'
    (harmless on POSIX); and its INITIAL list of curve points (pixel quantiles) is de-duplicated, because it
    asserts uniqueness and fails on maps with flat plateaus. Metric values are computed exactly as before;
    the evaluator's own refinement adds points as needed.
    """
    base = loco_maps_dir(cfg, seed)
    out = base / "metrics" / category
    script = PROJECT_ROOT / "third_party" / "mvtec_loco_ad_evaluation" / "evaluate_experiment.py"
    wrapper = "\n".join([
        "import os, sys, runpy",
        "if not hasattr(os, 'nice'):",
        "    os.nice = lambda inc: 0",
        "_j = os.path.join",
        "os.path.join = lambda *p: _j(*p).replace(chr(92), '/')",
        f"sys.path.insert(0, {str(script.parent)!r})",
        "import src.aggregation as _ag",
        "_orig = _ag.MetricsAggregator._get_initial_thresholds",
        "def _dedup(self, *a, **k):",
        "    return list(dict.fromkeys(_orig(self, *a, **k)))",
        "_ag.MetricsAggregator._get_initial_thresholds = _dedup",
        "sys.argv = sys.argv[1:]",
        "runpy.run_path(sys.argv[0], run_name='__main__')",
    ])
    cmd = [sys.executable, "-c", wrapper, str(script), "--object_name", category, "--dataset_base_dir",
           str(data_root(cfg, dataset)), "--anomaly_maps_dir", str(base), "--output_dir", str(out)]
    # the evaluator's progress bars are not in the Windows code page: read its output as UTF-8, replacing bad bytes
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=script.parent,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    mf = out / "metrics.json"
    if proc.returncode != 0 or not mf.exists():
        return {"au_spro": None, "au_spro_status": "UNAVAILABLE: official script failed",
                "stderr_tail": proc.stderr[-2000:]}
    m = json.loads(mf.read_text())
    spro = m.get("localization", {}).get("auc_spro", {})
    return {"au_spro_by_integration_limit": spro, "au_spro_mean_at_0.05": spro.get("mean", {}).get("0.05"),
            "official_classification_metrics_from_max_of_map": m.get("classification"),
            "au_spro_status": f"official MVTec LOCO evaluation v2.0 (shims: {LOCO_EVALUATOR_SHIMS})"}


def log_localization_access(cfg, dataset, category) -> dict:
    """Records the test-set access of a localization run; returns its pre-registration status."""
    from . import prereg
    status = prereg.check(cfg, dataset, category, exploratory=True)
    prereg.log_test_access({**cfg, "experiment": {**cfg["experiment"],
                                                  "name": cfg["experiment"]["name"] + " (localization)"}},
                           dataset, [category], status)
    return {"evaluation_status": "confirmatory" if status["confirmatory"] else "EXPLORATORY",
            "evaluation_reasons": status["reasons"]}


def au_pro(masks: list[np.ndarray], maps: list[np.ndarray], fpr_limit: float = 0.3) -> float:
    structure = np.ones((3, 3), int)
    region_w, normal_w, scores = [], [], []
    n_regions = 0
    n_normal = sum(int((~m).sum()) for m in masks)
    for m, a in zip(masks, maps):
        lab, n = ndimage.label(m, structure=structure)
        rw = np.zeros(m.shape, np.float64)
        for k in range(1, n + 1):
            sel = lab == k
            rw[sel] = 1.0 / sel.sum()
        n_regions += n
        region_w.append(rw.ravel())
        normal_w.append((~m).ravel().astype(np.float64))
        scores.append(a.ravel())
    if n_regions == 0 or n_normal == 0:
        return float("nan")
    s = np.concatenate(scores)
    order = np.argsort(-s, kind="stable")
    s = s[order]
    pro = np.cumsum(np.concatenate(region_w)[order]) / n_regions
    fpr = np.cumsum(np.concatenate(normal_w)[order]) / n_normal
    last = np.r_[np.diff(s) != 0, True]     # evaluate only at distinct thresholds
    fpr, pro = np.r_[0.0, fpr[last]], np.r_[0.0, pro[last]]
    keep = fpr <= fpr_limit
    x, y = fpr[keep], pro[keep]
    if x[-1] < fpr_limit:
        j = np.searchsorted(fpr, fpr_limit)
        y_lim = np.interp(fpr_limit, fpr[j - 1:j + 1], pro[j - 1:j + 1])
        x, y = np.r_[x, fpr_limit], np.r_[y, y_lim]
    return float(np.trapezoid(y, x) / fpr_limit)


def localize(cfg, dataset, category, seed) -> dict:
    res = {"dataset": dataset, "category": category, **log_localization_access(cfg, dataset, category)}
    if layout_of(dataset) == "ad":
        test, fused, info = fused_maps(cfg, dataset, category, seed)
        res.update(info)
        root = data_root(cfg, dataset)
        masks = []
        for r in test:
            m = load_mask(root, r)
            masks.append(np.zeros((r.height, r.width), bool) if m is None else m)
        y = np.concatenate([m.ravel() for m in masks])
        s = np.concatenate([f.ravel() for f in fused])
        res["pixel_auroc"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
        res["au_pro_0.3"] = au_pro(masks, fused, 0.3)
        res["note"] = "AU-PRO: re-implementation of the official MVTec AD algorithm (not the official script)"
    else:
        res.update(write_loco_maps(cfg, dataset, category, seed))
        res.update(run_loco_evaluator(cfg, dataset, category, seed))
    write_json(run_dir(cfg, dataset, category, seed) / "localization.json", res)
    return res
