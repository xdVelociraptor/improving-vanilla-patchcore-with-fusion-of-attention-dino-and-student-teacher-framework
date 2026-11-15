"""Fit -> score -> calibrate -> predict orchestration for one (experiment, dataset, category, seed).

Branch models and branch score files are shared between experiments: an experiment only
chooses which branch variants to combine and how. That makes the PatchCore scores of every
experiment *literally the same file*, which is how "PatchCore scores remain unchanged" is
enforced (and tested) as an implementation invariant.

Labels are never loaded here. `evaluate.py` joins predictions with manifest labels.
"""
from __future__ import annotations

import copy
import importlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .branches import BranchContext, BranchOutput
from .calibration import calibrate_normal_only, standardize, fuse, verify_binding, CalibrationError
from .config import load_yaml, deep_merge, stable_hash, project_path, CONFIG_DIR
from .manifests import load_manifest, select, CALIB_ROLES_SCALE, CALIB_ROLES_THRESHOLD, build_manifest
from .utils import get_device, seed_everything, write_json, read_json, sha256_file

BRANCH_MODULES = {
    "patchcore": ("hybridad.branches.patchcore", "PatchCoreBranch"),
    "local_st": ("hybridad.branches.local_st", "LocalSTBranch"),
    "global_attention": ("hybridad.branches.dino_recon", "DinoReconBranch"),
    "composition": ("hybridad.branches.composition", "CompositionBranch"),
}
CALIB_ROLES = ("calib_scale", "calib_threshold", "calib_shared")


class PipelineError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- configuration
def load_experiment(path: str | Path, extra: list[str | Path] | None = None) -> dict:
    cfg = load_yaml(path)
    local = CONFIG_DIR / "local.yaml"
    if local.exists():
        cfg = deep_merge(cfg, load_yaml(local))
    for p in extra or []:
        cfg = deep_merge(cfg, load_yaml(p))
    env = {"HYBRIDAD_WORK_DIR": ("paths", "work_dir"),
           "HYBRIDAD_MVTEC_AD": ("paths", "datasets", "mvtec_ad"),
           "HYBRIDAD_MVTEC_LOCO": ("paths", "datasets", "mvtec_loco")}
    for var, keys in env.items():
        if os.environ.get(var):
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = os.environ[var]
    exp = cfg.get("experiment")
    if not exp or "branches" not in exp:
        raise PipelineError(f"{path} is not an experiment config (needs experiment.branches)")
    cfg["experiment"]["branch_cfgs"] = {b: load_yaml(v) if isinstance(v, str) else v
                                        for b, v in exp["branches"].items()}
    for b, ov in (cfg.get("branch_overrides") or {}).items():
        if b in cfg["experiment"]["branch_cfgs"]:
            cfg["experiment"]["branch_cfgs"][b] = deep_merge(cfg["experiment"]["branch_cfgs"][b], ov)
    if cfg["experiment"].get("reproduction"):
        # Published-protocol reproduction: fit on ALL official normal training images. The model identity
        # changes (so it never shares memories/checkpoints with the paired fit-split models), calibration
        # images are inside the fitting data, and therefore no held-out calibration is done.
        for bc in cfg["experiment"]["branch_cfgs"].values():
            bc["_fit_roles"] = "all_official_train"
    for b, bc in cfg["experiment"]["branch_cfgs"].items():
        if bc.get("branch") != b:
            raise PipelineError(f"experiment maps slot {b} to a config for branch {bc.get('branch')}")
    cfg["_source"] = str(path)
    return cfg


def effective_branch_cfg(bcfg: dict, dataset: str, category: str) -> dict:
    out = copy.deepcopy(bcfg)
    ov = (bcfg.get("category_overrides") or {}).get(dataset, {}).get(category)
    if ov:
        out = deep_merge(out, ov)
    out.pop("category_overrides", None)
    if "transform" in out:
        if dataset not in out["transform"]:
            raise PipelineError(f"branch {bcfg.get('variant')} has no preprocessing defined for dataset {dataset}")
        out["transform"] = {dataset: out["transform"][dataset]}
    return out


def branch_enabled(bcfg: dict, cfg: dict, dataset: str, category: str) -> tuple[bool, str]:
    rule = (bcfg.get("disable_for") or {})
    if rule.get("textures") and category in cfg["datasets"].get(dataset, {}).get("textures", []):
        return False, "disabled for texture categories by pre-registered configuration"
    if category in (rule.get("categories") or {}).get(dataset, []):
        return False, "disabled for this category by pre-registered configuration"
    return True, ""


# Keys that only select among already-computed image-score aggregations. They do not change the
# fitted model or the score files, so they are excluded from the model/score identity and are
# bound into each calibration record instead.
SCORING_ONLY_KEYS = ("primary_score",)


def variant_id(bcfg: dict, dataset: str, category: str) -> str:
    eff = effective_branch_cfg(bcfg, dataset, category)
    ident = {k: v for k, v in eff.items() if k not in SCORING_ONLY_KEYS}
    return f"{eff['variant']}__{stable_hash(ident, n=10)}"


def seed_tag(bcfg: dict, seed: int) -> str:
    return f"fixedseed{bcfg['coreset_seed']}" if bcfg["branch"] == "patchcore" else f"seed{seed}"


def work_dir(cfg: dict) -> Path:
    return project_path(cfg["paths"]["work_dir"])


def data_root(cfg: dict, dataset: str) -> Path:
    root = project_path(cfg["paths"]["datasets"][dataset])   # relative paths resolve inside the project folder
    if not root.is_dir():
        raise PipelineError(f"dataset root for {dataset} not found: {root}. Set paths.datasets.{dataset} "
                            f"in configs/local.yaml or HYBRIDAD_{dataset.upper()}.")
    return root


def model_dir(cfg, bcfg, dataset, category, seed) -> Path:
    return work_dir(cfg) / "models" / dataset / category / variant_id(bcfg, dataset, category) / seed_tag(bcfg, seed)


def score_path(cfg, bcfg, dataset, category, seed, part: str) -> Path:
    return (work_dir(cfg) / "scores" / dataset / category / variant_id(bcfg, dataset, category)
            / seed_tag(bcfg, seed) / f"{part}.npz")


def run_dir(cfg, dataset, category, seed) -> Path:
    return project_path(cfg["paths"]["runs"]) / cfg["experiment"]["name"] / dataset / category / f"seed{seed}"


def make_branch(bcfg: dict, cfg: dict, dataset: str, category: str, seed: int, manifest_sha: str):
    mod, cls = BRANCH_MODULES[bcfg["branch"]]
    klass = getattr(importlib.import_module(mod), cls)
    ctx = BranchContext(dataset=dataset, category=category, data_root=data_root(cfg, dataset),
                        work_dir=work_dir(cfg), device=get_device(cfg["runtime"]["device"]), seed=seed,
                        manifest_sha256=manifest_sha, num_workers=int(cfg["runtime"].get("num_workers", 0)))
    eff = effective_branch_cfg(bcfg, dataset, category)
    eff["_paths"] = {"imagenette": str(project_path(cfg["paths"].get("imagenette", "")))}
    return klass(eff, ctx)


def manifests_for(cfg: dict, dataset: str, category: str, create: bool = True):
    mdir = project_path(cfg["paths"]["manifests"])
    if create:
        build_manifest(dataset, data_root(cfg, dataset), category, cfg["protocol"], mdir)
    return load_manifest(mdir, dataset, category)


# ----------------------------------------------------------------------------- fit + score
def fit_and_score(cfg: dict, dataset: str, category: str, seed: int, slots: list[str] | None = None,
                  parts: tuple[str, ...] = ("calib", "test"), log=print) -> dict:
    """Fit every active branch (resumable) and score calibration and/or test images."""
    records, msha = manifests_for(cfg, dataset, category)
    if cfg["experiment"].get("reproduction"):
        fit_recs = [r for r in records if r.source_split == "train" and r.split != "excluded_duplicate"]
        parts = tuple(p for p in parts if p == "test")
    else:
        fit_recs = select(records, "fit")
    part_recs = {"calib": select(records, CALIB_ROLES), "test": select(records, "test")}
    fit_ids = {r.rel_id for r in fit_recs}
    status = {}
    for slot, bcfg in cfg["experiment"]["branch_cfgs"].items():
        if slots and slot not in slots:
            continue
        on, why = branch_enabled(bcfg, cfg, dataset, category)
        if not on:
            status[slot] = {"enabled": False, "reason": why}
            continue
        missing = [p for p in parts if not score_path(cfg, bcfg, dataset, category, seed, p).exists()]
        if not missing:
            status[slot] = {"enabled": True, "cached": True}
            continue
        seed_everything(seed, cfg["runtime"].get("deterministic", True))
        t0 = time.time()
        branch = make_branch(bcfg, cfg, dataset, category, seed, msha)
        mdir = model_dir(cfg, bcfg, dataset, category, seed)
        log(f"[{dataset}/{category}] fit {slot} ({variant_id(bcfg, dataset, category)}) -> {mdir}")
        meta = branch.fit(fit_recs, mdir)
        t_fit = time.time() - t0
        for p in missing:
            recs = part_recs[p]
            leak = fit_ids & {r.rel_id for r in recs}
            if leak:
                raise PipelineError(f"fit images would be scored as {p}: {sorted(leak)[:3]}")
            t1 = time.time()
            out = branch.score(recs)
            if out.rel_ids != [r.rel_id for r in recs]:
                raise PipelineError(f"{slot}: scores are not in manifest order")
            if len(out.status) != len(recs):
                raise PipelineError(f"{slot}: expected one status per image")
            out.extra["model_meta_hash"] = stable_hash(meta)
            out.extra["variant_id"] = variant_id(bcfg, dataset, category)
            out.extra["manifest_sha256"] = msha
            out.extra["score_seconds"] = time.time() - t1
            out.save(score_path(cfg, bcfg, dataset, category, seed, p))
            log(f"[{dataset}/{category}] scored {slot} {p}: {len(recs)} images in {time.time() - t1:.1f}s")
        diag_path = mdir / "padding_diagnostic.json"
        if hasattr(branch, "padding_diagnostic") and "calib" in missing and not diag_path.exists():
            # normal calibration images only: does padding content move scores or light up borders?
            write_json(diag_path, branch.padding_diagnostic(part_recs["calib"]))
        status[slot] = {"enabled": True, "fit_seconds": t_fit}
        del branch
        _free_gpu()
    return status


def _free_gpu():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_scores(cfg, dataset, category, seed, part) -> dict[str, BranchOutput]:
    out = {}
    for slot, bcfg in cfg["experiment"]["branch_cfgs"].items():
        on, _ = branch_enabled(bcfg, cfg, dataset, category)
        if not on:
            continue
        p = score_path(cfg, bcfg, dataset, category, seed, part)
        if not p.exists():
            raise PipelineError(f"missing {part} scores for {slot}: {p}. Run `hybridad prepare` first.")
        out[slot] = BranchOutput.load(p)
    return out


def _primary(bcfg: dict, out: BranchOutput) -> np.ndarray:
    name = bcfg.get("primary_score", out.primary)
    if name not in out.scores:
        raise PipelineError(f"score aggregation {name} not produced by {bcfg['variant']} (has {list(out.scores)})")
    s = np.asarray(out.scores[name], dtype=np.float64).copy()
    failed = np.array([not st.startswith("ok") for st in out.status])
    s[failed] = np.nan
    return s


def _binding(cfg, dataset, category, seed, outs_calib: dict, outs_test: dict | None = None) -> dict:
    b = {
        "experiment": cfg["experiment"]["name"], "dataset": dataset, "category": category, "seed": seed,
        "fusion": cfg["experiment"]["fusion"], "or_alpha": cfg["calibration"]["or_alpha"],
        "branches": {slot: {"variant_id": variant_id(bc, dataset, category),
                            "primary_score": bc.get("primary_score")}
                     for slot, bc in cfg["experiment"]["branch_cfgs"].items()
                     if branch_enabled(bc, cfg, dataset, category)[0]},
        "calib_score_files": {slot: sha256_file(score_path(cfg, cfg["experiment"]["branch_cfgs"][slot], dataset,
                                                           category, seed, "calib")) for slot in outs_calib},
    }
    return b


def calibrate(cfg: dict, dataset: str, category: str, seed: int) -> dict:
    records, msha = manifests_for(cfg, dataset, category, create=False)
    outs = load_scores(cfg, dataset, category, seed, "calib")
    if not outs:
        raise PipelineError("no active branches")
    role = {r.rel_id: r.split for r in records}
    ids = next(iter(outs.values())).rel_ids
    for o in outs.values():
        if o.rel_ids != ids:
            raise PipelineError("branch calibration score files cover different images")
    scale_mask = np.array([role[i] in CALIB_ROLES_SCALE for i in ids])
    second_mask = np.array([role[i] in CALIB_ROLES_THRESHOLD for i in ids])
    shared = bool(any(role[i] == "calib_shared" for i in ids))
    bcfgs = cfg["experiment"]["branch_cfgs"]
    raw = {slot: _primary(bcfgs[slot], o) for slot, o in outs.items()}
    binding = _binding(cfg, dataset, category, seed, outs)
    binding["manifest_sha256"] = msha
    rec = calibrate_normal_only(
        {s: v[scale_mask] for s, v in raw.items()}, {s: v[second_mask] for s, v in raw.items()},
        fusion=cfg["experiment"]["fusion"], or_alpha=float(cfg["calibration"]["or_alpha"]),
        missing_policy=cfg["calibration"]["missing_score_policy"], binding=binding, shared_calibration=shared)
    rec["n_scale"] = int(scale_mask.sum())
    rec["n_second_half"] = int(second_mask.sum())
    rec["disabled_branches"] = {slot: branch_enabled(bc, cfg, dataset, category)[1]
                                for slot, bc in bcfgs.items() if not branch_enabled(bc, cfg, dataset, category)[0]}
    out = run_dir(cfg, dataset, category, seed)
    write_json(out / "calibration.json", rec)
    return rec


def predict(cfg: dict, dataset: str, category: str, seed: int, part: str = "test") -> pd.DataFrame:
    """Apply the stored calibration to scored images. Returns one row per image, no labels."""
    records, msha = manifests_for(cfg, dataset, category, create=False)
    rec = read_json(run_dir(cfg, dataset, category, seed) / "calibration.json")
    outs_c = load_scores(cfg, dataset, category, seed, "calib")
    binding = _binding(cfg, dataset, category, seed, outs_c)
    binding["manifest_sha256"] = msha
    verify_binding(rec, binding)
    outs = load_scores(cfg, dataset, category, seed, part)
    ids = next(iter(outs.values())).rel_ids
    bcfgs = cfg["experiment"]["branch_cfgs"]
    raw = {slot: _primary(bcfgs[slot], o) for slot, o in outs.items()}
    active = rec["active_branches"]
    z, missing = standardize({b: raw[b] for b in active}, rec["normalizers"], rec["missing_policy"])
    fused = fuse(z, rec["fusion"], rec.get("or_offsets"))
    df = pd.DataFrame({"dataset": dataset, "category": category, "rel_id": ids})
    for slot in ("patchcore", "local_st", "global_attention", "composition"):
        if slot in raw:
            df[f"score_{slot}"] = raw[slot]
            df[f"status_{slot}"] = outs[slot].status
            if slot in z:
                df[f"calibrated_{slot}"] = z[slot]
                df[f"missing_{slot}"] = missing[slot]
        else:
            df[f"score_{slot}"] = np.nan
            df[f"status_{slot}"] = "inactive"
    df["fused_score"] = fused
    df["fusion"] = rec["fusion"]
    df["active_branches"] = "+".join(active)
    df["calibration_binding_hash"] = rec["binding_hash"]
    df["experiment"] = cfg["experiment"]["name"]
    df["seed"] = seed
    expected = [r.rel_id for r in select(records, "test" if part == "test" else CALIB_ROLES)]
    if sorted(expected) != sorted(ids) or len(set(ids)) != len(ids):
        raise PipelineError(f"predictions do not cover exactly one row per expected {part} image")
    return df
