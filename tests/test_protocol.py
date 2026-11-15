"""Manifests, leakage, cache compatibility, one prediction per image, invariance and reload (synthetic)."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, requires_dino, requires_pdn, requires_wrn
from hybridad.cache import FeatureCache, IncompatibleCacheError
from hybridad.manifests import (build_manifest, load_manifest, check_leakage, ManifestError, Record, select,
                                manifest_path)
from hybridad.synthetic import make_synthetic_ad, make_synthetic_loco

PROTOCOL = {"ad_calibration_fraction": 0.3, "ad_calibration_min": 5, "separate_scale_threshold_min": 10,
            "split_salt": "test"}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    make_synthetic_ad(root / "synthetic_ad")
    make_synthetic_loco(root / "synthetic_loco")
    return root


def test_manifest_roles_disjoint_and_immutable(synth, tmp_path):
    meta = build_manifest("synthetic_ad", synth / "synthetic_ad", "blobs", PROTOCOL, tmp_path)
    recs, sha = load_manifest(tmp_path, "synthetic_ad", "blobs")
    assert sha == meta["manifest_sha256"]
    fit = {r.rel_id for r in select(recs, "fit")}
    cal = {r.rel_id for r in select(recs, ("calib_scale", "calib_threshold", "calib_shared"))}
    test = {r.rel_id for r in select(recs, "test")}
    assert not (fit & cal) and not (fit & test) and not (cal & test)
    assert all(r.label == 0 for r in select(recs, ("fit", "calib_scale", "calib_threshold")))
    # rebuilding gives the same bytes; tampering is detected
    build_manifest("synthetic_ad", synth / "synthetic_ad", "blobs", PROTOCOL, tmp_path)
    p = manifest_path(tmp_path, "synthetic_ad", "blobs")
    p.write_bytes(p.read_bytes().replace(b",fit,", b",test,", 1))
    with pytest.raises(ManifestError):
        load_manifest(tmp_path, "synthetic_ad", "blobs")


def test_duplicate_content_removed_from_fit(synth, tmp_path):
    src = synth / "synthetic_ad"
    dup = tmp_path / "dup"
    shutil.copytree(src, dup)
    shutil.copy(dup / "blobs/test/good/000.png", dup / "blobs/train/good/999.png")  # a test image leaks into train
    build_manifest("synthetic_ad", dup, "blobs", PROTOCOL, tmp_path / "m")
    recs, _ = load_manifest(tmp_path / "m", "synthetic_ad", "blobs")
    r = [x for x in recs if x.rel_id.endswith("train/good/999.png")][0]
    assert r.split == "excluded_duplicate"


def test_leakage_check_catches_shared_content():
    base = dict(dataset="d", category="c", source_split="train", label=0, anomaly_type="good", subgroup="good",
                mask_rel="", width=1, height=1, pixel_sha256="p")
    recs = [Record(rel_id="a", split="fit", file_sha256="x", **base),
            Record(rel_id="b", split="test", file_sha256="x", **base)]
    with pytest.raises(ManifestError):
        check_leakage(recs)


def test_cache_refuses_incompatible(tmp_path):
    c = FeatureCache(tmp_path, "ns")
    k = FeatureCache.make_key(image_sha256="a", image_id="i", transform={"s": 1}, augmentation=None, backbone="b",
                              weights_sha256="w", layers=["l"], params={}, precision="fp32")
    c.put(k, np.ones(3))
    assert np.array_equal(c.get(k), np.ones(3))
    k2 = dict(k, precision="fp16")
    assert c.get(k2) is None                         # different key -> miss, never a wrong hit
    arr_p, meta_p = c._paths(k)
    meta_p.write_text(meta_p.read_text().replace("fp32", "bf16"))
    with pytest.raises(IncompatibleCacheError):
        c.get(k)


@requires_wrn
@requires_pdn
@requires_dino
@pytest.mark.slow
def test_full_pipeline_invariance_reload_and_one_prediction(synth, tmp_path):
    """Synthetic end-to-end: fit -> score -> calibrate -> predict for A and G, then checks."""
    from hybridad.pipeline import load_experiment, fit_and_score, calibrate, predict, score_path, make_branch, \
        model_dir, manifests_for
    from hybridad.branches import BranchOutput
    from hybridad.evaluate import join_labels
    extra = [ROOT / "configs/smoke/override.yaml"]
    cfgs = {}
    for name in ("A_patchcore", "G_full"):
        cfg = load_experiment(ROOT / f"configs/experiments/{name}.yaml", extra)
        cfg["paths"].update({"work_dir": str(tmp_path / "work"), "manifests": str(tmp_path / "man"),
                             "runs": str(tmp_path / "runs")})
        cfg["paths"]["datasets"]["synthetic_loco"] = str(synth / "synthetic_loco")
        cfgs[name] = cfg
    ds, cat = "synthetic_loco", "pairs"
    for cfg in cfgs.values():
        fit_and_score(cfg, ds, cat, 0, log=lambda *a: None)
        calibrate(cfg, ds, cat, 0)
    pa, pg = predict(cfgs["A_patchcore"], ds, cat, 0), predict(cfgs["G_full"], ds, cat, 0)
    # (1) PatchCore scores are identical inside the full model (implementation invariant)
    assert np.array_equal(pa["score_patchcore"].to_numpy(), pg["score_patchcore"].to_numpy())
    # (2) exactly one prediction per expected test image
    recs, _ = manifests_for(cfgs["G_full"], ds, cat, create=False)
    assert len(join_labels(pg, recs)) == len(select(recs, "test"))
    with pytest.raises(ValueError):
        join_labels(pg.iloc[1:], recs)
    # (3) reload equivalence: fresh branch objects loaded from disk reproduce the saved scores
    test_recs = select(recs, "test")
    for slot, bcfg in cfgs["G_full"]["experiment"]["branch_cfgs"].items():
        br = make_branch(bcfg, cfgs["G_full"], ds, cat, 0, "")
        br.load(model_dir(cfgs["G_full"], bcfg, ds, cat, 0))
        fresh = br.score(test_recs)
        saved = BranchOutput.load(score_path(cfgs["G_full"], bcfg, ds, cat, 0, "test"))
        for k in saved.scores:
            np.testing.assert_allclose(fresh.scores[k], saved.scores[k], rtol=1e-5, atol=1e-6, err_msg=f"{slot}:{k}")
    # (4) the pipeline refuses to score fitting images
    from hybridad.pipeline import PipelineError
    fit_recs = select(recs, "fit")
    assert not ({r.rel_id for r in fit_recs} & set(pg["rel_id"]))
