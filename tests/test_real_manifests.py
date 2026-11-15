"""Leakage checks on the real manifests (skipped if they have not been built)."""
import pytest

from conftest import ROOT
from hybridad.manifests import load_manifest, select, MVTEC_AD_CATEGORIES, MVTEC_LOCO_CATEGORIES

CASES = [("mvtec_ad", c) for c in MVTEC_AD_CATEGORIES] + [("mvtec_loco", c) for c in MVTEC_LOCO_CATEGORIES]


@pytest.mark.data
@pytest.mark.parametrize("dataset,category", CASES)
def test_real_manifest_disjoint(dataset, category):
    mdir = ROOT / "manifests"
    if not (mdir / dataset / f"{category}.csv").exists():
        pytest.skip("manifest not built")
    recs, _ = load_manifest(mdir, dataset, category)          # re-verifies hash + check_leakage
    fit = select(recs, "fit")
    cal = select(recs, ("calib_scale", "calib_threshold", "calib_shared"))
    test = select(recs, "test")
    ids = [r.rel_id for r in fit + cal + test]
    assert len(ids) == len(set(ids))
    content = {}
    for r in fit + cal + test:
        role = "fit" if r in fit else "cal" if r in cal else "test"
        assert content.setdefault(r.pixel_sha256, role) == role
    assert all(r.label == 0 for r in fit + cal)
    assert all(r.source_split == "test" for r in test)
    if dataset == "mvtec_loco":
        assert all(r.source_split == "validation" for r in cal)
        assert all(r.source_split == "train" for r in fit)
