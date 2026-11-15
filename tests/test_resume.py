"""Interrupted training must resume from its checkpoint and finish (regression test for the RNG-state bug)."""
from pathlib import Path

import pytest
import torch

from conftest import ROOT, requires_dino
from hybridad.synthetic import make_synthetic_ad


@requires_dino
@pytest.mark.slow
def test_dino_training_resumes_after_interruption(tmp_path, monkeypatch):
    from hybridad.branches import BranchContext
    from hybridad.branches import dino_recon
    from hybridad.config import load_yaml
    from hybridad.manifests import build_manifest, load_manifest, select
    root = make_synthetic_ad(tmp_path / "synthetic_ad")
    proto = {"ad_calibration_fraction": 0.3, "ad_calibration_min": 5, "separate_scale_threshold_min": 10, "split_salt": "t"}
    build_manifest("synthetic_ad", root, "blobs", proto, tmp_path / "m")
    recs, sha = load_manifest(tmp_path / "m", "synthetic_ad", "blobs")
    cfg = load_yaml("branches/dino_recon_compact.yaml")
    cfg.update({"iterations": 6, "batch_size": 4, "micro_batch": 4, "warmup_iters": 1, "hm_warmup_iters": 2,
                "checkpoint_every": 2, "encoder_cache": True})
    cfg["transform"]["synthetic_ad"] = {"kind": "resize_square", "size": 112, "normalize": "imagenet"}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = BranchContext("synthetic_ad", "blobs", root, tmp_path / "work", dev, 0, sha)
    out = tmp_path / "model"
    calls = {"n": 0}
    real_step = dino_recon.StableAdamW.step

    def failing_step(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt("simulated interruption")
        return real_step(self, *a, **k)

    monkeypatch.setattr(dino_recon.StableAdamW, "step", failing_step)
    with pytest.raises(KeyboardInterrupt):
        dino_recon.DinoReconBranch(cfg, ctx).fit(select(recs, "fit"), out)
    assert (out / "train_state.pt").exists()
    monkeypatch.setattr(dino_recon.StableAdamW, "step", real_step)
    meta = dino_recon.DinoReconBranch(cfg, ctx).fit(select(recs, "fit"), out)
    assert meta["resumed_from"] == 4 and (out / "head_final.pt").exists()
