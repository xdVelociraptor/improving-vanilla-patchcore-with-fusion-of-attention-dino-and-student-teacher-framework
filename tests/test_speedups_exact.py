"""The two speed-ups must be exact: penalty preloading and cached encoder outputs."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from conftest import requires_dino
from hybridad.branches.local_st import _load_penalty, _penalty_from_bank, penalty_bank


def test_penalty_bank_identical_to_reference_loading(tmp_path):
    rng = np.random.default_rng(0)
    files = []
    for i in range(3):
        f = tmp_path / f"{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (97 + i * 13, 130, 3), dtype=np.uint8)).save(f, quality=90)
        files.append(f)
    bank = penalty_bank(files, 32, tmp_path / "work")
    for i, f in enumerate(files):
        for gray in (False, True):
            assert torch.equal(_penalty_from_bank(bank, i, gray), _load_penalty(f, 32, gray))


@requires_dino
def test_cached_encoder_targets_give_identical_loss_and_grads():
    from hybridad.branches import BranchContext
    from hybridad.branches.dino_recon import DinoReconBranch, init_trainable, global_cosine_hm_percent
    from hybridad.config import load_yaml
    cfg = load_yaml("branches/dino_recon_compact.yaml")
    cfg["transform"]["mvtec_ad"] = {"kind": "resize_square", "size": 112, "normalize": "imagenet"}
    ctx = BranchContext("mvtec_ad", "x", Path("."), Path("."), torch.device("cpu"), 0, "")
    b = DinoReconBranch(cfg, ctx)
    torch.manual_seed(0)
    b.head = b._head()
    init_trainable(b.head)
    b.head.eval()   # dropout off so both paths are deterministic
    x = torch.randn(2, 3, 112, 112)
    en1, de1 = b.forward_groups(x)
    l1 = global_cosine_hm_percent(en1, de1, 0.5, 0.1)
    l1.backward()
    g1 = [p.grad.clone() for p in b.head.parameters()]
    b.head.zero_grad()
    fused, en2 = b.encoder_targets(x)
    fused = torch.from_numpy(fused.numpy().astype(np.float32))            # round-trip through the float32 cache
    en2 = [torch.from_numpy(e.numpy().astype(np.float32)) for e in en2]
    l2 = global_cosine_hm_percent(en2, b.decode(fused), 0.5, 0.1)
    l2.backward()
    assert torch.equal(l1, l2)
    for a, c in zip(g1, (p.grad for p in b.head.parameters())):
        assert torch.equal(a, c)
