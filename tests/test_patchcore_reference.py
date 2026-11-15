"""The protected PatchCore branch must match the vendored author implementation."""
import sys

import numpy as np
import pytest
import torch

from conftest import ROOT, requires_wrn
from hybridad.backbones import WRNFeatures, load_wrn50
from hybridad.branches.patchcore import PatchMaker, embed_reference, ApproximateGreedyCoreset

AUTHOR_SRC = ROOT / "third_party" / "patchcore-inspection" / "src"


def _author():
    # the author backbones module imports timm/pretrainedmodels for backbones we do not use
    import types
    for mod in ("timm", "pretrainedmodels"):
        sys.modules.setdefault(mod, types.ModuleType(mod))
    if str(AUTHOR_SRC) not in sys.path:
        sys.path.insert(0, str(AUTHOR_SRC))
    import patchcore.patchcore as pc  # type: ignore
    import patchcore.sampler as ps  # type: ignore
    import patchcore.common as common  # type: ignore
    return pc, ps, common


@requires_wrn
@pytest.mark.reference
def test_embedding_matches_author_code():
    pc, _, common = _author()
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    ours_bb = WRNFeatures(("layer2", "layer3"))
    net = load_wrn50()  # separate instance: the author code installs forward hooks on it
    author = pc.PatchCore(torch.device("cpu"))
    author.load(backbone=net, layers_to_extract_from=["layer2", "layer3"], device=torch.device("cpu"),
                input_shape=(3, 224, 224), pretrain_embed_dimension=1024, target_embed_dimension=1024,
                patchsize=3, nn_method=common.FaissNN(False, 4))
    a = np.asarray(author._embed(x))
    fm = ours_bb(x)
    ours, hw = embed_reference([fm["layer2"], fm["layer3"]], PatchMaker(3, 1), 1024, 1024)
    assert hw == (28, 28)
    assert a.shape == tuple(ours.shape)
    np.testing.assert_allclose(ours.numpy(), a, rtol=1e-5, atol=1e-6)
    for h in net.hook_handles:
        h.remove()


@pytest.mark.reference
def test_coreset_matches_author_sampler():
    _, ps, _ = _author()
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(600, 64)).astype(np.float32)
    torch.manual_seed(5)
    np.random.seed(5)
    author = ps.ApproximateGreedyCoresetSampler(0.1, torch.device("cpu"))
    a_feats = author.run(feats)
    torch.manual_seed(5)
    np.random.seed(5)
    idx = ApproximateGreedyCoreset(0.1, torch.device("cpu")).run(feats)
    np.testing.assert_array_equal(feats[idx], a_feats)
