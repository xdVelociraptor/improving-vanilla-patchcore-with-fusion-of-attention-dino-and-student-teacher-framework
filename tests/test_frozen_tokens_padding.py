"""Frozen teachers, special-token exclusion, padding exclusion, receptive field, exact checkpointing."""
import numpy as np
import pytest
import torch
from PIL import Image

from conftest import requires_dino, requires_pdn, requires_wrn
from hybridad.backbones import DinoEncoder, SmallPDN, receptive_field, PDN_S_LAYERS, load_pdn_teacher, WRNFeatures
from hybridad.data import apply_transform, TransformSpec, grid_valid_fraction, map_to_original, geometry_for
from hybridad.utils import FrozenModule, module_fingerprint


def test_frozen_module_stays_in_eval_and_unchanged():
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.Dropout(0.5))
    fm = FrozenModule(net)
    parent = torch.nn.ModuleDict({"t": fm, "s": torch.nn.Linear(2, 2)})
    parent.train()                                   # surrounding model enters training mode
    assert not fm.module.training and not net[1].training
    before = module_fingerprint(net)
    x = torch.randn(2, 3, 8, 8)
    y1, y2 = fm(x), fm(x)
    assert torch.equal(y1, y2)                       # dropout inactive, BN uses running stats
    assert module_fingerprint(net) == before         # BN running buffers NOT updated
    fm.assert_unchanged()
    assert all(not p.requires_grad for p in net.parameters())


@requires_pdn
def test_pdn_teacher_frozen_during_student_step():
    teacher = FrozenModule(load_pdn_teacher())
    student = SmallPDN(384)
    opt = torch.optim.Adam(student.parameters(), 1e-3)
    x = torch.rand(1, 3, 64, 64)
    loss = ((teacher(x) - student(x)) ** 2).mean()
    loss.backward()
    opt.step()
    teacher.assert_unchanged()


def test_pdn_small_receptive_field_is_33():
    assert receptive_field(PDN_S_LAYERS) == (33, 4)
    # empirical: an impulse affects exactly a 33-wide window of inputs for one output cell
    net = SmallPDN(8)
    x = torch.zeros(1, 3, 80, 80, requires_grad=True)
    net(x)[0, :, 5, 5].sum().backward()
    ys, xs = torch.nonzero(x.grad.abs().sum(1)[0], as_tuple=True)
    assert ys.max() - ys.min() + 1 <= 33 and xs.max() - xs.min() + 1 <= 33
    assert ys.max() - ys.min() + 1 >= 30


@requires_dino
def test_dino_patch_grid_uses_only_patch_tokens():
    enc = DinoEncoder()
    x = torch.randn(1, 3, 112, 112)
    t = enc(x, [11])[11]
    assert t.shape == (1, 5 + 64, 768)
    g = enc.patch_grid(t, (8, 8))
    assert torch.equal(g[0, :, 0, 0], t[0, 5])       # first patch token, not CLS/register
    assert torch.equal(g[0, :, 1, 0], t[0, 5 + 8])   # row-major
    with pytest.raises(ValueError):
        enc.patch_grid(t[:, 1:], (8, 8))             # dropping one special token must be caught
    with pytest.raises(ValueError):
        enc(torch.randn(1, 3, 100, 100), [2])        # not divisible by 14
    enc.train()
    assert not enc.net.module.training


def test_letterbox_valid_mask_and_geometry():
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    x, v, geo = apply_transform(img, TransformSpec("letterbox", 64, None, "imagenet"))
    assert x.shape == (3, 64, 64) and v.shape == (1, 64, 64)
    assert v[0, 0].sum() == 0 and v[0, 32].sum() == 64    # padded rows at top, content in the middle
    assert torch.allclose(x[:, 0, 0], torch.zeros(3), atol=0.02)   # padding = ImageNet mean -> ~0
    frac = grid_valid_fraction(v[None], (4, 4))[0, 0]
    assert frac[0].max() == 0 and frac[1:3].min() == 1
    m = np.arange(16, dtype=np.float32).reshape(4, 4)
    back = map_to_original(m, geo)
    assert back.shape == (100, 200) and np.isfinite(back).all()
    crop = TransformSpec("resize_crop", 256, 224, "imagenet")
    back2 = map_to_original(np.ones((28, 28), np.float32), geometry_for(900, 900, crop))
    assert np.isnan(back2[0, 0]) and np.isfinite(back2[450, 450])   # centre-crop border is unobserved


@requires_dino
def test_dino_loss_ignores_padding_cells():
    from hybridad.branches.dino_recon import global_cosine_hm_percent
    en = [torch.randn(2, 8, 4, 4)]
    de = [torch.randn(2, 8, 4, 4, requires_grad=True)]
    valid = torch.ones(2, 1, 4, 4)
    valid[:, :, 0] = 0
    loss = global_cosine_hm_percent(en, de, p=0.5, factor=0.1, valid=valid)
    loss.backward()
    assert de[0].grad[:, :, 0].abs().max() == 0          # no gradient from padded cells
    en2 = [en[0].clone()]
    en2[0][:, :, 0] = 1e3                                  # arbitrary padding content
    assert torch.allclose(global_cosine_hm_percent(en2, [de[0].detach()], 0.5, 0.1, valid),
                          global_cosine_hm_percent(en, [de[0].detach()], 0.5, 0.1, valid))


def test_activation_checkpointing_is_exact():
    from hybridad.branches.dino_recon import ReconHead, init_trainable
    torch.manual_seed(0)
    a = ReconHead(64, 64, 4, 2, "linear", 0.0, activation_checkpointing=False)
    init_trainable(a)
    b = ReconHead(64, 64, 4, 2, "linear", 0.0, activation_checkpointing=True)
    b.load_state_dict(a.state_dict())
    a.train(), b.train()
    x = torch.randn(2, 10, 64)
    sa = sum(o.pow(2).sum() for o in a(x))
    sb = sum(o.pow(2).sum() for o in b(x))
    sa.backward(), sb.backward()
    for (n, p), q in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6), n


@requires_wrn
def test_wrn_extractor_frozen():
    w = WRNFeatures()
    w.train()
    assert not w.net.module.training
    w(torch.randn(1, 3, 64, 64))
    w.net.assert_unchanged()
