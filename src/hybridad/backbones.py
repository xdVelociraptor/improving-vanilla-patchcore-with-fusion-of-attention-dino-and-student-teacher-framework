"""Frozen pretrained backbones: WideResNet-50-2, DINOv2 ViT-B/14 with registers, EfficientAD PDN.

Layer indexing for DINOv2 is ZERO-BASED over `encoder.blocks`: layer i is the output of
`blocks[i]`, so Dinomaly's target layers [2..9] are the outputs of the 3rd..10th blocks.
Token layout of every block output: [CLS, REG_1..REG_R, PATCH_1..PATCH_N] with N = (H/14)*(W/14),
patches in row-major order. Only PATCH tokens are ever reshaped into a spatial grid.
"""
from __future__ import annotations

import sys
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from .config import PROJECT_ROOT
from .provenance import verify_weight
from .utils import FrozenModule

DINOV2_CODE = PROJECT_ROOT / "third_party" / "dinov2"

# Position-embedding interpolation settings. `dinomaly` reproduces Dinomaly's vit_encoder.load
# (antialias=False, offset=0.1); `official` reproduces facebookresearch/dinov2 hub `*_reg` models
# (antialias=True, offset=0.0).
POS_INTERP = {
    "dinomaly": {"interpolate_antialias": False, "interpolate_offset": 0.1},
    "official": {"interpolate_antialias": True, "interpolate_offset": 0.0},
}


# ----------------------------------------------------------------------------- WideResNet-50-2
def load_wrn50() -> nn.Module:
    import torchvision
    path = verify_weight("wide_resnet50_2")
    model = torchvision.models.wide_resnet50_2(weights=None)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class WRNFeatures(nn.Module):
    """Runs WRN-50-2 up to the last requested stage and returns stage outputs.

    Equivalent to the PatchCore author's forward hooks on `layer2[-1]` / `layer3[-1]`
    (the output of the last Bottleneck of each stage), stopping after the last requested stage.
    """

    STAGES = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, layers=("layer2", "layer3")):
        super().__init__()
        for l in layers:
            if l not in self.STAGES:
                raise ValueError(f"unsupported WRN layer {l}")
        self.layers = tuple(layers)
        self.net = FrozenModule(load_wrn50())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        m = self.net.module
        out = {}
        with torch.no_grad():
            h = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            last = max(self.STAGES.index(l) for l in self.layers)
            for i, name in enumerate(self.STAGES[: last + 1]):
                h = getattr(m, name)(h)
                if name in self.layers:
                    out[name] = h
        return out


# ----------------------------------------------------------------------------- DINOv2
def _import_dinov2():
    if str(DINOV2_CODE) not in sys.path:
        sys.path.insert(0, str(DINOV2_CODE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dinov2.models import vision_transformer as vits  # type: ignore
    return vits


def load_dinov2_vitb14_reg4(pos_interp: str = "dinomaly") -> nn.Module:
    if pos_interp not in POS_INTERP:
        raise ValueError(f"pos_interp must be one of {list(POS_INTERP)}")
    vits = _import_dinov2()
    model = vits.vit_base(patch_size=14, img_size=518, init_values=1.0, block_chunks=0,
                          num_register_tokens=4, **POS_INTERP[pos_interp])
    state = torch.load(verify_weight("dinov2_vitb14_reg4"), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True), None
    del missing, unexpected
    model.eval()
    return model


class DinoEncoder(nn.Module):
    """Frozen DINOv2 ViT-B/14-reg4 exposing raw block outputs (all tokens) by zero-based index."""

    patch_size = 14
    embed_dim = 768

    def __init__(self, pos_interp: str = "dinomaly"):
        super().__init__()
        self.pos_interp = pos_interp
        self.net = FrozenModule(load_dinov2_vitb14_reg4(pos_interp))
        self.num_register_tokens = self.net.module.num_register_tokens
        self.num_special = 1 + self.num_register_tokens
        assert self.num_register_tokens == 4

    def forward(self, x: torch.Tensor, layers: list[int], final_norm_for: tuple[int, ...] = ()) -> dict[int, torch.Tensor]:
        """Returns {layer_index: (B, 1+R+N, C)} for each requested zero-based block index."""
        m = self.net.module
        B, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"input {H}x{W} is not divisible by patch size {self.patch_size}")
        n_expected = (H // self.patch_size) * (W // self.patch_size)
        want = sorted(set(layers))
        if want[0] < 0 or want[-1] >= len(m.blocks):
            raise ValueError(f"layer indices must be in [0, {len(m.blocks) - 1}] (zero-based)")
        out = {}
        with torch.no_grad():
            t = m.prepare_tokens_with_masks(x)
            assert t.shape == (B, self.num_special + n_expected, self.embed_dim), t.shape
            for i, blk in enumerate(m.blocks):
                if i > want[-1]:
                    break
                t = blk(t)
                if i in want:
                    out[i] = m.norm(t) if i in final_norm_for else t
        return out

    def patch_grid(self, tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """(B, 1+R+N, C) -> (B, C, h, w) using ONLY the patch tokens."""
        B, T, C = tokens.shape
        h, w = hw
        if T != self.num_special + h * w:
            raise ValueError(f"token count {T} != {self.num_special} special + {h}x{w} patches")
        patches = tokens[:, self.num_special:, :]
        return patches.transpose(1, 2).reshape(B, C, h, w)


# ----------------------------------------------------------------------------- EfficientAD PDN
def imagenet_norm_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    return (x - mean) / std


class SmallPDN(nn.Module):
    """EfficientAD-S patch description network (architecture copied from anomalib torch_model.py).

    Without padding: conv4x4 -> avgpool2 -> conv4x4 -> avgpool2 -> conv3x3 -> conv4x4.
    Receptive field = 33 x 33 input pixels, stride 4 (verified by `receptive_field()` in tests).
    Input is in [0, 1]; ImageNet normalization happens inside, as in the reference.
    """

    def __init__(self, out_channels: int = 384, padding: bool = False):
        super().__init__()
        pad = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


def receptive_field(layers: list[tuple[int, int]]) -> tuple[int, int]:
    """(kernel, stride) sequence -> (receptive field, total stride)."""
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump
        jump *= s
    return rf, jump


PDN_S_LAYERS = [(4, 1), (2, 2), (4, 1), (2, 2), (3, 1), (4, 1)]


def load_pdn_teacher(size: str = "small") -> nn.Module:
    if size != "small":
        raise NotImplementedError("only the EfficientAD-S PDN is wired in this project")
    net = SmallPDN(out_channels=384)
    state = torch.load(verify_weight("pdn_small_teacher"), map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net
