"""Pretrained-weight registry, checksum verification and environment capture.

Every pretrained file used by the project is listed here with its SHA-256, origin and
whether it is the author's own release. Loading refuses a file whose checksum differs.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from .config import PROJECT_ROOT
from .utils import sha256_file

WEIGHTS = {
    "wide_resnet50_2": {
        "file": "weights/wide_resnet50_2-95faca4d.pth",
        "sha256": "95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8",
        "source": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth (torchvision IMAGENET1K_V1); "
                  "copied from the local torch hub cache, checksum prefix matches the torchvision file name",
        "used_by": "PatchCore author code via torchvision `wide_resnet50_2(pretrained=True)`",
        "official": "torchvision release (the backbone the PatchCore author code loads)",
    },
    "dinov2_vitb14_reg4": {
        "file": "weights/dinov2_vitb14_reg4_pretrain.pth",
        "sha256": "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
        "source": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
        "used_by": "facebookresearch/dinov2 hub `dinov2_vitb14_reg`; Dinomaly `dinov2reg_vit_base_14`",
        "official": "author (Meta AI) release",
    },
    "pdn_small_teacher": {
        "file": "weights/efficientad/efficientad_pretrained_weights/pretrained_teacher_small.pth",
        "sha256": "a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d",
        "source": "https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/"
                  "efficientad_pretrained_weights.zip (zip sha256 c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a)",
        "used_by": "anomalib EfficientAd (model_size=S)",
        "official": "THIRD-PARTY: hosted in anomalib's GitHub release, not published by the EfficientAD authors; "
                    "the distillation procedure behind this file was not verified in this project",
    },
    "pdn_medium_teacher": {
        "file": "weights/efficientad/efficientad_pretrained_weights/pretrained_teacher_medium.pth",
        "sha256": "f7356663c8e00ada12ae01fb8c8aad0a1de2f800f8eadf252a46d29bbdfdf718",
        "source": "same zip as pdn_small_teacher",
        "used_by": "anomalib EfficientAd (model_size=M)",
        "official": "THIRD-PARTY (anomalib)",
    },
}

AUX_DATA = {
    "imagenette2": {
        "archive": "external_data/imagenette2.tgz",
        "sha256": "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
        "source": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz (anomalib's IMAGENETTE_DOWNLOAD_INFO)",
        "use": "EfficientAD student penalty images (generic-image penalty). The EfficientAD paper uses ImageNet; "
               "anomalib substitutes ImageNette. This project follows anomalib.",
    }
}

REFERENCE_CODE = {
    "patchcore-inspection": {"url": "https://github.com/amazon-science/patchcore-inspection",
                             "commit": "fcaa92f124fb1ad74a7acf56726decd4b27cbcad", "author_code": True},
    "Dinomaly": {"url": "https://github.com/guojiajeremy/Dinomaly",
                 "commit": "1f252be03a918789b19848f0ca37166e7c28dada", "author_code": True},
    "dinov2": {"url": "https://github.com/facebookresearch/dinov2",
               "commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8", "author_code": True},
    "anomalib (efficient_ad)": {"url": "https://github.com/open-edge-platform/anomalib",
                                "commit": "d716bb94f86505eb28a2e473f4ce00f9e98dedf7",
                                "author_code": False, "note": "third-party reimplementation of EfficientAD"},
    "CSAD": {"url": "https://github.com/Tokichan/CSAD", "commit": "d46cf856b99bf056088b5f510fd984e04813583a",
             "author_code": True, "note": "cited reference only; not reproduced in this project"},
    "mvtec_loco_ad_evaluation": {"url": "https://www.mvtec.com/company/research/datasets/mvtec-loco",
                                 "version": "2.0 (README)", "author_code": True,
                                 "note": "copied from a previous local project, not re-downloaded"},
}

_VERIFIED: dict[tuple, str] = {}


class WeightError(RuntimeError):
    pass


def weight_path(name: str) -> Path:
    if name not in WEIGHTS:
        raise WeightError(f"unknown weight {name}")
    return PROJECT_ROOT / WEIGHTS[name]["file"]


def verify_weight(name: str) -> Path:
    p = weight_path(name)
    if not p.exists():
        raise WeightError(f"missing pretrained weights {name}: expected {p}. Source: {WEIGHTS[name]['source']}")
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _VERIFIED:
        _VERIFIED[key] = sha256_file(p)
    if _VERIFIED[key] != WEIGHTS[name]["sha256"]:
        raise WeightError(f"checksum mismatch for {name} at {p}: {_VERIFIED[key]} != {WEIGHTS[name]['sha256']}")
    return p


def environment_report() -> dict:
    import numpy
    import torch
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": numpy.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import faiss  # noqa: F401
        info["faiss"] = faiss.__version__
    except Exception:  # noqa: BLE001
        info["faiss"] = None
    for mod in ("sklearn", "scipy", "PIL", "yaml", "pandas", "matplotlib"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        info["pip_freeze"] = out.stdout.splitlines()
    except Exception as e:  # noqa: BLE001
        info["pip_freeze_error"] = str(e)
    return info
