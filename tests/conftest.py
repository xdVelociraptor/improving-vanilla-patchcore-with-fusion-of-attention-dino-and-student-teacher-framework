import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from hybridad.provenance import WEIGHTS  # noqa: E402


def weights_available(*names):
    return all((ROOT / WEIGHTS[n]["file"]).exists() for n in names)


requires_wrn = pytest.mark.skipif(not weights_available("wide_resnet50_2"), reason="WRN50 weights missing")
requires_dino = pytest.mark.skipif(not weights_available("dinov2_vitb14_reg4"), reason="DINOv2 weights missing")
requires_pdn = pytest.mark.skipif(not weights_available("pdn_small_teacher"), reason="PDN weights missing")
