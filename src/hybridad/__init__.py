"""Hybrid anomaly detection research package.

Four independently observable branch scores (PatchCore, local student-teacher,
DINOv2 global-attention reconstruction, composition evidence), calibrated on
held-out normal images and fused under a stored, versioned calibration.
"""

__version__ = "0.1.0"

BRANCHES = ("patchcore", "local_st", "global_attention", "composition")
SCORE_COLUMNS = tuple(f"score_{b}" for b in BRANCHES)
