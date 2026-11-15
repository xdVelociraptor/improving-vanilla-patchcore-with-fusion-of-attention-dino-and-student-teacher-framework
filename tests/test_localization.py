"""AU-PRO re-implementation vs a brute-force threshold sweep."""
import numpy as np
from scipy import ndimage

from hybridad.localization import au_pro


def _brute(masks, maps, limit=0.3):
    s_all = np.unique(np.concatenate([m.ravel() for m in maps]))[::-1]
    regions = []
    for m, a in zip(masks, maps):
        lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
        regions += [(a, lab == k) for k in range(1, n + 1)]
    normal = np.concatenate([(~m).ravel() for m in masks])
    scores = np.concatenate([a.ravel() for a in maps])
    fprs, pros = [0.0], [0.0]
    for t in s_all:
        fprs.append(((scores >= t) & normal).sum() / normal.sum())
        pros.append(np.mean([(a[r] >= t).mean() for a, r in regions]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= limit
    x, y = fprs[keep], pros[keep]
    if x[-1] < limit:
        j = np.searchsorted(fprs, limit)
        x, y = np.r_[x, limit], np.r_[y, np.interp(limit, fprs[j - 1:j + 1], pros[j - 1:j + 1])]
    return np.trapezoid(y, x) / limit


def test_au_pro_matches_bruteforce():
    rng = np.random.default_rng(0)
    masks, maps = [], []
    for i in range(4):
        m = np.zeros((20, 20), bool)
        if i:
            m[2 + i:6 + i, 3:8] = True
            m[12:15, 12:18] = i % 2 == 0
        masks.append(m)
        maps.append((rng.random((20, 20)) + 0.8 * m).round(2))   # rounding creates ties
    assert abs(au_pro(masks, maps) - _brute(masks, maps)) < 1e-9


def test_au_pro_perfect_map():
    m = np.zeros((10, 10), bool); m[2:5, 2:5] = True
    assert abs(au_pro([m], [m.astype(float)]) - 1.0) < 1e-9
