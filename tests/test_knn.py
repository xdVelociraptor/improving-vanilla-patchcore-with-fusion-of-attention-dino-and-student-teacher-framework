"""Nearest-neighbour backends agree with a float64 brute-force reference."""
import numpy as np
import pytest
import torch

from hybridad.knn import KNNIndex, brute_force_sq, HAVE_FAISS


@pytest.mark.parametrize("backend", ["torch"] + (["faiss"] if HAVE_FAISS else []))
def test_backend_matches_brute_force(backend):
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(1500, 48)).astype(np.float32)
    q = rng.normal(size=(300, 48)).astype(np.float32)
    idx = KNNIndex(backend, query_chunk=64, bank_chunk=257).fit(bank)
    d, i = idx.search(q, 3)
    rd, ri = brute_force_sq(q, bank, 3)
    np.testing.assert_allclose(d, rd, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(i, ri)


def test_distances_are_squared_euclidean():
    bank = np.array([[0.0, 0.0], [3.0, 4.0]], np.float32)
    idx = KNNIndex("torch").fit(bank)
    d, i = idx.search(np.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]], np.float32), 1)
    np.testing.assert_allclose(d[:, 0], [0.0, 0.0, 25.0])  # squared, not 5.0


def test_near_duplicate_stability():
    """||a||^2+||b||^2-2ab cancels badly for large-norm near-duplicates; fp64 refinement fixes it."""
    rng = np.random.default_rng(1)
    base = (rng.normal(size=(200, 1024)) * 30).astype(np.float32)
    q = base[:50] + np.float32(1e-3)
    d, i = KNNIndex("torch").fit(base).search(q, 1)
    rd, _ = brute_force_sq(q, base, 1)
    assert (d >= 0).all()
    np.testing.assert_array_equal(i[:, 0], np.arange(50))
    np.testing.assert_allclose(d, rd, rtol=1e-3, atol=1e-6)


@pytest.mark.skipif(not HAVE_FAISS, reason="faiss not installed")
def test_faiss_and_torch_agree_on_real_scale_features():
    rng = np.random.default_rng(2)
    bank = np.abs(rng.normal(size=(4000, 1024))).astype(np.float32)
    q = np.abs(rng.normal(size=(512, 1024))).astype(np.float32)
    df, i_f = KNNIndex("faiss").fit(bank).search(q, 1)
    dt, i_t = KNNIndex("torch", device="cuda" if torch.cuda.is_available() else "cpu").fit(bank).search(q, 1)
    np.testing.assert_allclose(df, dt, rtol=1e-4)
    assert (i_f == i_t).mean() > 0.999


def test_rejects_non_finite():
    idx = KNNIndex("torch").fit(np.zeros((3, 2), np.float32))
    with pytest.raises(FloatingPointError):
        idx.search(np.array([[np.nan, 0.0]], np.float32))
