"""
Tests for the two-cloud (cloud-to-cloud) query path added to the KNN backends.

`KNNBackend.query` gained an optional `reference` argument: when given, the
index is built on `reference` and `points` are the query set (indices returned
index into `reference`). When omitted it stays a self-query, exactly as before.
Only the CPU ScipyKNN backend is exercised here so the test runs without a GPU.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

from plugins.backends.knn_backends import ScipyKNN


def test_two_cloud_query_matches_ckdtree():
    """Indices from a two-cloud query must match a direct cKDTree ground truth."""
    rng = np.random.default_rng(0)
    ref = rng.random((500, 3)).astype(np.float32)
    target = rng.random((200, 3)).astype(np.float32)

    dist, idx = ScipyKNN().query(target, k=1, reference=ref)
    assert idx.shape == (200, 1), idx.shape
    assert dist.shape == (200, 1), dist.shape

    _, idx_gt = cKDTree(ref).query(target, k=1)
    assert np.array_equal(idx[:, 0], idx_gt.astype(np.int64))
    print("✓ two-cloud query (build on B, query A) matches cKDTree ground truth")


def test_self_query_unchanged():
    """reference=None must keep the original self-query behaviour."""
    rng = np.random.default_rng(1)
    pts = rng.random((300, 3)).astype(np.float32)

    dist, idx = ScipyKNN().query(pts, k=3)  # reference defaults to None
    assert idx.shape == (300, 3)
    assert np.array_equal(idx[:, 0], np.arange(300))   # nearest neighbour is self
    assert np.allclose(dist[:, 0], 0.0)
    print("✓ self-query (reference=None) unchanged")


def test_k1_returns_2d():
    """k=1 must still return (N, 1) arrays (scipy returns 1-D for k=1)."""
    rng = np.random.default_rng(2)
    pts = rng.random((50, 3)).astype(np.float32)
    dist, idx = ScipyKNN().query(pts, k=1)
    assert idx.shape == (50, 1) and dist.shape == (50, 1)
    print("✓ k=1 keeps the (N, k) contract")


if __name__ == "__main__":
    test_two_cloud_query_matches_ckdtree()
    test_self_query_unchanged()
    test_k1_returns_2d()
    print("\nAll KNN two-cloud backend tests passed.")
