"""Synthetic-data tests for PlaneModel RANSAC fitting.

Run directly: ``python unit_test/ransac/test_plane_fit.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.services.ransac import fit


def _generate_plane_cloud(
    n_inliers: int,
    n_outliers: int,
    noise: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    normal = np.array([0.3, -0.5, 0.8])
    normal /= np.linalg.norm(normal)
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(normal[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    point0 = np.array([1.0, 2.0, -1.0])

    a = rng.uniform(-5, 5, n_inliers)
    b = rng.uniform(-5, 5, n_inliers)
    inliers = (
        point0
        + a[:, None] * u
        + b[:, None] * v
        + rng.normal(0, noise, (n_inliers, 3))
    )
    outliers = rng.uniform(-10, 10, (n_outliers, 3))
    points = np.concatenate([inliers, outliers], axis=0)
    return points, normal


def test_plane_recovery():
    points, true_n = _generate_plane_cloud(500, 200, 0.01, seed=0)
    model, mask = fit(points, "plane", threshold=0.05, seed=0)
    assert model is not None, "expected a successful fit"
    cos = abs(float(model.normal @ true_n))
    assert cos > 0.999, f"normal off; cos={cos}"
    n_in = int(mask.sum())
    assert n_in > 450, f"expected most of the 500 inliers; got {n_in}"
    print(f"test_plane_recovery passed (cos={cos:.5f}, inliers={n_in})")


def test_plane_collinear_input_fails():
    # 3 collinear points: every minimal sample is degenerate, so no fit.
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    model, mask = fit(points, "plane", threshold=0.05, seed=0)
    assert model is None and mask is None
    print("test_plane_collinear_input_fails passed")


def test_plane_pure_noise_fails():
    rng = np.random.default_rng(1)
    points = rng.uniform(-10, 10, (500, 3))
    model, mask = fit(
        points, "plane", threshold=0.05, min_inlier_ratio=0.5, seed=0
    )
    assert model is None and mask is None, "expected no fit on pure noise"
    print("test_plane_pure_noise_fails passed")


def test_plane_reproducibility():
    points, _ = _generate_plane_cloud(500, 200, 0.01, seed=42)
    m1, mask1 = fit(points, "plane", threshold=0.05, seed=7)
    m2, mask2 = fit(points, "plane", threshold=0.05, seed=7)
    assert m1 is not None and m2 is not None
    assert np.allclose(m1.normal, m2.normal)
    assert np.array_equal(mask1, mask2)
    print("test_plane_reproducibility passed")


if __name__ == "__main__":
    test_plane_recovery()
    test_plane_collinear_input_fails()
    test_plane_pure_noise_fails()
    test_plane_reproducibility()
    print("OK")
