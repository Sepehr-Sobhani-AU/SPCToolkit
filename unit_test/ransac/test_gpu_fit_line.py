"""GPU-only line RANSAC tests. Skipped cleanly when CUDA is unavailable.

Run: ``python unit_test/ransac/test_gpu_fit_line.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.services.ransac import fit


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _generate_line_cloud(n_inliers, n_outliers, noise, seed):
    rng = np.random.default_rng(seed)
    direction = np.array([1.0, 2.0, 3.0])
    direction /= np.linalg.norm(direction)
    origin = np.array([5.0, -2.0, 1.0])
    t = rng.uniform(-10, 10, n_inliers)
    inliers = (
        origin
        + t[:, None] * direction
        + rng.normal(0, noise, (n_inliers, 3))
    )
    outliers = rng.uniform(-15, 15, (n_outliers, 3))
    points = np.concatenate([inliers, outliers], axis=0)
    return points, direction


def test_gpu_line_recovery():
    points, true_dir = _generate_line_cloud(2000, 500, 0.01, seed=0)
    model, mask = fit(points, "line", threshold=0.05, seed=0, backend="gpu")
    assert model is not None, "expected a successful fit"
    cos = abs(float(model.direction @ true_dir))
    assert cos > 0.999, f"direction off; cos={cos}"
    n_in = int(mask.sum())
    assert n_in > 1800, f"expected most of the 2000 inliers; got {n_in}"
    print(f"test_gpu_line_recovery passed (cos={cos:.5f}, inliers={n_in})")


def test_gpu_line_pure_noise_fails():
    rng = np.random.default_rng(1)
    points = rng.uniform(-10, 10, (3000, 3))
    model, mask = fit(
        points,
        "line",
        threshold=0.05,
        min_inlier_ratio=0.5,
        seed=0,
        backend="gpu",
    )
    assert model is None and mask is None
    print("test_gpu_line_pure_noise_fails passed")


def test_gpu_line_reproducibility():
    points, _ = _generate_line_cloud(2000, 500, 0.01, seed=42)
    m1, mask1 = fit(points, "line", threshold=0.05, seed=7, backend="gpu")
    m2, mask2 = fit(points, "line", threshold=0.05, seed=7, backend="gpu")
    assert m1 is not None and m2 is not None
    assert np.allclose(m1.direction, m2.direction, atol=1e-9)
    assert np.array_equal(mask1, mask2)
    print("test_gpu_line_reproducibility passed")


if __name__ == "__main__":
    if not _cuda_available():
        print("CUDA not available; GPU line tests skipped.")
        sys.exit(0)
    test_gpu_line_recovery()
    test_gpu_line_pure_noise_fails()
    test_gpu_line_reproducibility()
    print("OK")
