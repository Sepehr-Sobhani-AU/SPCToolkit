"""Synthetic-data tests for CylinderModel RANSAC fitting.

Run: ``python unit_test/ransac/test_cylinder_fit.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.services.ransac import fit


def _perp_basis(d):
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, helper); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


def _generate_cylinder_cloud(
    n_inliers,
    n_outliers,
    noise,
    axis_origin,
    axis_dir,
    radius,
    seed,
):
    rng = np.random.default_rng(seed)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    u, v = _perp_basis(axis_dir)
    t = rng.uniform(-5, 5, n_inliers)
    theta = rng.uniform(0, 2 * np.pi, n_inliers)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    inliers = (
        axis_origin[None]
        + t[:, None] * axis_dir[None]
        + radius * cos_t * u[None]
        + radius * sin_t * v[None]
        + rng.normal(0, noise, (n_inliers, 3))
    )
    # Outward normals at each surface point (before adding the noise jitter).
    normals_in = cos_t * u[None] + sin_t * v[None]
    normals_in = normals_in + rng.normal(0, noise, (n_inliers, 3))
    normals_in /= np.linalg.norm(normals_in, axis=1, keepdims=True)

    outlier_pts = rng.uniform(-10, 10, (n_outliers, 3))
    outlier_normals = rng.normal(0, 1, (n_outliers, 3))
    outlier_normals /= np.linalg.norm(outlier_normals, axis=1, keepdims=True)

    points = np.concatenate([inliers, outlier_pts])
    normals = np.concatenate([normals_in, outlier_normals])
    return points, normals, axis_dir, radius


def test_cylinder_recovery():
    axis_origin = np.array([1.0, -2.0, 0.5])
    axis_dir = np.array([1.0, 1.0, 2.0])
    radius = 0.7
    points, normals, true_dir, true_r = _generate_cylinder_cloud(
        800, 200, 0.005, axis_origin, axis_dir, radius, seed=0
    )
    model, mask = fit(
        points,
        "cylinder",
        threshold=0.03,
        normals=normals,
        max_iterations=1500,
        seed=0,
    )
    assert model is not None, "expected a successful fit"
    cos = abs(float(model.direction @ (true_dir / np.linalg.norm(true_dir))))
    assert cos > 0.999, f"axis direction off; cos={cos}"
    rel_r_err = abs(model.radius - true_r) / true_r
    assert rel_r_err < 0.01, f"radius off; relative error={rel_r_err}"
    n_in = int(mask.sum())
    assert n_in > 700, f"expected most inliers; got {n_in}"
    print(
        f"test_cylinder_recovery passed "
        f"(cos={cos:.5f}, r={model.radius:.4f} vs {true_r:.4f}, inliers={n_in})"
    )


def test_cylinder_requires_normals():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (100, 3))
    try:
        fit(points, "cylinder", threshold=0.05, seed=0)
    except ValueError as e:
        assert "normals" in str(e).lower()
        print("test_cylinder_requires_normals passed")
        return
    raise AssertionError("expected ValueError for missing normals")


def test_cylinder_pure_noise_fails():
    rng = np.random.default_rng(1)
    points = rng.uniform(-5, 5, (800, 3))
    normals = rng.normal(0, 1, (800, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    model, mask = fit(
        points,
        "cylinder",
        threshold=0.03,
        normals=normals,
        min_inlier_ratio=0.5,
        max_iterations=500,
        seed=0,
    )
    assert model is None and mask is None
    print("test_cylinder_pure_noise_fails passed")


def test_cylinder_reproducibility():
    points, normals, _, _ = _generate_cylinder_cloud(
        800, 200, 0.005, np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.5, seed=42
    )
    m1, mask1 = fit(
        points, "cylinder", threshold=0.03, normals=normals, seed=7
    )
    m2, mask2 = fit(
        points, "cylinder", threshold=0.03, normals=normals, seed=7
    )
    assert m1 is not None and m2 is not None
    assert np.allclose(m1.direction, m2.direction)
    assert abs(m1.radius - m2.radius) < 1e-9
    assert np.array_equal(mask1, mask2)
    print("test_cylinder_reproducibility passed")


if __name__ == "__main__":
    test_cylinder_recovery()
    test_cylinder_requires_normals()
    test_cylinder_pure_noise_fails()
    test_cylinder_reproducibility()
    print("OK")
