"""Synthetic-data tests for ConeModel RANSAC fitting.

Run: ``python unit_test/ransac/test_cone_fit.py``
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


def _generate_cone_cloud(
    n_inliers,
    n_outliers,
    noise,
    apex,
    axis_dir,
    half_angle,
    seed,
):
    rng = np.random.default_rng(seed)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    u, v = _perp_basis(axis_dir)
    s = rng.uniform(0.5, 5.0, n_inliers)              # along-axis position
    theta = rng.uniform(0, 2 * np.pi, n_inliers)      # around-axis angle
    radial = s * np.tan(half_angle)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    radial_dir = cos_t * u[None] + sin_t * v[None]    # (N, 3)
    inliers = (
        apex[None]
        + s[:, None] * axis_dir[None]
        + radial[:, None] * radial_dir
        + rng.normal(0, noise, (n_inliers, 3))
    )
    # Outward normal = cos(α)·radial_dir − sin(α)·axis_dir.
    normals_in = np.cos(half_angle) * radial_dir - np.sin(half_angle) * axis_dir[None]
    normals_in = normals_in + rng.normal(0, noise, (n_inliers, 3))
    normals_in /= np.linalg.norm(normals_in, axis=1, keepdims=True)

    outlier_pts = rng.uniform(-10, 10, (n_outliers, 3))
    outlier_normals = rng.normal(0, 1, (n_outliers, 3))
    outlier_normals /= np.linalg.norm(outlier_normals, axis=1, keepdims=True)

    points = np.concatenate([inliers, outlier_pts])
    normals = np.concatenate([normals_in, outlier_normals])
    return points, normals, axis_dir, half_angle, apex


def test_cone_recovery():
    apex = np.array([0.5, -1.0, 0.2])
    axis_dir = np.array([0.0, 0.0, 1.0])
    half_angle = np.deg2rad(25.0)
    points, normals, true_axis, true_alpha, _ = _generate_cone_cloud(
        800, 200, 0.005, apex, axis_dir, half_angle, seed=0
    )
    model, mask = fit(
        points,
        "cone",
        threshold=0.03,
        normals=normals,
        max_iterations=2000,
        seed=0,
    )
    assert model is not None, "expected a successful fit"
    cos = abs(float(model.axis @ true_axis))
    assert cos > 0.999, f"axis direction off; cos={cos}"
    alpha_err = abs(model.half_angle - true_alpha)
    assert alpha_err < 0.01, f"half-angle off; |Δα|={alpha_err:.5f} rad"
    n_in = int(mask.sum())
    assert n_in > 700, f"expected most inliers; got {n_in}"
    print(
        f"test_cone_recovery passed "
        f"(cos={cos:.5f}, α={np.rad2deg(model.half_angle):.3f}° vs "
        f"{np.rad2deg(true_alpha):.3f}°, inliers={n_in})"
    )


def test_cone_requires_normals():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (100, 3))
    try:
        fit(points, "cone", threshold=0.05, seed=0)
    except ValueError as e:
        assert "normals" in str(e).lower()
        print("test_cone_requires_normals passed")
        return
    raise AssertionError("expected ValueError for missing normals")


def test_cone_pure_noise_fails():
    rng = np.random.default_rng(1)
    points = rng.uniform(-5, 5, (800, 3))
    normals = rng.normal(0, 1, (800, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    model, mask = fit(
        points,
        "cone",
        threshold=0.03,
        normals=normals,
        min_inlier_ratio=0.5,
        max_iterations=500,
        seed=0,
    )
    assert model is None and mask is None
    print("test_cone_pure_noise_fails passed")


def test_cone_reproducibility():
    apex = np.zeros(3)
    axis_dir = np.array([0.0, 0.0, 1.0])
    half_angle = np.deg2rad(20.0)
    points, normals, _, _, _ = _generate_cone_cloud(
        800, 200, 0.005, apex, axis_dir, half_angle, seed=42
    )
    m1, mask1 = fit(points, "cone", threshold=0.03, normals=normals, seed=7)
    m2, mask2 = fit(points, "cone", threshold=0.03, normals=normals, seed=7)
    assert m1 is not None and m2 is not None
    assert np.allclose(m1.axis, m2.axis)
    assert abs(m1.half_angle - m2.half_angle) < 1e-9
    assert np.array_equal(mask1, mask2)
    print("test_cone_reproducibility passed")


def test_cone_vs_cylinder_discrimination():
    """Cylinder-shaped data fed to the cone fitter should either fail or
    converge to a half-angle near zero (i.e. degenerate / line-like cone).
    """
    rng = np.random.default_rng(7)
    axis_dir = np.array([0.0, 0.0, 1.0])
    radius = 0.5
    n = 600
    t = rng.uniform(-5, 5, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    u, v = _perp_basis(axis_dir)
    points = (
        t[:, None] * axis_dir[None]
        + radius * cos_t * u[None]
        + radius * sin_t * v[None]
        + rng.normal(0, 0.005, (n, 3))
    )
    normals = cos_t * u[None] + sin_t * v[None]
    normals = normals + rng.normal(0, 0.005, (n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    model, mask = fit(
        points,
        "cone",
        threshold=0.05,
        normals=normals,
        max_iterations=1000,
        seed=0,
    )
    # Acceptable outcomes: either no fit, or a half-angle very close to 0.
    if model is None:
        print("test_cone_vs_cylinder_discrimination passed (no fit)")
        return
    assert model.half_angle < 0.05, (
        f"cone fitted to cylindrical data should be near-degenerate; "
        f"got half_angle={np.rad2deg(model.half_angle):.3f}°"
    )
    print(
        f"test_cone_vs_cylinder_discrimination passed "
        f"(degenerate fit, half_angle={np.rad2deg(model.half_angle):.3f}°)"
    )


if __name__ == "__main__":
    test_cone_recovery()
    test_cone_requires_normals()
    test_cone_pure_noise_fails()
    test_cone_reproducibility()
    test_cone_vs_cylinder_discrimination()
    print("OK")
