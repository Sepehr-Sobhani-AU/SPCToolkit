"""Tests for fit_many — runs on whichever backend is available.

When CUDA is available, exercises the batched GPU path. Otherwise
exercises the sequential CPU fallback. The asserted invariants hold
for both.

Run: ``python unit_test/ransac/test_fit_many.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from core.services.ransac import fit_many


def _generate_line_cloud(n_inliers, n_outliers, noise, direction, origin, seed):
    rng = np.random.default_rng(seed)
    direction = direction / np.linalg.norm(direction)
    t = rng.uniform(-10, 10, n_inliers)
    inliers = (
        origin
        + t[:, None] * direction
        + rng.normal(0, noise, (n_inliers, 3))
    )
    outliers = rng.uniform(-15, 15, (n_outliers, 3))
    points = np.concatenate([inliers, outliers], axis=0)
    return points, direction


def _generate_plane_cloud(n_inliers, n_outliers, noise, normal, origin, seed):
    rng = np.random.default_rng(seed)
    normal = normal / np.linalg.norm(normal)
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(normal[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    a = rng.uniform(-5, 5, n_inliers)
    b = rng.uniform(-5, 5, n_inliers)
    inliers = (
        origin
        + a[:, None] * u
        + b[:, None] * v
        + rng.normal(0, noise, (n_inliers, 3))
    )
    outliers = rng.uniform(-10, 10, (n_outliers, 3))
    points = np.concatenate([inliers, outliers], axis=0)
    return points, normal


def test_fit_many_lines():
    rng = np.random.default_rng(0)
    B = 20
    point_sets = []
    true_dirs = []
    for i in range(B):
        direction = rng.normal(size=3)
        origin = rng.uniform(-5, 5, 3)
        n_in = int(rng.integers(120, 260))
        n_out = int(rng.integers(20, 80))
        points, d = _generate_line_cloud(n_in, n_out, 0.01, direction, origin, seed=i)
        point_sets.append(points)
        true_dirs.append(d)

    models, masks = fit_many(point_sets, "line", threshold=0.05, seed=0)

    n_recovered = 0
    for i in range(B):
        assert models[i] is not None, f"row {i} failed to fit"
        cos = abs(float(models[i].direction @ true_dirs[i]))
        assert cos > 0.999, f"row {i}: direction off; cos={cos}"
        n_recovered += 1
    print(f"test_fit_many_lines passed ({n_recovered}/{B} recovered)")


def test_fit_many_planes():
    rng = np.random.default_rng(1)
    B = 15
    point_sets = []
    true_normals = []
    for i in range(B):
        normal = rng.normal(size=3)
        origin = rng.uniform(-5, 5, 3)
        n_in = int(rng.integers(300, 600))
        n_out = int(rng.integers(100, 300))
        points, n = _generate_plane_cloud(n_in, n_out, 0.01, normal, origin, seed=i)
        point_sets.append(points)
        true_normals.append(n)

    models, masks = fit_many(point_sets, "plane", threshold=0.05, seed=0)

    for i in range(B):
        assert models[i] is not None, f"row {i} failed to fit"
        cos = abs(float(models[i].normal @ true_normals[i]))
        assert cos > 0.999, f"row {i}: normal off; cos={cos}"
    print(f"test_fit_many_planes passed (all {B} recovered)")


def test_fit_many_mixed_pass_fail():
    rng = np.random.default_rng(2)
    # Mix real lines and pure noise; expect noise rows to return None.
    point_sets = []
    is_real = []
    for i in range(8):
        if i % 2 == 0:
            direction = rng.normal(size=3)
            origin = rng.uniform(-5, 5, 3)
            points, _ = _generate_line_cloud(200, 30, 0.01, direction, origin, seed=i)
            point_sets.append(points)
            is_real.append(True)
        else:
            # pure noise of same size scale
            point_sets.append(rng.uniform(-10, 10, (300, 3)))
            is_real.append(False)

    models, masks = fit_many(
        point_sets, "line", threshold=0.05, min_inlier_ratio=0.5, seed=0
    )

    for i, real in enumerate(is_real):
        if real:
            assert models[i] is not None, f"row {i} (real line) failed to fit"
        else:
            assert models[i] is None, f"row {i} (noise) unexpectedly returned a model"
    print("test_fit_many_mixed_pass_fail passed")


def test_fit_many_empty():
    models, masks = fit_many([], "line", threshold=0.05)
    assert models == [] and masks == []
    print("test_fit_many_empty passed")


if __name__ == "__main__":
    test_fit_many_lines()
    test_fit_many_planes()
    test_fit_many_mixed_pass_fail()
    test_fit_many_empty()
    print("OK")
