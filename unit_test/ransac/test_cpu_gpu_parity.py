"""CPU/GPU parity tests: both backends should recover the same model
within numerical tolerance on the same input.

Note: identical *seeds* don't yield bit-identical results across
backends — the CPU path samples via numpy.random.Generator while the
GPU path samples via torch.Generator on CUDA. Parity is on the
recovered geometry (direction / normal / inlier count), not on the
internal random stream.

Skipped cleanly when CUDA is unavailable.

Run: ``python unit_test/ransac/test_cpu_gpu_parity.py``
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


def _make_line_cloud(seed):
    rng = np.random.default_rng(seed)
    direction = np.array([1.0, -2.0, 0.5])
    direction /= np.linalg.norm(direction)
    origin = np.array([0.0, 0.0, 0.0])
    t = rng.uniform(-10, 10, 3000)
    inliers = origin + t[:, None] * direction + rng.normal(0, 0.005, (3000, 3))
    outliers = rng.uniform(-15, 15, (500, 3))
    return np.concatenate([inliers, outliers]), direction


def _make_plane_cloud(seed):
    rng = np.random.default_rng(seed)
    normal = np.array([0.1, 0.9, -0.4])
    normal /= np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    a = rng.uniform(-5, 5, 5000)
    b = rng.uniform(-5, 5, 5000)
    inliers = a[:, None] * u + b[:, None] * v + rng.normal(0, 0.005, (5000, 3))
    outliers = rng.uniform(-10, 10, (1500, 3))
    return np.concatenate([inliers, outliers]), normal


def test_line_parity():
    points, true_dir = _make_line_cloud(seed=0)
    cpu_m, cpu_mask = fit(points, "line", threshold=0.05, seed=0, backend="cpu")
    gpu_m, gpu_mask = fit(points, "line", threshold=0.05, seed=0, backend="gpu")
    assert cpu_m is not None and gpu_m is not None
    cpu_cos = abs(float(cpu_m.direction @ true_dir))
    gpu_cos = abs(float(gpu_m.direction @ true_dir))
    assert cpu_cos > 0.999 and gpu_cos > 0.999
    cross_cos = abs(float(cpu_m.direction @ gpu_m.direction))
    assert cross_cos > 0.9999, f"CPU/GPU directions disagree; cos={cross_cos}"
    delta_inliers = abs(int(cpu_mask.sum()) - int(gpu_mask.sum()))
    assert delta_inliers < 50, f"inlier count mismatch: cpu={cpu_mask.sum()}, gpu={gpu_mask.sum()}"
    print(
        f"test_line_parity passed "
        f"(cpu_cos={cpu_cos:.5f}, gpu_cos={gpu_cos:.5f}, "
        f"cross_cos={cross_cos:.6f}, Δin={delta_inliers})"
    )


def test_plane_parity():
    points, true_n = _make_plane_cloud(seed=0)
    cpu_m, cpu_mask = fit(points, "plane", threshold=0.05, seed=0, backend="cpu")
    gpu_m, gpu_mask = fit(points, "plane", threshold=0.05, seed=0, backend="gpu")
    assert cpu_m is not None and gpu_m is not None
    cpu_cos = abs(float(cpu_m.normal @ true_n))
    gpu_cos = abs(float(gpu_m.normal @ true_n))
    assert cpu_cos > 0.999 and gpu_cos > 0.999
    cross_cos = abs(float(cpu_m.normal @ gpu_m.normal))
    assert cross_cos > 0.9999, f"CPU/GPU normals disagree; cos={cross_cos}"
    delta_inliers = abs(int(cpu_mask.sum()) - int(gpu_mask.sum()))
    assert delta_inliers < 100, f"inlier count mismatch: cpu={cpu_mask.sum()}, gpu={gpu_mask.sum()}"
    print(
        f"test_plane_parity passed "
        f"(cpu_cos={cpu_cos:.5f}, gpu_cos={gpu_cos:.5f}, "
        f"cross_cos={cross_cos:.6f}, Δin={delta_inliers})"
    )


if __name__ == "__main__":
    if not _cuda_available():
        print("CUDA not available; CPU/GPU parity tests skipped.")
        sys.exit(0)
    test_line_parity()
    test_plane_parity()
    print("OK")
