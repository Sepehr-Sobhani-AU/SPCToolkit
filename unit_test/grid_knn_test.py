"""
Tests for core.services.grid_knn.

``GridKNN`` claims to be *exact*: the grid only decides which points are looked
at, never which ones are returned. So ``cKDTree`` is the oracle for every test,
on the clouds that make a uniform grid work hardest — very uneven density, and
queries far outside the cloud, where the ring-by-ring scan has to keep going.
Distances are compared (not indices), because two points at the same distance
may come back in either order.

Needs CuPy and an NVIDIA GPU: the service has no CPU path by design.

Run:
    python unit_test/grid_knn_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

from core.services.grid_knn import GridKNN, MAX_K

_RNG = np.random.default_rng(23)
# float32 distances against a float64 tree: 1e-5 absolute, plus float32
# rounding (relative) for queries far from the cloud.
_TOL = 1e-5
_REL = 1e-6


def _uniform(n=20_000, extent=10.0):
    return (_RNG.random((n, 2)) * extent).astype(np.float32)


def _uneven(n_dense=15_000, n_sparse=500):
    """A dense patch in one corner and a sparse spread over the rest: most
    grid cells are empty, a few hold far more than the average."""
    dense = _RNG.normal([1.0, 1.0], 0.05, (n_dense, 2))
    sparse = _RNG.random((n_sparse, 2)) * 20.0
    return np.concatenate([dense, sparse]).astype(np.float32)


def _check(ref, query, k, label):
    index = GridKNN(ref)
    try:
        d, i = index.query(query, k)
    finally:
        index.release()
    kk = min(k, len(ref))
    td, _ti = cKDTree(ref.astype(np.float64)).query(query.astype(np.float64), k=kk)
    td = np.asarray(td).reshape(len(query), kk)
    assert d.shape == (len(query), kk) and i.shape == (len(query), kk), (d.shape, i.shape)
    assert d.dtype == np.float32 and i.dtype == np.int64, (d.dtype, i.dtype)
    assert np.all(i >= 0) and np.all(i < len(ref)), "an index points outside the cloud"
    # The returned indices really are at the returned distances.
    real = np.linalg.norm(ref[i].astype(np.float64) - query[:, None, :2].astype(np.float64), axis=2)
    assert (np.abs(real - d) / (1.0 + _REL / _TOL * real)).max() < _TOL, np.abs(real - d).max()
    # Nearest first.
    assert np.all(np.diff(d, axis=1) >= 0), "distances not sorted"
    # No point returned twice for one query.
    s = np.sort(i, axis=1)
    assert not np.any(s[:, 1:] == s[:, :-1]), "a neighbour returned twice"
    err = (np.abs(d.astype(np.float64) - td) / (1.0 + _REL / _TOL * td)).max()
    assert err < _TOL, f"{label}: max distance error {err}"
    print(f"  {label}: k={k}, {len(query):,} queries, max error {err:.1e}")


def test_uniform_cloud_matches_a_kdtree():
    ref = _uniform()
    for k in (1, 16, MAX_K):
        _check(ref, ref, k, "uniform, self-query")
    _check(ref, _uniform(3_000), 16, "uniform, other points")


def test_uneven_density_matches_a_kdtree():
    ref = _uneven()
    for k in (1, 16, MAX_K):
        _check(ref, ref, k, "uneven, self-query")
    # Queries spread evenly: most land in empty cells far from the dense patch.
    _check(ref, (_RNG.random((3_000, 2)) * 20.0).astype(np.float32), MAX_K, "uneven, spread queries")


def test_queries_outside_the_bounding_box_match_a_kdtree():
    ref = _uniform(10_000, 10.0)
    far = np.concatenate([
        _RNG.random((500, 2)) * 10.0 + [25.0, 0.0],     # off one side
        _RNG.random((500, 2)) * 10.0 - [0.0, 40.0],     # off another
        _RNG.random((500, 2)) * 200.0 - 100.0,          # anywhere, mostly far out
    ]).astype(np.float32)
    for k in (1, 16, MAX_K):
        _check(ref, far, k, "outside the box")


def test_k_larger_than_the_cloud_is_clipped():
    ref = _uniform(10, 1.0)
    _check(ref, ref, 20, "10 points, k=20")
    _check(ref, _uniform(50, 3.0), MAX_K, "10 points, k=32, other queries")
    index = GridKNN(ref[:1])
    d, i = index.query(ref, 5)
    index.release()
    assert d.shape == (10, 1) and np.all(i == 0), (d.shape, i)
    print("  k is clipped to the number of points (one point: every query gets row 0)")


def test_self_query_returns_self_first():
    ref = _uniform(5_000, 5.0)
    # A few exact duplicates: either copy may come first, both at distance 0.
    ref[100:110] = ref[0]
    index = GridKNN(ref)
    d, i = index.query(ref, 8)
    index.release()
    assert np.all(d[:, 0] == 0.0), "a point is not its own nearest at distance 0"
    first_is_self = i[:, 0] == np.arange(len(ref))
    same_spot = np.all(ref[i[:, 0]] == ref, axis=1)
    assert np.all(first_is_self | same_spot), "first neighbour is not the point itself"
    assert first_is_self[110:].all(), "a unique point did not come back first"
    print(f"  self first at distance 0 for all {len(ref):,} points (ties allowed)")


def test_k_above_the_limit_is_refused():
    index = GridKNN(_uniform(100))
    try:
        index.query(_uniform(10), MAX_K + 1)
    except ValueError:
        print(f"  k={MAX_K + 1} raises ValueError")
    else:
        raise AssertionError(f"k={MAX_K + 1} should raise ValueError")
    finally:
        index.release()


def test_degenerate_clouds():
    # All points on one line: the bounding box has no height. Queries stay
    # within 1 cm of the line: the cells are then ~0.1 mm wide and a query
    # far off the line scans (distance / cell)^2 cells — 1 m off takes ~8 s.
    line = np.column_stack([np.linspace(0, 10, 1_000), np.full(1_000, 2.0)]).astype(np.float32)
    near = np.column_stack([_RNG.random(500) * 10.0,
                            2.0 + _RNG.normal(0, 0.003, 500)]).astype(np.float32)
    _check(line, near, 8, "points on a line")
    # Extra columns are ignored (a (N,3) cloud is read as its XY).
    xyz = np.column_stack([_uniform(2_000), _RNG.random(2_000) * 100]).astype(np.float32)
    index = GridKNN(xyz)
    d, _i = index.query(xyz, 4)
    index.release()
    td, _ = cKDTree(xyz[:, :2].astype(np.float64)).query(xyz[:, :2].astype(np.float64), k=4)
    assert np.abs(d - td).max() < _TOL
    try:
        GridKNN(np.zeros((0, 2), np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("an empty cloud should raise ValueError")
    print("  a flat line, (N,3) input and an empty cloud are handled")


if __name__ == "__main__":
    try:
        import cupy
        cupy.zeros(1)
    except Exception as exc:
        print(f"Grid KNN tests need CuPy and a GPU ({type(exc).__name__}); skipped.")
        sys.exit(0)

    test_uniform_cloud_matches_a_kdtree()
    test_uneven_density_matches_a_kdtree()
    test_queries_outside_the_bounding_box_match_a_kdtree()
    test_k_larger_than_the_cloud_is_clipped()
    test_self_query_returns_self_first()
    test_k_above_the_limit_is_refused()
    test_degenerate_clouds()
    print("\nAll grid KNN tests passed.")
