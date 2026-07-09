# geometry_utils_test.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.services.geometry_utils import (
    unit, unit_rows, perp_basis, principal_axis, dominant_direction,
)


def test_unit_zero_safe():
    # A zero vector must not blow up; result stays finite and ~zero.
    out = unit(np.zeros(3))
    assert np.all(np.isfinite(out)), "unit() produced non-finite values"
    assert np.linalg.norm(out) < 1e-6, "unit(0) should stay near zero"
    # A non-zero vector normalises to length 1.
    out = unit(np.array([3.0, 0.0, 4.0]))
    assert abs(np.linalg.norm(out) - 1.0) < 1e-9
    print("unit: zero-safe and normalises")


def test_unit_rows():
    v = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    out = unit_rows(v)
    norms = np.linalg.norm(out, axis=1)
    assert abs(norms[0] - 1.0) < 1e-9 and abs(norms[2] - 1.0) < 1e-9
    assert norms[1] < 1e-6, "zero row must stay near zero"
    print("unit_rows: row-wise normalise, zero-safe")


def test_perp_basis_orthonormal():
    for d in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]):
        d = np.asarray(d, dtype=np.float64)
        u, v = perp_basis(d)
        dn = d / np.linalg.norm(d)
        assert abs(np.linalg.norm(u) - 1.0) < 1e-9
        assert abs(np.linalg.norm(v) - 1.0) < 1e-9
        assert abs(u @ v) < 1e-9, "u, v not orthogonal"
        assert abs(u @ dn) < 1e-9 and abs(v @ dn) < 1e-9, "basis not perpendicular to d"
    print("perp_basis: orthonormal and perpendicular")


def test_principal_axis_mean_centered():
    # Points on a line offset far from the origin: the mean-centred principal
    # axis is the line direction, unaffected by the offset.
    t = np.linspace(-1.0, 1.0, 50)[:, None]
    direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    pts = np.array([100.0, 100.0, 100.0]) + t * direction
    axis = principal_axis(pts)
    assert abs(abs(axis @ direction) - 1.0) < 1e-6, "principal_axis missed the line direction"
    print("principal_axis: recovers mean-shifted line direction")


def test_dominant_direction_vs_principal_axis():
    # A bundle of unit normals clustered around +z (about the origin, NOT their
    # mean): dominant_direction returns ~+z; principal_axis (mean-centred) would
    # instead return the spread direction. The two must differ here.
    rng = np.random.default_rng(0)
    base = np.array([0.0, 0.0, 1.0])
    normals = base + 0.05 * rng.normal(size=(200, 3))
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    dom = dominant_direction(normals)
    assert abs(abs(dom @ base) - 1.0) < 1e-2, "dominant_direction missed the normal bundle axis"
    pa = principal_axis(normals)
    assert abs(abs(pa @ base) - 1.0) > 1e-2, (
        "principal_axis unexpectedly matched the bundle axis — the two functions "
        "should differ for direction bundles"
    )
    print("dominant_direction: raw second moment differs from mean-centred principal_axis")


if __name__ == "__main__":
    test_unit_zero_safe()
    test_unit_rows()
    test_perp_basis_orthonormal()
    test_principal_axis_mean_centered()
    test_dominant_direction_vs_principal_axis()
    print("\nAll geometry_utils tests passed.")
