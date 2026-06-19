"""Headless tests for the Fit Cylinder / Cone plugin.

Covers the two parts that carry real risk:
  1. RANSAC recovery — synthesize a cylinder and a cone (with analytic
     normals) and confirm core.services.ransac.fit recovers the parameters.
  2. The pure geometry helpers (truncate + tessellate) produce well-formed
     wireframe arrays.

Run: python unit_test/fit_cylinder_cone_test.py
"""

import os
import sys
import importlib.util

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.services.ransac import fit


# ── Load the plugin module by path (numeric-prefixed dir isn't importable) ──

def _load_plugin_module():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "plugins", "060_Infrastructure", "040_fit_cylinder_cone_plugin.py",
    )
    spec = importlib.util.spec_from_file_location("fit_cylinder_cone_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_plugin = _load_plugin_module()


# ── Synthetic generators ───────────────────────────────────────────────────

def _perp_basis(axis):
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _make_cylinder(radius, height, axis, center, n=800, seed=0):
    rng = np.random.default_rng(seed)
    axis = axis / np.linalg.norm(axis)
    u, v = _perp_basis(axis)
    theta = rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(0, height, n)
    radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    points = center + radius * radial + h[:, None] * axis
    normals = radial  # outward radial — exactly what the cylinder solver needs
    return points, normals


def _make_cone(half_angle_deg, s_lo, s_hi, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.radians(half_angle_deg)
    apex = np.array([0.0, 0.0, 0.0]); axis = np.array([0.0, 0.0, 1.0])
    u, v = _perp_basis(axis)
    theta = rng.uniform(0, 2 * np.pi, n)
    s = rng.uniform(s_lo, s_hi, n)
    r = s * np.tan(a)
    radial = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    points = apex + s[:, None] * axis + r[:, None] * radial
    normals = np.cos(a) * radial - np.sin(a) * axis[None, :]  # outward cone normal
    return points, normals, a


# ── Tests ───────────────────────────────────────────────────────────────────

def test_cylinder_recovery_and_geometry():
    radius, height = 0.10, 4.0
    axis = np.array([0.1, 0.0, 1.0])
    center = np.array([1.0, 2.0, 0.0])
    points, normals = _make_cylinder(radius, height, axis, center, n=800, seed=1)

    model, mask = fit(points, "cylinder", threshold=0.005, normals=normals,
                      max_iterations=2000, min_inlier_ratio=0.4, seed=0)
    assert model is not None, "cylinder fit returned no model"
    assert abs(model.radius - radius) < 0.01, f"radius off: {model.radius} vs {radius}"
    cos = abs(float(np.dot(model.direction / np.linalg.norm(model.direction),
                           axis / np.linalg.norm(axis))))
    assert cos > 0.99, f"axis off: cos={cos}"

    geom = _plugin._truncate_cylinder(model, points[mask])
    assert geom["base"][2] <= geom["top"][2]
    verts, faces, edges = _plugin._tessellate_frustum(
        geom["base"], geom["top"], geom["radius_base"], geom["radius_top"])
    assert verts.shape == (2 * _plugin._N_SEGMENTS, 3)
    assert edges.shape == (3 * _plugin._N_SEGMENTS, 2)
    assert len(faces) == _plugin._N_SEGMENTS
    print("  cylinder: radius/axis recovered, wireframe well-formed")


def test_cone_recovery_and_geometry():
    half_angle_deg = 18.0
    points, normals, true_a = _make_cone(half_angle_deg, s_lo=1.0, s_hi=4.0, n=1200, seed=2)

    model, mask = fit(points, "cone", threshold=0.01, normals=normals,
                      max_iterations=4000, min_inlier_ratio=0.4, seed=0)
    assert model is not None, "cone fit returned no model"
    cos = abs(float(np.dot(model.axis / np.linalg.norm(model.axis),
                           np.array([0.0, 0.0, 1.0]))))
    assert cos > 0.98, f"cone axis off: cos={cos}"
    assert abs(np.degrees(model.half_angle) - half_angle_deg) < 4.0, \
        f"half-angle off: {np.degrees(model.half_angle)} vs {half_angle_deg}"

    geom = _plugin._truncate_cone(model, points[mask])
    assert geom["primitive"] == "cone"
    assert geom["half_angle"] is not None
    verts, faces, edges = _plugin._tessellate_frustum(
        geom["base"], geom["top"], geom["radius_base"], geom["radius_top"])
    assert verts.shape == (2 * _plugin._N_SEGMENTS, 3)
    assert edges.shape == (3 * _plugin._N_SEGMENTS, 2)
    print("  cone: axis/half-angle recovered, frustum well-formed")


def test_build_feature_carries_analytic_keys():
    import uuid
    radius, height = 0.08, 3.0
    points, normals = _make_cylinder(radius, height, np.array([0.0, 0.0, 1.0]),
                                     np.array([0.0, 0.0, 0.0]), n=600, seed=3)
    model, mask = fit(points, "cylinder", threshold=0.005, normals=normals,
                      max_iterations=2000, min_inlier_ratio=0.4, seed=0)
    assert model is not None
    feature, quality = _plugin._build_feature(points, model, mask, "cylinder", uuid.uuid4())
    g = feature.geometry
    for key in ("vertices", "faces", "edges", "primitive", "base", "top",
                "axis", "radius_base", "radius_top", "height", "fit"):
        assert key in g, f"missing geometry key: {key}"
    assert feature.geometry_type == "mesh"
    assert feature.symbol_type == "cylinder"
    assert quality["inlier_ratio"] > 0.4
    print(f"  build_feature: analytic keys present, inliers={quality['inlier_ratio']*100:.0f}%")


if __name__ == "__main__":
    print("Testing Fit Cylinder / Cone:")
    test_cylinder_recovery_and_geometry()
    test_cone_recovery_and_geometry()
    test_build_feature_carries_analytic_keys()
    print("All tests passed.")
