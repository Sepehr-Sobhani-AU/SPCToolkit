# crease_tracer_test.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.services.crease_tracer import (
    CreaseTracer,
    vertices_to_polyline_feature,
    debug_vector_features,
)


def _flip_random_signs(normals, rng):
    """Plane normals carry a sign ambiguity; flip half at random so the test
    exercises the tracer's sign-invariant normal clustering."""
    flip = rng.random(len(normals)) < 0.5
    normals = normals.copy()
    normals[flip] *= -1.0
    return normals


def _kerb_crease_cloud(noise=0.004, seed=0):
    """A straight kerb: horizontal road (z=0) meets a vertical face (y=0) along
    the line y=0, z=0, x in [0, 10]."""
    rng = np.random.default_rng(seed)
    m = 3600
    road = np.stack(
        [rng.uniform(0, 10, m), rng.uniform(0, 2, m), np.zeros(m)], axis=1
    )
    road_n = np.tile([0.0, 0.0, 1.0], (m, 1))
    face = np.stack(
        [rng.uniform(0, 10, m), np.zeros(m), rng.uniform(0, 1, m)], axis=1
    )
    face_n = np.tile([0.0, 1.0, 0.0], (m, 1))

    points = np.vstack([road, face]) + rng.normal(0, noise, (2 * m, 3))
    normals = _flip_random_signs(np.vstack([road_n, face_n]), rng)
    return points.astype(np.float64), normals.astype(np.float64)


def _corner_crease_cloud(noise=0.004, seed=1):
    """A building corner: two vertical walls (x=0 and y=0) meet along the
    vertical line x=0, y=0, z in [0, 5]. Validates a *vertical* crease, which a
    2-D (XY-only) grid would miss."""
    rng = np.random.default_rng(seed)
    m = 3600
    wall_a = np.stack(
        [np.zeros(m), rng.uniform(0, 3, m), rng.uniform(0, 5, m)], axis=1
    )
    wall_a_n = np.tile([1.0, 0.0, 0.0], (m, 1))
    wall_b = np.stack(
        [rng.uniform(0, 3, m), np.zeros(m), rng.uniform(0, 5, m)], axis=1
    )
    wall_b_n = np.tile([0.0, 1.0, 0.0], (m, 1))

    points = np.vstack([wall_a, wall_b]) + rng.normal(0, noise, (2 * m, 3))
    normals = _flip_random_signs(np.vstack([wall_a_n, wall_b_n]), rng)
    return points.astype(np.float64), normals.astype(np.float64)


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rot_y(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _principal(n):
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    _, v = np.linalg.eigh(n.T @ n)
    return v[:, -1]


def _monotonic(values):
    d = np.diff(values)
    return np.all(d >= -1e-6) or np.all(d <= 1e-6)


def test_kerb_crease_traces_edge_line():
    points, normals = _kerb_crease_cloud()
    tracer = CreaseTracer(
        points, normals, cell_size=1.0, min_points_per_cell=10,
        min_dihedral_deg=20.0, ransac_threshold=0.03, ransac_iterations=100,
        backend="cpu", seed=0,
    )
    v = tracer.trace()
    assert len(v) >= 5, f"expected several vertices, got {len(v)}"

    # The analytic edge is the line y=0, z=0 -> deviation is sqrt(y^2 + z^2).
    dev = np.sqrt(v[:, 1] ** 2 + v[:, 2] ** 2)
    assert dev.max() < 0.05, f"vertices stray from edge: max dev {dev.max():.3f} m"
    assert v[:, 0].max() - v[:, 0].min() > 7.0, "edge does not span the kerb length"
    assert _monotonic(v[:, 0]), "vertices not ordered along the crease"
    print(f"kerb crease: {len(v)} vertices, max edge deviation {dev.max()*1000:.1f} mm")


def test_oblique_crease_traces_edge_line():
    """A kerb crease rotated off the world axes. With a world-aligned grid the
    cells would cut the crease unevenly and starve some plane fits; the
    crease-aligned grid must still recover the edge cleanly."""
    points, normals = _kerb_crease_cloud()
    rot = _rot_z(0.5) @ _rot_y(0.3)          # arbitrary off-axis orientation
    points = points @ rot.T
    normals = normals @ rot.T

    tracer = CreaseTracer(
        points, normals, cell_size=1.0, min_points_per_cell=10,
        min_dihedral_deg=20.0, ransac_threshold=0.03, ransac_iterations=100,
        backend="cpu", seed=0,
    )
    v = tracer.trace()
    assert len(v) >= 5, f"expected several vertices, got {len(v)}"

    # The analytic edge is the original x-axis line through the origin, rotated:
    # direction = rot @ (1,0,0) = first column of rot, passing through 0.
    d = rot[:, 0]
    perp = v - np.outer(v @ d, d)            # component off the edge line
    dev = np.linalg.norm(perp, axis=1)
    assert dev.max() < 0.05, f"oblique vertices stray from edge: max dev {dev.max():.3f} m"
    assert (v @ d).max() - (v @ d).min() > 7.0, "edge does not span the kerb length"
    assert _monotonic(v @ d), "vertices not ordered along the crease"
    assert tracer._global_axes is not None, "global crease orientation should have been found"
    print(f"oblique crease: {len(v)} vertices, max edge deviation {dev.max()*1000:.1f} mm")


def _curved_crease_cloud(radius=5.0, noise=0.004, seed=4):
    """A curved kerb: a vertical wall on a 90-degree arc (radius R, z in [0,1])
    meets the flat ground (z=0) along the arc at z=0. The crease is the arc — no
    single global frame fits it, so it exercises the march's per-step rotation."""
    rng = np.random.default_rng(seed)
    m = 4000
    th = rng.uniform(0.0, np.pi / 2, m)
    wall = np.stack(
        [radius * np.cos(th), radius * np.sin(th), rng.uniform(0, 1, m)], axis=1
    )
    wall_n = np.stack([np.cos(th), np.sin(th), np.zeros(m)], axis=1)  # radial

    th2 = rng.uniform(0.0, np.pi / 2, m)
    rr = rng.uniform(radius, radius + 1.5, m)
    road = np.stack([rr * np.cos(th2), rr * np.sin(th2), np.zeros(m)], axis=1)
    road_n = np.tile([0.0, 0.0, 1.0], (m, 1))

    points = np.vstack([wall, road]) + rng.normal(0, noise, (2 * m, 3))
    normals = _flip_random_signs(np.vstack([wall_n, road_n]), rng)
    return points.astype(np.float64), normals.astype(np.float64)


def test_curved_crease_follows_arc():
    radius = 5.0
    points, normals = _curved_crease_cloud(radius=radius)
    tracer = CreaseTracer(
        points, normals, cell_size=0.5, min_points_per_cell=8,
        min_dihedral_deg=20.0, ransac_threshold=0.03, ransac_iterations=100,
        backend="cpu", seed=0,
    )
    v = tracer.trace()
    assert len(v) >= 8, f"expected to follow the arc, got {len(v)} vertices"

    # The analytic edge is the arc radius=R at z=0.
    r = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    assert np.abs(r - radius).max() < 0.06, \
        f"vertices stray from the arc: max |r-R| {np.abs(r - radius).max():.3f} m"
    assert np.abs(v[:, 2]).max() < 0.06, f"vertices off z=0: {np.abs(v[:, 2]).max():.3f} m"
    # Covers a good span of the 90-degree (1.57 rad) arc.
    ang = np.arctan2(v[:, 1], v[:, 0])
    assert ang.max() - ang.min() > 1.0, "edge does not span the arc"
    print(f"curved crease: {len(v)} vertices, max |r-R| {np.abs(r - radius).max()*1000:.1f} mm, "
          f"arc span {np.degrees(ang.max() - ang.min()):.0f} deg")


def test_corner_crease_traces_vertical_edge():
    points, normals = _corner_crease_cloud()
    tracer = CreaseTracer(
        points, normals, cell_size=1.0, min_points_per_cell=10,
        min_dihedral_deg=20.0, ransac_threshold=0.03, ransac_iterations=100,
        backend="cpu", seed=0,
    )
    v = tracer.trace()
    assert len(v) >= 3, f"expected several vertices, got {len(v)}"

    # The analytic edge is the vertical line x=0, y=0 -> deviation is sqrt(x^2+y^2).
    dev = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    assert dev.max() < 0.05, f"vertices stray from edge: max dev {dev.max():.3f} m"
    assert v[:, 2].max() - v[:, 2].min() > 3.5, "edge does not span the corner height"
    assert _monotonic(v[:, 2]), "vertices not ordered along the vertical crease"
    print(f"corner crease: {len(v)} vertices, max edge deviation {dev.max()*1000:.1f} mm")


def test_single_plane_yields_no_edge():
    """One flat surface is not a crease: every cell resolves to one plane, so the
    dihedral gate rejects them all and no edge is produced."""
    rng = np.random.default_rng(3)
    m = 5000
    pts = np.stack(
        [rng.uniform(0, 10, m), rng.uniform(0, 10, m), np.zeros(m)], axis=1
    )
    pts += rng.normal(0, 0.004, pts.shape)
    nrm = _flip_random_signs(np.tile([0.0, 0.0, 1.0], (m, 1)), rng)

    tracer = CreaseTracer(
        pts.astype(np.float64), nrm.astype(np.float64),
        cell_size=1.0, min_points_per_cell=10, min_dihedral_deg=20.0,
        ransac_threshold=0.03, ransac_iterations=100, backend="cpu", seed=0,
    )
    v = tracer.trace()
    assert len(v) == 0, f"a single plane should yield no crease, got {len(v)} vertices"
    print("single plane: no crease (correct)")


def test_debug_overlays_recorded():
    points, normals = _kerb_crease_cloud()
    tracer = CreaseTracer(
        points, normals, cell_size=1.0, min_points_per_cell=10,
        min_dihedral_deg=20.0, ransac_threshold=0.03, ransac_iterations=100,
        backend="cpu", seed=0, record_debug=True,
    )
    v = tracer.trace()
    assert len(v) >= 5, f"expected several vertices, got {len(v)}"

    # The march visits a corridor of cubes; each cube records its oriented box,
    # its two planes, and the side of every point it covered.
    assert tracer.debug_point_cell is not None
    assert (tracer.debug_point_cell >= 0).any(), "no points assigned to march cubes"
    assert len(tracer.debug_cells) > 0, "no cells recorded"
    assert len(tracer.debug_cells) == len(tracer.debug_planes), "one plane-pair per cell"
    assert len(tracer.debug_cells) >= len(v), "at least one cell per vertex"

    # Normals split into two colour branches by global cluster; both surfaces
    # present and the two clusters capture distinctly-oriented normals.
    side = tracer.debug_point_side
    assert side is not None
    assert (side == 0).any() and (side == 1).any(), "both normal clusters expected"
    assert abs(_principal(normals[side == 0]) @ _principal(normals[side == 1])) < 0.5, \
        "the two normal clusters should be distinct surfaces"

    feats = debug_vector_features(tracer, True, True, True)
    names = [n for n, _ in feats]
    assert names == ["debug_cells", "debug_planes",
                     "debug_normals_a", "debug_normals_b"], names
    for _, vf in feats:
        assert vf.geometry_type == "mesh"
        assert vf.geometry["vertices"].shape[1] == 3
        assert vf.geometry["edges"].shape[1] == 2
    assert debug_vector_features(tracer, False, False, False) == []
    print(f"debug overlays: {names}; {len(tracer.debug_cells)} cells, "
          f"{len(tracer.debug_planes)} planes, "
          f"normals split {int((side==0).sum())}/{int((side==1).sum())}")


def test_polyline_feature_built():
    v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    vf = vertices_to_polyline_feature(v)
    assert vf is not None
    assert vf.geometry_type == "polyline"
    assert vf.geometry["closed"] is False
    assert vf.geometry["vertices"].shape == (3, 3)
    assert vf.symbol_type == "crease_edge"
    # Fewer than two vertices is nothing to draw.
    assert vertices_to_polyline_feature(v[:1]) is None
    print("polyline VectorFeature: built and validated")


if __name__ == "__main__":
    test_kerb_crease_traces_edge_line()
    test_oblique_crease_traces_edge_line()
    test_curved_crease_follows_arc()
    test_corner_crease_traces_vertical_edge()
    test_single_plane_yields_no_edge()
    test_debug_overlays_recorded()
    test_polyline_feature_built()
    print("\nAll crease_tracer tests passed.")
