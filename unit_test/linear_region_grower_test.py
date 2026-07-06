# linear_region_grower_test.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.services.linear_region_grower import (
    LinearRegionGrower, AXIS_TRACE, LINEARITY_CONNECTED, HYBRID,
    debug_vector_features,
)


def _straight_line_cloud():
    """A straight line along +x with small noise, plus off-line clutter."""
    rng = np.random.default_rng(0)
    n_line = 400
    x = np.linspace(0, 20, n_line)
    line = np.stack([x, np.zeros(n_line), np.zeros(n_line)], axis=1)
    line += rng.normal(0, 0.005, line.shape)
    clutter = rng.uniform([0, 2, 2], [20, 6, 6], size=(300, 3))
    points = np.vstack([line, clutter]).astype(np.float64)
    return points, np.arange(n_line)


def test_axis_trace_collects_line():
    points, line_idx = _straight_line_cloud()
    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE,
        ransac_threshold=0.05, cylinder_radius=0.1, cylinder_length=1.0,
        min_points=3, max_angle_deg=20.0,
    )
    grown = set(grower.grow(line_idx[:12]).tolist())
    line_set = set(line_idx.tolist())
    recovered = len(grown & line_set) / len(line_set)
    leaked = len(grown - line_set)
    print(f"axis_trace: recovered {recovered:.0%} of line, leaked {leaked} clutter pts")
    assert recovered > 0.9, f"axis_trace recovered only {recovered:.0%} of the line"
    assert leaked == 0, f"axis_trace leaked {leaked} clutter points"
    # Debug geometry recorded for the show-cylinders / show-lines overlays.
    assert len(grower.debug_cylinders) > 0, "axis_trace recorded no debug cylinders"
    assert len(grower.debug_lines) > 0, "axis_trace recorded no debug lines"
    print(f"  recorded {len(grower.debug_cylinders)} cylinders, "
          f"{len(grower.debug_lines)} line segments")


def test_axis_trace_long_curved_seeds():
    """Seeds spanning a long curved arc (span >> cylinder_length).

    A single global line over such seeds is a chord, badly misaligned with the
    feature's local tangents; the march must instead start from a local tangent
    and traverse the seed body step by step. Recovery should stay high.
    """
    rng = np.random.default_rng(2)
    R = 5.0
    n_arc = 600
    theta = np.linspace(np.radians(30), np.radians(150), n_arc)  # 120-degree arc
    arc = np.stack([R * np.cos(theta), R * np.sin(theta), np.zeros(n_arc)], axis=1)
    arc += rng.normal(0, 0.003, arc.shape)
    clutter = rng.uniform([-6, -6, 3], [6, 6, 8], size=(300, 3))
    points = np.vstack([arc, clutter]).astype(np.float64)
    arc_idx = np.arange(n_arc)

    # 12 seeds spread across the WHOLE arc -> span (~10 m) >> cylinder_length.
    seeds = arc_idx[::n_arc // 12]
    span = float(np.linalg.norm(points[seeds].max(0) - points[seeds].min(0)))

    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE,
        ransac_threshold=0.05, cylinder_radius=0.2, cylinder_length=0.5,
        min_points=3, max_angle_deg=25.0,
    )
    grown = set(grower.grow(seeds).tolist())
    arc_set = set(arc_idx.tolist())
    recovered = len(grown & arc_set) / len(arc_set)
    leaked = len(grown - arc_set)
    print(f"curved seeds (span {span:.1f} m >> 0.5 m cyl): "
          f"recovered {recovered:.0%} of arc, leaked {leaked} clutter pts")
    assert recovered > 0.9, f"curved-seed trace recovered only {recovered:.0%} of the arc"
    assert leaked == 0, f"curved-seed trace leaked {leaked} clutter points"


def test_axis_trace_cylinders_match_search():
    """The drawn search cylinders must be the ACTUAL selection cylinders: each
    has the full cylinder_length and cylinder_radius the march searched with —
    not a shortened tip-to-tip segment — so the overlay matches the real
    selection region even when overlap > 0 shortens the per-step advance."""
    rng = np.random.default_rng(3)
    R = 8.0
    n = 500
    theta = np.linspace(np.radians(40), np.radians(140), n)
    arc = np.stack([R * np.cos(theta), R * np.sin(theta), np.zeros(n)], axis=1)
    arc += rng.normal(0, 0.003, arc.shape)
    seeds = np.arange(n)[::n // 12]

    cyl_len, cyl_rad = 0.5, 0.2
    g = LinearRegionGrower(
        arc, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=cyl_rad, cylinder_length=cyl_len, overlap=0.5,  # step != length
        min_points=3, max_angle_deg=25.0,
    )
    g.grow(seeds)
    cyls = g.debug_cylinders
    assert len(cyls) > 4, f"expected several cylinders, got {len(cyls)}"

    for _base, direction, radius, length in cyls:
        assert abs(length - cyl_len) < 1e-9, f"cylinder length {length} != search length {cyl_len}"
        assert abs(radius - cyl_rad) < 1e-9, f"cylinder radius {radius} != search radius {cyl_rad}"
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6, "cylinder axis must be unit length"
    print(f"search cylinders: all {len(cyls)} match length={cyl_len} radius={cyl_rad}")


def test_linearity_connected_stays_on_feature():
    """A linear kerb embedded in a planar ground patch; linearity separates them."""
    rng = np.random.default_rng(1)
    n_kerb = 300
    x = np.linspace(0, 15, n_kerb)
    kerb = np.stack([x, np.zeros(n_kerb), np.zeros(n_kerb)], axis=1)
    kerb += rng.normal(0, 0.005, kerb.shape)
    gx, gy = np.meshgrid(np.linspace(0, 15, 60), np.linspace(-3, 3, 40))
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    points = np.vstack([kerb, ground]).astype(np.float64)
    kerb_idx = np.arange(n_kerb)

    linearity = np.zeros(len(points))
    linearity[:n_kerb] = 0.9  # high on the kerb, 0 on the ground

    grower = LinearRegionGrower(
        points, mode=LINEARITY_CONNECTED, linearity=linearity,
        linearity_threshold=0.4, neighbor_k=16,
    )
    grown = set(grower.grow(kerb_idx[:10]).tolist())
    kerb_set = set(kerb_idx.tolist())
    recovered = len(grown & kerb_set) / len(kerb_set)
    leaked = len(grown - kerb_set)
    print(f"linearity_connected: recovered {recovered:.0%} of kerb, leaked {leaked} ground pts")
    assert recovered > 0.9, f"linearity_connected recovered only {recovered:.0%} of the kerb"
    assert leaked == 0, f"linearity_connected leaked {leaked} ground points"


def test_linearity_mode_requires_linearity():
    points, _ = _straight_line_cloud()
    for mode in (LINEARITY_CONNECTED, HYBRID):
        try:
            LinearRegionGrower(points, mode=mode)  # no linearity provided
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for mode '{mode}' without linearity")
    print("linearity modes correctly require a linearity array")


def test_overlap_increases_cylinder_count():
    points, line_idx = _straight_line_cloud()

    def n_cylinders(overlap):
        g = LinearRegionGrower(
            points, mode=AXIS_TRACE, overlap=overlap,
            ransac_threshold=0.05, cylinder_radius=0.1, cylinder_length=1.0,
            min_points=3, max_angle_deg=20.0,
        )
        g.grow(line_idx[:12])
        return len(g.debug_cylinders)

    n0, n_half = n_cylinders(0.0), n_cylinders(0.5)
    print(f"overlap 0.0 -> {n0} cylinders, 0.5 -> {n_half} cylinders")
    assert n_half > n0, f"overlap=0.5 ({n_half}) should yield more cylinders than 0.0 ({n0})"


def _two_lines_cloud():
    """Two parallel straight lines along +x, 5 m apart, with small noise."""
    rng = np.random.default_rng(5)
    n = 300
    x = np.linspace(0, 20, n)
    line_a = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)      # y = 0
    line_b = np.stack([x, np.full(n, 5.0), np.zeros(n)], axis=1)  # y = 5
    for ln in (line_a, line_b):
        ln += rng.normal(0, 0.005, ln.shape)
    points = np.vstack([line_a, line_b]).astype(np.float64)
    return points, np.arange(n), np.arange(n, 2 * n)


def test_grow_lines_keeps_separate_lines():
    """Two seed groups on two distinct lines must stay two separate lines."""
    points, a_idx, b_idx = _two_lines_cloud()
    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.1, cylinder_length=1.0, min_points=3, max_angle_deg=20.0,
    )
    lines = grower.grow_lines([a_idx[:8], b_idx[:8]])
    assert len(lines) == 2, f"expected 2 separate lines, got {len(lines)}"
    for line in lines:
        assert line.centerline is not None and len(line.centerline) >= 2
    print("grow_lines: kept 2 separate lines")


def test_grow_lines_parallel_offset_not_merged():
    """Two parallel lines close together but laterally offset must NOT be joined
    into one — this is the zig-zag centerline bug. Growth from one line's cluster
    never reaches the other (their offset exceeds the cylinder radius), so the
    greedy consume loop leaves them as two distinct features."""
    rng = np.random.default_rng(7)
    n = 300
    z = np.linspace(0, 20, n)
    line_a = np.stack([np.zeros(n), np.zeros(n), z], axis=1)      # x = 0
    line_b = np.stack([np.full(n, 0.5), np.zeros(n), z], axis=1)  # x = 0.5, parallel
    for ln in (line_a, line_b):
        ln += rng.normal(0, 0.003, ln.shape)
    points = np.vstack([line_a, line_b]).astype(np.float64)
    a_idx, b_idx = np.arange(n), np.arange(n, 2 * n)

    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.1, cylinder_length=1.0, min_points=3, max_angle_deg=20.0,
    )
    lines = grower.grow_lines([a_idx[:8], b_idx[:8]])
    assert len(lines) == 2, f"parallel-offset lines must stay separate, got {len(lines)}"
    print("grow_lines: parallel-offset lines kept separate (no zig-zag)")


def test_grow_lines_joins_split_groups():
    """One physical line seeded by two disjoint groups (a gap in the picks)
    must come back as ONE joined line with a single continuous centerline."""
    points, line_idx = _straight_line_cloud()
    left, right = line_idx[:8], line_idx[-8:]  # same line, far apart along it
    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.1, cylinder_length=1.0, min_points=3, max_angle_deg=20.0,
    )
    lines = grower.grow_lines([left, right])
    assert len(lines) == 1, f"expected the two groups joined into 1, got {len(lines)}"
    cl = lines[0].centerline
    assert cl is not None and len(cl) >= 2
    span_x = float(cl[:, 0].max() - cl[:, 0].min())
    assert span_x > 15.0, f"joined centerline spans only {span_x:.1f} m"
    print(f"grow_lines: joined 2 split groups into 1 line "
          f"(centerline spans {span_x:.1f} m)")


def test_debug_vector_features_built():
    points, line_idx = _straight_line_cloud()
    g = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.1, cylinder_length=1.0, min_points=3,
    )
    g.grow(line_idx[:12])

    feats = debug_vector_features(g, show_cylinders=True, show_lines=True)
    names = [n for n, _ in feats]
    assert names == ["search_cylinders", "centerlines"], names
    for _, vf in feats:
        assert vf.geometry_type == "mesh"
        assert vf.geometry["vertices"].shape[1] == 3
        assert vf.geometry["edges"].shape[1] == 2
    assert debug_vector_features(g, False, False) == []
    print(f"debug_vector_features: built {names}")


if __name__ == "__main__":
    test_axis_trace_collects_line()
    test_axis_trace_long_curved_seeds()
    test_axis_trace_cylinders_match_search()
    test_linearity_connected_stays_on_feature()
    test_linearity_mode_requires_linearity()
    test_overlap_increases_cylinder_count()
    test_grow_lines_keeps_separate_lines()
    test_grow_lines_parallel_offset_not_merged()
    test_grow_lines_joins_split_groups()
    test_debug_vector_features_built()
    print("\nAll linear_region_grower tests passed.")
