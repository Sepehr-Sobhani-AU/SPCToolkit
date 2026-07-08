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


def test_axis_trace_survives_clutter_in_tube():
    """A clear straight line that the search tube shares with a patch of
    non-linear clutter in the middle. The clutter makes the line a MINORITY of
    the tube's points there — the old "line must be >=20% of the tube" ratio gate
    reported 'fit failed' and stopped the march mid-line, even though the line is
    obviously present. MSAC still locks onto the line (scattered clutter never
    accumulates line-inliers), so the march must cross the patch and finish."""
    rng = np.random.default_rng(11)
    n_line = 400
    x = np.linspace(0, 20, n_line)
    line = np.stack([x, np.zeros(n_line), np.zeros(n_line)], axis=1)
    line += rng.normal(0, 0.004, line.shape)

    # A dense blob straddling the line around x=10, offset 0.04-0.09 m sideways:
    # inside the 0.1 m search tube, but well outside the 0.03 m fit threshold, so
    # it is gathered by the tube yet never counts as a line inlier. ~95 clutter
    # points per 0.5 m step vs ~10 line points -> line is <15% of the tube.
    n_clut = 150
    cx = rng.uniform(9.6, 10.4, n_clut)
    r = rng.uniform(0.04, 0.09, n_clut)
    a = rng.uniform(0, 2 * np.pi, n_clut)
    clutter = np.stack([cx, r * np.cos(a), r * np.sin(a)], axis=1)
    points = np.vstack([line, clutter]).astype(np.float64)
    line_idx = np.arange(n_line)

    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.03,
        cylinder_radius=0.1, cylinder_length=0.5, overlap=0.0,
        min_points=3, max_angle_deg=20.0,
    )
    grown = set(grower.grow(line_idx[:10]).tolist())          # seed at the x=0 end
    line_set = set(line_idx.tolist())
    recovered = len(grown & line_set) / len(line_set)
    max_x = float(points[sorted(grown), 0].max())
    print(f"clutter-in-tube: recovered {recovered:.0%} of line, reached x={max_x:.1f}")
    assert recovered > 0.9, (
        f"march stopped early in clutter: recovered only {recovered:.0%} "
        "(the old inlier-ratio gate would false-fail here)")
    assert max_x > 18.0, f"march did not cross the clutter patch (reached x={max_x:.1f})"


def test_plain_pca_step_resists_dense_near_patch():
    """Documents why the per-step heading uses plain variance-weighted PCA (not a
    density-normalized fit). The feature runs +x; near the seed sits a dense,
    compact patch tilted 20 deg off-axis (a stub / bracket). Because PCA weights
    by spread, the compact patch has low leverage and the long +x run wins — the
    heading stays within a few degrees of +x. (Equal-per-section density
    normalization was measured to make this WORSE, re-inflating the patch.)"""
    rng = np.random.default_rng(21)
    far = np.stack([np.linspace(0.15, 0.60, 12),          # sparse true +x run
                    np.zeros(12), np.zeros(12)], axis=1)
    s = np.linspace(0.0, 0.15, 200)                        # dense tilted patch,
    tilt = np.radians(20.0)                                # 20 deg off +x
    patch = np.stack([s * np.cos(tilt), s * np.sin(tilt), np.zeros(200)], axis=1)
    patch += rng.normal(0, 0.002, patch.shape)
    pts = np.vstack([far, patch])

    d = LinearRegionGrower._principal_axis(pts)
    d = d / np.linalg.norm(d)
    off = np.degrees(np.arccos(min(1.0, abs(float(d @ np.array([1.0, 0.0, 0.0]))))))
    print(f"plain PCA step: heading {off:.1f} deg off +x despite the dense patch")
    assert off < 5.0, f"plain PCA heading {off:.1f} deg off +x — patch hijacked it"


def test_gap_bridging_crosses_fragment_gap():
    """A straight line with a 1 m gap in the middle. With reach > gap the march
    bridges it and reaches the far end; with reach = fit window it stops at the
    gap. Proves the fit-window / search-reach decoupling."""
    rng = np.random.default_rng(23)
    n = 400
    x = np.linspace(0, 20, n)
    line = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    line += rng.normal(0, 0.004, line.shape)
    pts = line[(x < 9.5) | (x > 10.5)]                   # 1.0 m gap in the middle
    seed = np.where(pts[:, 0] < 1.0)[0]                  # seeds at the x=0 end

    def run(reach_factor):
        g = LinearRegionGrower(
            pts, mode=AXIS_TRACE, ransac_threshold=0.05, cylinder_radius=0.1,
            cylinder_length=0.5, reach_factor=reach_factor, min_points=3,
            max_angle_deg=20.0,
        )
        grown = sorted(g.grow(seed))
        return float(pts[grown, 0].max())

    max_x3 = run(3.0)   # reach 1.5 m > 1.0 m gap  -> bridges
    max_x1 = run(1.0)   # reach 0.5 m < 1.0 m gap  -> stops
    print(f"gap bridging: reach x3 -> max_x {max_x3:.1f}; reach x1 -> max_x {max_x1:.1f}")
    assert max_x3 > 18.0, f"reach=3 failed to cross the gap (max_x {max_x3:.1f})"
    assert max_x1 < 10.5, f"reach=1 should stop at the gap but reached {max_x1:.1f}"


def test_gap_bridging_stays_on_own_line():
    """Two parallel lines 0.5 m apart, the near one broken by a gap. Bridging the
    gap must follow the SAME line longitudinally, never hop to the neighbour."""
    rng = np.random.default_rng(29)
    n = 300
    x = np.linspace(0, 20, n)
    a = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)       # line A, y = 0
    b = np.stack([x, np.full(n, 0.5), np.zeros(n)], axis=1)   # line B, y = 0.5
    for ln in (a, b):
        ln += rng.normal(0, 0.004, ln.shape)
    a = a[(x < 9.5) | (x > 10.5)]                             # gap in A only
    pts = np.vstack([a, b])
    n_a = len(a)
    a_seed = np.where((pts[:, 0] < 1.0) & (np.arange(len(pts)) < n_a))[0]
    b_seed = np.where((pts[:, 0] < 1.0) & (np.arange(len(pts)) >= n_a))[0]

    g = LinearRegionGrower(
        pts, mode=AXIS_TRACE, ransac_threshold=0.05, cylinder_radius=0.1,
        cylinder_length=0.5, reach_factor=3.0, min_points=3, max_angle_deg=20.0,
    )
    lines = g.grow_lines([a_seed, b_seed])
    assert len(lines) == 2, f"expected 2 lines, got {len(lines)}"
    for line in lines:
        ys = pts[line.indices, 1]
        assert ys.max() - ys.min() < 0.2, "a line absorbed the parallel neighbour"
    spans = [float(pts[line.indices, 0].max() - pts[line.indices, 0].min())
             for line in lines]
    assert max(spans) > 18.0, "the gapped line did not bridge across its gap"
    print("gap bridging: stayed on own line, bridged the gap")


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
    test_axis_trace_survives_clutter_in_tube()
    test_plain_pca_step_resists_dense_near_patch()
    test_gap_bridging_crosses_fragment_gap()
    test_gap_bridging_stays_on_own_line()
    test_linearity_connected_stays_on_feature()
    test_linearity_mode_requires_linearity()
    test_overlap_increases_cylinder_count()
    test_grow_lines_keeps_separate_lines()
    test_grow_lines_parallel_offset_not_merged()
    test_grow_lines_joins_split_groups()
    test_debug_vector_features_built()
    print("\nAll linear_region_grower tests passed.")
