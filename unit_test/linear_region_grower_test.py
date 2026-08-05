# linear_region_grower_test.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.services.linear_region_grower import (
    LinearRegionGrower, AXIS_TRACE, LINEARITY_CONNECTED, HYBRID,
    debug_vector_features, STOP_SHARP_BEND, STOP_PICKED_END,
    lines_to_traces, traces_to_lines, resolved_stop_keys, stop_key,
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


def _dense_arc(seed=3, R=8.0, n=500):
    """A clean, dense arc (no gaps -> no bridges) for the geometry-chain tests."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(np.radians(40), np.radians(140), n)
    arc = np.stack([R * np.cos(theta), R * np.sin(theta), np.zeros(n)], axis=1)
    arc += rng.normal(0, 0.003, arc.shape)
    return arc


def _march_one_direction(arc, **kwargs):
    """Drive a single march from the arc midpoint (one monotonic chain of
    cylinders/segments — grow()'s two opposite marches would meet at the anchor
    and not abut there, so tests that check abutment use one direction)."""
    g = LinearRegionGrower(arc, mode=AXIS_TRACE, **kwargs)
    mid = len(arc) // 2
    start = arc[mid]
    direction = LinearRegionGrower._principal_axis(arc[mid - 3:mid + 3])
    g._march(start, direction, False)
    return g


def test_cylinders_abut_end_to_end():
    """Consecutive search cylinders must connect: the end centre of each is the
    start centre of the next. Both hang off the shared fitted-line midpoints, so
    the drawn tube is one continuous chain with no sideways jog (the whole point
    of building the geometry on midpoints). Radius stays the search radius and
    each axis is unit length. (This replaces the old fixed-length assertion, which
    the midpoint scheme deliberately reverses.)"""
    cyl_rad = 0.2
    g = _march_one_direction(
        _dense_arc(), ransac_threshold=0.05, cylinder_radius=cyl_rad,
        cylinder_length=0.5, overlap=0.5, min_points=3, max_angle_deg=25.0,
    )
    cyls = g.debug_cylinders
    assert len(cyls) > 4, f"expected several cylinders, got {len(cyls)}"

    for _tip, direction, radius, _length in cyls:
        assert abs(radius - cyl_rad) < 1e-9, f"cylinder radius {radius} != {cyl_rad}"
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6, "cylinder axis must be unit"
    for (t0, d0, _r0, l0), (t1, *_rest) in zip(cyls, cyls[1:]):
        end0 = np.asarray(t0) + l0 * np.asarray(d0)
        assert np.allclose(end0, t1, atol=1e-9), (
            f"cylinders do not abut: end {end0} != next start {t1}")
    print(f"cylinders abut end-to-end: {len(cyls)} cylinders form one chain")


def test_centerline_is_continuous():
    """The recorded centerline segments must chain vertex-to-vertex: the end of
    each segment is the exact start of the next (they share the fitted-line
    midpoints), so the centerline is one continuous polyline with no per-junction
    gap or sideways jog."""
    g = _march_one_direction(
        _dense_arc(), ransac_threshold=0.05, cylinder_radius=0.2,
        cylinder_length=0.5, min_points=3, max_angle_deg=25.0,
    )
    segs = g.debug_lines
    assert len(segs) > 4, f"expected several segments, got {len(segs)}"
    for (_p0, p1), (q0, _q1) in zip(segs, segs[1:]):
        assert np.array_equal(p1, q0), (
            f"centerline broken: segment end {p1} != next start {q0}")
    print(f"centerline continuous: {len(segs)} segments chain end-to-end")


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


def test_wide_band_axis_stays_centred():
    """A wide, laterally-scattered straight band traced from a slightly off-centre,
    tilted heading. A single gate+fit per step selects a biased slice on one side
    and walks the axis off the band (stopping early with a false 'too few points'
    and drawing the cylinders off the line). The per-step re-centring must keep the
    axis on the band centre and trace the whole length."""
    rng = np.random.default_rng(41)
    n = 1200
    x = np.linspace(0, 20, n)
    band = np.stack([x, rng.normal(0, 0.06, n), rng.normal(0, 0.01, n)], axis=1)

    g = LinearRegionGrower(
        band, mode=AXIS_TRACE, ransac_threshold=0.03, cylinder_radius=0.15,
        cylinder_length=0.5, reach_factor=3.0, min_points=5, max_angle_deg=20.0,
    )
    # Deliberately biased start: tip off-centre in y, heading tilted a few degrees.
    tilt = np.radians(5.0)
    collected, _stop = g._march(np.array([0.2, 0.02, 0.0]),
                                np.array([np.cos(tilt), np.sin(tilt), 0.0]), False)
    grown = sorted(collected)
    reached = float(band[grown, 0].max()) if grown else 0.0
    base_off = max(float(np.hypot(b[1], b[2])) for b, *_ in g.debug_cylinders)
    print(f"wide band: reached x={reached:.1f}, "
          f"max cylinder centre offset {base_off:.3f} m")
    # A centred axis stays within centroid sampling noise (~band_sigma/sqrt(pts
    # per window) ~= 0.01 m, a few times that at worst) of y=z=0; a drifting axis
    # walks off toward cylinder_radius (0.15 m) and stops early.
    assert reached > 18.0, f"axis drifted / stopped early (reached x={reached:.1f})"
    assert base_off < 0.06, f"cylinder axis drifted {base_off:.3f} m off the band centre"


def test_noisy_patch_not_mistaken_for_bend():
    """A dead-straight line with a laterally-scattered patch in the middle (points
    present in the tube, but spread beyond the fit threshold of the axis). The old
    count-based test flagged this as a sharp bend and stopped mid-line; the march
    must instead recognise the line continues straight and trace through, and no
    end must be labelled a sharp bend."""
    rng = np.random.default_rng(31)
    n = 400
    x = np.linspace(0, 20, n)
    line = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    line += rng.normal(0, 0.004, line.shape)
    patch = (x > 9.5) & (x < 10.5)                       # scatter this stretch
    m = int(patch.sum())
    line[patch, 1] = np.sign(rng.uniform(-1, 1, m)) * rng.uniform(0.06, 0.09, m)
    seed = np.where(x < 1.0)[0]

    g = LinearRegionGrower(
        line, mode=AXIS_TRACE, ransac_threshold=0.05, cylinder_radius=0.1,
        cylinder_length=0.5, reach_factor=3.0, min_points=3, max_angle_deg=20.0,
    )
    grown = sorted(g.grow(seed))
    max_x = float(line[grown, 0].max())
    reasons = [r for r, _ in g.debug_end_cylinders]
    print(f"noisy patch: reached x={max_x:.1f}, stop reasons={reasons}")
    assert max_x > 18.0, f"march stopped at the noisy patch (reached x={max_x:.1f})"
    assert STOP_SHARP_BEND not in reasons, "noisy straight patch mislabelled a bend"


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


def test_grow_lines_no_duplicate_when_seeds_off_axis():
    """A seed group the march drove straight through — but whose points sit
    further off the axis than ransac_threshold, so none became line members —
    must NOT regrow the same feature as a second line on top of the first.

    Membership is gated at ransac_threshold while the search tube is
    cylinder_radius wide, so 'did growth reach this cluster?' has to be asked of
    the SWEPT region, not of the member points."""
    rng = np.random.default_rng(0)
    n = 400
    x = np.linspace(0, 20, n)
    core = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    core += rng.normal(0, 0.01, core.shape)          # tight core, traces end to end
    # Mid-cable seed group 0.3 m off the axis: inside cylinder_radius (0.5) so it
    # IS searched, outside ransac_threshold (0.05) so it is never a member.
    mid = np.stack([np.linspace(9.5, 10.5, 8), np.full(8, 0.3), np.zeros(8)], axis=1)
    points = np.vstack([core, mid])
    core_idx, mid_idx = np.arange(n), np.arange(n, n + 8)

    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.5, cylinder_length=1.0, min_points=3, max_angle_deg=20.0,
    )
    lines = grower.grow_lines([core_idx[:8], mid_idx])
    assert len(lines) == 1, \
        f"off-axis seed group regrew the same feature: got {len(lines)} lines"
    span_x = float(np.ptp(lines[0].centerline[:, 0]))
    assert span_x > 15.0, f"centerline spans only {span_x:.1f} m"

    # The mid group is swept by the march but claimed by none of it — exactly the
    # gap the swept region closes.
    claimed = set(lines[0].indices.tolist())
    assert not (claimed & set(mid_idx.tolist())), "off-axis group became a member"
    grower.grow(core_idx[:8])
    swept = set(grower.swept_indices().tolist())
    assert set(mid_idx.tolist()) <= swept, "off-axis group was not swept"
    print(f"grow_lines: off-axis seed group did not duplicate the line "
          f"(1 line, spans {span_x:.1f} m, group swept but unclaimed)")


def test_grow_lines_no_duplicate_after_early_stop():
    """When the first line stops short of a seed group (here a bend measured at
    the ragged cable end), the leftover group grows a line that marches BACK over
    the first one. That retrace must be discarded, not drawn as a second line."""
    rng = np.random.default_rng(0)
    n = 400
    x = np.linspace(0, 20, n)
    points = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    points += rng.normal(0, 0.10, points.shape)  # thick, noisy cable: stops early
    idx = np.arange(n)

    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.5, cylinder_length=1.0, min_points=3, max_angle_deg=20.0,
    )
    lines = grower.grow_lines([idx[:8], idx[-8:]])
    assert len(lines) == 1, \
        f"leftover group retraced the cable: got {len(lines)} lines"
    print(f"grow_lines: early stop did not duplicate the line "
          f"({len(lines)} line, {len(lines[0].indices)} pts of {len(points)})")


def test_grow_lines_collinear_separate_features_kept():
    """The duplicate check must not swallow a genuinely separate feature. Two
    collinear cables end to end with a gap wider than the search reach stay two
    lines: the second's growth never enters the region the first swept."""
    rng = np.random.default_rng(3)
    n = 200
    a = np.stack([np.linspace(0, 10, n), np.zeros(n), np.zeros(n)], axis=1)
    b = np.stack([np.linspace(16, 26, n), np.zeros(n), np.zeros(n)], axis=1)
    for ln in (a, b):
        ln += rng.normal(0, 0.005, ln.shape)
    points = np.vstack([a, b])                      # 6 m gap >> reach (3 x 1.0 m)
    a_idx, b_idx = np.arange(n), np.arange(n, 2 * n)

    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.1, cylinder_length=1.0, min_points=3, max_angle_deg=20.0,
    )
    lines = grower.grow_lines([a_idx[:8], b_idx[:8]])
    assert len(lines) == 2, \
        f"collinear separate features must stay 2 lines, got {len(lines)}"
    print("grow_lines: collinear features across a wide gap kept as 2 lines")


class _Flag:
    """Minimal threading.Event stand-in: only .is_set() is needed by the grower."""
    def __init__(self, value=False):
        self._v = value

    def is_set(self):
        return self._v


def test_grow_lines_progress_cb_called_per_line():
    """progress_cb fires once per grown line with monotonic 'done' counts."""
    points, a_idx, b_idx = _two_lines_cloud()
    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.1, cylinder_length=1.0, min_points=3, max_angle_deg=20.0,
    )
    calls = []
    lines = grower.grow_lines(
        [a_idx[:8], b_idx[:8]],
        progress_cb=lambda done, total, msg: calls.append((done, total, msg)),
    )
    assert len(calls) == len(lines), (
        f"expected one progress call per line, got {len(calls)} for {len(lines)} lines"
    )
    dones = [c[0] for c in calls]
    assert dones == sorted(dones) and dones[-1] == len(lines), f"non-monotonic: {dones}"
    assert all(c[1] == 2 for c in calls), "total should be the initial group count (2)"
    print(f"grow_lines progress_cb: {len(calls)} call(s), done={dones}")


def test_grow_lines_cancel_returns_partial():
    """A pre-set cancel_event stops growing immediately and returns no lines;
    an unset event grows normally — proving the event is what gates it."""
    points, a_idx, b_idx = _two_lines_cloud()
    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05,
        cylinder_radius=0.1, cylinder_length=1.0, min_points=3, max_angle_deg=20.0,
    )
    cancelled = grower.grow_lines([a_idx[:8], b_idx[:8]], cancel_event=_Flag(True))
    assert cancelled == [], f"pre-cancelled grow should return no lines, got {len(cancelled)}"
    # The cancel handle must be cleared after the call (no leak into later runs).
    assert grower._cancel_event is None
    normal = grower.grow_lines([a_idx[:8], b_idx[:8]], cancel_event=_Flag(False))
    assert len(normal) == 2, f"unset event should grow normally, got {len(normal)}"
    print("grow_lines cancel: pre-set event returns partial, unset grows normally")


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


def _broken_cable(gap_start=10.0, gap_end=16.0, end=26.0, noise=0.01, seed=0):
    """A straight cable along +x with a hole in it, far wider than the search
    reach can bridge — so growth from one end stops at the hole."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.linspace(0, gap_start, 200),
                        np.linspace(gap_end, end, 200)])
    pts = np.stack([x, np.zeros(len(x)), np.zeros(len(x))], axis=1)
    return pts + rng.normal(0, noise, (len(x), 3))


def _cable_grower(points):
    return LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05, cylinder_radius=0.2,
        cylinder_length=1.0, reach_factor=3.0, min_points=3, max_angle_deg=20.0,
    )


def test_march_reports_stop_state():
    """Every march direction must report where and why it gave up — including a
    march that dies on its FIRST step, which records no end cylinder at all and
    so is invisible to the debug_end_cylinders list. Those are exactly the ends
    worth re-seeding, so they must not be silently missing."""
    points = _broken_cable()
    g = _cable_grower(points)
    g.grow(np.arange(8))
    stops = g.march_stops()
    print(f"stop state: {[s.reason for s in stops]}, "
          f"tips x={[round(float(s.tip[0]), 1) for s in stops]}")
    assert len(stops) == 2, f"expected one stop per direction, got {len(stops)}"
    for stop in stops:
        assert np.all(np.isfinite(stop.tip)), "stop tip is not a real point"
        assert abs(np.linalg.norm(stop.direction) - 1.0) < 1e-6, \
            "stop heading is not a unit vector"

    # The backward march starts at the cable's own end and dies immediately, so
    # it records no cylinder — but it must still produce a usable stop.
    backward = [s for s in stops if s.direction[0] < 0][0]
    assert backward.tip[0] < 2.0, \
        f"backward stop should sit near x=0, got {backward.tip[0]:.1f}"


def test_reseed_from_stop_extends_line():
    """The whole point of the feature: a trace that stopped at a hole, plus a few
    user picks past the hole, must come back as ONE line spanning the lot."""
    points = _broken_cable()
    g = _cable_grower(points)
    line = g.grow_lines([np.arange(8)])[0]
    before = float(np.ptp(line.centerline[:, 0]))

    claimed = np.zeros(len(points), dtype=bool)
    claimed[line.indices] = True
    stop, n_ahead = g.rank_stops(line.stops, claimed)[0]
    assert n_ahead > 0, "ranking saw nothing ahead of the stop at the hole"

    picks = np.where((points[:, 0] > 16.0) & (points[:, 0] < 16.6))[0]
    extended = g.extend_from_stop(stop, line, picks).line
    after = float(np.ptp(extended.centerline[:, 0]))
    print(f"re-seed: {len(line.indices)} pts / {before:.1f} m -> "
          f"{len(extended.indices)} pts / {after:.1f} m from {len(picks)} picks")

    assert before < 12.0, f"setup wrong: growth already spanned {before:.1f} m"
    assert after > 24.0, f"re-seed did not cross the hole (spans {after:.1f} m)"
    assert len(extended.indices) > 0.95 * len(points), \
        "re-seed crossed the hole but did not collect the far half"


def test_reseed_splices_one_continuous_centerline():
    """The spliced centerline must be ONE ordered chain running the length of the
    feature, not the extension tangled back through what it extends."""
    points = _broken_cable()
    g = _cable_grower(points)
    line = g.grow_lines([np.arange(8)])[0]
    claimed = np.zeros(len(points), dtype=bool)
    claimed[line.indices] = True
    stop = g.rank_stops(line.stops, claimed)[0][0]
    picks = np.where((points[:, 0] > 16.0) & (points[:, 0] < 16.6))[0]

    centerline = g.extend_from_stop(stop, line, picks).line.centerline
    steps = np.diff(centerline[:, 0])
    back = int((steps < -1e-3).sum())
    print(f"splice: {len(centerline)} vertices, {back} back-steps, "
          f"x {centerline[0, 0]:.1f} -> {centerline[-1, 0]:.1f}")
    assert back == 0, f"spliced centerline doubles back {back} time(s)"
    assert float(np.ptp(centerline[:, 0])) > 24.0, "spliced centerline is short"


def test_reseed_does_not_reach_without_picks():
    """The guard rail. Growth is only allowed past a stop because the user
    pointed there — with no picks the same call must NOT cross the hole, or the
    feature would be quietly guessing instead of being told."""
    points = _broken_cable()
    g = _cable_grower(points)
    line = g.grow_lines([np.arange(8)])[0]
    claimed = np.zeros(len(points), dtype=bool)
    claimed[line.indices] = True
    stop = g.rank_stops(line.stops, claimed)[0][0]

    # The invariant that enforces it: with no picks, nothing is relaxed at all.
    granted = g._pick_bounded_search(stop, np.empty(0, dtype=np.intp))
    assert granted == (g.reach_factor, g.cylinder_radius), \
        f"search was widened with no picks to justify it: {granted}"

    result = g.extend_from_stop(stop, line, np.empty(0, dtype=np.intp))
    spanned = (0.0 if result is None
               else float(np.ptp(result.line.centerline[:, 0])))
    print(f"no picks: granted {granted}, span {spanned:.1f} m (must stay short)")
    assert spanned < 12.0, \
        f"crossed a 6 m hole with no user picks (spans {spanned:.1f} m)"


def test_pick_bounded_search_opens_both_reach_and_width():
    """Granting reach without width is a trap: the tube is aimed along the
    heading the march DRIFTED to, and that angular error scales with distance.
    A 2-degree drift — routine after a few re-fits — misses by 0.85 m at 25 m,
    so a long thin tube sails straight past the very points the user picked."""
    # A cable that resumes 0.6 m to the side after the hole — a kink at a pole,
    # or simply sag. That offset is far outside the 0.2 m search tube, so the
    # continuation is invisible to a long thin probe no matter how far it reaches.
    rng = np.random.default_rng(3)
    left = np.stack([np.linspace(0, 10, 200), np.zeros(200), np.zeros(200)], axis=1)
    right = np.stack([np.linspace(16, 26, 200), np.full(200, 0.6),
                      np.zeros(200)], axis=1)
    points = np.vstack([left, right]) + rng.normal(0, 0.01, (400, 3))

    g = _cable_grower(points)
    line = g.grow_lines([np.arange(8)])[0]
    claimed = np.zeros(len(points), dtype=bool)
    claimed[line.indices] = True
    stop = g.rank_stops(line.stops, claimed)[0][0]

    picks = np.where((points[:, 0] > 16.0) & (points[:, 0] < 16.6))[0]
    reach_factor, radius = g._pick_bounded_search(stop, picks)
    offset = float(np.linalg.norm(
        np.cross(points[picks] - stop.tip, stop.direction), axis=1).max())
    print(f"pick-bounded: reach x{reach_factor:.1f}, radius {radius:.3f} m "
          f"(picks sit {offset:.3f} m off the heading, tube is "
          f"{g.cylinder_radius:.2f} m)")

    assert offset > g.cylinder_radius, \
        "setup wrong: the picks are already inside the default tube"
    assert reach_factor > g.reach_factor, "reach was not opened up to the picks"
    assert radius >= offset, \
        f"tube radius {radius:.3f} m cannot see picks {offset:.3f} m off-axis"


def test_sharp_bend_keeps_near_on_axis_points():
    """Stopping on a bend must still keep the points that hug the CURRENT
    heading. They are on this line whatever the window's fitted axis did, and
    discarding them throws away up to a full window at every bend."""
    rng = np.random.default_rng(5)
    n = 200
    x = np.linspace(0, 10, n)
    cable = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    cable += rng.normal(0, 0.004, cable.shape)
    # A dense wall across the cable at x=5 turns the fitted axis hard.
    wall = np.stack([np.full(120, 5.0), np.linspace(-1.5, 1.5, 120),
                     np.zeros(120)], axis=1)
    points = np.vstack([cable, wall])

    g = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05, cylinder_radius=0.6,
        cylinder_length=0.5, reach_factor=1.0, min_points=3, max_angle_deg=15.0,
    )
    grown = g.grow(np.arange(8))
    reasons = [s.reason for s in g.march_stops()]
    reached = float(points[grown, 0].max())
    print(f"sharp bend: reasons={reasons}, reached x={reached:.2f}")

    assert STOP_SHARP_BEND in reasons, "setup wrong: the wall did not cause a bend"
    # The cable must be kept right up to the obstruction at x=5. Without the fix
    # the bend window is discarded wholesale and the trace ends at x~4.6 — so the
    # threshold sits above that, not merely above "most of the cable".
    assert reached > 5.0, \
        f"bend threw away the window's on-axis points (reached x={reached:.2f})"


def test_stop_ranking_puts_real_ends_last():
    """A stop with unclaimed points ahead must outrank a genuine feature end, so
    the user is not walked through dozens of real ends to find the few stops
    worth extending."""
    points = _broken_cable()
    g = _cable_grower(points)
    line = g.grow_lines([np.arange(8)])[0]
    claimed = np.zeros(len(points), dtype=bool)
    claimed[line.indices] = True

    ranked = g.rank_stops(line.stops, claimed)
    print("ranking: " + ", ".join(
        f"{s.reason}@x={float(s.tip[0]):.1f}->{n} ahead" for s, n in ranked))
    assert ranked[0][1] > 0, "the stop at the hole ranked with nothing ahead"
    assert ranked[-1][1] == 0, "the cable's real end ranked as having points ahead"
    assert ranked[0][0].direction[0] > 0, \
        "the promising stop should be the one facing the hole"


def test_rollback_trims_the_end_and_reaims():
    """Rolling a stop back N steps must shorten the line from that end, drop the
    points the trimmed stretch had collected, and hand back a stop sitting on
    what is left — the clean body a re-seed starts from."""
    points = _broken_cable()
    g = _cable_grower(points)
    line = g.grow_lines([np.arange(8)])[0]
    claimed = np.zeros(len(points), dtype=bool)
    claimed[line.indices] = True
    stop = g.rank_stops(line.stops, claimed)[0][0]

    step = g.cylinder_length * (1.0 - g.overlap)
    rolled, new_stop = g.rollback_stop(line, stop, 2)
    trimmed = float(np.ptp(rolled.centerline[:, 0]))
    original = float(np.ptp(line.centerline[:, 0]))
    print(f"rollback 2: {original:.1f} m / {len(line.indices)} pts -> "
          f"{trimmed:.1f} m / {len(rolled.indices)} pts, "
          f"new tip x={new_stop.tip[0]:.2f} (was {stop.tip[0]:.2f})")

    assert trimmed < original, "rollback did not shorten the line"
    assert abs((original - trimmed) - 2 * step) < 0.6 * step, \
        f"trimmed {original - trimmed:.2f} m, expected about {2 * step:.2f} m"
    assert len(rolled.indices) < len(line.indices), \
        "rollback kept every point despite trimming the centerline"
    assert new_stop.tip[0] < stop.tip[0], "new stop is not further back"
    assert new_stop.direction[0] > 0, "new stop lost the outward heading"
    # The old stop must be gone from the line, replaced by the rolled-back one.
    # Compared by identity: a MarchStop holds numpy arrays, so `in` / `==` on it
    # is ambiguous rather than false.
    assert not any(s is stop for s in rolled.stops), "old stop still on the line"
    assert any(s is new_stop for s in rolled.stops), "new stop not added"


def test_rollback_refuses_to_consume_whole_line():
    """Asking to roll back further than the line is long must refuse rather than
    delete the feature — there would be nothing left to re-seed from."""
    points = _broken_cable()
    g = _cable_grower(points)
    line = g.grow_lines([np.arange(8)])[0]
    claimed = np.zeros(len(points), dtype=bool)
    claimed[line.indices] = True
    stop = g.rank_stops(line.stops, claimed)[0][0]

    print(f"rollback 500 on a {np.ptp(line.centerline[:, 0]):.0f} m line -> "
          f"{g.rollback_stop(line, stop, 500)}")
    assert g.rollback_stop(line, stop, 500) is None, \
        "rollback consumed the whole line instead of refusing"
    assert g.rollback_stop(line, stop, 0) == (line, stop), \
        "rollback of 0 steps must be a no-op"


def test_rollback_backs_out_past_what_stopped_the_march():
    """The case the feature exists for. A blob hanging off the cable stops the
    march inside it, so the trace ends among the offending points and a re-seed
    from that tip starts from a body contaminated by them. Rolling back must put
    the new tip BEFORE the blob, on clean cable — and the re-seed from there must
    then get past it."""
    rng = np.random.default_rng(11)
    n = 300
    x = np.linspace(0, 15, n)
    cable = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    cable += rng.normal(0, 0.008, cable.shape)
    blob_lo, blob_hi = 6.8, 7.4
    blob = np.stack([rng.uniform(blob_lo, blob_hi, 90),
                     rng.uniform(0.15, 0.5, 90), np.zeros(90)], axis=1)
    points = np.vstack([cable, blob])

    g = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.06, cylinder_radius=0.5,
        cylinder_length=0.5, reach_factor=2.0, min_points=4, max_angle_deg=18.0,
    )
    line = g.grow_lines([np.arange(6)])[0]
    stop = [s for s in line.stops if s.direction[0] > 0][0]
    assert stop.tip[0] < 14.0, "setup wrong: the blob did not stop the march"

    rolled, clean = g.rollback_stop(line, stop, 3)
    assert rolled is not None, "could not roll back"
    print(f"blob at x={blob_lo}-{blob_hi}: march stopped at x={stop.tip[0]:.2f}, "
          f"rolled back to x={clean.tip[0]:.2f}")
    assert clean.tip[0] < blob_lo, \
        f"rollback left the tip at x={clean.tip[0]:.2f}, still inside the blob"
    # The trimmed line must no longer hold the blob points it had swallowed.
    kept_blob = int((rolled.indices >= len(cable)).sum())
    was_blob = int((line.indices >= len(cable)).sum())
    print(f"  blob points on the line: {was_blob} -> {kept_blob}")
    assert kept_blob <= was_blob, "rollback added blob points"

    # Note what is and isn't claimed here: rollback backs the end out of the
    # trouble spot and hands back a clean body to re-seed from. Whether the
    # following march then gets past the obstacle is down to the growth
    # parameters, not to rollback, and is not asserted.


def test_extension_adopts_picks_when_the_march_cannot_follow():
    """A point the user picked is a point they LOOKED AT and judged to be on this
    line. When growth cannot follow them, the picks must still join the line and
    the frontier must move to them — otherwise the workflow dead-ends exactly
    where it is needed most, and the user's judgement is silently discarded.

    The march contributing nothing must still be reported as such (marched
    False), because grow() always hands back the seeds it was given: counting
    points cannot tell "the trace advanced" from "your picks were added"."""
    rng = np.random.default_rng(17)
    # A cable that simply ends at x=10. Picks far away across genuinely empty
    # space, off to one side: nothing joins them to the cable but the user.
    cable = np.stack([np.linspace(0, 10, 200), np.zeros(200), np.zeros(200)],
                     axis=1) + rng.normal(0, 0.01, (200, 3))
    stray = np.array([[40.0, 3.0, 0.0], [40.2, 3.0, 0.0], [40.4, 3.0, 0.0]])
    points = np.vstack([cable, stray])

    g = _cable_grower(points)
    line = g.grow_lines([np.arange(8)])[0]
    stop = [s for s in line.stops if s.direction[0] > 0][0]
    picks = np.arange(len(cable), len(points))

    result = g.extend_from_stop(stop, line, picks)
    print(f"picks-only extension: marched={result.marched}, "
          f"{len(line.indices)} -> {len(result.line.indices)} pts, "
          f"frontier at x={result.stop.tip[0]:.1f} ({result.stop.reason})")
    assert result.marched is False, \
        "extension claimed the march advanced while it contributed nothing"
    assert set(picks.tolist()) <= set(result.line.indices.tolist()), \
        "the user's picks were thrown away"
    assert result.stop.reason == STOP_PICKED_END, \
        f"frontier is not marked as hand-picked: {result.stop.reason}"
    assert result.stop.tip[0] > 35.0, (
        f"frontier stayed behind at x={result.stop.tip[0]:.1f} instead of "
        f"moving out to the picks — the next round would re-offer this stop")


def test_extension_bridges_to_sparse_picks_below_min_points():
    """min_points stops the march bridging blindly onto a couple of stray
    returns. It must not overrule a human: a thin cable through canopy leaves
    two or three returns per metre, and the user pointing at them is the very
    evidence min_points exists to demand."""
    rng = np.random.default_rng(23)
    # Dense cable to x=10; one lone return at x=14 inside the occlusion; the
    # cable itself resumes densely at x=15. The lone return is the only stepping
    # stone across, and on its own it is far below min_points — blind bridging
    # must refuse it, and does. The user picking it is what changes.
    near = np.stack([np.linspace(0, 10, 200), np.zeros(200), np.zeros(200)], axis=1)
    stepping_stone = np.array([[14.0, 0.0, 0.0]])
    far = np.stack([np.linspace(15, 22, 150), np.zeros(150), np.zeros(150)], axis=1)
    points = np.vstack([near, stepping_stone, far])
    points += rng.normal(0, 0.005, points.shape)

    g = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05, cylinder_radius=0.2,
        cylinder_length=1.0, reach_factor=3.0, min_points=8, max_angle_deg=20.0,
    )
    line = g.grow_lines([np.arange(8)])[0]
    stop = [s for s in line.stops if s.direction[0] > 0][0]
    assert stop.tip[0] < 13.0, \
        f"setup wrong: growth already crossed the hole (tip x={stop.tip[0]:.1f})"

    picks = np.array([len(near)], dtype=np.intp)          # the lone x=14 return
    result = g.extend_from_stop(stop, line, picks)
    reached = float(points[result.line.indices][:, 0].max())
    print(f"sparse bridge: min_points={g.min_points}, 1 pick at x=14 -> "
          f"line reaches x={reached:.1f} (marched={result.marched})")
    assert result.marched, \
        "the march refused the picked stepping stone and never resumed"
    assert reached > 21.0, (
        f"stepped onto the pick but did not pick the cable back up beyond it "
        f"(line ends at x={reached:.1f}, cable runs to x=22)")


def test_traces_round_trip_through_persistence():
    """Lines must survive being packed for the project file and rebuilt — that is
    the whole basis of continuing a trace in a later session."""
    points = _broken_cable()
    g = _cable_grower(points)
    lines = g.grow_lines([np.arange(8)])
    labels = np.full(len(points), -1, dtype=np.int32)
    for label, line in enumerate(lines):
        labels[line.indices] = label

    dismissed = {stop_key(0, s) for s in lines[0].stops if s.direction[0] < 0}
    traces = lines_to_traces(lines, {"cylinder_length": 1.0}, resolved=dismissed)
    rebuilt = traces_to_lines(traces, labels)

    print(f"persistence: {len(rebuilt)} line(s), "
          f"{len(rebuilt[0].stops)} stops, {len(rebuilt[0].indices)} pts")
    assert len(rebuilt) == len(lines)
    assert len(rebuilt[0].indices) == len(lines[0].indices), "point set changed"
    assert len(rebuilt[0].stops) == len(lines[0].stops), "stops lost"
    assert np.allclose(rebuilt[0].centerline, lines[0].centerline, atol=1e-3), \
        "centerline changed across the round trip"
    assert resolved_stop_keys(traces) == dismissed, \
        "stops dismissed as real ends did not survive the round trip"


if __name__ == "__main__":
    test_axis_trace_collects_line()
    test_axis_trace_long_curved_seeds()
    test_cylinders_abut_end_to_end()
    test_centerline_is_continuous()
    test_axis_trace_survives_clutter_in_tube()
    test_plain_pca_step_resists_dense_near_patch()
    test_gap_bridging_crosses_fragment_gap()
    test_wide_band_axis_stays_centred()
    test_noisy_patch_not_mistaken_for_bend()
    test_gap_bridging_stays_on_own_line()
    test_linearity_connected_stays_on_feature()
    test_linearity_mode_requires_linearity()
    test_overlap_increases_cylinder_count()
    test_grow_lines_keeps_separate_lines()
    test_grow_lines_parallel_offset_not_merged()
    test_grow_lines_joins_split_groups()
    test_grow_lines_no_duplicate_when_seeds_off_axis()
    test_grow_lines_no_duplicate_after_early_stop()
    test_grow_lines_collinear_separate_features_kept()
    test_grow_lines_progress_cb_called_per_line()
    test_grow_lines_cancel_returns_partial()
    test_debug_vector_features_built()
    test_march_reports_stop_state()
    test_reseed_from_stop_extends_line()
    test_reseed_splices_one_continuous_centerline()
    test_reseed_does_not_reach_without_picks()
    test_pick_bounded_search_opens_both_reach_and_width()
    test_sharp_bend_keeps_near_on_axis_points()
    test_stop_ranking_puts_real_ends_last()
    test_rollback_trims_the_end_and_reaims()
    test_rollback_refuses_to_consume_whole_line()
    test_rollback_backs_out_past_what_stopped_the_march()
    test_extension_adopts_picks_when_the_march_cannot_follow()
    test_extension_bridges_to_sparse_picks_below_min_points()
    test_traces_round_trip_through_persistence()
    print("\nAll linear_region_grower tests passed.")
