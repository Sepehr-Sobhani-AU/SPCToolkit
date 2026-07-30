# contour_tracer_test.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading

import numpy as np
from core.services.contour_tracer import (
    ContourTracer,
    contours_to_vector_feature,
    suggest_spacing,
)


def _grid(extent=10.0, spacing=0.25, jitter=0.25, seed=0):
    """A flat scatter on z=0. Jittered off the lattice on purpose: a perfect grid
    is cocircular, so Delaunay picks its diagonals arbitrarily."""
    rng = np.random.default_rng(seed)
    steps = int(extent / spacing) + 1
    xs, ys = np.meshgrid(np.linspace(0, extent, steps), np.linspace(0, extent, steps))
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    pts[:, :2] += rng.uniform(-jitter * spacing, jitter * spacing, (len(pts), 2))
    return pts


def _report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail else ''}")
    return ok


def test_straight_contour():
    """Field = x on a plane: the level set at x=5 is a straight line along y."""
    pts = _grid()
    tracer = ContourTracer(pts, values=pts[:, 0], level=5.0,
                           proximity=0.8, max_triangle_edge=0.6)
    lines = tracer.trace(seed_point=np.array([5.0, 5.0, 0.0]))

    ok = _report("straight: one line", len(lines) == 1, f"got {len(lines)}")
    if not lines:
        return False
    line = lines[0]
    max_dev = float(np.abs(line[:, 0] - 5.0).max())
    ok &= _report("straight: vertices sit on x=5", max_dev < 1e-9,
                  f"max deviation {max_dev:.2e}")
    span = float(line[:, 1].max() - line[:, 1].min())
    ok &= _report("straight: spans the full cloud", span > 9.5, f"span {span:.2f}")
    return ok


def test_closed_loop():
    """Field = radius on a plane: the level set at r=3 is a circle, and it must
    close on itself without any distance tolerance."""
    pts = _grid()
    center = np.array([5.0, 5.0])
    radius = np.linalg.norm(pts[:, :2] - center, axis=1)
    tracer = ContourTracer(pts, values=radius, level=3.0,
                           proximity=0.8, max_triangle_edge=0.6)
    lines = tracer.trace(seed_point=np.array([8.0, 5.0, 0.0]))

    ok = _report("loop: one line", len(lines) == 1, f"got {len(lines)}")
    if not lines:
        return False
    line = lines[0]
    ok &= _report("loop: closes on itself", bool(np.allclose(line[0], line[-1])))
    r = np.linalg.norm(line[:, :2] - center, axis=1)
    ok &= _report("loop: vertices sit on r=3",
                  float(np.abs(r - 3.0).max()) < 0.01,
                  f"max radial error {float(np.abs(r - 3.0).max()):.4f}")
    return ok


def test_keeps_every_line_in_the_ball():
    """Field = |x-5| has TWO level sets at 1 (x=4 and x=6). With a ball wide
    enough to see both, the flood must keep both — not only the seeded one."""
    pts = _grid()
    values = np.abs(pts[:, 0] - 5.0)

    wide = ContourTracer(pts, values=values, level=1.0,
                         proximity=2.5, max_triangle_edge=0.6)
    both = wide.trace(seed_point=np.array([4.0, 5.0, 0.0]))
    ok = _report("keep-all: wide ball finds both lines", len(both) == 2,
                 f"got {len(both)}")
    if len(both) == 2:
        xs = sorted(float(np.median(line[:, 0])) for line in both)
        ok &= _report("keep-all: the two lines are x=4 and x=6",
                      abs(xs[0] - 4.0) < 0.01 and abs(xs[1] - 6.0) < 0.01,
                      f"got x={xs[0]:.3f} and x={xs[1]:.3f}")

    # A ball too narrow to ever see x=6 must not invent it.
    narrow = ContourTracer(pts, values=values, level=1.0,
                           proximity=0.8, max_triangle_edge=0.6)
    one = narrow.trace(seed_point=np.array([4.0, 5.0, 0.0]))
    ok &= _report("keep-all: narrow ball finds only the seeded line",
                  len(one) == 1, f"got {len(one)}")
    return ok


def test_contour_lands_on_a_curved_surface():
    """Contouring Z on a dome: vertices must be interpolated in 3-D and so land
    back on the surface, not on the flattened local plane."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(-4, 4, (12000, 2))
    z = 5.0 - 0.2 * (xy[:, 0] ** 2 + xy[:, 1] ** 2)
    pts = np.column_stack([xy, z])

    tracer = ContourTracer(pts, values=pts[:, 2], level=3.0,
                           proximity=0.9, max_triangle_edge=0.7)
    lines = tracer.trace(seed_point=np.array([np.sqrt(10.0), 0.0, 3.0]))

    ok = _report("dome: one closed contour",
                 len(lines) == 1 and bool(np.allclose(lines[0][0], lines[0][-1])),
                 f"got {len(lines)} line(s)")
    if not lines:
        return False
    line = lines[0]
    # z=3 on the dome means 0.2*r^2 = 2, i.e. r = sqrt(10).
    r = np.linalg.norm(line[:, :2], axis=1)
    ok &= _report("dome: contour follows r=sqrt(10)",
                  float(np.abs(r - np.sqrt(10.0)).max()) < 0.05,
                  f"max radial error {float(np.abs(r - np.sqrt(10.0)).max()):.4f}")
    on_surface = np.abs(line[:, 2] - 3.0).max()
    ok &= _report("dome: vertices sit at z=3", float(on_surface) < 1e-9,
                  f"max z error {float(on_surface):.2e}")
    return ok


def test_max_triangle_edge_stops_gap_crossing():
    """A gap in the cloud must break the contour, not be bridged by the fake
    surface Delaunay invents to fill its convex hull."""
    pts = _grid()
    pts = pts[(pts[:, 1] < 4.0) | (pts[:, 1] > 6.0)]  # cut a 2 m strip out

    # A 0.8 m ball cannot step over a 2 m hole, so only the seeded side is found —
    # and the contour must stop at the hole rather than be drawn across it.
    tight = ContourTracer(pts, values=pts[:, 0], level=5.0,
                          proximity=0.8, max_triangle_edge=0.6)
    stopped = tight.trace(seed_point=np.array([5.0, 2.0, 0.0]))
    ok = _report("gap: only the seeded side is traced", len(stopped) == 1,
                 f"got {len(stopped)}")
    if stopped:
        reach = float(max(line[:, 1].max() for line in stopped))
        ok &= _report("gap: nothing is drawn across the hole", reach < 4.1,
                      f"reached y={reach:.2f}")

    # Reaching across the hole on purpose must join both sides into one line.
    loose = ContourTracer(pts, values=pts[:, 0], level=5.0,
                          proximity=3.0, max_triangle_edge=2.5)
    joined = loose.trace(seed_point=np.array([5.0, 2.0, 0.0]))
    ok &= _report("gap: a wide edge cut bridges it deliberately", len(joined) == 1,
                  f"got {len(joined)}")
    if joined:
        span = float(joined[0][:, 1].max() - joined[0][:, 1].min())
        ok &= _report("gap: the bridged line spans both sides", span > 9.5,
                      f"span {span:.2f}")
    return ok


def test_overlapping_balls_do_not_double_the_line():
    """Balls overlap heavily by design. Edge-keyed crossings must collapse onto
    each other, so no vertex is drawn twice and the line has no junctions."""
    pts = _grid()
    tracer = ContourTracer(pts, values=pts[:, 0], level=5.0,
                           proximity=1.5, max_triangle_edge=0.6)
    lines = tracer.trace(seed_point=np.array([5.0, 5.0, 0.0]))

    degrees = [len(n) for n in tracer._adjacency.values()]
    ok = _report("dedup: still one line under heavy overlap", len(lines) == 1,
                 f"got {len(lines)}")
    ok &= _report("dedup: no vertex has more than two segments",
                  max(degrees) <= 2, f"max degree {max(degrees)}")
    ok &= _report("dedup: every vertex is unique",
                  len(tracer._crossings) == len(set(tracer._crossings)))
    return ok


def test_cancel_returns_partial():
    """Cancelling mid-flood returns what was traced so far, not nothing. Needs a
    line long enough to still be growing at the first progress report."""
    pts = _grid(extent=30.0)
    cancel = threading.Event()
    tracer = ContourTracer(pts, values=pts[:, 0], level=15.0,
                           proximity=0.8, max_triangle_edge=0.6)

    def _progress(done, total, message):
        cancel.set()  # stop at the first progress report

    lines = tracer.trace(seed_point=np.array([15.0, 15.0, 0.0]),
                         progress_cb=_progress, cancel_event=cancel)
    span = max((float(l[:, 1].max() - l[:, 1].min()) for l in lines), default=0.0)
    return _report("cancel: partial line returned",
                   len(lines) >= 1 and 0.0 < span < 30.0,
                   f"{len(lines)} line(s), span {span:.2f} of 30.00")


def test_vector_feature_packs_lines_without_bridging():
    """All contours go in ONE branch, each its own edge chain — no edge may bridge
    one line to the next."""
    line_a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    line_b = np.array([[0, 5, 0], [1, 5, 0]], dtype=np.float64)
    feature = contours_to_vector_feature([line_a, line_b])

    ok = _report("feature: built", feature is not None)
    if feature is None:
        return False
    ok &= _report("feature: is a mesh wireframe", feature.geometry_type == "mesh")
    edges = feature.geometry["edges"]
    ok &= _report("feature: edge count is per-line chains only",
                  len(edges) == (len(line_a) - 1) + (len(line_b) - 1),
                  f"got {len(edges)}")
    # The bridging edge would be (2, 3): last vertex of A to first of B.
    bridged = any(tuple(e) == (2, 3) for e in edges)
    ok &= _report("feature: no edge bridges the two lines", not bridged)
    ok &= _report("feature: empty result returns None",
                  contours_to_vector_feature([]) is None)
    return ok


def test_suggest_spacing():
    pts = _grid(spacing=0.25, jitter=0.0)
    spacing = suggest_spacing(pts, np.array([5.0, 5.0, 0.0]))
    return _report("suggest_spacing: recovers the grid spacing",
                   abs(spacing - 0.25) < 0.02, f"got {spacing:.4f}")


def test_rejects_bad_input():
    pts = _grid(extent=1.0)
    ok = True
    try:
        ContourTracer(pts, values=np.zeros(3), level=0.0,
                      proximity=0.5, max_triangle_edge=0.3)
        ok &= _report("guards: mismatched values rejected", False)
    except ValueError:
        ok &= _report("guards: mismatched values rejected", True)
    try:
        ContourTracer(pts, values=pts[:, 0], level=0.0,
                      proximity=0.0, max_triangle_edge=0.3)
        ok &= _report("guards: zero proximity rejected", False)
    except ValueError:
        ok &= _report("guards: zero proximity rejected", True)
    return ok


if __name__ == "__main__":
    tests = [
        ("Straight contour", test_straight_contour),
        ("Closed loop", test_closed_loop),
        ("Keeps every line in the ball", test_keeps_every_line_in_the_ball),
        ("Contour on a curved surface", test_contour_lands_on_a_curved_surface),
        ("Max triangle edge stops gap crossing", test_max_triangle_edge_stops_gap_crossing),
        ("Overlapping balls do not double the line", test_overlapping_balls_do_not_double_the_line),
        ("Cancel returns partial", test_cancel_returns_partial),
        ("VectorFeature packs lines without bridging", test_vector_feature_packs_lines_without_bridging),
        ("Suggest spacing", test_suggest_spacing),
        ("Rejects bad input", test_rejects_bad_input),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n{name}:")
        try:
            if not fn():
                failed.append(name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}/{len(tests)}): " + ", ".join(failed))
        sys.exit(1)
    print(f"All {len(tests)} contour tracer tests passed.")
