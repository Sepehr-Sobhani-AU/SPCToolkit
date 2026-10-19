# kalman_line_tracer_test.py
#
# The Kalman Trace engine on made-up scenes where the true lines are known:
# it must measure a line's width from its own points (thin cable and hollow
# pipe alike), follow a line across a gap and through a clutter blob without
# jumping to a neighbour, trace a line once however many seed groups lie on
# it, end a line rather than reach across empty space onto an unrelated one,
# and never take points of an existing line. Uses the GPU shape query
# service, like the axis march.
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.services.kalman_line_tracer import KalmanLineTracer, line_profile

RNG = np.random.default_rng(3)


def _wire(x0, x1, y=0.0, z=0.0, per_m=40, sd=0.008, gaps=()):
    """Points along a straight line in x, with gaussian jitter across it."""
    x = np.sort(RNG.uniform(x0, x1, int((x1 - x0) * per_m)))
    for a, b in gaps:
        x = x[(x < a) | (x > b)]
    pts = np.stack([x, np.full_like(x, y), np.full_like(x, z)], 1)
    pts[:, 1:] += RNG.normal(0, sd, (len(x), 2))
    return pts


def _pipe(x0, x1, radius, per_m=60):
    """Points on the surface of a pipe along x (a hollow ring in section)."""
    x = np.sort(RNG.uniform(x0, x1, int((x1 - x0) * per_m)))
    a = RNG.uniform(0, 2 * np.pi, len(x))
    return np.stack([x, radius * np.cos(a), radius * np.sin(a)], 1) \
        + RNG.normal(0, 0.003, (len(x), 3))


def _seed(pts, x0, x1, offset=0):
    return offset + np.flatnonzero((pts[:, 0] >= x0) & (pts[:, 0] <= x1))


def _off_axis(line, y=0.0, z=0.0):
    c = np.asarray(line.centerline)
    return np.hypot(c[:, 1] - y, c[:, 2] - z)


def test_profile_measures_thin_and_hollow():
    thin = line_profile(_wire(0, 5, sd=0.01))
    pipe = line_profile(_pipe(0, 5, radius=0.10))
    # Solid line: radius ~ 2.5 sd of one axis covers ~96% of its points.
    assert 0.018 < thin["radius"] < 0.035, thin
    # Hollow pipe: its radius plus the jitter, not a thin line.
    assert 0.10 < pipe["radius"] < 0.13, pipe
    print(f"profile: thin {thin['radius'] * 100:.1f} cm, pipe {pipe['radius'] * 100:.1f} cm")


def test_follows_gap_and_clutter_not_neighbour():
    line = _wire(0, 60, gaps=[(25, 29)])
    neighbour = _wire(0, 60, y=0.5)
    blob = RNG.normal(0, 1.0, (3000, 3)) + [45, 0, 0]
    pts = np.vstack([line, neighbour, blob]).astype(np.float32)
    tracer = KalmanLineTracer(pts)
    lines = tracer.grow_lines([_seed(line, 0, 3)])
    tracer.release()
    assert len(lines) == 1, len(lines)
    c = lines[0].centerline
    assert c[:, 0].max() > 59 and c[:, 0].min() < 1, (c[:, 0].min(), c[:, 0].max())
    assert np.mean(_off_axis(lines[0]) < 0.15) > 0.97, "left the line"
    assert not np.isin(np.arange(len(line), len(line) + len(neighbour)),
                       lines[0].indices).any(), "took the neighbour's points"
    print(f"gap + clutter: one line {c[:, 0].min():.1f}..{c[:, 0].max():.1f} m, "
          f"neighbour untouched")


def test_one_line_from_many_seed_groups():
    line = _wire(0, 40)
    pts = line.astype(np.float32)
    groups = [_seed(line, a, a + 2) for a in (2, 12, 22, 33)]
    tracer = KalmanLineTracer(pts)
    lines = tracer.grow_lines(groups)
    tracer.release()
    assert len(lines) == 1, f"{len(lines)} lines - the same line drawn more than once"
    assert len(lines[0].indices) > 0.9 * len(line)
    print(f"4 seed groups on one line -> 1 line, {len(lines[0].indices)} of {len(line)} points")


def test_ends_instead_of_reaching_an_unrelated_line():
    line = _wire(0, 20)                                  # ends at x = 20
    across = _wire(-5, 5, per_m=40)                      # an unrelated line ...
    across = across[:, [1, 0, 2]] + [32, 0, 0.6]         # ... across its path, 12 m on
    pts = np.vstack([line, across]).astype(np.float32)
    tracer = KalmanLineTracer(pts)
    lines = tracer.grow_lines([_seed(line, 0, 3)])
    tracer.release()
    reached = lines[0].centerline[:, 0].max()
    assert reached < 21, f"reached x = {reached:.1f} - attached to the unrelated line"
    assert any(s.reason for s in lines[0].stops)
    print(f"ends at x = {reached:.1f} m, not on the unrelated line at x = 32")


def test_sparse_picks_in_tiny_groups_still_trace():
    # A sparse selection that grouping split into groups of 2-4 picks, each
    # too few to measure a line from: the picks around them are borrowed.
    line = _wire(0, 30, per_m=15)
    pts = line.astype(np.float32)
    picks = _seed(line, 2, 6)
    groups = [picks[i:i + 3] for i in range(0, len(picks), 3)]
    tracer = KalmanLineTracer(pts)
    lines = tracer.grow_lines(groups)
    tracer.release()
    assert len(lines) == 1, f"{len(lines)} lines from {len(groups)} tiny groups"
    c = lines[0].centerline
    assert c[:, 0].max() > 29 and c[:, 0].min() < 1, (c[:, 0].min(), c[:, 0].max())
    print(f"{len(groups)} groups of <= 3 picks -> 1 line, {c[:, 0].min():.1f}..{c[:, 0].max():.1f} m")


def test_starts_on_a_tight_curve():
    # A quarter circle of radius 4 m: started straight, a tracer loses it
    # within a window; the bend is measured from the seed.
    th = np.sort(RNG.uniform(0, np.pi / 2, 1500))
    arc = np.stack([4 * np.sin(th), 4 - 4 * np.cos(th), np.zeros_like(th)], 1) \
        + RNG.normal(0, 0.008, (len(th), 3))
    tracer = KalmanLineTracer(arc.astype(np.float32), min_bend_radius=2.0)
    lines = tracer.grow_lines([np.flatnonzero(th < 0.25)])
    tracer.release()
    end = np.linalg.norm(lines[0].centerline - [4, 4, 0], axis=1).min()
    assert end < 1.0, f"stopped {end:.1f} m short of the curve's end"
    print(f"tight curve R=4 m: traced to {end * 100:.0f} cm of its end")


def test_scattered_picks_on_lines_side_by_side():
    # Picks scattered along 5 lines 0.3 m apart, all in one pool: each line
    # traced once, as its own line - no line taking its neighbours' picks
    # (measuring the width from picks or from everything near them took in
    # the neighbours).
    lines, picks, which = [], [], []
    for i in range(5):
        pts = _wire(0, 40, y=0.3 * i, per_m=30)
        lines.append(pts)
    offset = 0
    for i, pts in enumerate(lines):
        for x0 in RNG.uniform(10, 28, 8):
            j = offset + int(np.argmin(np.abs(pts[:, 0] - x0)))
            picks.append(j); which.append(i)
        offset += len(pts)
    cloud = np.vstack(lines).astype(np.float32)
    picks, which = np.array(picks), np.array(which)
    tracer = KalmanLineTracer(cloud)
    traced = tracer.grow_lines([picks])
    tracer.release()
    for i in range(5):
        owners = {k for k, l in enumerate(traced) if np.isin(picks[which == i], l.indices).mean() > 0.5}
        assert len(owners) == 1, f"line {i}: picks on {len(owners)} traced lines"
        l = traced[owners.pop()]
        others = np.isin(picks[which != i], l.indices).sum()
        assert others == 0, f"line {i}'s trace took {others} picks of other lines"
    print(f"5 lines 0.3 m apart, 40 scattered picks in one pool -> each traced once, apart")


def test_existing_line_is_seen_not_taken():
    line = _wire(0, 40)
    pts = line.astype(np.float32)
    blocked = np.zeros(len(pts), dtype=bool)
    blocked[pts[:, 0] > 25] = True                       # already traced beyond 25 m
    tracer = KalmanLineTracer(pts)
    lines = tracer.grow_lines([_seed(line, 0, 3)], blocked=blocked)
    tracer.release()
    assert not blocked[lines[0].indices].any(), "took points of an existing line"
    assert lines[0].centerline[:, 0].max() < 27
    print(f"stops at the existing line (x = {lines[0].centerline[:, 0].max():.1f} m), "
          f"took none of its points")


if __name__ == "__main__":
    test_profile_measures_thin_and_hollow()
    test_follows_gap_and_clutter_not_neighbour()
    test_one_line_from_many_seed_groups()
    test_ends_instead_of_reaching_an_unrelated_line()
    test_sparse_picks_in_tiny_groups_still_trace()
    test_starts_on_a_tight_curve()
    test_scattered_picks_on_lines_side_by_side()
    test_existing_line_is_seen_not_taken()
    print("\nAll Kalman line tracer tests passed.")
