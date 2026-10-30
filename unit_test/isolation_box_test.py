"""
Tests for the isolation box: core.services.isolation_box and the level-of-detail
fill in RenderingCoordinator.

The clip planes are what the graphics card uses to hide points; ``contains`` is
what picking, the lasso and the zoom window use to skip them. The first test
holds the two to the same answer, which is the whole promise: a point is
reported inside exactly when it is drawn.

Run:
    python unit_test/isolation_box_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.services.isolation_box import IsolationBox, CENTRE_HANDLE, MIN_HALF_EXTENT
from application.rendering_coordinator import RenderingCoordinator

_RNG = np.random.default_rng(7)


def _random_axes():
    q, _ = np.linalg.qr(_RNG.normal(size=(3, 3)))
    return q


def _cloud(n=200_000, extent=10.0):
    return (_RNG.random((n, 3)) * extent - extent / 2).astype(np.float32)


def test_planes_agree_with_contains():
    """Level and turned boxes: the six plane tests and contains() agree."""
    pts = _cloud()
    for axes in (np.eye(3), _random_axes()):
        box = IsolationBox([0.3, -0.2, 0.5], [1.5, 0.7, 2.2], axes)
        homo = np.c_[pts.astype(np.float64), np.ones(len(pts))]
        by_planes = np.all(homo @ box.planes().T >= 0, axis=1)
        assert np.array_equal(by_planes, box.contains(pts)), "planes and contains disagree"
        assert 0 < by_planes.sum() < len(pts)
    print("  planes agree with contains: OK")


def test_cube_and_contains_point():
    box = IsolationBox.cube([1, 2, 3], 3.0)
    assert np.allclose(box.half_extents, 1.5)
    assert box.contains_point([2.4, 2, 3])
    assert not box.contains_point([2.6, 2, 3])
    assert box.contains(np.empty((0, 3), np.float32)).shape == (0,)
    print("  cube and contains_point: OK")


def test_drag_face_even_and_one_sided():
    box = IsolationBox.cube([0, 0, 0], 2.0)
    box.drag_face(0, 0.5, one_sided=False)            # +X face out 0.5
    assert np.allclose(box.half_extents, 1.5), "box did not grow in every direction"
    assert np.allclose(box.centre, 0)

    box = IsolationBox([0, 0, 0], [1.0, 2.0, 0.5])    # keeps its shape
    box.drag_face(4, -0.25, one_sided=False)          # +Z face in: half size
    assert np.allclose(box.half_extents, [0.5, 1.0, 0.25])

    box = IsolationBox([0, 0, 0], [1.0, 1.0, 0.1])    # a slab keeps its thickness
    box.drag_face(0, 1.0, one_sided=False, scale_axes=(0, 1))
    assert np.allclose(box.half_extents, [2.0, 2.0, 0.1])

    box = IsolationBox.cube([0, 0, 0], 2.0)
    box.drag_face(0, -10.0, one_sided=False)          # cannot shrink to nothing
    assert np.all(box.half_extents >= MIN_HALF_EXTENT - 1e-12)

    box = IsolationBox.cube([0, 0, 0], 2.0)
    box.drag_face(0, 0.5, one_sided=True)             # only +X moves
    assert np.isclose(box.half_extents[0], 1.25)
    assert np.isclose(box.centre[0] - box.half_extents[0], -1.0), "opposite face moved"
    assert np.isclose(box.centre[0] + box.half_extents[0], 1.5)

    box = IsolationBox.cube([0, 0, 0], 2.0)
    box.drag_face(1, 0.4, one_sided=True)             # -X face out 0.4
    assert np.isclose(box.centre[0] + box.half_extents[0], 1.0), "opposite face moved"
    assert np.isclose(box.centre[0] - box.half_extents[0], -1.4)

    box = IsolationBox.cube([0, 0, 0], 2.0)
    box.drag_face(2, -10.0, one_sided=True)           # cannot turn inside out
    assert box.half_extents[2] >= MIN_HALF_EXTENT
    assert np.isclose(box.centre[2] - box.half_extents[2], -1.0)
    print("  face drags: OK")


def test_handles():
    box = IsolationBox([1, 1, 1], [1, 2, 3], _random_axes())
    h = box.handle_positions()
    assert h.shape == (7, 3) and np.allclose(h[CENTRE_HANDLE], box.centre)
    for k in range(6):
        assert box.contains_point(h[k] + 1e-9 * (box.centre - h[k]))
    assert box.edges().shape == (12, 2, 3)
    lengths = np.linalg.norm(box.edges()[:, 1] - box.edges()[:, 0], axis=1)
    assert np.allclose(sorted(lengths), sorted([2] * 4 + [4] * 4 + [6] * 4))
    print("  handles and edges: OK")


def test_slab_steps_cover_every_point():
    """Stepping a slab by half its thickness through a region misses nothing."""
    pts = _cloud(50_000, extent=4.0)
    view = _random_axes()
    slab = IsolationBox.cube([0, 0, 0], 4.0).slab(view, thickness=0.3)
    assert np.isclose(slab.half_extents[2], 0.15)
    assert np.allclose(slab.axes, view)

    # Start well behind the cloud and step to well past it.
    slab.step(2, -4.0)
    seen = np.zeros(len(pts), dtype=bool)
    for _ in range(int(8.0 / slab.half_extents[2]) + 2):
        seen |= slab.contains(pts)
        slab.step(2, slab.half_extents[2])
    # The slab is as wide as the box's largest side, so corners of the cube
    # can fall outside it sideways; within that width nothing is skipped.
    local = (pts - 0) @ view.T
    sideways_in = np.all(np.abs(local[:, :2]) <= 2.0, axis=1)
    assert seen[sideways_in].all(), "a slab step skipped points"
    print("  slab steps cover every point: OK")


def test_focus_indices():
    """Box rows always kept; the rest sampled; result sorted and unique."""
    inside = np.array([5, 9, 10, 400], dtype=np.int64)
    rows = RenderingCoordinator._focus_indices(100_000, inside, 0.05)
    assert np.all(np.diff(rows) > 0)
    assert np.isin(inside, rows).all()
    assert 4_000 < len(rows) < 6_000

    rows = RenderingCoordinator._focus_indices(100_000, inside, 0.0)
    assert np.array_equal(rows, inside)

    assert len(RenderingCoordinator._focus_indices(100_000, None, 0.0)) == 0
    print("  focus indices: OK")


class _Cloud:
    def __init__(self, pts):
        self.points = pts


class _Node:
    data_type = "point_cloud"


class _Nodes:
    def get_node(self, _uid):
        return _Node()


class _Recon:
    def __init__(self, clouds):
        self.clouds = clouds

    def reconstruct(self, uid):
        return _Cloud(self.clouds[uid])


def test_focus_rows_budget():
    """The box gets the budget first; over budget, the box itself is thinned."""
    import uuid
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    clouds = {a: _cloud(100_000), b: _cloud(100_000)}
    rc = RenderingCoordinator(_Nodes(), _Recon(clouds), cache_service=None)
    rc._total_visible_points = 200_000
    box = IsolationBox.cube([0, 0, 0], 4.0)          # ~6.4% of a 10 m cube
    expected = {u: np.flatnonzero(box.contains(p)) for u, p in clouds.items()}
    n_in = sum(len(r) for r in expected.values())

    rc._point_budget = 50_000
    rows = rc._focus_rows([a, b], box, sample_rate=0.25)
    assert not rc.focus_thinned and rc.focus_points_inside == n_in
    for u in (a, b):
        assert np.array_equal(rows[u][0], expected[u])
    outside_rate = rows[a][1]
    assert np.isclose(outside_rate, (50_000 - n_in) / 200_000)

    rc._point_budget = n_in // 2
    rows = rc._focus_rows([a, b], box, sample_rate=0.25)
    assert rc.focus_thinned
    kept = sum(len(r) for r, _ in rows.values())
    assert kept <= rc._point_budget and kept > 0.9 * rc._point_budget
    for u in (a, b):
        assert np.isin(rows[u][0], expected[u]).all()
    assert rows[a][1] < 0.001
    print("  focus rows within budget: OK")


def test_isolated_run_helpers():
    """Which menus are offered an isolated run; selection and clicks carried
    into the mask branch's rows."""
    from application.isolated_run import processes_points, child_selection, child_picks

    assert processes_points("020_Points/020_Clustering")
    assert processes_points("080_CAD/000_Boundary")
    assert processes_points("050_Processing")
    for menu in ("010_View", "040_Clusters", "030_Selection", "015_Branch", None, ""):
        assert not processes_points(menu), menu

    inside = np.array([False, True, True, False, True])
    parent = np.array([True, True, False, True, False])
    assert np.array_equal(child_selection(parent, inside), [True, False, False])
    assert child_selection(np.array([True, False, False, True, False]), inside) is None
    assert child_selection(None, inside) is None
    assert child_selection(parent[:3], inside) is None          # wrong cloud

    picks = [("a", 4), ("b", 1), ("a", 0), ("a", 2), ("a", 99)]
    assert child_picks(picks, "a", inside) == [2, 1]           # click order kept
    print("  isolated run helpers: OK")


if __name__ == "__main__":
    print("isolation_box_test")
    test_planes_agree_with_contains()
    test_cube_and_contains_point()
    test_drag_face_even_and_one_sided()
    test_handles()
    test_slab_steps_cover_every_point()
    test_focus_indices()
    test_focus_rows_budget()
    test_isolated_run_helpers()
    print("All isolation box tests passed.")
