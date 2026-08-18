# selection_edge_cases_test.py
#
# Regression tests for four defects found reviewing the picking and lasso code.
# Each is a state that is easy to reintroduce because it only shows up after a
# re-render, a hidden branch, a thread interleaving, or one bad coordinate —
# never in a straightforward click-and-lasso run.
#
#   1. picked_points_indices outlive set_branches() on purpose, so after LOD
#      drops a level they address rows past the end of the shorter buffer.
#      Deselecting with a lasso indexed straight into it.
#   2. A lasso already in progress survives its branches being hidden, so
#      self.points is None by the time the polygon closes.
#   3. A pick grid is built on a background thread. Publishing it after
#      checking the branch still holds the same rows was two statements, so a
#      re-render in between could pair a grid with rows it does not describe.
#   4. One non-finite coordinate made the grid's bounding box NaN on that axis,
#      collapsing every cell on it. The plain scan this replaced shrugged NaN
#      off, so it was a regression.
#
# Qt is needed because the fixes live on the real widget, but nothing is shown.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import warnings

import numpy as np
from PyQt5.QtWidgets import QApplication

from config.config import global_variables
from core.services.point_grid import PointGrid
from gui.widgets.pcd_viewer.pcd_viewer_widget import PCDViewerWidget
from plugins.backends.selection_backends import CuPySelection, NumpySelection

_app = QApplication.instance() or QApplication(sys.argv)

_MV = np.eye(4)
_MV[3, :3] = [0.0, 0.0, -140.0]
_PROJ = np.zeros((4, 4))
_PROJ[0, 0], _PROJ[1, 1] = 1.06, 1.7
_PROJ[2, 2], _PROJ[2, 3] = -1.0002, -1.0
_PROJ[3, 2] = -1.0001

_FULL_SCREEN = [(0, 0), (1280, 0), (1280, 800), (0, 800)]


class _NoController:
    """The viewer asks the controller about branch selection and cluster locks.

    Answering "nothing selected, no nodes" leaves the selection filters as
    pass-through, so these tests measure the paths under test and not the
    filtering, which pick_focus_labels_test already covers.
    """

    selected_branches = []

    def get_node(self, uid):
        return None


def _cloud(n, seed=0):
    """An (n, 6) render slice: xyz spread over the view, rgb all white."""
    rng = np.random.default_rng(seed)
    slc = np.empty((n, 6), dtype=np.float32)
    slc[:, :3] = rng.uniform(-40, 40, (n, 3))
    slc[:, 3:] = 1.0
    return slc


def _viewer():
    global_variables.global_application_controller = _NoController()
    v = PCDViewerWidget()
    v.resize(1280, 800)
    v.model_view_matrix, v.projection_matrix = _MV, _PROJ
    v.viewport = (0, 0, 1280, 800)
    v.max_extent = 80.0
    v.center = np.array([0.0, 0.0, 0.0])
    return v


def _close_lasso(viewer, vertices, deselect=False):
    viewer._polygon_mode = True
    viewer._polygon_deselect_mode = deselect
    viewer._polygon_vertices = list(vertices)
    if deselect:
        viewer._close_polygon_and_deselect()
    else:
        viewer._close_polygon_and_select()


def test_deselect_after_the_buffer_shrinks():
    """A deselect lasso must survive picks that outlived their rows.

    Zooming out far enough makes AUTO-LOD hand the viewer a smaller slice while
    picked_points_indices still holds rows from the larger one. This used to
    raise IndexError the moment the deselect polygon closed.
    """
    v = _viewer()
    big, small = _cloud(50_000, seed=1), _cloud(10_000, seed=2)

    v.set_branches({"A": big}, ["A"])
    _close_lasso(v, _FULL_SCREEN)
    assert len(v.picked_points_indices) == 50_000, len(v.picked_points_indices)

    v.set_branches({"A": small}, ["A"])          # AUTO-LOD step-down
    assert max(v.picked_points_indices) >= len(v.points)

    _close_lasso(v, [(300, 200), (900, 200), (900, 600), (300, 600)], deselect=True)

    survivors = np.asarray(v.picked_points_indices, dtype=np.int64)
    assert survivors.size == 0 or survivors.max() < len(v.points), \
        "stale picks were kept after a deselect"
    assert not v._polygon_mode, "polygon mode was left on"
    print(f"  deselect after shrink: {survivors.size:,} picks kept, all in range")


def test_closing_a_lasso_with_nothing_visible():
    """Hiding every branch mid-lasso must not crash when the polygon closes.

    enter_polygon_mode() refuses to start without points, but it cannot stop the
    branches disappearing under a lasso that is already being drawn.
    """
    for deselect in (False, True):
        v = _viewer()
        v.set_branches({"A": _cloud(20_000, seed=3)}, ["A"])
        if deselect:
            _close_lasso(v, _FULL_SCREEN)        # something to deselect
        v.set_branches({}, [])                   # user hides everything
        assert v.points is None

        _close_lasso(v, [(100, 100), (900, 100), (900, 700)], deselect=deselect)
        assert not v._polygon_mode, "polygon mode was left on"
    print("  lasso closed with nothing visible: select and deselect both fine")


def test_a_stale_pick_grid_is_never_used():
    """A grid published for rows that are no longer drawn must be ignored.

    The build runs on a background thread. This drives the interleaving the
    identity check could not exclude — the grid finishes for the old slice
    *after* set_branches() has swapped in a new one — and asserts the pick falls
    back to the scan rather than indexing the wrong array.
    """
    v = _viewer()
    big, small = _cloud(50_000, seed=4), _cloud(10_000, seed=5)

    v.set_branches({"A": big}, ["A"])
    v._build_pick_grid("A", big)                 # grid for the 50,000 rows
    assert v._pick_grid_for("A") is not None

    v.set_branches({"A": small}, ["A"])          # rows replaced under it
    assert v._pick_grid_for("A") is None, "a grid for the old rows was handed out"

    # Worse case: the build finishes *after* the swap and publishes anyway.
    v._build_pick_grid("A", big)
    assert v._pick_grid_for("A") is None, "a late grid for the old rows was used"

    target = small[7, :3]
    got = v._nearest_point_within(target, v.max_extent)
    expected = _nearest_by_brute_force(small, target)
    assert got == expected, f"picked {got}, expected {expected}"
    print("  stale grid ignored; pick fell back to the scan and was correct")


def test_one_bad_coordinate_does_not_degrade_the_grid():
    """A NaN must not poison the bounding box or hide the nearest point.

    np.minimum spreads NaN across the whole axis, and np.argmin returns the
    index of a NaN rather than skipping it — so one bad point used to collapse
    the grid to a handful of cells and make nearby clicks find nothing.
    """
    clean = _cloud(20_000, seed=6)
    dirty = clean.copy()
    dirty[123, 0] = np.nan
    dirty[456, 2] = np.inf

    backends = [NumpySelection()]
    try:
        import cupy  # noqa: F401
        backends.append(CuPySelection())
    except Exception:
        pass

    rng = np.random.default_rng(11)
    for backend in backends:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            grid = PointGrid.build(dirty, backend=backend)

        cells = np.unique(grid.cell_ids).size
        assert cells == 242, f"{backend.name}: only {cells} of 242 cells used"

        for _ in range(60):
            target = (dirty[rng.integers(len(dirty)), :3] if rng.random() < 0.5
                      else rng.uniform(-40, 40, 3).astype(np.float32))
            if not np.isfinite(target).all():
                continue
            row, _sq = grid.nearest(dirty, target, max_dist=1e9)
            assert row == _nearest_by_brute_force(dirty, target), \
                f"{backend.name}: wrong point next to a bad coordinate"
        print(f"  one NaN + one inf, {backend.name}: {cells}/242 cells, picks correct")


def _nearest_by_brute_force(points, target):
    """Nearest row, with non-finite distances excluded — the scan's behaviour."""
    offset = np.asarray(points[:, :3], dtype=np.float32) - np.float32(target)
    sq = np.einsum('ij,ij->i', offset, offset)
    sq[~np.isfinite(sq)] = np.inf
    return int(np.argmin(sq))


if __name__ == "__main__":
    test_deselect_after_the_buffer_shrinks()
    test_closing_a_lasso_with_nothing_visible()
    test_a_stale_pick_grid_is_never_used()
    test_one_bad_coordinate_does_not_degrade_the_grid()
    print("\nAll selection edge-case tests passed.")
