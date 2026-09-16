# selection_edge_cases_test.py
#
# Regression tests for four defects found reviewing the picking and lasso code.
# Each is a state that is easy to reintroduce because it only shows up after a
# re-render, a hidden branch, a thread interleaving, or one bad coordinate —
# never in a straightforward click-and-lasso run.
#
#   1. The selection is held in CLOUD space, one boolean mask per branch, so an
#      LOD change or a branch toggle cannot alter it — it used to be a list of
#      rendered rows that outlived the buffer they indexed, and a deselect lasso
#      indexed straight into the shorter one.
#   2. A lasso already in progress survives its branches being hidden, so
#      self.points is None by the time the polygon closes.
#   3. A coarse spatial index is built on a background thread. Publishing it after
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

import logging
import time
import warnings

import numpy as np
from PyQt5.QtWidgets import QApplication

from config.config import global_variables
from core.services.spatial_grid import SpatialGrid
from gui.widgets.pcd_viewer.pcd_viewer_widget import PCDViewerWidget
from plugins.backends.selection_backends import CuPySelection, NumpySelection
from plugins.backends.grid_backends import CuPyGrid, NumpyGrid

_app = QApplication.instance() or QApplication(sys.argv)

_MV = np.eye(4)
_MV[3, :3] = [0.0, 0.0, -140.0]
_PROJ = np.zeros((4, 4))
_PROJ[0, 0], _PROJ[1, 1] = 1.06, 1.7
_PROJ[2, 2], _PROJ[2, 3] = -1.0002, -1.0
_PROJ[3, 2] = -1.0001

_FULL_SCREEN = [(0, 0), (1280, 0), (1280, 800), (0, 800)]


# uid -> the branch's full-resolution points, for the stub controller to
# reconstruct. The selection lives in cloud space now, so the viewer needs a
# controller that can hand it a cloud.
_CLOUDS = {}


class _Cloud:
    def __init__(self, points):
        self.points = points


class _NoController:
    """The viewer asks the controller about branch selection and cluster locks.

    Answering "nothing selected, no nodes" leaves the selection filters as
    pass-through, so these tests measure the paths under test and not the
    filtering, which pick_focus_labels_test already covers.
    """

    selected_branches = []

    def get_node(self, uid):
        return None

    def reconstruct(self, uid):
        return _Cloud(_CLOUDS[uid])


def _cloud(n, seed=0):
    """An (n, 6) render slice: xyz spread over the view, rgb all white."""
    rng = np.random.default_rng(seed)
    slc = np.empty((n, 6), dtype=np.float32)
    slc[:, :3] = rng.uniform(-40, 40, (n, 3))
    slc[:, 3:] = 1.0
    return slc


def _tree():
    """A real TreeStructureWidget, installed as the global.

    The viewer keeps no selection masks of its own — they live on the branch's
    tree item — so every test that selects anything needs a tree with the branch
    in it, exactly as the running app has.
    """
    from gui.widgets.tree_structure_widget import TreeStructureWidget

    tree = global_variables.global_tree_structure_widget
    if tree is None:
        tree = TreeStructureWidget()
        global_variables.global_tree_structure_widget = tree
    return tree


def _show(viewer, uid, full, rendered_rows=None):
    """Draw branch *uid*, whose full cloud is *full*.

    Pass *rendered_rows* to simulate LOD drawing only a subset — which is what
    a step-down really is: the cloud is unchanged, only the slice handed to the
    viewer shrinks.
    """
    tree = _tree()
    if uid not in tree.branches_dict:
        tree.blockSignals(True)          # add_branch emits; nothing is listening
        tree.add_branch(uid, None, "branch", is_root=True)
        tree.blockSignals(False)
    # _apply_point_count would set this from the controller; the stub controller
    # here has no node table, so stand in for it.
    tree.branches_dict[uid].point_count = len(full)
    _CLOUDS[uid] = full[:, :3]
    if rendered_rows is None:
        viewer.set_branches({uid: full}, [uid])
    else:
        viewer.set_branches({uid: full[rendered_rows]}, [uid],
                            sample_indices_by_uid={uid: rendered_rows})


def _viewer():
    _CLOUDS.clear()
    global_variables.global_tree_structure_widget = None
    global_variables.global_application_controller = _NoController()
    v = PCDViewerWidget()
    v.resize(1280, 800)
    v.model_view_matrix, v.projection_matrix = _MV, _PROJ
    v.viewport = (0, 0, 1280, 800)
    v.max_extent = 80.0
    v.center = np.array([0.0, 0.0, 0.0])
    return v


def _centre_of(vertices):
    """A point comfortably inside the given polygon."""
    xs = [x for x, _y in vertices]
    ys = [y for _x, y in vertices]
    return _Pos(sum(xs) / len(xs), sum(ys) / len(ys))


class _Pos:
    """Stands in for a QPoint: _close_polygon only asks for x() and y()."""

    def __init__(self, x, y):
        self._x, self._y = x, y

    def x(self):
        return self._x

    def y(self):
        return self._y


def _close_lasso(viewer, vertices, deselect=False, at=None):
    """Close a lasso and wait for the masks it kicks off to be built.

    *at* is where the closing double-click lands; the centre of the shape by
    default, which is the "act on what it encloses" gesture.

    The build runs on a worker thread so the window never freezes on a large
    cloud; the app waits for it in ``MainWindow._when_selection_ready`` before
    launching a plugin, and the tests wait here for the same reason.
    """
    viewer._polygon_mode = True
    viewer._polygon_vertices = list(vertices)
    if at is None:
        at = _centre_of(vertices)
    if deselect:
        viewer._close_polygon_and_deselect(at)
    else:
        viewer._close_polygon_and_select(at)

    for _ in range(400):
        if viewer.selection_ready():
            return
        time.sleep(0.01)
    raise AssertionError("selection masks were never finished")


def test_a_selection_survives_an_lod_step_down():
    """An LOD change must not alter what is selected.

    Zooming out makes AUTO-LOD hand the viewer a smaller slice. The selection is
    held against the CLOUD, not the slice, so it is untouched — only the subset
    that can be *highlighted* shrinks. This used to be the other way round: the
    selection was a list of rendered rows, so a step-down left it naming rows
    past the end of the shorter buffer, and every path that indexed with it had
    to clamp. One did not, and raised IndexError the moment a deselect polygon
    closed.
    """
    v = _viewer()
    big = _cloud(50_000, seed=1)

    _show(v, "A", big)
    _close_lasso(v, _FULL_SCREEN)
    assert v.selection_count() == 50_000, v.selection_count()

    drawn = np.arange(0, 50_000, 5)              # AUTO-LOD step-down to 1 in 5
    _show(v, "A", big, rendered_rows=drawn)
    assert v.selection_count() == 50_000, \
        "the LOD step-down changed the selection"
    assert len(v._selection_draw_rows("A")) == len(drawn), \
        "the highlight should cover every drawn point of a fully selected cloud"

    _close_lasso(v, [(300, 200), (900, 200), (900, 600), (300, 600)], deselect=True)

    mask = v.selection_mask_for("A")
    assert mask is None or len(mask) == 50_000, \
        "the mask stopped describing the cloud"
    assert v.selection_count() < 50_000, "the deselect lasso removed nothing"
    assert not v._polygon_mode, "polygon mode was left on"
    print(f"  LOD step-down kept the selection; deselect left "
          f"{v.selection_count():,} of 50,000")


def test_closing_a_lasso_with_nothing_visible():
    """Hiding every branch mid-lasso must not crash when the polygon closes.

    enter_polygon_mode() refuses to start without points, but it cannot stop the
    branches disappearing under a lasso that is already being drawn.
    """
    for deselect in (False, True):
        v = _viewer()
        _show(v, "A", _cloud(20_000, seed=3))
        if deselect:
            _close_lasso(v, _FULL_SCREEN)        # something to deselect
        v.set_branches({}, [])                   # user hides everything
        assert v.points is None

        _close_lasso(v, [(100, 100), (900, 100), (900, 700)], deselect=deselect)
        assert not v._polygon_mode, "polygon mode was left on"
    print("  lasso closed with nothing visible: select and deselect both fine")


def test_a_stale_coarse_index_is_never_used():
    """A grid published for rows that are no longer drawn must be ignored.

    The build runs on a background thread. This drives the interleaving the
    identity check could not exclude — the grid finishes for the old slice
    *after* set_branches() has swapped in a new one — and asserts the pick falls
    back to the scan rather than indexing the wrong array.
    """
    v = _viewer()
    big, small = _cloud(50_000, seed=4), _cloud(10_000, seed=5)

    _show(v, "A", big)
    v._build_coarse_index("A", big)                 # grid for the 50,000 rows
    assert v._coarse_index_for("A") is not None

    _show(v, "A", small)                         # rows replaced under it
    assert v._coarse_index_for("A") is None, "a grid for the old rows was handed out"

    # Worse case: the build finishes *after* the swap and tries to publish. It
    # must not land at all — an entry for a branch that no longer draws those
    # rows is unreachable, so it would leak the grid and pin the whole slice.
    v._build_coarse_index("A", big)
    stored = v._coarse_indexes.get("A")
    assert stored is None or stored[0] is v._branch_vertices["A"], \
        "a grid for the old rows was published"

    # Whatever is handed out from here on must describe the rows now drawn,
    # whether that is a fresh grid or None while one builds.
    for _ in range(200):
        grid = v._coarse_index_for("A")
        if grid is not None:
            assert grid.n_points == len(small), \
                f"grid covers {grid.n_points} rows, buffer has {len(small)}"
            break
        time.sleep(0.05)

    target = small[7, :3]
    got = v._nearest_point_within(target, v.max_extent)
    expected = _nearest_by_brute_force(small, target)
    assert got == expected, f"picked {got}, expected {expected}"
    print("  stale grid never published; the pick was correct either way")


def test_a_failed_build_is_not_retried_every_click():
    """A build that raises must not respawn a thread on every click.

    Each attempt redoes the full O(N) numbering, so a cloud whose build runs out
    of memory would stack up a fresh multi-second thread every time the user
    clicks, forever.
    """
    v = _viewer()
    slc = _cloud(5_000, seed=8)
    _show(v, "A", slc)

    from core.services import spatial_grid
    original = spatial_grid.SpatialGrid.build
    attempts = []

    def exploding_build(*args, **kwargs):
        attempts.append(1)
        raise MemoryError("simulated")

    spatial_grid.SpatialGrid.build = staticmethod(exploding_build)
    # The viewer logs the failure with a traceback, which is correct — GPU and
    # build errors are reported, never swallowed. Quiet it here so the expected
    # traceback does not read like a test failure.
    grid_log = logging.getLogger("gui.widgets.pcd_viewer._data_management")
    previous_level = grid_log.level
    grid_log.setLevel(logging.CRITICAL)
    try:
        for _ in range(10):
            v._nearest_point_within(slc[0, :3], v.max_extent)
            time.sleep(0.05)
    finally:
        spatial_grid.SpatialGrid.build = original
        grid_log.setLevel(previous_level)

    assert len(attempts) == 1, f"{len(attempts)} build attempts over 10 clicks"
    assert "A" in v._coarse_index_failed

    # New rows must get a fresh attempt.
    v.set_branches({"A": _cloud(5_000, seed=9)}, ["A"])
    assert "A" not in v._coarse_index_failed, "the failure outlived the rows it happened for"
    print(f"  failed build attempted {len(attempts)}x over 10 clicks, reset on re-render")


def test_every_visible_branch_starts_building_on_one_click():
    """One click must start every branch's build, not one per click.

    Returning as soon as a branch was missing meant N visible branches needed N
    clicks — each a full scan of the whole combined buffer — before the grid
    path engaged at all.
    """
    v = _viewer()
    branches = {f"B{i}": _cloud(20_000, seed=20 + i) for i in range(4)}
    v.set_branches(branches, list(branches))

    v._nearest_via_coarse_indexes(branches["B0"][0, :3], v.max_extent)   # one click
    for _ in range(200):
        if all(u in v._coarse_indexes for u in branches):
            break
        time.sleep(0.05)

    ready = sum(1 for u in branches if u in v._coarse_indexes)
    assert ready == len(branches), f"only {ready}/{len(branches)} grids built after one click"

    got, _ = v._nearest_via_coarse_indexes(branches["B0"][0, :3], v.max_extent)
    assert got is True, "the grid path did not engage once every grid was ready"
    print(f"  one click started all {len(branches)} builds; grid path engaged next click")


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

    backends = [NumpyGrid()]
    try:
        import cupy  # noqa: F401
        backends.append(CuPyGrid())
    except Exception:
        pass

    rng = np.random.default_rng(11)
    for backend in backends:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            grid = SpatialGrid.build_coarse_spatial_index(dirty, backend=backend)

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


def test_deselect_cluster_after_an_lod_step_down():
    """Ctrl+Shift+Right must be unaffected by an LOD step-down.

    Sibling of test_a_selection_survives_an_lod_step_down. This path used to
    index the render buffer with picks that outlived it and raised IndexError;
    it now works in cloud rows, where the question does not arise. The branch
    here carries no cluster labels, so the click finds no cluster and the
    selection must simply be left alone.
    """
    v = _viewer()
    big = _cloud(50_000, seed=10)
    _show(v, "A", big)
    _close_lasso(v, _FULL_SCREEN)

    drawn = np.arange(0, 50_000, 5)
    _show(v, "A", big, rendered_rows=drawn)

    # Stand in for the depth-buffer unprojection, which needs a live GL context.
    v._unproject_mouse_to_nearest_point = lambda _pos: (0, v.points[0, :3])
    v.deselect_cluster_at(None)

    assert v.selection_count() == 50_000, \
        "a click on a branch with no clusters changed the selection"
    print("  cluster deselect after an LOD step-down: no crash, nothing changed")


def test_the_selection_honours_the_viewer_filters():
    """What a plugin receives must match what the viewer highlighted.

    The filters are applied once, in cloud space, as the lasso closes. There is
    no second, ungated path left for a plugin to reach by accident — which is
    what the ``allowed=`` argument every caller had to remember to pass was
    patching over.
    """
    from core.entities.clusters import Clusters

    labels = np.array([0, 0, 0, -1, -1, 5])
    clusters = Clusters(labels=labels)
    clusters.locked_clusters = {5: {"select"}}

    class _Node:
        data_type = "cluster_labels"
        uid = "A"
        data = clusters

    class _Controller:
        selected_branches = ["A"]

        def get_node(self, uid):
            return _Node()

        def reconstruct(self, uid):
            return _Cloud(_CLOUDS[uid])

    global_variables.global_application_controller = _Controller()
    xyz = np.column_stack([np.arange(6), np.zeros(6), np.zeros(6)]).astype(np.float32)
    slc = np.hstack([xyz, np.ones((6, 3), dtype=np.float32)])

    _CLOUDS.clear()
    global_variables.global_tree_structure_widget = None
    v = PCDViewerWidget()
    v.resize(1280, 800)
    v.model_view_matrix, v.projection_matrix = _MV, _PROJ
    v.viewport = (0, 0, 1280, 800)
    v.max_extent = 10.0
    v.center = np.array([0.0, 0.0, 0.0])
    _show(v, "A", np.hstack([xyz, np.ones((6, 3), dtype=np.float32)]))
    _CLOUDS["A"] = xyz
    v.set_branches({"A": slc}, ["A"], sample_indices_by_uid={"A": np.arange(6)})

    _close_lasso(v, _FULL_SCREEN)

    selected = sorted(np.flatnonzero(v.selection_mask_for("A")).tolist())
    assert selected == [0, 1, 2], selected        # noise and locked refused

    highlighted = sorted(v._selection_draw_rows("A").tolist())
    assert highlighted == selected, \
        f"viewer shows {highlighted}, plugins would get {selected}"

    from application.selection_gate import selected_cloud_indices
    plugin_sees = selected_cloud_indices(v, "A", xyz)
    assert sorted(plugin_sees.tolist()) == selected, \
        f"plugin got {plugin_sees.tolist()}, viewer showed {selected}"
    print(f"  selection {selected} — noise and the locked cluster refused, and "
          f"the viewer, the mask and the plugin all agree")


def _same_colour_clusters_viewer():
    """Two clusters (labels 1 and 2) drawn in the same RGB, plus noise and a
    select-locked cluster, with LOD dropping every other cloud row."""
    from core.entities.clusters import Clusters

    labels = np.array([1, 1, 2, 2, 1, 2, -1, -1, 7, 7, 1, 2])
    clusters = Clusters(labels=labels)
    clusters.locked_clusters = {7: {"select"}}

    class _Node:
        data_type = "cluster_labels"
        uid = "A"
        data = clusters

    class _Controller:
        selected_branches = ["A"]

        def get_node(self, uid):
            return _Node()

        def reconstruct(self, uid):
            return _Cloud(_CLOUDS[uid])

    global_variables.global_application_controller = _Controller()
    kept = np.arange(0, 12, 2)                           # cloud rows 0,2,4,6,8,10
    full = np.column_stack([np.arange(12), np.zeros(12),
                            np.zeros(12)]).astype(np.float32)
    xyz = full[kept]
    slc = np.hstack([xyz, np.ones((6, 3), dtype=np.float32)])   # all white
    _CLOUDS.clear()
    global_variables.global_tree_structure_widget = None

    v = PCDViewerWidget()
    v.resize(1280, 800)
    v.model_view_matrix, v.projection_matrix = _MV, _PROJ
    v.viewport = (0, 0, 1280, 800)
    v.max_extent = 10.0
    v.center = np.array([0.0, 0.0, 0.0])
    _show(v, "A", np.hstack([full, np.ones((12, 3), dtype=np.float32)]))
    v.set_branches({"A": slc}, ["A"], sample_indices_by_uid={"A": kept})
    # Rendered rows -> labels: 0:1  1:2  2:1  3:-1  4:7  5:1
    return v


def _selected(v):
    mask = v.selection_mask_for("A")
    return [] if mask is None else sorted(np.flatnonzero(mask).tolist())


def test_cluster_select_and_deselect_match_by_label_not_colour():
    """Ctrl+Shift+Left/Right must act on the clicked cluster's label.

    Matching by RGB grabbed or dropped every cluster sharing the colour.

    It must also take the WHOLE cluster, not just the drawn part. Labels are
    [1,1,2,2,1,2,-1,-1,7,7,1,2] and LOD draws every second row, so cluster 1 is
    cloud rows 0, 1, 4, 10 while only 0, 4 and 10 are on screen. The widening
    used to happen separately inside each plugin, by convention; a plugin that
    read the geometric selection instead got just the drawn subset.
    """
    v = _same_colour_clusters_viewer()

    # Click rendered row 2 = cloud row 4, label 1 -> cloud rows 0, 1, 4, 10.
    v._unproject_mouse_to_nearest_point = lambda _pos: (2, v.points[2, :3])
    v.select_cluster_at(None)
    assert _selected(v) == [0, 1, 4, 10], _selected(v)
    assert 1 in _selected(v), "the undrawn point of the cluster was left out"

    v.select_cluster_at(None)
    assert _selected(v) == [0, 1, 4, 10], "re-select changed the selection"

    # Add label 2, then deselect label 1: label 2 must survive the shared colour.
    v._unproject_mouse_to_nearest_point = lambda _pos: (1, v.points[1, :3])
    v.select_cluster_at(None)
    assert _selected(v) == [0, 1, 2, 3, 4, 5, 10, 11], _selected(v)

    v._unproject_mouse_to_nearest_point = lambda _pos: (5, v.points[5, :3])
    v.deselect_cluster_at(None)
    assert _selected(v) == [2, 3, 5, 11], _selected(v)

    # Clicking noise or a locked cluster selects nothing.
    for row in (3, 4):
        v._unproject_mouse_to_nearest_point = lambda _pos, r=row: (r, v.points[r, :3])
        v.select_cluster_at(None)
    assert _selected(v) == [2, 3, 5, 11], _selected(v)
    print("  cluster select/deselect follow the label, cover undrawn points, "
          "and respect noise and locks")


def test_polygon_double_click_decides_select_or_deselect():
    """One polygon mode: left double-click selects, right double-click deselects.

    A single right-click must not close the polygon, and Shift+P no longer
    enters a separate deselect mode.
    """
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QKeyEvent, QMouseEvent

    def mouse(v, kind, button, x, y):
        event = QMouseEvent(kind, QPointF(x, y), button, button, Qt.NoModifier)
        if kind == QEvent.MouseButtonDblClick:
            v.mouseDoubleClickEvent(event)
        else:
            v.mousePressEvent(event)

    # A box over the left half of the view, so there is an inside and an
    # outside and the two can be told apart by the numbers.
    box = [(100, 100), (600, 100), (600, 700), (100, 700)]
    inside_pt, outside_pt = (350, 400), (1100, 400)

    def draw(v, close_button, close_at, vertices=box):
        v.enter_polygon_mode()
        for x, y in vertices:
            mouse(v, QEvent.MouseButtonPress, Qt.LeftButton, x, y)
        # A double-click arrives as a press, then the double-click event.
        mouse(v, QEvent.MouseButtonPress, close_button, *close_at)
        assert v._polygon_mode, "a single click closed the polygon"
        assert len(v._polygon_vertices) == len(vertices) + (
            1 if close_button == Qt.LeftButton else 0), \
            "the closing press did not behave as an ordinary click"
        mouse(v, QEvent.MouseButtonDblClick, close_button, *close_at)
        assert not v._polygon_mode, "double-click did not close the polygon"
        for _ in range(400):
            if v.selection_ready():
                break
            time.sleep(0.01)

    v = _viewer()
    _show(v, "A", _cloud(5_000, seed=20))

    # Closing INSIDE acts on what the shape encloses.
    draw(v, Qt.LeftButton, inside_pt)
    enclosed = v.selection_count()
    assert 0 < enclosed < 5_000, f"setup wrong: the box caught {enclosed:,}"

    draw(v, Qt.RightButton, inside_pt)
    assert v.selection_count() == 0, v.selection_count()

    # Closing OUTSIDE acts on everything the shape does not enclose.
    draw(v, Qt.LeftButton, outside_pt)
    assert v.selection_count() == 5_000 - enclosed, (
        f"closing outside selected {v.selection_count():,}, expected "
        f"{5_000 - enclosed:,}")

    draw(v, Qt.RightButton, outside_pt)
    assert v.selection_count() == 0, v.selection_count()

    v.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_P, Qt.ShiftModifier))
    assert not v._polygon_mode, "Shift+P still enters polygon mode"
    print(f"  closing inside took {enclosed:,}, closing outside took the other "
          f"{5_000 - enclosed:,}; right button removes either way")


def test_the_closing_click_is_not_part_of_the_shape():
    """The double-click ends the tracing; it does not add a corner.

    A double-click arrives as a press and then the double-click event, and that
    press goes through the ordinary add-a-vertex path. Left as it was, every
    lasso gained a stray vertex wherever the user happened to finish — and when
    they finished OUTSIDE the shape, which is now how you ask for the points
    around it, that vertex dragged the outline out to meet the cursor.
    """
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent

    def mouse(v, kind, x, y):
        event = QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, Qt.LeftButton,
                            Qt.NoModifier)
        if kind == QEvent.MouseButtonDblClick:
            v.mouseDoubleClickEvent(event)
        else:
            v.mousePressEvent(event)

    box = [(100, 100), (600, 100), (600, 700), (100, 700)]

    v = _viewer()
    _show(v, "A", _cloud(5_000, seed=21))

    # Trace the box, then finish far outside it.
    v.enter_polygon_mode()
    for x, y in box:
        mouse(v, QEvent.MouseButtonPress, x, y)
    mouse(v, QEvent.MouseButtonPress, 1200, 400)
    assert len(v._polygon_vertices) == 5, \
        "the closing press should add a vertex like any other click"

    traced = []
    real_close = v._close_polygon

    def capture(op, at=None):
        traced.append(list(v._polygon_vertices))
        return real_close(op, at)

    v._close_polygon = capture
    mouse(v, QEvent.MouseButtonDblClick, 1200, 400)

    assert traced and traced[0] == box, \
        f"the closing click stayed in the shape: {traced[0]}"
    print("  the closing double-click ends the tracing without adding a corner")




def test_the_same_lasso_selects_the_same_points_at_any_lod():
    """The point of holding the selection in cloud space.

    The same screen-space lasso, drawn once with the branch fully drawn and once
    with LOD showing a tenth of it, must select exactly the same cloud rows. It
    used to select only what was drawn, so the answer changed with the zoom
    level — and ``Separate Selected Points`` produced a subsample full of holes.
    """
    full = _cloud(20_000, seed=31)
    lasso = [(300, 150), (1000, 150), (1000, 650), (300, 650)]

    v = _viewer()
    _show(v, "A", full)
    _close_lasso(v, lasso)
    whole = np.flatnonzero(v.selection_mask_for("A"))

    v = _viewer()
    _show(v, "A", full, rendered_rows=np.arange(0, 20_000, 10))
    _close_lasso(v, lasso)
    lod = np.flatnonzero(v.selection_mask_for("A"))

    assert whole.size > 0, "setup wrong: the lasso caught nothing"
    assert np.array_equal(whole, lod), (
        f"LOD changed the selection: {whole.size:,} points drawn whole, "
        f"{lod.size:,} at 1-in-10")
    print(f"  same lasso, same {whole.size:,} points at full resolution and at "
          f"1-in-10 LOD")


def test_deselecting_does_not_collapse_the_selection():
    """Deselecting one thing must not silently shrink everything else.

    Every deselect path used to drop the stored polygons, which is what the
    full-resolution widening depended on. After one stray right-click a lasso
    that had handed a plugin a million points handed it the LOD subset instead —
    and nothing on screen changed, because the highlight only ever drew the
    rendered picks. The mask makes deselect an ordinary boolean subtraction.
    """
    full = _cloud(20_000, seed=32)
    v = _viewer()
    _show(v, "A", full, rendered_rows=np.arange(0, 20_000, 10))

    _close_lasso(v, _FULL_SCREEN)
    before = v.selection_count()
    assert before == 20_000, before          # the whole cloud, not the 2,000 drawn

    _close_lasso(v, [(0, 0), (640, 0), (640, 400), (0, 400)], deselect=True)
    after = v.selection_count()

    assert after < before, "the deselect lasso removed nothing"
    assert after > 0, "the deselect lasso removed everything"
    # The collapse this guards against replaced the full-resolution selection
    # with the rendered subset — 2,000 points, a tenth of the cloud.
    assert after > 2_000, (
        f"the selection collapsed to about the LOD subset: {before:,} -> {after:,}")
    print(f"  deselect removed {before - after:,} of {before:,} — no collapse")


def test_the_selection_lives_on_the_branch_in_the_tree():
    """A selection belongs to the branch, so it is held on the branch's tree item.

    Not on the DataNode, which is pickled whole into the project file — a
    selection is about this session, not about the data. Not on the viewer
    either, which discards and rebuilds branches on every LOD change, visibility
    toggle and cache toggle while the selection has to survive all of them.

    Living on the tree item also settles the lifetime question for free: remove
    the branch and its selection goes with it, with nothing else to remember.
    """
    v = _viewer()
    try:
        full = _cloud(5_000, seed=33)
        _show(v, "A", full)
        tree = global_variables.global_tree_structure_widget
        _close_lasso(v, _FULL_SCREEN)

        stored = tree.selection_mask("A")
        assert stored is not None, "the mask was not stored on the branch"
        assert int(stored.sum()) == 5_000, int(stored.sum())
        assert v.selection_count() == 5_000

        # An LOD change re-renders the branch. The selection is about the
        # branch, not the render, so it must not notice.
        _show(v, "A", full, rendered_rows=np.arange(0, 5_000, 10))
        assert v.selection_count() == 5_000, "a re-render lost the selection"

        tree.remove_branch("A")
        assert tree.selection_mask("A") is None, \
            "the selection outlived the branch it belonged to"
        assert v.selection_count() == 0
        print("  the mask lives on the tree item, survives a re-render, and is "
              "removed with the branch")
    finally:
        global_variables.global_tree_structure_widget = None


def test_a_cache_toggle_keeps_the_selection():
    """Unchecking Cache must not lose the user's selection.

    Cache invalidation says the cached *reconstruction* is stale, not that the
    branch's points changed: replaying the transformers yields the same points
    in the same order, so a mask built against them is still exactly right.
    Dropping it would mean a checkbox silently threw away the selection.

    This drives the real CacheService notification, so a listener added later
    that clears the selection fails here.
    """
    import uuid as _uuid

    from core.services.cache_service import CacheService

    uid = str(_uuid.uuid4())             # CacheService resolves uids as UUIDs
    v = _viewer()
    try:
        _show(v, uid, _cloud(5_000, seed=34))
        _close_lasso(v, _FULL_SCREEN)
        before = v.selection_count()
        assert before == 5_000

        class _Node:
            is_cached = True
            cached_point_cloud = object()
            cache_timestamp = 0.0
            parent_uid = None

        node = _Node()
        node.uid = _uuid.UUID(uid)

        class _Nodes:
            data_nodes = {_uuid.UUID(uid): node}

            def get_node(self, wanted):
                return self.data_nodes.get(wanted)

        cache = CacheService(_Nodes())
        global_variables.global_pcd_viewer_widget = v
        try:
            cache.invalidate(uid)            # exactly what unchecking Cache does
        finally:
            global_variables.global_pcd_viewer_widget = None

        assert v.selection_count() == before, (
            f"a cache toggle lost the selection: {before:,} -> "
            f"{v.selection_count():,}")
        print(f"  cache toggle kept all {before:,} selected points")
    finally:
        global_variables.global_tree_structure_widget = None


def test_click_picks_belong_to_the_branch_too():
    """Click order lives on the branches, not on the viewer.

    The mask says what is selected; the picks say which point was clicked first,
    which a mask cannot express — so they are separate, but they are the same
    kind of fact and belong in the same place. Held on the viewer instead, they
    outlived the branches they named: removing one branch took its mask with it
    and left picks addressing a cloud that no longer existed.

    The order is global, not per branch: clicks in two branches interleave, and
    a measurement across both has to come back in the order the user made it.
    """
    v = _viewer()
    try:
        _show(v, "A", _cloud(50, seed=40))
        _show(v, "B", _cloud(50, seed=41))
        tree = global_variables.global_tree_structure_widget

        # Interleave clicks between the two branches.
        for uid, row in (("A", 5), ("B", 9), ("A", 2), ("B", 1)):
            tree.add_pick(uid, row)

        assert v.picked_points == [("A", 5), ("B", 9), ("A", 2), ("B", 1)], \
            f"click order was not preserved across branches: {v.picked_points}"
        assert v.first_pick() == 5
        assert v.first_pick("B") == 9, "first_pick ignored the branch asked for"

        tree.remove_branch("A")
        assert v.picked_points == [("B", 9), ("B", 1)], \
            f"picks outlived the branch they named: {v.picked_points}"
        print("  picks keep global click order and are removed with their branch")
    finally:
        global_variables.global_tree_structure_widget = None


def test_the_tree_shows_selected_over_total():
    """The Selected Points column reads ``selected/total``.

    The selected half is the full-resolution number — what a plugin would
    actually receive. That figure used to be invisible: the viewer highlighted
    the LOD subset while the plugin got the whole region, and the two were never
    reconciled anywhere the user could look.
    """
    v = _viewer()
    try:
        _show(v, "A", _cloud(5_000, seed=42))
        tree = global_variables.global_tree_structure_widget
        item = tree.branches_dict["A"]

        tree.refresh_selected_counts()
        assert item.text(2) == "0/5,000", item.text(2)

        box = [(100, 100), (600, 100), (600, 700), (100, 700)]
        _close_lasso(v, box)
        v.refresh_selection_readout()

        selected = v.selection_count()
        assert 0 < selected < 5_000, selected
        assert item.text(2) == f"{selected:,}/5,000", item.text(2)

        v.clear_selection()
        assert item.text(2) == "0/5,000", item.text(2)
        print(f"  tree column read 0/5,000 -> {selected:,}/5,000 -> 0/5,000")
    finally:
        global_variables.global_tree_structure_widget = None


def test_a_mask_for_a_different_cloud_is_refused():
    """The safety net for a branch whose cloud really did change length.

    Nothing drops the mask eagerly any more, so the read has to notice. A mask
    that does not describe the cloud the caller is holding is refused rather
    than returned misaligned.
    """
    v = _viewer()
    _show(v, "A", _cloud(1_000, seed=35))
    _close_lasso(v, _FULL_SCREEN)
    assert v.selection_mask_for_cloud("A", np.zeros((1_000, 3))) is not None

    shorter = np.zeros((400, 3), dtype=np.float32)
    assert v.selection_mask_for_cloud("A", shorter) is None, \
        "a mask describing a different cloud was handed out"
    print("  a mask that does not fit the caller's cloud is refused")


def _nearest_by_brute_force(points, target):
    """Nearest row, with non-finite distances excluded — the scan's behaviour."""
    offset = np.asarray(points[:, :3], dtype=np.float32) - np.float32(target)
    sq = np.einsum('ij,ij->i', offset, offset)
    sq[~np.isfinite(sq)] = np.inf
    return int(np.argmin(sq))


if __name__ == "__main__":
    test_a_selection_survives_an_lod_step_down()
    test_deselect_cluster_after_an_lod_step_down()
    test_cluster_select_and_deselect_match_by_label_not_colour()
    test_closing_a_lasso_with_nothing_visible()
    test_polygon_double_click_decides_select_or_deselect()
    test_the_closing_click_is_not_part_of_the_shape()
    test_the_selection_honours_the_viewer_filters()
    test_the_same_lasso_selects_the_same_points_at_any_lod()
    test_deselecting_does_not_collapse_the_selection()
    test_the_selection_lives_on_the_branch_in_the_tree()
    test_a_cache_toggle_keeps_the_selection()
    test_click_picks_belong_to_the_branch_too()
    test_the_tree_shows_selected_over_total()
    test_a_mask_for_a_different_cloud_is_refused()
    test_a_stale_coarse_index_is_never_used()
    test_a_failed_build_is_not_retried_every_click()
    test_every_visible_branch_starts_building_on_one_click()
    test_one_bad_coordinate_does_not_degrade_the_grid()
    print("\nAll selection edge-case tests passed.")
