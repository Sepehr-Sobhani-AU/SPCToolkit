# line_extension_window_test.py
#
# End-to-end cover for the guided line-extension window, against the REAL
# ApplicationController — which is the part that was missing when the candidate
# cluster shipped: the labelling was right, but a second point branch left
# visible drew a grey, unlabelled copy of the same cloud on top of it and took
# every click. The window logic looked correct from every angle except the one
# that mattered.
#
# Runs headless: Qt is created with the offscreen platform, and the viewer and
# tree are stood in for (they are OpenGL and QTreeWidget). Everything the
# assertions are about — the controller, the Clusters branch, the grower, the
# window itself — is real.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

from config.config import global_variables
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from plugins.plugin_manager import PluginManager
from application.application_controller import ApplicationController
from application.selection_gate import picked_cloud_indices
from core.services.linear_region_grower import LinearRegionGrower, AXIS_TRACE

_app = QApplication.instance() or QApplication([])


class _FakeItem:
    """Stands in for a QTreeWidgetItem: only its checked/selected state matters."""

    def __init__(self):
        self.checked = True
        self.selected = False

    def setCheckState(self, _col, state):
        self.checked = bool(state)

    def setSelected(self, value):
        self.selected = bool(value)


class _FakeTree:
    def __init__(self, uids):
        self.branches_dict = {uid: _FakeItem() for uid in uids}
        self.visibility_status = {uid: True for uid in uids}

    def blockSignals(self, _flag):
        pass

    def clearSelection(self):
        for item in self.branches_dict.values():
            item.selected = False

    def add_branch(self, uid, *_a, **_k):
        self.branches_dict.setdefault(uid, _FakeItem())
        self.visibility_status[uid] = True

    def remove_branch(self, uid):
        self.branches_dict.pop(uid, None)
        self.visibility_status.pop(uid, None)


class _FakeViewer:
    """Only what the window touches: which branches draw points, and the picks."""

    def __init__(self, offsets, points=None):
        self._branch_offsets = offsets
        self.picked_points_indices = []
        self._selection_polygons = []
        self.focused_on = None
        self.points = points
        self.polygon_mask = None       # what retest_polygon_selection returns
        self.emphasis = {}             # uid -> (emphasised, faded)
        self.line_width = 1.0          # the viewer's real default

    def clear_selection(self):
        """Mirrors the real viewer: picks and stored polygons go together.

        Callers use this rather than emptying picked_points_indices by hand,
        precisely so a stale polygon cannot outlive the picks it produced.
        """
        self.picked_points_indices.clear()
        self._selection_polygons.clear()
        self.polygon_mask = None

    def set_point_emphasis(self, uid, emphasised=None, faded=None):
        if emphasised is None and faded is None:
            self.emphasis.pop(uid, None)
        else:
            self.emphasis[uid] = (np.asarray(emphasised), np.asarray(faded))

    def focus_on(self, point, extent, preserve_rotation=True):
        self.focused_on = np.asarray(point)

    def retest_polygon_selection(self, points_3d):
        return self.polygon_mask

    def update(self):
        pass


def _rendered_colours(clusters, points):
    """The colours the viewer would actually draw for this branch.

    Goes through the real ClustersTransformer rather than reading
    ``clusters.colors``, because for a NAMED Clusters that array is not what
    gets drawn — and a growth result is always named ("Line 1", "Line 2").
    """
    from core.transformers.clusters_transformer import ClustersTransformer
    cloud = PointCloud(points=points.astype(np.float32))
    return ClustersTransformer(cloud, clusters).execute().colors


def _scene():
    """A cable with a hole in it, grown — the state the plugin hands the window."""
    rng = np.random.default_rng(4)
    cable = np.vstack([
        np.stack([np.linspace(0, 10, 400), np.zeros(400), np.full(400, 8.0)], 1),
        np.stack([np.linspace(16, 30, 560), np.zeros(560), np.full(560, 8.0)], 1)])
    ground = np.column_stack([rng.uniform(-5, 35, 4000), rng.uniform(-8, 8, 4000),
                              rng.normal(0, 0.03, 4000)])
    points = np.vstack([cable + rng.normal(0, 0.01, cable.shape), ground])

    grower = LinearRegionGrower(
        points, mode=AXIS_TRACE, ransac_threshold=0.05, cylinder_radius=0.2,
        cylinder_length=1.0, reach_factor=3.0, min_points=5, max_angle_deg=20.0)
    lines = grower.grow_lines([np.arange(8)])
    labels = np.full(len(points), -1, dtype=np.int32)
    for label, line in enumerate(lines):
        labels[line.indices] = label
    # Named, exactly as _build_result_branch makes it — which is the whole
    # reason the candidates have to be named too.
    clusters = Clusters(labels=labels,
                        cluster_names={k: f"Line {k + 1}" for k in range(len(lines))})
    clusters.set_random_color()
    return points, lines, grower, clusters


def _open_window(points, lines, grower, clusters):
    """Wire a real controller + branch, then open the real window on it."""
    from plugins.dialogs.line_extension_window import LineExtensionWindow

    controller = ApplicationController.create(PluginManager())
    global_variables.global_application_controller = controller
    global_variables.global_main_window = None

    cloud_uid = controller.add_point_cloud(PointCloud(points=points.astype(np.float32)),
                                           "cloud")
    cloud_node = controller.get_node(cloud_uid)
    result_uid = controller.add_analysis_result(
        clusters, "cluster_labels", [cloud_node.uid], cloud_node,
        "linear_region_growing", {})

    # Both branches on screen — the situation that broke it.
    n = len(points)
    viewer = _FakeViewer({result_uid: (0, n), cloud_uid: (n, 2 * n)},
                         points=points)
    tree = _FakeTree([cloud_uid, result_uid])
    global_variables.global_pcd_viewer_widget = viewer
    global_variables.global_tree_structure_widget = tree
    controller.set_selected_branches([cloud_uid])   # what the tree really holds

    window = LineExtensionWindow(result_uid, points, lines, grower,
                                 {"cylinder_length": 1.0})
    return window, controller, tree, cloud_uid, result_uid


def test_window_claims_the_viewport():
    """The bug that made the candidate cluster look broken: the input cloud is
    still visible, drawing a grey unlabelled copy of every point on top of the
    result. Those copies take the clicks — measured on the real controller, with
    the input cloud visible and selected, EVERY one of its points is selectable
    and none of the result's are. The window has to clear the viewport."""
    points, lines, grower, clusters = _scene()
    window, controller, tree, cloud_uid, result_uid = _open_window(
        points, lines, grower, clusters)

    print(f"after opening: input cloud visible={tree.visibility_status[cloud_uid]}, "
          f"result visible={tree.visibility_status[result_uid]}, "
          f"selection={controller.selected_branches}")
    assert tree.visibility_status[cloud_uid] is False, \
        "the input cloud is still drawing a second copy of the points"
    assert tree.branches_dict[cloud_uid].checked is False, \
        "the tree still shows the input cloud as visible"
    assert tree.visibility_status[result_uid] is True, \
        "the result branch must be on screen — hiding everything else would " \
        "otherwise leave an empty viewport"
    assert controller.selected_branches == [result_uid], \
        "picking is filtered to the tree selection; it must be the result branch"

    window.close()
    print(f"after closing: input cloud visible={tree.visibility_status[cloud_uid]}")
    assert tree.visibility_status[cloud_uid] is True, \
        "the input cloud was not put back the way the user had it"


def test_candidates_are_drawn_in_the_candidate_colour():
    """What the user should SEE, taken from the renderer rather than from the
    per-point colour array.

    A growth result names its clusters, and for a named Clusters the renderer
    colours by name and never reads that array — so writing yellow into it left
    the candidates the unnamed 0.7 grey default on precisely the branches this
    window opens on. Checking `clusters.colors` would have passed throughout."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    marked = window.marked_indices
    assert marked is not None and len(marked), "no points were offered for picking"
    label = int(clusters.labels[marked[0]])
    drawn = _rendered_colours(clusters, points)
    noise = np.where(clusters.labels == -1)[0][0]
    print(f"offered {len(marked)} points as cluster {label} "
          f"({clusters.cluster_names.get(label)!r}); the renderer draws them "
          f"{drawn[marked[0]].tolist()}, the rest {drawn[noise].tolist()}")

    assert label == max(clusters.labels), \
        "the candidate cluster must sort last or it re-colours every line"
    assert np.allclose(drawn[marked], [1.0, 1.0, 0.0]), \
        "the candidates are not DRAWN in the candidate colour"
    assert not np.allclose(drawn[noise], [1.0, 1.0, 0.0]), \
        "the rest of the cloud is drawn the same as the candidates"

    window.close()
    print(f"after closing: labels {sorted(set(clusters.labels.tolist()))}, "
          f"names {sorted(clusters.cluster_names.values())}")
    assert max(clusters.labels) == len(lines) - 1, \
        "the candidate cluster outlived the window and would reach classification"
    assert "Pick candidates" not in clusters.cluster_names.values(), \
        "the candidate cluster's NAME outlived the window — it would show up as " \
        "a real class in classification and export"


def test_the_offer_is_exactly_what_the_panel_counts():
    """The panel reports one number — the unclaimed points lying ahead — and the
    yellow on screen must be that same set, point for point.

    An earlier version offered a much wider cone than the count measured, so a
    tree canopy lit up beside a count of 77 and read as a bug. Two numbers for
    one stop was the mistake; there is now one."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    _label, stop = window._current()
    counted = grower.unclaimed_ahead(stop, window._claimed_mask())
    offered = window.marked_indices
    print(f"{len(counted)} counted as ahead, {len(offered)} offered in yellow")

    assert set(offered.tolist()) == set(counted.tolist()), \
        "the yellow points are not the points the panel counts"
    window.close()


def test_a_polygon_cannot_drag_in_what_was_never_offered():
    """A polygon selection is re-tested against the FULL cloud
    (``picked_cloud_indices``) so it covers everything it encloses rather than
    only the points LOD drew — and that re-test does not go through the viewer's
    selection filters. Drawn over the corridor it therefore comes back holding
    every point inside it: measured on real data the viewer honestly reported 32
    points picked while the extension consumed thousands, and since picks are
    always adopted, a whole bush joined the line.

    Only what is on offer may be picked, whichever code path found it."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)
    viewer = window.viewer

    offered = set(window.marked_indices.tolist())
    # One honest click on an offered point, plus a polygon over a big region
    # that happens to enclose a great deal more.
    viewer.picked_points_indices = [int(window.marked_indices[0])]
    sprawl = np.zeros(len(points), dtype=bool)
    sprawl[::3] = True                       # a third of the whole cloud
    viewer.polygon_mask = sprawl

    raw = picked_cloud_indices(viewer, points, grower.kdtree)
    used = window._picked_indices()
    print(f"polygon returned {len(raw):,} points; {len(used)} of them were on "
          f"offer and used")

    assert len(raw) > 10 * len(offered), \
        "setup wrong: the polygon was expected to sweep up far more than the offer"
    assert set(used.tolist()) <= offered, \
        "points that were never offered got picked up by the polygon"
    assert len(used), "the honest click was dropped along with the sprawl"
    window.close()


class _SilencedDialogs:
    """Swap the window's QMessageBox for a recorder.

    Anything that refuses to act tells the user why, and a real message box
    blocks on exec() forever with no one to click it — so a headless test that
    exercises a refusal has to stand in for the user. The messages are kept so
    the test can check that something was actually said.
    """

    def __enter__(self):
        from plugins.dialogs import line_extension_window as module
        self._module = module
        self._real = module.QMessageBox
        self.shown = []
        recorder = self

        class _Recorder:
            @staticmethod
            def information(_parent, title, text):
                recorder.shown.append((title, text))

            warning = information
            critical = information

        module.QMessageBox = _Recorder
        return self

    def __exit__(self, *_exc):
        self._module.QMessageBox = self._real
        return False


def _click(window, points, target):
    """Put one viewer pick on the cloud point nearest *target*."""
    row = int(np.argmin(np.linalg.norm(points - np.asarray(target), axis=1)))
    window.viewer.picked_points_indices.append(row)
    return row


def test_the_offer_is_drawn_bigger_and_the_clutter_faded():
    """Colour alone did not separate the offer from the clutter on real data: at
    the zoom a line is reviewed at, one yellow point is the same few pixels as
    the grey one beside it. The offer is drawn larger and the unclaimed points
    see-through — and the traced lines are left alone, because those are what
    Trim, Delete and Join are aimed at."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, result_uid = _open_window(
        points, lines, grower, clusters)

    emphasised, faded = window.viewer.emphasis[result_uid]
    on_a_line = np.concatenate([np.asarray(line.indices) for line in lines])
    print(f"{len(emphasised)} points drawn big, {len(faded)} faded, "
          f"{len(on_a_line)} on a traced line and left alone")

    assert set(emphasised.tolist()) == set(window.marked_indices.tolist()), \
        "the points drawn bigger are not the points on offer"
    assert not set(faded.tolist()) & set(emphasised.tolist()), \
        "an offered point was also faded"
    assert not set(faded.tolist()) & set(on_a_line.tolist()), \
        "the traced lines were faded — they are what the edit buttons aim at"
    assert len(faded) > len(emphasised), \
        "setup wrong: the clutter should outnumber the offer"

    window.close()
    assert result_uid not in window.viewer.emphasis, \
        "the emphasis outlived the window and stays on the branch"


def test_trim_cuts_the_clicked_segment_out():
    """Trim removes the clicked segment and releases its points. A cut in the
    middle leaves TWO lines: a GrownLine carries one polyline, so a line with a
    hole in it is not something that can be drawn honestly."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    line = window.lines[0]
    middle = np.asarray(line.centerline)[len(line.centerline) // 2]
    _click(window, points, middle)
    before_lines, before_claimed = len(window.lines), int(window._claimed_mask().sum())

    window._trim()
    after_claimed = int(window._claimed_mask().sum())
    print(f"trim at {middle.round(1).tolist()}: {before_lines} -> "
          f"{len(window.lines)} line(s), {before_claimed} -> {after_claimed} "
          f"claimed points | {window.edit_status.text()}")

    assert len(window.lines) == before_lines + 1, \
        "a mid-line cut must leave two lines"
    assert after_claimed < before_claimed, "no points were released by the trim"
    # Checked per line rather than against labels.max(): the branch also carries
    # the candidate cluster for the stop under review, which sorts above them all.
    for label, line in enumerate(window.lines):
        assert set(clusters.labels[np.asarray(line.indices)].tolist()) == {label}, \
            f"line {label}'s points do not carry label {label} on the branch"
    named_lines = {k: n for k, n in clusters.cluster_names.items()
                   if n != "Pick candidates"}
    assert named_lines == {k: f"Line {k + 1}" for k in range(len(window.lines))}, \
        f"cluster names were not rebuilt for the new line list: {named_lines}"

    window._undo()
    print(f"after undo: {len(window.lines)} line(s), "
          f"{int(window._claimed_mask().sum())} claimed points")
    assert len(window.lines) == before_lines, "undo did not put the line back"
    assert int(window._claimed_mask().sum()) == before_claimed, \
        "undo did not reclaim the trimmed points"
    window.close()


def test_delete_drops_the_whole_line():
    """Delete takes the line, its geometry and its claim on the points."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    line = window.lines[0]
    _click(window, points, np.asarray(line.centerline)[1])
    doomed = set(np.asarray(line.indices).tolist())
    before = len(window.lines)

    window._delete_line()
    still_claimed = doomed & set(np.where(window._claimed_mask())[0].tolist())
    print(f"delete: {before} -> {len(window.lines)} line(s), "
          f"{len(doomed)} points released, {len(still_claimed)} still claimed")

    assert len(window.lines) == before - 1, "the line was not removed"
    assert not still_claimed, "the deleted line's points are still on a line"
    assert all(clusters.labels[i] == -1 for i in list(doomed)[:50]), \
        "released points did not go back to -1 on the branch"
    window.close()


def test_join_keeps_the_first_line_clicked():
    """Join folds the second clicked line into the first and chains the two
    centrelines into one, whichever way round each was traced."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    # Trim the line in two first, so there are two lines to join back up.
    first = window.lines[0]
    _click(window, points, np.asarray(first.centerline)[len(first.centerline) // 2])
    window._trim()
    assert len(window.lines) >= 2, "setup wrong: the trim did not split the line"
    window._clear_picks()

    a, b = window.lines[0], window.lines[1]
    span_a, span_b = len(a.indices), len(b.indices)
    _click(window, points, np.asarray(a.centerline)[0])
    _click(window, points, np.asarray(b.centerline)[-1])
    before = len(window.lines)

    window._join()
    joined = window.lines[0]
    chain = np.asarray(joined.centerline)
    print(f"join: {before} -> {len(window.lines)} line(s), "
          f"{span_a}+{span_b} -> {len(joined.indices)} points, "
          f"{len(chain)} vertices | {window.edit_status.text()}")

    assert len(window.lines) == before - 1, "the two lines did not become one"
    assert len(joined.indices) == span_a + span_b, \
        "the joined line does not hold both lines' points"
    steps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    assert len(chain) >= len(a.centerline) + len(b.centerline), \
        "the joined centreline lost vertices"
    assert steps.max() < 10.0, \
        f"the centrelines were chained at the wrong ends (a {steps.max():.1f} m " \
        f"jump between consecutive vertices)"
    window.close()


def test_a_click_beside_a_line_still_names_it():
    """Picking resolves the depth buffer at exactly the pixel clicked, so a
    click three pixels off a one-pixel centreline lands on whatever is behind
    it — grey clutter, a yellow candidate, or nothing. Those clicks have to name
    the line anyway, or aiming at a hairline is the whole interaction."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    # Trim a piece out, which leaves real UNCLAIMED points lying along the line
    # — exactly what a click that misses the centreline lands on.
    first = window.lines[0]
    _click(window, points, np.asarray(first.centerline)[len(first.centerline) // 2])
    window._trim()
    window._clear_picks()

    claimed = window._claimed_mask()
    released = [i for i in np.asarray(first.indices) if not claimed[i]]
    assert released, "setup wrong: the trim released nothing to click on"

    row = int(released[len(released) // 2])
    window.viewer.picked_points_indices = [row]
    picked = window._picked_lines_in_order()
    reach = grower.cylinder_length * grower.reach_factor
    print(f"a click on a released (unclaimed) point beside the line names "
          f"line {picked[0][0] if picked else None} — anything within "
          f"{reach:.1f} m counts")

    assert picked, "a click beside the line named no line at all"

    # Just off the centreline counts; far from every line does not, or every
    # stray click would edit something.
    near = np.asarray(window.lines[0].centerline)[1] + np.array([0.0, 0.25, 0.0])
    far = np.asarray(window.lines[0].centerline)[1] + np.array([0.0, 0.0, -8.0])
    print(f"0.25 m off the centreline -> {window._line_under(near)}, "
          f"8 m away -> {window._line_under(far)}")
    assert window._line_under(near) == 0, "a click just off the line named nothing"
    assert window._line_under(far) is None, \
        "a click far from every line still named one"
    window.close()


def test_the_panel_names_the_line_before_you_press():
    """Trim and Delete are irreversible-looking single clicks, and Join silently
    depends on which line was clicked FIRST. The panel has to say which line the
    buttons are aimed at, before they are pressed."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    window._refresh_edit_target()
    idle = window.edit_label.text()

    first = window.lines[0]
    _click(window, points, np.asarray(first.centerline)[2])
    window._refresh_edit_target()
    one = window.edit_label.text()
    print(f"nothing picked: {idle!r}\none line picked: {one!r}")

    assert "Line 1" not in idle, "the panel named a line before one was picked"
    assert "Line 1" in one, "the panel does not say which line is picked"

    # Split it, then pick both halves — the readout must say which Join keeps.
    window._trim()
    window._clear_picks()
    _click(window, points, np.asarray(window.lines[1].centerline)[1])
    _click(window, points, np.asarray(window.lines[0].centerline)[1])
    window._refresh_edit_target()
    two = window.edit_label.text()
    print(f"two lines picked: {two!r}")

    assert "Line 2" in two and "Line 1" in two, "the panel names only one line"
    assert two.index("Line 2") < two.index("Line 1"), \
        "the panel does not report them in click order, so Join reads as a guess"
    assert "keeps Line 2" in two, "the panel does not say which line Join keeps"
    window.close()


def test_the_centreline_is_thickened_while_the_window_is_open():
    """A one-pixel centreline is aimed at by pixel-hunting, and a miss reads the
    depth buffer back empty and picks nothing at all."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    while_open = window.viewer.line_width
    window.close()
    print(f"line width: 1.0 -> {while_open} while open -> "
          f"{window.viewer.line_width} after closing")

    assert while_open > 1.0, "the centreline is still a one-pixel hair"
    assert window.viewer.line_width == 1.0, \
        "the window kept the viewer's line width after closing"


def test_editing_needs_a_line_under_the_click():
    """A click on the clutter or on the yellow candidates names no line, and the
    edit buttons have to say so rather than acting on line 1 by default."""
    points, lines, grower, clusters = _scene()
    window, _controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    before = list(window.lines)
    window.viewer.picked_points_indices = [int(window.marked_indices[0])]
    picked = window._picked_lines_in_order()
    print(f"a click on an offered (unclaimed) point resolves to {picked} line(s)")

    assert picked == [], "an unclaimed point was taken to name a traced line"
    with _SilencedDialogs() as dialogs:
        window._trim()
        window._delete_line()
        window._join()
    print(f"all three refused and said so: {[t for t, _ in dialogs.shown]}")

    assert window.lines == before, "an edit went ahead with no line picked"
    assert len(dialogs.shown) == 3, \
        f"an edit refused silently: {[t for t, _ in dialogs.shown]}"
    window.close()


def test_the_marker_branch_is_really_gone_afterwards():
    """The green "you are here" marker is a branch the window creates and must
    take away again. Its removal went through ApplicationController.remove_node,
    which called a DataNodes method that has never existed — the failure was
    caught and logged, so the tree row vanished while the node stayed in the
    project for good."""
    points, lines, grower, clusters = _scene()
    window, controller, _tree, _cloud_uid, _result_uid = _open_window(
        points, lines, grower, clusters)

    def markers():
        return [n for n in controller.data_nodes.data_nodes.values()
                if n.params == "stop_under_review"]

    print(f"marker branches while open: {len(markers())}")
    assert markers(), "setup wrong: no marker branch was drawn"

    window.close()
    print(f"marker branches after closing: {len(markers())}")
    assert not markers(), \
        "the marker branch outlived the window and stays in the project"


if __name__ == "__main__":
    test_window_claims_the_viewport()
    test_candidates_are_drawn_in_the_candidate_colour()
    test_the_offer_is_exactly_what_the_panel_counts()
    test_a_polygon_cannot_drag_in_what_was_never_offered()
    test_the_offer_is_drawn_bigger_and_the_clutter_faded()
    test_trim_cuts_the_clicked_segment_out()
    test_delete_drops_the_whole_line()
    test_join_keeps_the_first_line_clicked()
    test_a_click_beside_a_line_still_names_it()
    test_the_panel_names_the_line_before_you_press()
    test_the_centreline_is_thickened_while_the_window_is_open()
    test_editing_needs_a_line_under_the_click()
    test_the_marker_branch_is_really_gone_afterwards()
    print("\nAll line-extension window tests passed.")
