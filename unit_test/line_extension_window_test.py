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

    def __init__(self, offsets):
        self._branch_offsets = offsets
        self.picked_points_indices = []
        self.focused_on = None

    def focus_on(self, point, extent, preserve_rotation=True):
        self.focused_on = np.asarray(point)

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
    viewer = _FakeViewer({result_uid: (0, n), cloud_uid: (n, 2 * n)})
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
    test_the_marker_branch_is_really_gone_afterwards()
    print("\nAll line-extension window tests passed.")
