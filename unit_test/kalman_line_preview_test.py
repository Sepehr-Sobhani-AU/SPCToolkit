# kalman_line_preview_test.py
#
# The Kalman preview reads a Linear Region Growing result and draws, next to
# the original, the tracker's smoothed lines and predictions. No Qt widgets:
# this checks it reads a saved result the way Extend Traced Lines does, refuses
# anything else, and builds uncertainty cones that widen with the prediction.
import sys
import os
import importlib
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.entities.clusters import Clusters, UNASSIGNED_LABEL
from core.services.linear_region_grower import GrownLine, MarchStop, lines_to_traces
from core.services.path_tracker import track_polyline

module = importlib.import_module(
    "plugins.020_Points.020_Clustering.080_kalman_line_preview_plugin")
Plugin = module.KalmanLinePreviewPlugin


def _result_node(n_points=200):
    R = 2.0
    t = np.linspace(0, np.pi / 2, 25)
    centerline = np.stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)],
                          axis=1).astype(np.float32)
    stop = MarchStop(centerline[-1], np.array([-1.0, 0, 0]), "too_few_points")
    line = GrownLine(np.arange(0, 50), centerline, [], [], [stop])
    labels = np.full(n_points, UNASSIGNED_LABEL, dtype=np.int32)
    labels[:50] = 0
    clusters = Clusters(labels, cluster_names={0: "Line 1"},
                        line_traces=lines_to_traces([line], {}))
    return SimpleNamespace(uid="result", data_type="cluster_labels", data=clusters)


def _controller(node, selected=("result",)):
    return SimpleNamespace(selected_branches=list(selected),
                           get_node=lambda uid: node if uid == "result" else None)


def test_reads_a_linear_result():
    node = _result_node()
    prepared = Plugin()._prepare(_controller(node), None)
    assert prepared is not None
    uid, _node, lines = prepared
    assert uid == "result" and len(lines) == 1
    assert len(lines[0].centerline) == 25
    print("prepare: reads the saved centerline of a linear result")


def test_refuses_other_branches():
    import PyQt5.QtWidgets as qw
    shown = []
    original = qw.QMessageBox.warning
    module.QMessageBox.warning = lambda *a, **k: shown.append(a[1])
    try:
        dbscan = SimpleNamespace(uid="result", data_type="cluster_labels",
                                 data=Clusters(np.array([0, 1, -1])))
        assert Plugin()._prepare(_controller(dbscan), None) is None
        assert Plugin()._prepare(_controller(_result_node(), selected=()), None) is None
    finally:
        module.QMessageBox.warning = original
    assert shown == ["Not a Traced Result", "Select One Branch"], shown
    print("prepare: refuses a non-linear branch and a missing selection")


def test_uncertainty_cones_widen_ahead():
    node = _result_node()
    _uid, _n, lines = Plugin()._prepare(_controller(node), None)
    result = track_polyline(lines[0].centerline, point_sigma=0.01)
    positions, _t, sigma = result.tracker.predict_path(3.0, 0.1)
    cones = module._cone_segments(positions, sigma)
    assert len(cones) == len(positions) - 1
    starts = [c[2] for c in cones]
    assert all(b >= a - 1e-12 for a, b in zip(starts, starts[1:])), \
        "cone narrowed further along"
    for tip, direction, r0, r1, length in cones:
        assert abs(np.linalg.norm(direction) - 1) < 1e-9 and r1 >= r0 - 1e-12
    print(f"cones: {len(cones)} segments, {starts[0] * 100:.1f} cm -> "
          f"{cones[-1][3] * 100:.1f} cm wide (2 sigma)")


if __name__ == "__main__":
    test_reads_a_linear_result()
    test_refuses_other_branches()
    test_uncertainty_cones_widen_ahead()
    print("\nAll Kalman preview tests passed.")
