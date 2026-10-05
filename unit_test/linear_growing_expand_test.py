# linear_growing_expand_test.py
#
# Linear region growing writes into ONE result branch: the first run creates it,
# later runs with that branch selected add lines to it. These tests pin what
# that relies on, without Qt widgets:
#
#   - adding lines keeps the existing ones exactly as they were (labels, names,
#     settled stops) and leaves a one-step undo behind;
#   - lines are locked against selection, untinted, and the rest is UNASSIGNED;
#   - a result saved before all this (rest = -1, lines unlocked) is brought up
#     to date when the project loads.
import sys
import os
import copy
import pickle
import importlib
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.entities.clusters import Clusters, UNASSIGNED_LABEL, NOISE_LABEL
from core.services.linear_region_grower import (
    GrownLine, MarchStop, lines_to_traces, stop_key,
)

_plugin_module = importlib.import_module(
    "plugins.020_Points.020_Clustering.040_linear_region_growing_plugin")
Plugin = _plugin_module.LinearRegionGrowingPlugin

N = 30


def _line(indices, x):
    centerline = np.array([[x, 0, 0], [x, 1, 0]], dtype=np.float32)
    stop = MarchStop(np.array([x, 1.2, 0.0]), np.array([0.0, 1.0, 0.0]),
                     "too_few_points")
    return GrownLine(np.asarray(indices, dtype=np.intp), centerline, [], [], [stop])


def _existing_result():
    """A result branch holding two lines; line 0 classified "Cable", and one of
    line 1's stops already dismissed as a real end."""
    lines = [_line(range(0, 5), 0.0), _line(range(5, 10), 1.0)]
    resolved = {stop_key(1, lines[1].stops[0])}
    clusters = Clusters(Plugin._labels_for(lines, N),
                        cluster_names={0: "Cable", 1: "Line 2"},
                        line_traces=lines_to_traces(lines, {"cylinder_length": 0.5},
                                                    resolved=resolved),
                        tint_locked=False)
    Plugin._lock_lines(clusters, range(2))
    clusters.set_random_color()
    return clusters


class _Cache:
    def __init__(self):
        self.invalidated = []

    def invalidate(self, uid):
        self.invalidated.append(uid)

    def invalidate_descendants(self, uid):
        self.invalidated.append(("descendants", uid))


def test_expand_adds_lines_and_keeps_the_old_ones():
    clusters = _existing_result()
    colours_before = clusters.get_named_colors()[:10].copy()
    node = SimpleNamespace(uid="result", data=clusters)
    controller = SimpleNamespace(_cluster_undo={}, cache_service=_Cache())

    new = [_line(range(20, 26), 3.0)]
    all_lines, labels, resolved = Plugin()._expand_result_branch(
        controller, node, new, {"cylinder_length": 0.7})

    assert len(all_lines) == 3
    assert labels[:5].tolist() == [0] * 5 and labels[5:10].tolist() == [1] * 5
    assert labels[20:26].tolist() == [2] * 6, "new line not labelled after the old"
    assert (labels[10:20] == UNASSIGNED_LABEL).all() and not (labels == NOISE_LABEL).any()
    assert clusters.cluster_names == {0: "Cable", 1: "Line 2", 2: "Line 3"}, \
        clusters.cluster_names
    assert all("select" in clusters.locked_clusters.get(k, set()) for k in range(3))
    assert np.allclose(clusters.get_named_colors()[:10], colours_before), \
        "existing lines changed colour"
    assert len(resolved) == 1, "a settled stop was forgotten"
    stored = [s for line in clusters.line_traces["lines"] for s in line["stops"]]
    assert sum(s["resolved"] for s in stored) == 1
    assert clusters.line_traces["params"] == {"cylinder_length": 0.7}

    undo = controller._cluster_undo["result"]
    assert undo is not clusters and undo.labels[20:26].tolist() == [UNASSIGNED_LABEL] * 6, \
        "undo does not hold the branch as it was before the run"
    print("expand: 2 -> 3 lines, old labels/names/colours/stops kept, undo stored")


def test_is_linear_result():
    result = SimpleNamespace(data_type="cluster_labels", data=_existing_result())
    dbscan = SimpleNamespace(data_type="cluster_labels",
                             data=Clusters(np.array([0, 1, -1])))
    cloud = SimpleNamespace(data_type="point_cloud", data=None)
    assert Plugin._is_linear_result(result)
    assert not Plugin._is_linear_result(dbscan)
    assert not Plugin._is_linear_result(cloud)
    print("is_linear_result: only branches carrying line traces")


def test_locked_lines_are_not_tinted():
    clusters = _existing_result()
    plain = copy.deepcopy(clusters)
    plain.locked_clusters = {}
    plain.set_random_color()
    clusters.set_random_color()
    assert np.allclose(clusters.colors, plain.colors), "locked lines were tinted"

    tinted = Clusters(np.array([0, 0, 1]), locked_clusters={0: {"select"}})
    untinted = Clusters(np.array([0, 0, 1]))
    tinted.set_random_color()
    untinted.set_random_color()
    assert not np.allclose(tinted.colors[0], untinted.colors[0]), \
        "a user-chosen lock elsewhere lost its tint"
    print("tint: off on linear results, still on for other locked clusters")


def test_old_result_is_upgraded_on_load():
    """Saved before the rest was UNASSIGNED: rest -1, lines unlocked."""
    lines = [_line(range(0, 5), 0.0)]
    labels = np.full(N, NOISE_LABEL, dtype=np.int32)
    labels[:5] = 0
    old = Clusters(labels, cluster_names={0: "Line 1"},
                   line_traces=lines_to_traces(lines, {}))
    old.__dict__.pop("tint_locked")          # the attribute did not exist then

    loaded = pickle.loads(pickle.dumps(old))
    assert (loaded.labels[5:] == UNASSIGNED_LABEL).all(), "rest is still noise"
    assert loaded.labels[:5].tolist() == [0] * 5
    assert loaded.locked_clusters == {0: {"select"}}, loaded.locked_clusters
    assert loaded.tint_locked is False

    # Any other cluster branch loads exactly as saved.
    dbscan = Clusters(np.array([0, 1, -1]))
    assert pickle.loads(pickle.dumps(dbscan)).labels.tolist() == [0, 1, -1]
    print("load: old linear result -> rest unassigned, lines locked; DBSCAN untouched")


def test_extend_window_lifts_line_locks_and_restores_them():
    """Trim, Delete and Join are aimed by clicking lines, so the Extend window
    lifts their select locks while open and puts them back on close. Any other
    lock the user set stays."""
    from plugins.dialogs.line_extension_window import LineExtensionWindow as W
    clusters = _existing_result()
    clusters.locked_clusters[0].add("delete")
    node = SimpleNamespace(data=clusters)
    window = SimpleNamespace(
        result_uid="result", lines=[None, None, None],     # a third line added
        controller=SimpleNamespace(get_node=lambda uid: node))

    W._set_line_locks(window, False)
    assert clusters.locked_clusters == {0: {"delete"}}, clusters.locked_clusters
    W._set_line_locks(window, True)
    assert clusters.locked_clusters == {0: {"delete", "select"}, 1: {"select"},
                                        2: {"select"}}, clusters.locked_clusters
    print("extend window: select locks lifted while open, restored on close")


if __name__ == "__main__":
    test_expand_adds_lines_and_keeps_the_old_ones()
    test_is_linear_result()
    test_locked_lines_are_not_tinted()
    test_old_result_is_upgraded_on_load()
    test_extend_window_lifts_line_locks_and_restores_them()
    print("\nAll linear growing expand tests passed.")
