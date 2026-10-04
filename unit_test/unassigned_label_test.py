# unassigned_label_test.py
#
# Two reserved labels: NOISE (-1), a point a clustering algorithm rejected, and
# UNASSIGNED (-2), a point nothing has labelled YET. They must behave
# differently everywhere a label is read:
#
#   colour     noise -> dark grey;  unassigned -> the parent branch's colours
#   selection  noise -> never;      unassigned -> yes (it is what is left to pick)
#   cluster    neither is a real cluster
#
# No Qt: colours live on Clusters / ClustersTransformer, and the selection rule
# on selection_gate.
import sys
import os
import importlib
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.entities.clusters import (
    Clusters, NOISE_LABEL, UNASSIGNED_LABEL, UNASSIGNED_COLOR, is_cluster,
)
from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode
from core.transformers.clusters_transformer import ClustersTransformer
from application.selection_gate import selectable_cloud_indices


def _colour_of(clusters, label):
    return clusters.colors[clusters.labels == label][0]


def test_is_cluster():
    assert is_cluster(0) and is_cluster(np.int32(7))
    assert not is_cluster(NOISE_LABEL) and not is_cluster(UNASSIGNED_LABEL)
    got = is_cluster(np.array([-2, -1, 0, 3]))
    assert got.tolist() == [False, False, True, True], got
    print("is_cluster: -2 and -1 are not clusters, >= 0 are")


def test_unassigned_does_not_shift_cluster_colours():
    """UNASSIGNED sorts before every other label. Given a palette slot it would
    move every cluster's colour, so the lines would recolour the moment some
    unassigned points appeared."""
    base = Clusters(np.array([-1, -1, 0, 0, 1, 1]))
    base.set_random_color()
    mixed = Clusters(np.array([-2, -2, -1, -1, 0, 0, 1, 1]))
    mixed.set_random_color()

    for label in (NOISE_LABEL, 0, 1):
        assert np.allclose(_colour_of(base, label), _colour_of(mixed, label)), \
            f"label {label} changed colour when unassigned points were added"
    assert np.allclose(_colour_of(mixed, UNASSIGNED_LABEL), UNASSIGNED_COLOR)
    assert np.allclose(_colour_of(mixed, NOISE_LABEL), 0.2), "noise is no longer dark grey"
    print("set_random_color: unassigned takes no palette slot, noise stays grey")


def test_named_colours_give_unassigned_the_fallback():
    c = Clusters(np.array([-2, -1, 0]), cluster_names={0: "Line 1"})
    c.set_random_color()
    colours = c.get_named_colors()
    assert np.allclose(colours[0], UNASSIGNED_COLOR), colours[0]
    print("get_named_colors: unassigned -> fallback colour")


def test_transformer_keeps_parent_colours_on_unassigned():
    """The cloud nothing has labelled yet must look exactly as it did."""
    points = np.random.default_rng(0).random((4, 3)).astype(np.float32)
    parent_colours = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
                              dtype=np.float32)
    clusters = Clusters(np.array([-2, -2, -1, 0]), cluster_names={0: "Line 1"})
    clusters.set_random_color()
    before = clusters.colors.copy()

    out = ClustersTransformer(PointCloud(points=points, colors=parent_colours),
                              clusters).execute()
    assert np.allclose(out.colors[:2], parent_colours[:2]), "unassigned lost the parent colours"
    assert np.allclose(out.colors[2], 0.7), "noise must not take the parent colour"
    assert not np.allclose(out.colors[3], parent_colours[3]), "a cluster took the parent colour"
    assert np.array_equal(clusters.colors, before), "transformer modified the Clusters"

    # A parent without colours: unassigned falls back to the plain-cloud colour.
    bare = ClustersTransformer(PointCloud(points=points), clusters).execute()
    assert np.allclose(bare.colors[:2], UNASSIGNED_COLOR)
    print("transformer: unassigned keeps parent colours, falls back to white")


def test_unassigned_is_selectable_noise_is_not():
    labels = np.array([-2, -2, -1, 0, 1])
    clusters = Clusters(labels, locked_clusters={1: {"select"}})
    node = SimpleNamespace(data_type="cluster_labels", data=clusters)
    rows = selectable_cloud_indices(node, len(labels)).tolist()
    assert rows == [0, 1, 3], rows
    print(f"selection: admissible rows {rows} (unassigned yes, noise no, locked no)")


def test_cluster_size_filter_passes_unassigned_through():
    """Unassigned is the rest of the cloud, not one huge cluster to judge."""
    module = importlib.import_module(
        "plugins.020_Points.020_Clustering.020_cluster_size_filter_plugin")
    plugin_cls = next(obj for name, obj in vars(module).items()
                      if name.endswith("Plugin") and hasattr(obj, "execute")
                      and obj.__module__ == module.__name__)
    labels = np.array([-2] * 3 + [-1] * 2 + [0] * 5 + [1] * 1)
    pc = PointCloud(points=np.zeros((len(labels), 3), dtype=np.float32))
    pc.add_attribute("cluster_labels", labels)
    node = DataNode(data=pc, data_type="point_cloud")

    mask, _, _ = plugin_cls().execute(node, {"min_points": 4, "include_noise": False})
    expected = [True] * 3 + [False] * 2 + [True] * 5 + [False]
    assert mask.mask.tolist() == expected, mask.mask.tolist()
    print("cluster size filter: unassigned kept, small cluster and noise dropped")


if __name__ == "__main__":
    test_is_cluster()
    test_unassigned_does_not_shift_cluster_colours()
    test_named_colours_give_unassigned_the_fallback()
    test_transformer_keeps_parent_colours_on_unassigned()
    test_unassigned_is_selectable_noise_is_not()
    test_cluster_size_filter_passes_unassigned_through()
    print("\nAll unassigned label tests passed.")
