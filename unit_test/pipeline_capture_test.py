"""
Tests for pipeline (macro) capture + save/load round-trip.

Covers the pure logic in ``core/services/pipeline.py``:
- lineage walk reconstructs steps in root -> branch order,
- params are carried verbatim,
- non-replayable (action) steps are reported as unsupported and omitted,
- JSON save/load preserves the chain.

The replay orchestration (PipelineRunner) needs Qt and the live app, so it is
exercised manually, not here.
"""

import sys
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import tempfile
import unittest

from core.entities.data_node import DataNode
from core.entities.data_nodes import DataNodes
from core.services.pipeline import (
    capture_pipeline, save_pipeline, load_pipeline,
)

SUPPORTED = {"sor", "dbscan", "cluster_size_filter"}


def _build_chain():
    """root(pc) -> sor -> dbscan -> classify_cluster(action) -> size_filter."""
    nodes = DataNodes()

    root = DataNode(params="root", data_type="point_cloud", parent_uid=None, tags=[])
    nodes.add_node(root)

    sor = DataNode(params="sor", data_type="point_cloud", parent_uid=root.uid,
                   tags=["sor", {"std_ratio": 2.0, "k": 50}])
    nodes.add_node(sor)

    dbscan = DataNode(params="dbscan", data_type="cluster_labels", parent_uid=sor.uid,
                      tags=["dbscan", {"eps": 0.5, "min_samples": 10}])
    nodes.add_node(dbscan)

    # Action-plugin step: recorded with tags here for the test, but its plugin
    # name is not in SUPPORTED -> must be reported as unsupported.
    classify = DataNode(params="classify", data_type="cluster_labels", parent_uid=dbscan.uid,
                        tags=["classify_cluster", {"class_name": "Ground"}])
    nodes.add_node(classify)

    size = DataNode(params="size", data_type="cluster_labels", parent_uid=classify.uid,
                    tags=["cluster_size_filter", {"min_size": 100}])
    nodes.add_node(size)

    return nodes, root, size


class CaptureTest(unittest.TestCase):

    def test_capture_order_and_params(self):
        nodes, _root, leaf = _build_chain()
        pipeline, unsupported = capture_pipeline(leaf.uid, nodes, SUPPORTED)

        self.assertEqual([s.plugin for s in pipeline.steps],
                         ["sor", "dbscan", "cluster_size_filter"])
        self.assertEqual(unsupported, ["classify_cluster"])
        # params carried verbatim
        self.assertEqual(pipeline.steps[0].params, {"std_ratio": 2.0, "k": 50})
        self.assertEqual(pipeline.steps[1].params["eps"], 0.5)
        self.assertEqual(pipeline.steps[0].produces, "point_cloud")

    def test_producer_action_step_captured(self):
        # scale is an action plugin but creates a transform_matrix child node, so
        # when it is in the supported set it must be captured as a normal step.
        nodes = DataNodes()
        root = DataNode(params="root", data_type="point_cloud", parent_uid=None, tags=[])
        nodes.add_node(root)
        scale = DataNode(params="scale", data_type="transform_matrix", parent_uid=root.uid,
                         tags=["scale", {"direction": "z", "scale": 5.0}])
        nodes.add_node(scale)
        dbscan = DataNode(params="dbscan", data_type="cluster_labels", parent_uid=scale.uid,
                          tags=["dbscan", {"eps": 0.5}])
        nodes.add_node(dbscan)

        pipeline, unsupported = capture_pipeline(
            dbscan.uid, nodes, {"scale", "dbscan"})

        self.assertEqual(unsupported, [])
        self.assertEqual([s.plugin for s in pipeline.steps], ["scale", "dbscan"])
        self.assertEqual(pipeline.steps[0].params, {"direction": "z", "scale": 5.0})
        self.assertEqual(pipeline.steps[0].produces, "transform_matrix")

    def test_branch_reference_stored_as_hint_not_uuid(self):
        # A param that points at another branch by UUID must be stripped to a
        # hint -- the session UUID must never reach the saved file.
        nodes = DataNodes()
        root = DataNode(params="root", data_type="point_cloud", parent_uid=None, tags=[])
        nodes.add_node(root)
        ground = DataNode(params="Ground", data_type="masks", parent_uid=root.uid,
                          tags=["sor", {"k": 50}])
        nodes.add_node(ground)
        pd = DataNode(params="pd", data_type="values", parent_uid=root.uid,
                      tags=["projected_distance",
                            {"reference_node": str(ground.uid), "max_radius": 1.0}])
        nodes.add_node(pd)

        pipeline, _ = capture_pipeline(
            pd.uid, nodes, {"projected_distance", "sor"})
        step = next(s for s in pipeline.steps if s.plugin == "projected_distance")

        self.assertEqual(step.params, {"max_radius": 1.0})   # UUID param removed
        self.assertEqual(step.bindings, {"reference_node": "Ground"})  # hint kept

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            save_pipeline(pipeline, path)
            with open(path) as f:
                text = f.read()
            loaded = load_pipeline(path)

        self.assertNotIn(str(ground.uid), text)  # no session UUID persisted
        lstep = next(s for s in loaded.steps if s.plugin == "projected_distance")
        self.assertEqual(lstep.bindings, {"reference_node": "Ground"})

    def test_root_has_no_steps(self):
        nodes, root, _leaf = _build_chain()
        pipeline, unsupported = capture_pipeline(root.uid, nodes, SUPPORTED)
        self.assertEqual(pipeline.steps, [])
        self.assertEqual(unsupported, [])

    def test_save_load_roundtrip(self):
        nodes, _root, leaf = _build_chain()
        pipeline, _ = capture_pipeline(leaf.uid, nodes, SUPPORTED)
        pipeline.name = "demo"

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "demo.json")
            save_pipeline(pipeline, path)
            loaded = load_pipeline(path)

        self.assertEqual(loaded.name, "demo")
        self.assertEqual([s.plugin for s in loaded.steps],
                         [s.plugin for s in pipeline.steps])
        self.assertEqual([s.params for s in loaded.steps],
                         [s.params for s in pipeline.steps])


if __name__ == "__main__":
    unittest.main()
