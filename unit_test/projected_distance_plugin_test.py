"""
Test for the projected_distance plugin: vertical (Z-parallel) distance computed
via a 2-D XY nearest-neighbour search on the KNN backend.

The algorithm (per the plugin spec):
  1. take only A's XY,
  2. find B's nearest point in the XY plane (within a lateral radius),
  3. the distance is the signed height difference Az - Bz.

So the result is naturally signed (A above B -> +, below -> -), and the
``max_radius`` cap is a *horizontal* (XY) limit -- it does NOT clip a tall
vertical gap, only A points with no B point laterally beneath them.

The plugin module lives under a digit-prefixed directory that is not a valid
Python package name, so it is loaded directly from its file path with importlib.
"""

import sys
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import uuid
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from plugins.backends.knn_backends import ScipyKNN
from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode
from core.entities.values import Values

_PLUGIN_PATH = os.path.join(
    REPO, "plugins", "020_Points", "030_Analysis", "050_projected_distance_plugin.py"
)
_spec = importlib.util.spec_from_file_location("projected_distance_plugin_mod", _PLUGIN_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ProjectedDistancePlugin = _mod.ProjectedDistancePlugin


class TestProjectedDistanceZParallel(unittest.TestCase):
    """Vertical (Z) distance via a 2-D XY nearest-neighbour search."""

    def setUp(self):
        self.plugin = ProjectedDistancePlugin()

        # Target A: points above and below a flat z=0 reference plane.
        self.target_points = np.array([
            [0.0, 0.0, 1.0],    # 1.0 above
            [1.0, 0.0, 2.0],    # 2.0 above
            [0.0, 1.0, 0.5],    # 0.5 above
            [0.5, 0.5, -0.3],   # 0.3 below  -> negative
        ], dtype=np.float32)
        self.target_pc = PointCloud(points=self.target_points)
        self.target_node = DataNode(params="A", data=self.target_pc,
                                    data_type="point_cloud")
        self.target_uid = self.target_node.uid

        # Reference B: a z=0 point directly under each target, so the XY-nearest
        # reference sits straight below and Az - Bz == the point's z height.
        ref_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        self.ref_pc = PointCloud(points=ref_points)
        self.ref_uid = uuid.uuid4()

        # Signed vertical distance Az - Bz == the point's z (B is the z=0 plane).
        self.expected = np.array([1.0, 2.0, 0.5, -0.3], dtype=np.float32)

    @patch('config.config.global_variables')
    def test_signed_vertical_distance(self, mock_globals):
        mock_globals.global_data_nodes.get_node.return_value = object()  # non-None
        mock_globals.global_application_controller.reconstruct.return_value = self.ref_pc
        mock_globals.global_backend_registry.get_knn.return_value = ScipyKNN()

        # max_radius=0 disables the lateral cap so the raw Z math is under test.
        params = {"reference_node": str(self.ref_uid), "max_radius": 0.0}
        result, result_type, deps = self.plugin.execute(self.target_node, params)

        self.assertEqual(result_type, "values")
        self.assertIsInstance(result, Values)
        self.assertEqual(len(result.values), len(self.target_points))
        # Value is the signed height gap Az - Bz: above(+) / below(-).
        np.testing.assert_allclose(result.values, self.expected, atol=1e-4)
        self.assertEqual(deps, [self.target_uid, self.ref_uid])
        mock_globals.global_backend_registry.get_knn.assert_called_once()

    @patch('config.config.global_variables')
    def test_distance_is_vertical_not_slant(self, mock_globals):
        """The value is the Z gap to the XY-nearest B point, not the 3-D slant
        distance. A target offset sideways from B still reports only Az - Bz."""
        # A is at (3,4,10); the only B point is at the origin (0,0,0).
        target = PointCloud(points=np.array([[3.0, 4.0, 10.0]], dtype=np.float32))
        node = DataNode(params="A", data=target, data_type="point_cloud")
        ref = PointCloud(points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

        mock_globals.global_data_nodes.get_node.return_value = object()
        mock_globals.global_application_controller.reconstruct.return_value = ref
        mock_globals.global_backend_registry.get_knn.return_value = ScipyKNN()

        # XY distance is 5 (the 3-4-5 triangle); max_radius=0 disables the cap.
        result, _, _ = self.plugin.execute(
            node, {"reference_node": str(uuid.uuid4()), "max_radius": 0.0},
        )
        # Vertical gap only: 10 - 0 = 10 (NOT the 3-D slant sqrt(3^2+4^2+10^2)).
        np.testing.assert_allclose(result.values, [10.0], atol=1e-4)

    @patch('config.config.global_variables')
    def test_lateral_radius_caps_far_points_to_nan(self, mock_globals):
        """The cap is horizontal: an A point whose XY-nearest B is beyond
        max_radius laterally is NaN, even though a tall vertical gap is kept."""
        target = PointCloud(points=np.array([
            [0.0, 0.0, 5.0],   # XY 0.0 from B  <= 1.0  -> measured (gap 5.0)
            [3.0, 0.0, 0.2],   # XY 3.0 from B  >  1.0  -> NaN (overhangs B)
        ], dtype=np.float32))
        node = DataNode(params="A", data=target, data_type="point_cloud")
        ref = PointCloud(points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

        mock_globals.global_data_nodes.get_node.return_value = object()
        mock_globals.global_application_controller.reconstruct.return_value = ref
        mock_globals.global_backend_registry.get_knn.return_value = ScipyKNN()

        result, _, _ = self.plugin.execute(
            node, {"reference_node": str(uuid.uuid4()), "max_radius": 1.0},
        )
        vals = np.asarray(result.values)
        # Tall vertical gap is NOT clipped by the lateral cap.
        self.assertAlmostEqual(float(vals[0]), 5.0, places=4)
        # Laterally-distant point has no B beneath it -> no-measurement.
        self.assertTrue(np.isnan(vals[1]))


if __name__ == "__main__":
    unittest.main()
