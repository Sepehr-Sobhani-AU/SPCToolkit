# test_subtract_plugin.py
import unittest
import numpy as np
import uuid
from unittest.mock import MagicMock, patch

# Import the plugin to test
from plugins.analysis.subtract_plugin import SubtractPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class TestSubtractPlugin(unittest.TestCase):
    """Unit tests for the SubtractPlugin class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create the plugin instance
        self.plugin = SubtractPlugin()

        # Create mock point clouds with known points
        # First point cloud: a simple grid of 3x3 points in the xy plane
        self.points1 = np.array([
            [0, 0, 0], [0, 1, 0], [0, 2, 0],
            [1, 0, 0], [1, 1, 0], [1, 2, 0],
            [2, 0, 0], [2, 1, 0], [2, 2, 0]
        ], dtype=np.float32)

        # Second point cloud: a subset of the first (top-right 2x2 grid)
        self.points2 = np.array([
            [1, 1, 0], [1, 2, 0],
            [2, 1, 0], [2, 2, 0]
        ], dtype=np.float32)

        # Third point cloud: slightly offset from first points (for tolerance testing)
        self.points3 = np.array([
            [0.005, 0.005, 0.005],  # Within 0.01 tolerance of [0,0,0]
            [1.02, 1.02, 0.02]  # Outside 0.01 tolerance of [1,1,0]
        ], dtype=np.float32)

        # Create mock PointCloud objects
        self.pc1 = MagicMock(spec=PointCloud)
        self.pc1.points = self.points1

        self.pc2 = MagicMock(spec=PointCloud)
        self.pc2.points = self.points2

        self.pc3 = MagicMock(spec=PointCloud)
        self.pc3.points = self.points3

        # Create mock DataNode objects
        self.node1 = MagicMock(spec=DataNode)
        self.node1.data = self.pc1
        self.node1.uid = uuid.uuid4()

        self.node2 = MagicMock(spec=DataNode)
        self.node2.data = self.pc2
        self.node2.uid = uuid.uuid4()

        self.node3 = MagicMock(spec=DataNode)
        self.node3.data = self.pc3
        self.node3.uid = uuid.uuid4()

        # Set up mock for global_data_nodes manager
        self.mock_data_nodes = MagicMock()
        self.mock_data_nodes.get_node.side_effect = lambda uid: {
            str(self.node1.uid): self.node1,
            str(self.node2.uid): self.node2,
            str(self.node3.uid): self.node3
        }.get(str(uid))

    # FIXED: Patch the correct import path
    @patch('config.config.global_variables')
    def test_basic_subtraction(self, mock_globals):
        """Test basic subtraction of one point cloud from another."""
        # Set up the mock global variables
        mock_globals.global_data_nodes = self.mock_data_nodes

        # Set up parameters for the subtraction operation
        params = {
            "subtract_node": str(self.node2.uid),
            "tolerance": 0.01
        }

        # Execute the subtract operation (subtract pc2 from pc1)
        result, result_type, dependencies = self.plugin.execute(self.node1, params)

        # Check result type
        self.assertEqual(result_type, "masks")

        # Check dependencies
        self.assertEqual(len(dependencies), 2)
        self.assertIn(self.node1.uid, dependencies)
        self.assertIn(self.node2.uid, dependencies)

        # Check that the result is a Masks object
        self.assertIsInstance(result, Masks)

        # Check the mask values (should be True for the 5 points not in pc2, False for the 4 points in pc2)
        # The 5 points not in pc2 are: [0,0,0], [0,1,0], [0,2,0], [1,0,0], [2,0,0]
        expected_mask = np.array([True, True, True, True, False, False, True, False, False])
        np.testing.assert_array_equal(result.mask, expected_mask)

    # Fix the remaining test methods similarly
    @patch('config.config.global_variables')
    def test_tolerance_handling(self, mock_globals):
        """Test subtraction with points at the edge of the tolerance threshold."""
        mock_globals.global_data_nodes = self.mock_data_nodes

        params = {
            "subtract_node": str(self.node3.uid),
            "tolerance": 0.01
        }

        result, result_type, dependencies = self.plugin.execute(self.node1, params)

        expected_mask = np.array([False, True, True, True, True, True, True, True, True])
        np.testing.assert_array_equal(result.mask, expected_mask)

    @patch('config.config.global_variables')
    def test_empty_result(self, mock_globals):
        """Test subtraction that results in all points being subtracted."""
        # Create a point cloud with identical points to pc1
        same_pc = MagicMock(spec=PointCloud)
        same_pc.points = self.points1.copy()

        same_node = MagicMock(spec=DataNode)
        same_node.data = same_pc
        same_node.uid = uuid.uuid4()

        # Update mock for global_data_nodes manager
        mock_data_nodes_extended = MagicMock()
        mock_data_nodes_extended.get_node.side_effect = lambda uid: {
            str(self.node1.uid): self.node1,
            str(self.node2.uid): self.node2,
            str(self.node3.uid): self.node3,
            str(same_node.uid): same_node
        }.get(str(uid))

        mock_globals.global_data_nodes = mock_data_nodes_extended

        params = {
            "subtract_node": str(same_node.uid),
            "tolerance": 0.001
        }

        result, result_type, dependencies = self.plugin.execute(self.node1, params)

        expected_mask = np.zeros(9, dtype=bool)
        np.testing.assert_array_equal(result.mask, expected_mask)

    @patch('config.config.global_variables')
    def test_no_subtraction(self, mock_globals):
        """Test subtraction with no overlapping points."""
        # Create a point cloud with points far from any in pc1
        far_points = np.array([
            [10, 10, 10], [11, 11, 11], [12, 12, 12]
        ], dtype=np.float32)

        far_pc = MagicMock(spec=PointCloud)
        far_pc.points = far_points

        far_node = MagicMock(spec=DataNode)
        far_node.data = far_pc
        far_node.uid = uuid.uuid4()

        # Update mock for global_data_nodes manager
        mock_data_nodes_extended = MagicMock()
        mock_data_nodes_extended.get_node.side_effect = lambda uid: {
            str(self.node1.uid): self.node1,
            str(self.node2.uid): self.node2,
            str(self.node3.uid): self.node3,
            str(far_node.uid): far_node
        }.get(str(uid))

        mock_globals.global_data_nodes = mock_data_nodes_extended

        params = {
            "subtract_node": str(far_node.uid),
            "tolerance": 0.1
        }

        result, result_type, dependencies = self.plugin.execute(self.node1, params)

        expected_mask = np.ones(9, dtype=bool)
        np.testing.assert_array_equal(result.mask, expected_mask)

    @patch('config.config.global_variables')
    def test_different_tolerance_values(self, mock_globals):
        """Test subtraction with different tolerance values."""
        mock_globals.global_data_nodes = self.mock_data_nodes

        # Test with a very small tolerance
        small_params = {
            "subtract_node": str(self.node3.uid),
            "tolerance": 0.001
        }

        result_small, _, _ = self.plugin.execute(self.node1, small_params)

        expected_mask_small = np.ones(9, dtype=bool)
        np.testing.assert_array_equal(result_small.mask, expected_mask_small)

        # Test with a larger tolerance
        large_params = {
            "subtract_node": str(self.node3.uid),
            "tolerance": 0.1
        }

        result_large, _, _ = self.plugin.execute(self.node1, large_params)

        expected_mask_large = np.array([False, True, True, True, False, True, True, True, True])
        np.testing.assert_array_equal(result_large.mask, expected_mask_large)


if __name__ == '__main__':
    unittest.main()