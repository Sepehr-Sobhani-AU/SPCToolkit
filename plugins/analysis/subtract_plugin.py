# plugins/analysis/subtract_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.spatial import cKDTree
import uuid

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class SubtractPlugin(AnalysisPlugin):
    """
    Plugin for subtracting one point cloud from another.

    Creates a mask that identifies points in the first point cloud
    that are not present in the second point cloud (within a tolerance).
    Uses KD-tree for efficient spatial querying.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The name "subtract"
        """
        return "subtract"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for the subtraction operation.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        # Get the data nodes manager from global variables
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes

        # Get all node UUIDs and names for the dropdown
        node_options = {}
        for node_uid, node in data_nodes.data_nodes.items():
            node_options[str(node_uid)] = node.params

        # Set default if options exist
        default_value = ""
        if node_options:
            default_value = next(iter(node_options))

        return {
            "subtract_node": {
                "type": "dropdown",
                "options": node_options,
                "default": default_value,
                "label": "Branch to Subtract",
                "description": "Branch to subtract from the selected branch"
            },
            "tolerance": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 10.0,
                "label": "Distance Tolerance",
                "description": "Maximum distance for points to be considered the same"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the subtraction between two point clouds.

        Args:
            data_node (DataNode): The target data node (branch to subtract from)
            params (Dict[str, Any]): Parameters for the operation

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the result of subtraction
                - Result type identifier "masks"
                - List containing the data_node UIDs as dependencies
        """
        # Get the data nodes manager from global variables
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes
        data_manager = global_variables.global_data_manager

        # The data_node passed to us should already be a point cloud (reconstructed if needed)
        # by the DataManager.apply_analysis method before it reached this plugin
        target_pc = data_node.data
        target_points = target_pc.points

        # Get the subtract node and reconstruct it to a point cloud using DataManager
        try:
            subtract_uid = uuid.UUID(params["subtract_node"])
            subtract_node = data_nodes.get_node(subtract_uid)

            if subtract_node is None:
                raise ValueError(f"Branch with UUID {subtract_uid} not found")

            # Use the data_manager to reconstruct the point cloud from the subtract_node
            subtract_pc = data_manager.reconstruct_branch(subtract_uid)
            subtract_points = subtract_pc.points

        except Exception as e:
            raise ValueError(f"Error processing branch to subtract: {str(e)}")

        # Get the tolerance
        tolerance = params["tolerance"]

        # Build a KD-tree for the subtract point cloud for efficient spatial queries
        tree = cKDTree(subtract_points)

        # For each point in target, find the distance to the nearest point in subtract
        distances, _ = tree.query(target_points, k=1)

        # Create a mask where True means the point is not within tolerance of any subtract point
        mask = distances > tolerance

        # Create a Masks object with the result
        result_mask = Masks(mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid, subtract_uid]
        return result_mask, "masks", dependencies