# plugins/analysis/subtract_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np
import uuid
from scipy.spatial import cKDTree

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class SubtractPlugin(AnalysisPlugin):
    """
    Plugin for subtracting one point cloud from another.

    Creates a mask that identifies points in the first point cloud
    that are not present in the second point cloud.
    Supports both exact matching and distance-based subtraction.
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
            "exact_match": {
                "type": "bool",
                "default": True,
                "label": "Exact Match Only",
                "description": "If enabled, uses exact point matching; otherwise uses distance tolerance"
            },
            "tolerance": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 10.0,
                "label": "Distance Tolerance",
                "description": "Maximum distance for points to be considered the same (only used if 'Exact Match' is disabled)"
            },
            "precision": {
                "type": "int",
                "default": 6,
                "min": 1,
                "max": 10,
                "label": "Decimal Precision",
                "description": "Number of decimal places to consider when comparing points (only used with exact matching)"
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

        # Get the target point cloud
        target_pc = data_node.data
        target_points = target_pc.points

        # Get the subtract node and reconstruct it to a point cloud
        try:
            subtract_uid = uuid.UUID(params["subtract_node"])
            subtract_node = data_nodes.get_node(subtract_uid)

            if subtract_node is None:
                raise ValueError(f"Branch with UUID {subtract_uid} not found")

            # Use the data_manager to reconstruct the point cloud
            subtract_pc = data_manager.reconstruct_branch(subtract_uid)
            subtract_points = subtract_pc.points

        except Exception as e:
            raise ValueError(f"Error processing branch to subtract: {str(e)}")

        # Get parameters
        exact_match = params.get("exact_match", True)

        # Create mask based on the selected method
        if exact_match:
            # Exact matching - find points in target that don't exactly match any point in subtract
            precision = params["precision"]

            # Round both point clouds to the specified precision
            target_rounded = np.round(target_points, precision)
            subtract_rounded = np.round(subtract_points, precision)

            # Vectorized exact matching approach
            # Convert points to structured arrays for efficient comparison
            target_dtype = [(f'dim{i}', 'float64') for i in range(target_points.shape[1])]
            subtract_dtype = [(f'dim{i}', 'float64') for i in range(subtract_points.shape[1])]

            # Create structured arrays
            target_struct = np.array([tuple(p) for p in target_rounded], dtype=target_dtype)
            subtract_struct = np.array([tuple(p) for p in subtract_rounded], dtype=subtract_dtype)

            # Find which points in target exist in subtract
            # This is not truly vectorized but is memory-efficient
            mask = np.ones(len(target_struct), dtype=bool)

            # Use np.unique to first remove duplicate points for efficiency
            unique_subtract, _ = np.unique(subtract_struct, return_index=True)

            # Fast vectorized set operation to find matching points
            matches = np.array([p in unique_subtract for p in target_struct])
            mask = ~matches

        else:
            # Distance-based subtraction using KD-tree (original implementation)
            tolerance = params["tolerance"]
            tree = cKDTree(subtract_points)
            distances, _ = tree.query(target_points, k=1)
            mask = distances > tolerance

        # Create a Masks object with the result
        result_mask = Masks(mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid, subtract_uid]
        return result_mask, "masks", dependencies