# plugins/analysis/subtract_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np
import uuid

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.masks import Masks


class SubtractPlugin(Plugin):
    """
    Plugin for subtracting one point cloud from another.

    Creates a mask that identifies points in the first point cloud
    that are not present in the second point cloud.
    Only supports exact point matching.
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
            node_options[str(node_uid)] = node.alias or node.params

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
        controller = global_variables.global_application_controller

        # Get the target point cloud
        target_pc = data_node.data
        target_points = target_pc.points

        # Get the subtract node and reconstruct it to a point cloud
        try:
            subtract_uid = uuid.UUID(params["subtract_node"])
            subtract_node = data_nodes.get_node(subtract_uid)

            if subtract_node is None:
                raise ValueError(f"Branch with UUID {subtract_uid} not found")

            # Use the controller to reconstruct the point cloud (thread-safe: read-only)
            global_variables.global_progress = (None, "Reconstructing subtract branch...")
            subtract_pc = controller.reconstruct(subtract_uid)
            subtract_points = subtract_pc.points

        except Exception as e:
            raise ValueError(f"Error processing branch to subtract: {str(e)}")

        # Ensure the point clouds have the same dimensionality
        if target_points.shape[1] != subtract_points.shape[1]:
            raise ValueError("Point dimensions do not match between the two point clouds")

        global_variables.global_progress = (50, f"Comparing {len(target_points):,} vs {len(subtract_points):,} points...")

        # Convert rows to structured dtype to enable vectorized comparison
        dtype = [('f0', target_points.dtype), ('f1', target_points.dtype), ('f2', target_points.dtype)]

        # Create structured views of the arrays for exact comparison
        target_view = target_points.view(dtype).reshape(-1)
        subtract_view = subtract_points.view(dtype).reshape(-1)

        mask = ~np.isin(target_view, subtract_view)

        # Create a Masks object with the result
        result_mask = Masks(mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid, subtract_uid]
        return result_mask, "masks", dependencies