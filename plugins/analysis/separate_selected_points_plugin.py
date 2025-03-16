# plugins/analysis/separate_selected_points_plugin.py
from typing import Dict, Any, List, Tuple

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class SeparateSelectedPointsPlugin(AnalysisPlugin):
    """
    Plugin for separating selected points into a new branch.

    Creates a mask based on the currently selected points in the viewer
    and uses it to create a new point cloud.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The name "separate_selected_points"
        """
        return "separate_selected_points"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for separating selected points.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        return {
            "new_branch_name": {
                "type": "string",
                "default": "Selected Points",
                "label": "New Branch Name",
                "description": "Name for the new branch containing selected points"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the separation of selected points.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for the operation

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the selection mask
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        # Get the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Get the global viewer widget to access selected points
        from config.config import global_variables
        viewer_widget = global_variables.global_pcd_viewer_widget

        # Create a mask based on the selected points
        # The size of the mask should match the number of points in the point cloud
        selected_indices = viewer_widget.picked_points_indices
        total_points = point_cloud.size

        # Create a boolean mask where True indicates a selected point
        import numpy as np
        selection_mask = np.zeros(total_points, dtype=bool)
        for idx in selected_indices:
            if idx < total_points:  # Ensure index is valid
                selection_mask[idx] = True

        # Create a Masks object with the result
        mask = Masks(selection_mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return mask, "masks", dependencies