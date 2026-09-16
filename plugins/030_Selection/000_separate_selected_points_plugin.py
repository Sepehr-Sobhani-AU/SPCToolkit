# plugins/analysis/separate_selected_points_plugin.py
from typing import Dict, Any, List, Tuple

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.masks import Masks


class SeparateSelectedPointsPlugin(Plugin):
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

    def requires_selection(self) -> str:
        return "points"

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

        # The selection is already a boolean mask over this branch's
        # full-resolution cloud, in the same row order, which is exactly what a
        # "masks" result is. So this plugin is now a hand-off rather than a
        # derivation: no re-test, no coordinate matching, no gate to remember —
        # noise and select-locked clusters were excluded when the selection was
        # made.
        selection_mask = viewer_widget.selection_mask_for_cloud(
            data_node.uid, point_cloud.points)
        if selection_mask is None:
            raise ValueError(
                "Nothing is selected in this branch. Select points in the "
                "viewer (Shift+Click, or P for polygon select), then run this "
                "plugin."
            )

        # Create a Masks object with the result
        mask = Masks(selection_mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return mask, "masks", dependencies