# plugins/analysis/filtering_plugin.py
from typing import Dict, Any, List, Tuple

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class FilteringPlugin(AnalysisPlugin):
    """
    Plugin for filtering point cloud data based on a condition.

    Applies a filtering condition to the point cloud and returns a mask
    identifying which points meet the condition.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The name "filtering"
        """
        return "filtering"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for filtering.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        return {
            "condition": {
                "type": "string",
                "default": "point_cloud.normals[:, 2] >= 0.95",
                "label": "Filter Condition",
                "description": "Python expression that evaluates to a boolean mask"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute filtering on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for filtering (condition)

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the filter mask
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        filter_condition = params["condition"]
        point_cloud: PointCloud = data_node.data

        # Use a dictionary to capture the result of the exec statement
        local_vars = {"point_cloud": point_cloud}

        # Execute the condition in the provided namespace
        exec(f"filter_mask = {filter_condition}", globals(), local_vars)

        # Extract the filter_mask from the namespace
        filter_mask = local_vars.get("filter_mask")

        # Create a Masks object with the result
        mask = Masks(filter_mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return mask, "masks", dependencies