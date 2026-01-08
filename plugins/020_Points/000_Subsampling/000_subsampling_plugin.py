# plugins/analysis/subsampling_plugin.py
from typing import Dict, Any, List, Tuple

from plugins.interfaces import Plugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class SubsamplingPlugin(Plugin):
    """
    Plugin for subsampling point cloud data.

    Reduces the number of points in a point cloud by selecting a random subset
    based on the specified sampling rate.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The name "subsampling"
        """
        return "subsampling"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for subsampling.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        return {
            "rate": {
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "label": "Sampling Rate",
                "description": "Fraction of points to keep (0.1 = 10%)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute subsampling on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for subsampling (rate)

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the subsample mask
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Perform subsampling
        subsample_mask = point_cloud.subsample(rate=params["rate"], boolean=True)

        # Create a Masks object with the result
        mask = Masks(subsample_mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return mask, "masks", dependencies