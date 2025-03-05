# plugins/analysis/average_distance_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.values import Values


class AverageDistancePlugin(AnalysisPlugin):
    """
    Plugin to calculate the average distance of each point to its k nearest neighbors.

    This plugin computes the average Euclidean distance between each point in the point cloud
    and its k nearest neighbors. This can be useful for outlier detection, surface analysis,
    and point density evaluation.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "average_distance"
        """
        return "average_distance"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for average distance calculation.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "k_neighbors": {
                "type": "int",
                "default": 5,
                "min": 2,
                "max": 100,
                "label": "Number of Neighbors (k)",
                "description": "The number of nearest neighbors to consider for average distance calculation"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Calculate the average distance to k nearest neighbors for each point.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for calculation (k_neighbors)

        Returns:
            Tuple[Values, str, List]:
                - Values object containing average distances values
                - Result type identifier "attribute"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Get k from parameters
        k = params["k_neighbors"] + 1  # Add 1 to account for the point itself

        # Use PointCloud's KNN method to get distances and indices
        distances, indices = point_cloud.KNN(k=k)

        # Calculate average distance, excluding the first distance (which is to the point itself)
        # The first element (distances[:,0]) is the distance to the point itself (always 0)
        average_distances = np.mean(distances[:, 1:], axis=1).astype(np.float32)

        # Create a Values object with the calculated values
        values = Values(average_distances)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return values, "values", dependencies