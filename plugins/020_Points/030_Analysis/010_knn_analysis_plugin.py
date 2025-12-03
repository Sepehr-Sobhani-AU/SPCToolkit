# plugins/Points/Analysis/knn_analysis_plugin.py
"""
Plugin for K-Nearest Neighbors (KNN) analysis of point clouds.

This plugin computes various statistics based on the k-nearest neighbors for each point.
The computed statistics can be used for:
- Outlier detection (points with unusually high/low neighbor distances)
- Point density analysis
- Surface roughness estimation
- Feature extraction for machine learning
- Local geometry characterization
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import Plugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.values import Values


class KNNAnalysisPlugin(Plugin):
    """
    Plugin for computing K-Nearest Neighbors statistics.

    This plugin analyzes the local neighborhood of each point by computing
    various distance statistics to its k-nearest neighbors. Different statistics
    provide different insights into the point cloud structure.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "knn_analysis"
        """
        return "knn_analysis"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for KNN analysis.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "k_neighbors": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 100,
                "label": "Number of Neighbors (k)",
                "description": "Number of nearest neighbors to analyze for each point"
            },
            "statistic": {
                "type": "choice",
                "options": [
                    "Average Distance",
                    "Maximum Distance",
                    "Minimum Distance",
                    "Std Deviation",
                    "Distance to Kth Neighbor",
                    "Sum of Distances"
                ],
                "default": "Average Distance",
                "label": "Distance Statistic",
                "description": "The statistical measure to compute from neighbor distances"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute KNN analysis on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for KNN analysis (k_neighbors and statistic)

        Returns:
            Tuple[Values, str, List]:
                - Values object containing computed statistics for each point
                - Result type identifier "values"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Get parameters
        k = params["k_neighbors"]
        statistic = params["statistic"]

        # Perform KNN search
        # k+1 because the first neighbor is the point itself (distance = 0)
        distances, indices = point_cloud.KNN(k=k + 1)

        # Remove the first column (distance to itself, which is always 0)
        neighbor_distances = distances[:, 1:]

        # Compute the requested statistic
        if statistic == "Average Distance":
            result_values = np.mean(neighbor_distances, axis=1).astype(np.float32)

        elif statistic == "Maximum Distance":
            result_values = np.max(neighbor_distances, axis=1).astype(np.float32)

        elif statistic == "Minimum Distance":
            result_values = np.min(neighbor_distances, axis=1).astype(np.float32)

        elif statistic == "Std Deviation":
            result_values = np.std(neighbor_distances, axis=1).astype(np.float32)

        elif statistic == "Distance to Kth Neighbor":
            # Distance to the farthest of the k neighbors
            result_values = neighbor_distances[:, -1].astype(np.float32)

        elif statistic == "Sum of Distances":
            result_values = np.sum(neighbor_distances, axis=1).astype(np.float32)

        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        # Create a Values object with the calculated statistics
        values = Values(result_values)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return values, "values", dependencies