
from typing import Dict, Any, List, Tuple

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.clusters import Clusters


class DBSCANPlugin(AnalysisPlugin):
    """
    DBSCAN clustering algorithm plugin.

    This plugin implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    algorithm for clustering point clouds. It identifies core points, connects them to form clusters,
    and labels points as belonging to clusters or as noise.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "dbscan"
        """
        return "dbscan"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for DBSCAN.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "eps": {
                "type": "float",
                "default": 0.5,
                "min": 0.01,
                "max": 100.0,
                "label": "Epsilon (Neighborhood Size)",
                "description": "The maximum distance between two points for them to be considered neighbors"
            },
            "min_samples": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "label": "Minimum Samples",
                "description": "The minimum number of points required to form a dense region"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute DBSCAN clustering on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for DBSCAN (eps and min_samples)

        Returns:
            Tuple[Clusters, str, List]:
                - Clusters object with the clustering results
                - Result type identifier "cluster_labels"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Perform DBSCAN clustering
        cluster_labels = point_cloud.dbscan(eps=params["eps"], min_points=params["min_samples"])

        # Create a Clusters object and set random colors
        clusters = Clusters(cluster_labels)
        clusters.set_random_color()

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return clusters, "cluster_labels", dependencies