# plugins/analysis/hdbscan_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters


class HDBSCANPlugin(Plugin):
    """
    HDBSCAN clustering algorithm plugin.

    This plugin implements the Hierarchical Density-Based Spatial Clustering of Applications
    with Noise (HDBSCAN) algorithm for clustering point clouds. HDBSCAN extends DBSCAN by
    handling varying density clusters more effectively and automatically selecting the
    appropriate density threshold.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "hdbscan"
        """
        return "hdbscan"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for HDBSCAN.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "min_cluster_size": {
                "type": "int",
                "default": 5,
                "min": 2,
                "max": 1000,
                "label": "Minimum Cluster Size",
                "description": "The minimum size of clusters to be detected"
            },
            "min_samples": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "label": "Minimum Samples",
                "description": "Number of samples in a neighborhood for a point to be considered a core point"
            },
            "cluster_selection_epsilon": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "label": "Cluster Selection Epsilon",
                "description": "Distance threshold for cluster extraction (0.0 for automatic selection)"
            },
            "alpha": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "label": "Alpha",
                "description": "Normalization factor for the mutual reachability distance calculation"
            }
        }

    # plugins/analysis/hdbscan_plugin.py
    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute HDBSCAN clustering on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for HDBSCAN

        Returns:
            Tuple[Clusters, str, List]:
                - Clusters object with the clustering results
                - Result type identifier "cluster_labels"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Check if point_cloud has points
        if not hasattr(point_cloud, 'points') or point_cloud.points is None:
            raise ValueError("Point cloud has no points data")

        # Check if HDBSCAN is available
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            raise ImportError("HDBSCAN package is not installed. Please install it with: pip install hdbscan")

        # Get points from the point cloud
        points = point_cloud.points

        # Extract parameters
        min_cluster_size = params["min_cluster_size"]
        min_samples = params["min_samples"]
        cluster_selection_epsilon = params["cluster_selection_epsilon"]
        alpha = params["alpha"]

        # IMPORTANT: Do NOT set cluster_selection_epsilon to None
        # The HDBSCAN implementation requires a float value >= 0
        # If you want automatic selection, the documentation suggests using 0.0

        # Initialize HDBSCAN clusterer
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,  # Use the float value directly
            alpha=alpha
        )

        # Perform clustering
        print(f"[HDBSCAN] Starting clustering with {len(points)} points...")
        cluster_labels = clusterer.fit_predict(points)
        print(
            f"[HDBSCAN] Clustering completed. Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")

        # Create a Clusters object and set random colors
        clusters = Clusters(cluster_labels)
        clusters.set_random_color()

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return clusters, "cluster_labels", dependencies