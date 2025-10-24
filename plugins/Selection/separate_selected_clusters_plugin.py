# plugins/analysis/separate_selected_clusters_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import Plugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class SeparateSelectedClustersPlugin(Plugin):
    """
    Plugin for separating selected clusters into a new branch.

    Creates a mask based on the clusters that contain selected points
    and uses it to create a new point cloud.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The name "separate_selected_clusters"
        """
        return "separate_selected_clusters"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for separating selected clusters.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        return {
            "new_branch_name": {
                "type": "string",
                "default": "Selected Clusters",
                "label": "New Branch Name",
                "description": "Name for the new branch containing selected clusters"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the separation of selected clusters.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for the operation

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the cluster selection mask
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        # Get the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Get the global viewer widget to access selected points
        from config.config import global_variables
        viewer_widget = global_variables.global_pcd_viewer_widget

        # Create a mask based on the clusters that contain selected points
        selected_indices = viewer_widget.picked_points_indices

        # Check if the point cloud has cluster labels
        if not hasattr(point_cloud, 'cluster_labels') or point_cloud.cluster_labels is None:
            raise ValueError("Point cloud has no cluster labels. Clustering must be performed first.")

        # Get unique cluster IDs of the selected points
        selected_cluster_ids = set()
        for idx in selected_indices:
            if idx < len(point_cloud.cluster_labels):
                selected_cluster_ids.add(point_cloud.cluster_labels[idx])

        # Create a boolean mask where True indicates a point in a selected cluster
        total_points = point_cloud.size
        cluster_mask = np.zeros(total_points, dtype=bool)

        for i in range(total_points):
            if point_cloud.cluster_labels[i] in selected_cluster_ids:
                cluster_mask[i] = True

        # Create a Masks object with the result
        mask = Masks(cluster_mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return mask, "masks", dependencies