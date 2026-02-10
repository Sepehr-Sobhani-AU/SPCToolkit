"""
Plugin for removing selected clusters from a branch.

Creates a mask where all points in selected clusters are excluded
(True = keep, False = remove). The inverse of Separate Selected Clusters.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any, List, Tuple

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.masks import Masks


class RemoveSelectedClustersPlugin(Plugin):

    def get_name(self) -> str:
        return "remove_selected_clusters"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "new_branch_name": {
                "type": "string",
                "default": "Remaining Clusters",
                "label": "New Branch Name",
                "description": "Name for the new branch excluding selected clusters"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud: PointCloud = data_node.data

        from config.config import global_variables
        viewer_widget = global_variables.global_pcd_viewer_widget

        selected_indices = viewer_widget.picked_points_indices
        total_points = point_cloud.size

        # Check if the point cloud has cluster labels
        cluster_labels = point_cloud.get_attribute("cluster_labels")
        if cluster_labels is None:
            raise ValueError("Point cloud has no cluster labels. Clustering must be performed first.")

        # Get 3D coordinates of picked points from the viewer's combined vertex buffer.
        # picked_points_indices are indices into the combined buffer (all visible branches),
        # so we match by coordinates to find the correct local indices.
        picked_coords = []
        for idx in selected_indices:
            if idx < len(viewer_widget.points):
                picked_coords.append(viewer_widget.points[idx, :3])

        if not picked_coords:
            # No valid picks — return a keep-all mask
            return Masks(np.ones(total_points, dtype=bool)), "masks", [data_node.uid]

        picked_coords = np.array(picked_coords, dtype=np.float32)

        # Match picked coordinates to the point cloud via KDTree
        tree = cKDTree(point_cloud.points)
        distances, local_indices = tree.query(picked_coords)

        # Get unique cluster IDs of the matched points
        selected_cluster_ids = set()
        for local_idx in local_indices:
            if local_idx < len(cluster_labels):
                selected_cluster_ids.add(cluster_labels[local_idx])

        # Create mask: True = keep (not in selected clusters), False = remove
        cluster_mask = ~np.isin(cluster_labels, list(selected_cluster_ids))

        return Masks(cluster_mask), "masks", [data_node.uid]
