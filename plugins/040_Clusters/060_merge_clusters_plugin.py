"""
Plugin for merging multiple clusters into one.

Workflow:
1. User selects a cluster_labels branch, picks one point from each cluster to merge
2. Runs Clusters > Merge Clusters
3. All points in the affected clusters get the lowest cluster ID from the set
4. Colors regenerated, view refreshed, picked points cleared
"""

import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.clusters import Clusters


class MergeClustersPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "merge_clusters"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget

        # Validate: exactly one branch selected
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a cluster branch first.")
            return
        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch at a time.")
            return

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)

        # Validate: must be cluster_labels
        if node is None or node.data_type != "cluster_labels":
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Please select a cluster_labels branch.")
            return

        # Validate: points must be selected
        selected_indices = viewer_widget.picked_points_indices
        if not selected_indices:
            QMessageBox.warning(main_window, "No Points Selected",
                                "Please select one point from each cluster to merge.")
            return

        # Reconstruct to get cluster_labels attribute
        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{str(e)}")
            return

        cluster_labels = point_cloud.get_attribute("cluster_labels")
        if cluster_labels is None:
            QMessageBox.warning(main_window, "No Cluster Labels",
                                "The reconstructed point cloud has no cluster labels.")
            return

        labels = cluster_labels.copy()

        # Find affected cluster IDs (excluding noise -1)
        affected_cluster_ids = set()
        for idx in selected_indices:
            if idx < len(labels):
                cid = labels[idx]
                if cid != -1:
                    affected_cluster_ids.add(cid)

        if not affected_cluster_ids:
            QMessageBox.warning(main_window, "No Valid Clusters",
                                "Selected points do not belong to any valid clusters (all noise).")
            return

        if len(affected_cluster_ids) < 2:
            QMessageBox.warning(main_window, "Not Enough Clusters",
                                "Need at least 2 different clusters to merge.\n"
                                "Select points from different clusters.")
            return

        # Merge: all affected clusters get the lowest cluster ID
        target_id = min(affected_cluster_ids)
        new_labels = labels.copy()
        for cluster_id in affected_cluster_ids:
            if cluster_id != target_id:
                new_labels[labels == cluster_id] = target_id

        # Build new Clusters, preserving cluster_names for unaffected clusters
        old_clusters = node.data
        new_cluster_names = {}
        new_cluster_colors = {}

        if old_clusters.cluster_names:
            for cid, name in old_clusters.cluster_names.items():
                if cid not in affected_cluster_ids:
                    # Unaffected cluster — keep as-is
                    new_cluster_names[cid] = name
                elif cid == target_id:
                    # Keep the target cluster's name for the merged result
                    new_cluster_names[cid] = name
                # Other affected cluster names are dropped (merged away)
            new_cluster_colors = old_clusters.cluster_colors.copy()

        new_clusters = Clusters(
            labels=new_labels,
            cluster_names=new_cluster_names if new_cluster_names else None,
            cluster_colors=new_cluster_colors if new_cluster_colors else None,
        )
        new_clusters.set_random_color()

        # In-place update + cache invalidation
        node.data = new_clusters
        controller.cache_service.invalidate(str(node.uid))
        controller.cache_service.invalidate_descendants(str(node.uid))
        main_window.render_visible_data(zoom_extent=False)

        # Clear selection
        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()
