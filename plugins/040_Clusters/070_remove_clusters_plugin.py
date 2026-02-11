"""
Plugin for removing (deleting) selected clusters by setting them to noise (-1).

Workflow:
1. User selects a cluster_labels branch, picks one point from each cluster to remove
2. Runs Clusters > Remove Clusters
3. Plugin finds all cluster IDs containing selected points (excluding -1)
4. All points in those clusters are set to -1 (noise)
5. Colors regenerated, view refreshed, picked points cleared
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.clusters import Clusters


class RemoveClustersPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "remove_clusters"

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
                                "Please select points in clusters to remove "
                                "using Shift+Click or Polygon selection.")
            return

        # Get 3D coordinates of picked points from the viewer's combined vertex buffer.
        # picked_points_indices are indices into the viewer's combined buffer (all visible
        # branches), so we must match by coordinates rather than using indices directly.
        picked_coords = []
        for idx in selected_indices:
            if idx < len(viewer_widget.points):
                picked_coords.append(viewer_widget.points[idx, :3])
        if not picked_coords:
            QMessageBox.warning(main_window, "No Points Selected",
                                "Could not retrieve coordinates for selected points.")
            return
        picked_coords = np.array(picked_coords, dtype=np.float32)

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

        # Match picked coordinates to the reconstructed point cloud via KDTree
        tree = cKDTree(point_cloud.points)
        distances, local_indices = tree.query(picked_coords)

        # Find affected cluster IDs (excluding noise -1)
        affected_cluster_ids = set()
        for local_idx in local_indices:
            if local_idx < len(labels):
                cid = labels[local_idx]
                if cid != -1:
                    affected_cluster_ids.add(cid)

        if not affected_cluster_ids:
            QMessageBox.warning(main_window, "No Valid Clusters",
                                "Selected points do not belong to any valid clusters (all noise).")
            return

        # Filter out clusters locked against delete
        locked = {cid for cid in affected_cluster_ids
                  if "delete" in node.data.locked_clusters.get(cid, set())}
        if locked:
            locked_str = ", ".join(str(c) for c in sorted(locked))
            if locked == affected_cluster_ids:
                QMessageBox.warning(main_window, "All Clusters Locked",
                                    f"All selected clusters are locked against delete: {locked_str}")
                return
            QMessageBox.information(main_window, "Skipping Locked Clusters",
                                    f"Clusters locked against delete will be skipped: {locked_str}")
            affected_cluster_ids -= locked

        # Remove: set all points in affected clusters to noise (-1)
        new_labels = labels.copy()
        for cluster_id in affected_cluster_ids:
            new_labels[labels == cluster_id] = -1

        # Build new Clusters, dropping cluster_names for removed clusters
        old_clusters = node.data
        new_cluster_names = {}
        new_cluster_colors = {}

        if old_clusters.cluster_names:
            for cid, name in old_clusters.cluster_names.items():
                if cid not in affected_cluster_ids:
                    new_cluster_names[cid] = name
            new_cluster_colors = old_clusters.cluster_colors.copy()

        # Carry over locks, dropping removed clusters
        new_locked = {cid: locks for cid, locks in old_clusters.locked_clusters.items()
                      if cid not in affected_cluster_ids}

        new_clusters = Clusters(
            labels=new_labels,
            cluster_names=new_cluster_names if new_cluster_names else None,
            cluster_colors=new_cluster_colors if new_cluster_colors else None,
            locked_clusters=new_locked if new_locked else None,
        )
        new_clusters.set_random_color()

        # Save for undo, then update
        controller._cluster_undo[str(node.uid)] = old_clusters
        node.data = new_clusters
        controller.cache_service.invalidate(str(node.uid))
        controller.cache_service.invalidate_descendants(str(node.uid))
        main_window.render_visible_data(zoom_extent=False)

        # Clear selection
        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()
