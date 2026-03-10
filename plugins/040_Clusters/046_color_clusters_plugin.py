"""
Plugin for assigning a custom color to selected clusters.

Workflow:
1. User selects a cluster_labels branch, picks one point from each cluster to color
2. Runs Clusters > Color Clusters
3. Plugin finds affected cluster IDs from picked points
4. Shows a QColorDialog to pick a color
5. Stores the color in the Clusters custom_colors dict (persists through other operations)
6. Regenerates colors, refreshes view, clears picked points
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox, QColorDialog
from PyQt5.QtGui import QColor

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class ColorClustersPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "color_clusters"

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
                                "Please select points in clusters to color "
                                "using Shift+Click or Polygon selection.")
            return

        # Get 3D coordinates of picked points from the viewer's combined vertex buffer
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

        labels = cluster_labels

        # Match picked coordinates to the reconstructed point cloud via KDTree
        tree = cKDTree(point_cloud.points)
        distances, local_indices = tree.query(picked_coords)

        # Find affected cluster IDs (excluding noise -1)
        affected_cluster_ids = set()
        for local_idx in local_indices:
            if local_idx < len(labels):
                cid = labels[local_idx]
                if cid != -1:
                    affected_cluster_ids.add(int(cid))

        if not affected_cluster_ids:
            QMessageBox.warning(main_window, "No Valid Clusters",
                                "Selected points do not belong to any valid clusters (all noise).")
            return

        # Pre-select current custom color if all affected clusters share one
        clusters_data = node.data
        initial_color = QColor(255, 255, 255)
        existing_colors = [clusters_data.custom_colors.get(cid) for cid in affected_cluster_ids]
        if all(c is not None for c in existing_colors):
            first = existing_colors[0]
            if all(np.array_equal(c, first) for c in existing_colors):
                initial_color = QColor(int(first[0] * 255), int(first[1] * 255), int(first[2] * 255))

        # Show color picker
        color = QColorDialog.getColor(initial_color, main_window, "Select Cluster Color")
        if not color.isValid():
            return

        # Convert QColor to normalized RGB array
        rgb = np.array([color.redF(), color.greenF(), color.blueF()], dtype=np.float32)

        # Store custom color for each affected cluster
        for cid in affected_cluster_ids:
            clusters_data.custom_colors[cid] = rgb.copy()

        # Regenerate colors (custom colors are applied automatically in set_random_color)
        clusters_data.set_random_color()

        # Invalidate cache and re-render
        controller.cache_service.invalidate(str(node.uid))
        controller.cache_service.invalidate_descendants(str(node.uid))
        main_window.render_visible_data(zoom_extent=False)

        # Clear selection
        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()
