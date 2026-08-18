"""
Plugin for splitting selected points out of their cluster(s) into new clusters.

Workflow:
1. User selects a cluster_labels branch, picks points (Shift+Click or Polygon)
2. Runs Clusters > Split Clusters
3. Selected points are removed from their original clusters and assigned new cluster IDs
4. Each affected cluster produces one new cluster from its split-out points
5. Colors regenerated, view refreshed, picked points cleared
"""

import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from application.selection_gate import (
    picked_cloud_indices, selectable_cloud_indices)
from core.entities.clusters import Clusters


class SplitClustersPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "split_clusters"

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
                                "Please select points to cut using Shift+Click or Polygon selection.")
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

        # Map the picks onto rows of the reconstructed cloud. The shared helper
        # coordinate-matches the clicks *and* re-tests any lasso at full
        # resolution, which is what the two branches this replaces did by hand.
        # `allowed` keeps that widening inside what the viewer would have let
        # the user pick, so a lasso cannot cut into a locked cluster.
        allowed = selectable_cloud_indices(node, len(point_cloud.points))
        picked_rows = picked_cloud_indices(viewer_widget, point_cloud.points,
                                           allowed=allowed)
        if picked_rows is None or len(picked_rows) == 0:
            QMessageBox.warning(main_window, "No Points Selected",
                                "Could not retrieve coordinates for selected points.")
            return

        # Targeted cluster IDs, and the points to cut — noise belongs to no
        # cluster, so it is never cut and never names a target.
        picked_rows = picked_rows[picked_rows < len(labels)]
        picked_labels = labels[picked_rows]
        selected_rows = picked_rows[picked_labels != -1]
        affected_cluster_ids = {cid for cid in np.unique(picked_labels) if cid != -1}

        if not affected_cluster_ids:
            QMessageBox.warning(main_window, "No Valid Clusters",
                                "Selected points do not belong to any valid clusters (all noise).")
            return

        # Filter out clusters locked against cut
        locked = {cid for cid in affected_cluster_ids
                  if "cut" in node.data.locked_clusters.get(cid, set())}
        if locked:
            locked_str = ", ".join(str(c) for c in sorted(locked))
            if locked == affected_cluster_ids:
                QMessageBox.warning(main_window, "All Clusters Locked",
                                    f"All selected clusters are locked against cut: {locked_str}")
                return
            QMessageBox.information(main_window, "Skipping Locked Clusters",
                                    f"Clusters locked against cut will be skipped: {locked_str}")
            affected_cluster_ids -= locked

        # Cut: for each affected cluster, move selected points to a new cluster ID
        new_labels = labels.copy()
        new_label_id = int(labels.max()) + 1

        # One boolean mask of the selection, built once. It used to be rebuilt
        # from a Python set *inside* the per-cluster loop below, so a lasso
        # holding a million points paid that loop once for every cluster it
        # touched.
        selected_mask = np.zeros(len(labels), dtype=bool)
        selected_mask[selected_rows] = True

        for cluster_id in sorted(affected_cluster_ids):
            # Mask: points in this cluster AND in the selection
            mask = selected_mask & (labels == cluster_id)

            if np.any(mask):
                new_labels[mask] = new_label_id
                new_label_id += 1

        # Build new Clusters, preserving cluster_names for unaffected clusters
        old_clusters = node.data
        new_cluster_names = {}
        new_cluster_colors = {}

        if old_clusters.cluster_names:
            for cid, name in old_clusters.cluster_names.items():
                if cid not in affected_cluster_ids:
                    new_cluster_names[cid] = name
                else:
                    # Keep name for the remainder of the affected cluster (uncut part)
                    if np.any(new_labels == cid):
                        new_cluster_names[cid] = name
            new_cluster_colors = old_clusters.cluster_colors.copy()

        # Carry over locks for original cluster IDs (new cut-off clusters are unlocked)
        new_locked = {cid: locks for cid, locks in old_clusters.locked_clusters.items()}

        # Carry over custom colors for original cluster IDs
        new_custom = {cid: c for cid, c in old_clusters.custom_colors.items()}

        new_clusters = Clusters(
            labels=new_labels,
            cluster_names=new_cluster_names if new_cluster_names else None,
            cluster_colors=new_cluster_colors if new_cluster_colors else None,
            locked_clusters=new_locked if new_locked else None,
            custom_colors=new_custom if new_custom else None,
        )
        new_clusters.set_random_color()

        # Save for undo, then update
        controller._cluster_undo[str(node.uid)] = old_clusters
        node.data = new_clusters
        controller.cache_service.invalidate(str(node.uid))
        controller.cache_service.invalidate_descendants(str(node.uid))
        main_window.render_visible_data(zoom_extent=False)

        # Clear selection
        viewer_widget.clear_selection()
