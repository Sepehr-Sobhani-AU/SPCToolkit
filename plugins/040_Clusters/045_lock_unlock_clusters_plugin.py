"""
Plugin for locking/unlocking clusters against remove, cut, and merge operations.

Workflow:
1. User selects a cluster_labels branch, picks one point from each cluster to lock/unlock
2. Runs Clusters > Lock Unlock Clusters
3. Plugin finds affected cluster IDs from picked points
4. Shows a dialog with 3 checkboxes: Remove, Cut, Merge
5. Checkboxes pre-populated: checked if ALL selected clusters currently have that lock
6. On OK: updates locked_clusters dict on the Clusters data object
7. Regenerates colors (locked clusters get a visual tint), refreshes view, clears picked points
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox, QLabel

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class LockUnlockDialog(QDialog):
    def __init__(self, parent, affected_ids, locked_clusters):
        super().__init__(parent)
        self.setWindowTitle("Lock / Unlock Clusters")

        layout = QVBoxLayout(self)

        ids_str = ", ".join(str(cid) for cid in sorted(affected_ids))
        count = len(affected_ids)
        layout.addWidget(QLabel(f"{count} cluster(s) selected — IDs: {ids_str}"))
        layout.addWidget(QLabel("Lock selected clusters against:"))

        # Pre-populate: checked only if ALL affected clusters have that lock
        all_have_select = all(
            "select" in locked_clusters.get(cid, set()) for cid in affected_ids
        )
        all_have_delete = all(
            "delete" in locked_clusters.get(cid, set()) for cid in affected_ids
        )
        all_have_cut = all(
            "cut" in locked_clusters.get(cid, set()) for cid in affected_ids
        )
        all_have_merge = all(
            "merge" in locked_clusters.get(cid, set()) for cid in affected_ids
        )

        self.cb_select = QCheckBox("Selection")
        self.cb_select.setChecked(all_have_select)
        layout.addWidget(self.cb_select)

        self.cb_delete = QCheckBox("Remove")
        self.cb_delete.setChecked(all_have_delete)
        layout.addWidget(self.cb_delete)

        self.cb_cut = QCheckBox("Cut")
        self.cb_cut.setChecked(all_have_cut)
        layout.addWidget(self.cb_cut)

        self.cb_merge = QCheckBox("Merge")
        self.cb_merge.setChecked(all_have_merge)
        layout.addWidget(self.cb_merge)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class LockUnlockClustersPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "lock_unlock_clusters"

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
                                "Please select points in clusters to lock/unlock "
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

        # Show lock dialog with pre-populated state
        clusters_data = node.data
        dialog = LockUnlockDialog(
            main_window, affected_cluster_ids, clusters_data.locked_clusters
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        # Build desired lock set from checkboxes
        desired_locks = set()
        if dialog.cb_select.isChecked():
            desired_locks.add("select")
        if dialog.cb_delete.isChecked():
            desired_locks.add("delete")
        if dialog.cb_cut.isChecked():
            desired_locks.add("cut")
        if dialog.cb_merge.isChecked():
            desired_locks.add("merge")

        # Update locked_clusters for each affected cluster
        for cid in affected_cluster_ids:
            if desired_locks:
                clusters_data.locked_clusters[cid] = desired_locks.copy()
            else:
                clusters_data.locked_clusters.pop(cid, None)

        # Regenerate colors so locked clusters get visual tint
        clusters_data.set_random_color()

        # Invalidate cache and re-render to show updated colors
        controller.cache_service.invalidate(str(node.uid))
        controller.cache_service.invalidate_descendants(str(node.uid))
        main_window.render_visible_data(zoom_extent=False)

        # Clear selection
        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()
