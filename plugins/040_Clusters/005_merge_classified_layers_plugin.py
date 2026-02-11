"""
Plugin for merging multiple branches into a single classified PointCloud.

Workflow:
1. User separates features into different branches (e.g., Power Lines, Pole Support, etc.)
2. User Ctrl+Clicks to select 2+ branches in the tree
3. Runs Clusters > Merge Classified Layers
4. Each branch becomes a class — the branch display name is the class label
5. Creates a new root PointCloud with a 'classification' attribute (integer per point)
   and stores the class name mapping in metadata
"""

import numpy as np
from typing import Dict, Any, List

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode


class MergeClassifiedLayersPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "merge_classified_layers"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        selected_branches = controller.selected_branches
        if len(selected_branches) < 2:
            QMessageBox.warning(
                main_window, "Not Enough Branches",
                "Please select 2 or more branches to merge (Ctrl+Click)."
            )
            return

        # --- Collect branch info and reconstruct each ---
        branch_data: List[dict] = []
        for uid_str in selected_branches:
            node = controller.get_node(uid_str)
            if node is None:
                QMessageBox.warning(
                    main_window, "Invalid Selection",
                    f"Branch '{uid_str[:8]}...' not found."
                )
                return

            # Get display name from tree item
            item = tree_widget.branches_dict.get(uid_str)
            display_name = item.text(0) if item else (node.params or uid_str[:8])

            try:
                point_cloud = controller.reconstruct(uid_str)
            except Exception as e:
                QMessageBox.critical(
                    main_window, "Reconstruction Error",
                    f"Failed to reconstruct '{display_name}':\n{str(e)}"
                )
                return

            branch_data.append({
                "uid": uid_str,
                "name": display_name,
                "point_cloud": point_cloud,
            })

        # --- Build confirmation ---
        summary_lines = []
        for bd in branch_data:
            summary_lines.append(f"  {bd['name']}  ({bd['point_cloud'].size:,} points)")
        summary = "\n".join(summary_lines)

        reply = QMessageBox.question(
            main_window, "Merge Classified Layers",
            f"Merge {len(branch_data)} branches into one classified PointCloud?\n\n"
            f"{summary}\n\n"
            f"Each branch becomes a class using its name.",
            QMessageBox.Ok | QMessageBox.Cancel,
        )
        if reply != QMessageBox.Ok:
            return

        # --- Concatenate points, colors, normals; build classification ---
        all_points = []
        all_colors = []
        all_normals = []
        all_labels = []
        class_names = {}  # {int_label: class_name}

        has_colors = all(bd["point_cloud"].colors is not None for bd in branch_data)
        has_normals = all(bd["point_cloud"].normals is not None for bd in branch_data)

        for class_id, bd in enumerate(branch_data):
            pc = bd["point_cloud"]
            n = pc.size

            all_points.append(pc.points)
            if has_colors:
                all_colors.append(pc.colors)
            if has_normals:
                all_normals.append(pc.normals)

            all_labels.append(np.full(n, class_id, dtype=np.int32))
            class_names[class_id] = bd["name"]

        merged_points = np.concatenate(all_points, axis=0)
        merged_colors = np.concatenate(all_colors, axis=0) if has_colors else None
        merged_normals = np.concatenate(all_normals, axis=0) if has_normals else None
        merged_labels = np.concatenate(all_labels, axis=0)

        # --- Create new PointCloud ---
        merged_pc = PointCloud(
            points=merged_points,
            colors=merged_colors,
            normals=merged_normals,
        )
        merged_pc.attributes["classification"] = merged_labels
        merged_pc.metadata = {"class_names": class_names}

        # --- Add as new root PointCloud node ---
        uid_str = controller.add_point_cloud(merged_pc, "Merged Classification")

        # Add to tree as root
        tree_widget.blockSignals(True)
        try:
            tree_widget.add_branch(uid_str, None, "Merged Classification", is_root=True)
            item = tree_widget.branches_dict.get(uid_str)
            if item:
                item.setCheckState(0, Qt.Checked)
                tree_widget.visibility_status[uid_str] = True
        finally:
            tree_widget.blockSignals(False)

        main_window.render_visible_data(zoom_extent=False)

        total = merged_points.shape[0]
        QMessageBox.information(
            main_window, "Merge Complete",
            f"Created merged PointCloud with {total:,} points.\n\n"
            f"Classes ({len(class_names)}):\n"
            + "\n".join(f"  {cid}: {name}" for cid, name in class_names.items())
        )
