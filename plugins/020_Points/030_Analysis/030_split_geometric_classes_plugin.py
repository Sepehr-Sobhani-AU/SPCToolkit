"""
Plugin for splitting a named Clusters node into separate ClassReference branches.

Works on any cluster_labels node that has cluster_names (e.g., output of Geometric
Classification). No parameters needed — reads the existing classification directly.

For each named cluster, creates a ClassReference child branch that filters to only
the points belonging to that class.
"""

import uuid
from typing import Dict, Any

import numpy as np
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.data_node import DataNode
from core.entities.clusters import Clusters
from core.entities.class_reference import ClassReference


class SplitGeometricClassesPlugin(ActionPlugin):
    """
    Split a named Clusters node into one ClassReference branch per class.

    Select a cluster_labels node (e.g., from Geometric Classification) and run this
    plugin. It reads the existing labels/names/colors and creates child branches —
    no re-classification needed.
    """

    def get_name(self) -> str:
        return "split_geometric_classes"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # --- Validate selection ---
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(
                main_window, "No Branch Selected",
                "Please select a cluster_labels branch (e.g., from Geometric Classification)."
            )
            return

        if len(selected_branches) > 1:
            QMessageBox.warning(
                main_window, "Multiple Branches",
                "Please select only ONE branch at a time."
            )
            return

        selected_uid = selected_branches[0]
        selected_node = data_nodes.get_node(uuid.UUID(selected_uid))

        if selected_node is None:
            QMessageBox.warning(main_window, "Error", "Selected node not found.")
            return

        # --- Validate that the selected node is a named Clusters node ---
        clusters = selected_node.data
        if not isinstance(clusters, Clusters):
            QMessageBox.warning(
                main_window, "Invalid Selection",
                "Selected branch is not a Clusters node.\n"
                "Run Geometric Classification first, then select the resulting node."
            )
            return

        if not clusters.has_names():
            QMessageBox.warning(
                main_window, "No Class Names",
                "Selected Clusters node has no class names.\n"
                "This plugin requires named clusters (e.g., from Geometric Classification)."
            )
            return

        # --- Read existing classification ---
        labels = clusters.labels
        cluster_names = clusters.cluster_names
        cluster_colors = clusters.cluster_colors
        n_points = len(labels)

        # Count points per class
        counts = {}
        for label_id in cluster_names:
            counts[label_id] = int(np.sum(labels == label_id))

        print(f"\n{'='*80}")
        print(f"Split Geometric Classes")
        print(f"{'='*80}")
        for label_id, name in sorted(cluster_names.items()):
            count = counts[label_id]
            pct = 100.0 * count / n_points if n_points > 0 else 0.0
            print(f"  {name:12s}: {count:>8,d} points ({pct:5.1f}%)")

        # --- Create ClassReference branches ---
        clusters_uid = uuid.UUID(selected_uid)
        created_branches = []

        for label_id, name in sorted(cluster_names.items()):
            count = counts[label_id]
            if count == 0:
                continue

            color = cluster_colors.get(name, np.array([0.7, 0.7, 0.7], dtype=np.float32))
            if not isinstance(color, np.ndarray):
                color = np.asarray(color, dtype=np.float32)

            class_reference = ClassReference(
                class_id=label_id,
                class_name=name,
                color=color,
                cluster_ids=[label_id]
            )

            branch_name = f"{name} ({count:,} pts)"
            class_node = DataNode(
                params=branch_name,
                data=class_reference,
                data_type="class_reference",
                parent_uid=clusters_uid,
                depends_on=[clusters_uid],
                tags=["classification", "class", name]
            )

            class_uid = data_nodes.add_node(class_node)

            tree_widget.blockSignals(True)
            try:
                tree_widget.add_branch(str(class_uid), str(clusters_uid), branch_name)
                class_item = tree_widget.branches_dict.get(str(class_uid))
                if class_item:
                    class_item.setCheckState(0, Qt.Unchecked)
                    tree_widget.visibility_status[str(class_uid)] = False
            finally:
                tree_widget.blockSignals(False)

            created_branches.append(branch_name)
            print(f"  Created branch: {branch_name}")

        print(f"\n{'='*80}")
        print(f"Split Complete — {len(created_branches)} branches created")
        print(f"{'='*80}")

        # --- Summary dialog ---
        summary_msg = (
            f"Created {len(created_branches)} class branches.\n\n"
            f"All branches are unchecked by default for performance.\n"
            f"Check individual branches in the tree to view specific classes.\n\n"
            + "\n".join(f"  - {name}" for name in created_branches)
        )

        QMessageBox.information(
            main_window,
            "Split Geometric Classes",
            summary_msg
        )
