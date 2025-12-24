"""
Plugin for splitting FeatureClasses into separate class branches.

Workflow:
1. User runs PointNet classification (creates FeatureClasses branch)
2. User selects the FeatureClasses branch
3. User runs this plugin
4. Plugin analyzes classes and shows selection dialog
5. Plugin creates a "Classes" container branch
6. For each selected class, creates a lightweight ClassReference branch
7. All class branches are unchecked (invisible) by default for performance
"""

import numpy as np
import uuid
from typing import Dict, Any, List, Tuple
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QMessageBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.feature_classes import FeatureClasses
from core.class_reference import ClassReference
from core.data_node import DataNode


class ClassSelectionDialog(QDialog):
    """Dialog for selecting which classes to split into branches."""

    def __init__(self, parent, class_info: List[Tuple[int, str, int]]):
        """
        Initialize the dialog.

        Args:
            parent: Parent widget
            class_info: List of (class_id, class_name, cluster_count) tuples
        """
        super().__init__(parent)
        self.setWindowTitle("Split Classes")
        self.setModal(True)
        self.setMinimumWidth(400)

        self.class_info = class_info
        self.list_widget = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Select classes to create branches for:")
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title)

        # Class list with checkboxes
        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(300)

        # Add each class as a checkable item
        for class_id, class_name, cluster_count in self.class_info:
            item = QListWidgetItem(f"{class_name} ({cluster_count} clusters)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)  # All selected by default

            # Store class info in item data
            item.setData(Qt.UserRole, (class_id, class_name, cluster_count))

            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        # Selection buttons
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        selection_layout.addWidget(select_all_btn)
        selection_layout.addWidget(deselect_all_btn)
        selection_layout.addStretch()
        layout.addLayout(selection_layout)

        # Container name
        container_layout = QHBoxLayout()
        container_layout.addWidget(QLabel("Container Name:"))
        self.container_name_edit = QLineEdit("Classes")
        container_layout.addWidget(self.container_name_edit)
        layout.addLayout(container_layout)

        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _select_all(self):
        """Select all items in the list."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked)

    def _deselect_all(self):
        """Deselect all items in the list."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Unchecked)

    def get_selected_classes(self) -> List[Tuple[int, str, int]]:
        """
        Get the selected classes.

        Returns:
            List of (class_id, class_name, cluster_count) for selected classes
        """
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                class_id, class_name, cluster_count = item.data(Qt.UserRole)
                selected.append((class_id, class_name, cluster_count))
        return selected

    def get_container_name(self) -> str:
        """Get the container name."""
        return self.container_name_edit.text().strip() or "Classes"


class SplitClassesPlugin(ActionPlugin):
    """
    Action plugin for splitting FeatureClasses into separate class branches.

    Creates a lightweight ClassReference branch for each selected class.
    Each branch contains only a small reference object (~100 bytes) rather than copying data.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "split_classes"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters - uses custom dialog.

        Returns:
            Empty dict (custom dialog is shown in execute)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the split classes action.

        Args:
            main_window: The main application window
            params: Parameters (unused - custom dialog instead)
        """
        # Get global instances
        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Validate branch selection
        selected_branches = data_manager.selected_branches

        if not selected_branches:
            QMessageBox.warning(
                main_window,
                "No Branch Selected",
                "Please select a FeatureClasses branch before running this plugin."
            )
            return

        if len(selected_branches) > 1:
            QMessageBox.warning(
                main_window,
                "Multiple Branches",
                "Please select only ONE branch at a time."
            )
            return

        selected_uid = selected_branches[0]

        # Get the selected data node
        selected_node = data_nodes.get_node(uuid.UUID(selected_uid))
        if selected_node is None:
            QMessageBox.critical(
                main_window,
                "Node Not Found",
                f"Could not find data node with UID: {selected_uid}"
            )
            return

        # Validate that it's a FeatureClasses node
        if not isinstance(selected_node.data, FeatureClasses):
            QMessageBox.warning(
                main_window,
                "Invalid Branch Type",
                f"Selected branch is not a FeatureClasses branch.\n\n"
                f"Expected: FeatureClasses\n"
                f"Got: {type(selected_node.data).__name__}\n\n"
                f"Please select a FeatureClasses branch created by ML classification."
            )
            return

        try:
            feature_classes = selected_node.data

            # Need to get cluster_labels from the parent point cloud
            # Reconstruct the branch to get the point cloud with cluster_labels
            point_cloud = data_manager.reconstruct_branch(selected_uid)

            if not hasattr(point_cloud, 'cluster_labels') or point_cloud.cluster_labels is None:
                QMessageBox.warning(
                    main_window,
                    "No Cluster Labels",
                    "Cannot count clusters - the point cloud has no cluster_labels.\n\n"
                    "This typically means the classification was not done on clustered data."
                )
                return

            cluster_labels = point_cloud.cluster_labels

            # Analyze classes and count clusters (not points!)
            unique_class_ids = np.unique(feature_classes.labels)
            class_info = []

            for class_id in unique_class_ids:
                class_name = feature_classes.class_mapping.get(int(class_id), "Unknown")

                # Find points of this class
                class_mask = (feature_classes.labels == class_id)

                # Get cluster IDs for points of this class
                class_cluster_ids = cluster_labels[class_mask]

                # Count unique clusters (excluding noise -1)
                unique_clusters = np.unique(class_cluster_ids)
                unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude noise
                cluster_count = len(unique_clusters)

                class_info.append((int(class_id), class_name, cluster_count))

            # Sort by class name
            class_info.sort(key=lambda x: x[1])

            # Show selection dialog
            dialog = ClassSelectionDialog(main_window, class_info)
            if dialog.exec_() != QDialog.Accepted:
                return  # User cancelled

            selected_classes = dialog.get_selected_classes()
            container_name = dialog.get_container_name()

            if not selected_classes:
                QMessageBox.warning(
                    main_window,
                    "No Classes Selected",
                    "Please select at least one class to split."
                )
                return

            # Disable UI
            main_window.disable_menus()
            main_window.disable_tree()
            main_window.tree_overlay.show_processing("Creating class branches...")

            print(f"\n{'='*80}")
            print(f"Splitting FeatureClasses into Class Branches")
            print(f"{'='*80}")
            print(f"Creating {len(selected_classes)} class branches")

            # Create container branch
            parent_uuid = uuid.UUID(selected_uid)
            container_node = DataNode(
                params=container_name,
                data=None,  # Container has no data
                data_type="container",
                parent_uid=parent_uuid,
                depends_on=[parent_uuid],
                tags=["classification", "classes"]
            )

            container_uid = data_nodes.add_node(container_node)

            # Add container to tree (unchecked)
            tree_widget.blockSignals(True)
            try:
                tree_widget.add_branch(str(container_uid), str(selected_uid), container_name)
                container_item = tree_widget.branches_dict.get(str(container_uid))
                if container_item:
                    container_item.setCheckState(0, Qt.Unchecked)
                    tree_widget.visibility_status[str(container_uid)] = False
            finally:
                tree_widget.blockSignals(False)

            # Create class branches for selected classes
            created_branches = []
            for class_id, class_name, cluster_count in selected_classes:
                # Get color for this class
                color = feature_classes.class_colors.get(class_name, np.array([0.5, 0.5, 0.5]))

                # Create ClassReference object
                class_reference = ClassReference(
                    class_id=class_id,
                    class_name=class_name,
                    color=color
                )

                # Create DataNode with cluster count in name
                branch_name = f"{class_name} ({cluster_count})"
                class_node = DataNode(
                    params=branch_name,
                    data=class_reference,
                    data_type="class_reference",
                    parent_uid=container_uid,
                    depends_on=[parent_uuid],  # Depends on original FeatureClasses
                    tags=["classification", "class", class_name]
                )

                class_uid = data_nodes.add_node(class_node)

                # Add to tree (unchecked for performance)
                tree_widget.blockSignals(True)
                try:
                    tree_widget.add_branch(str(class_uid), str(container_uid), branch_name)
                    class_item = tree_widget.branches_dict.get(str(class_uid))
                    if class_item:
                        class_item.setCheckState(0, Qt.Unchecked)
                        tree_widget.visibility_status[str(class_uid)] = False
                finally:
                    tree_widget.blockSignals(False)

                created_branches.append(branch_name)
                print(f"Created branch: {branch_name}")

            print(f"\n{'='*80}")
            print(f"Class Split Complete!")
            print(f"{'='*80}")
            print(f"Created {len(created_branches)} class branches")
            print(f"All branches are unchecked (invisible) by default")
            print(f"Check individual class branches to view them")
            print(f"{'='*80}")

            # Show summary message
            summary_msg = (
                f"Successfully created {len(created_branches)} class branches!\n\n"
                f"All branches are unchecked (invisible) by default for performance.\n"
                f"Check individual class branches in the tree to view specific classes.\n\n"
                f"Created branches:\n" + "\n".join(f"  - {name}" for name in created_branches)
            )

            QMessageBox.information(
                main_window,
                "Class Split Complete",
                summary_msg
            )

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"\nERROR during class split:\n{error_msg}")

            QMessageBox.critical(
                main_window,
                "Split Error",
                f"An error occurred during class split:\n\n{str(e)}"
            )

        finally:
            # Re-enable UI
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()
