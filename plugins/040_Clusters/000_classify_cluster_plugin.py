"""
Plugin for manually classifying clusters.

Workflow:
1. User runs DBSCAN clustering
2. User makes clusters visible in viewer
3. User clicks on points in clusters to select them (Shift+Click)
4. User runs this plugin
5. Dialog asks for class label
6. Selected clusters are classified and stored in FeatureClasses DataNode
"""

import numpy as np
from typing import Dict, Any, Set, Optional, Tuple
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.feature_classes import FeatureClasses


class ClassifyClusterPlugin(ActionPlugin):
    """
    Action plugin for manually classifying clusters.

    Uses the existing point selection mechanism (Shift+Click) to select clusters.
    Finds which clusters contain the selected points and assigns them a class label.
    Creates or updates a FeatureClasses DataNode to store the classifications.
    """

    # Default class colors (RGB in [0, 1] range)
    DEFAULT_CLASS_COLORS = {
        "Tree": np.array([0.0, 0.8, 0.0]),          # Green
        "Building": np.array([0.7, 0.3, 0.1]),      # Brown
        "Car": np.array([0.0, 0.0, 1.0]),           # Blue
        "Pole": np.array([0.5, 0.5, 0.5]),          # Gray
        "Ground": np.array([0.6, 0.4, 0.2]),        # Tan
        "Vegetation": np.array([0.2, 0.6, 0.2]),    # Light green
        "Cable": np.array([0.0, 0.0, 0.0]),         # Black
        "Sign": np.array([1.0, 1.0, 0.0]),          # Yellow
        "Traffic_Light": np.array([1.0, 0.0, 0.0]), # Red
        "Fence": np.array([0.6, 0.3, 0.0]),         # Dark brown
        "Pedestrian": np.array([1.0, 0.5, 0.0]),    # Orange
        "Moving_Car": np.array([0.0, 0.5, 1.0]),    # Cyan
        "Other": np.array([0.9, 0.9, 0.9]),         # Light gray
    }

    def get_name(self) -> str:
        """Return the plugin name."""
        return "classify_cluster"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define parameters for the classification dialog.

        Returns:
            Dict[str, Any]: Parameter schema with class selection
        """
        return {
            "class_name": {
                "type": "choice",
                "options": list(self.DEFAULT_CLASS_COLORS.keys()),
                "default": "Tree",
                "label": "Class Label",
                "description": "Select or type a class name for the selected clusters"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the classify cluster action.

        Args:
            main_window: The main application window
            params: Parameters from the dialog (class_name)
        """
        # Get global instances
        data_manager = global_variables.global_data_manager
        viewer_widget = global_variables.global_pcd_viewer_widget

        # Get selected branches
        selected_branches = data_manager.selected_branches

        # Validate branch selection
        if not selected_branches:
            QMessageBox.warning(
                main_window,
                "No Branch Selected",
                "Please select a cluster branch before classifying."
            )
            return

        if len(selected_branches) > 1:
            QMessageBox.warning(
                main_window,
                "Multiple Branches",
                "Please select only ONE branch at a time."
            )
            return

        # Reconstruct the branch to get PointCloud
        selected_uid = selected_branches[0]
        try:
            point_cloud = data_manager.reconstruct_branch(selected_uid)
        except Exception as e:
            QMessageBox.critical(
                main_window,
                "Reconstruction Error",
                f"Failed to reconstruct branch:\n{str(e)}"
            )
            return

        # Check if point cloud has cluster labels
        if not hasattr(point_cloud, 'cluster_labels') or point_cloud.cluster_labels is None:
            QMessageBox.warning(
                main_window,
                "No Cluster Labels",
                "The selected branch has no cluster labels.\n\n"
                "Please:\n"
                "1. Run DBSCAN clustering\n"
                "2. Select the cluster branch\n"
                "3. Make it visible in viewer\n"
                "4. Click on clusters to select them (Shift+Click)"
            )
            return

        # Get selected point indices from viewer
        selected_indices = viewer_widget.picked_points_indices

        if not selected_indices:
            QMessageBox.warning(
                main_window,
                "No Points Selected",
                "No clusters are selected.\n\n"
                "Please click on points in the clusters you want to classify.\n"
                "Use Shift+Click to select points."
            )
            return

        # Find which clusters contain the selected points
        selected_cluster_ids = set()
        for idx in selected_indices:
            if idx < len(point_cloud.cluster_labels):
                cluster_id = point_cloud.cluster_labels[idx]
                # Ignore noise points (cluster_id == -1)
                if cluster_id != -1:
                    selected_cluster_ids.add(cluster_id)

        if not selected_cluster_ids:
            QMessageBox.warning(
                main_window,
                "No Valid Clusters",
                "Selected points do not belong to any valid clusters (all noise points)."
            )
            return

        # Get class name from parameters
        class_name = params["class_name"].strip()

        # Get data_nodes and tree_widget from global variables
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Check if FeatureClasses already exists as a child of this branch
        existing_fc_node, existing_feature_classes = self._find_existing_feature_classes(
            data_nodes, selected_uid
        )

        # Create or update FeatureClasses
        feature_classes = self.create_or_update_feature_classes(
            point_cloud.cluster_labels,
            selected_cluster_ids,
            class_name,
            existing_feature_classes
        )

        # Create or update the DataNode
        if existing_fc_node is not None:
            # Update existing node's data
            existing_fc_node.data = feature_classes

            # Make the feature_classes branch visible to show updated classifications
            from PyQt5.QtCore import Qt
            fc_item = tree_widget.branches_dict.get(str(existing_fc_node.uid))
            if fc_item:
                fc_item.setCheckState(0, Qt.Checked)
                tree_widget.visibility_status[str(existing_fc_node.uid)] = True

            # Manually trigger visibility update to refresh the visualization
            data_manager._render_visible_data(tree_widget.visibility_status, zoom_extent=False)
        else:
            # Create new DataNode for FeatureClasses
            from core.data_node import DataNode
            import uuid

            # Convert selected_uid from string to UUID
            parent_uuid = uuid.UUID(selected_uid) if isinstance(selected_uid, str) else selected_uid

            fc_node = DataNode(
                params="feature_classes",
                data=feature_classes,
                data_type="feature_classes",
                parent_uid=parent_uuid,
                depends_on=[parent_uuid],
                tags=["classification"]
            )

            # Add to data_nodes collection
            fc_uid = data_nodes.add_node(fc_node)

            # Add to tree widget and make it visible
            # CRITICAL: Block signals BEFORE adding branch to prevent auto-check from triggering render
            from PyQt5.QtCore import Qt
            tree_widget.blockSignals(True)

            try:
                # Add branch (it will be auto-checked inside add_branch, but signals are blocked)
                tree_widget.add_branch(str(fc_uid), str(selected_uid), "feature_classes")

                # Get the newly added item and set it to checked (visible)
                fc_item = tree_widget.branches_dict.get(str(fc_uid))
                if fc_item:
                    fc_item.setCheckState(0, Qt.Checked)
                    tree_widget.visibility_status[str(fc_uid)] = True
            finally:
                # Always re-enable signals
                tree_widget.blockSignals(False)

            # Manually trigger visibility update since signals were blocked
            data_manager._render_visible_data(tree_widget.visibility_status, zoom_extent=False)

        # Clear selection after classification
        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()

        # Show success message
        cluster_list = ", ".join(str(cid) for cid in sorted(selected_cluster_ids))
        QMessageBox.information(
            main_window,
            "Classification Complete",
            f"Classified {len(selected_cluster_ids)} cluster(s) as '{class_name}':\n"
            f"Cluster IDs: {cluster_list}\n\n"
            f"Total classified clusters: {len(feature_classes.class_mapping)}"
        )

    def create_or_update_feature_classes(
        self,
        cluster_labels: np.ndarray,
        selected_cluster_ids: Set[int],
        class_name: str,
        existing_feature_classes: Optional[FeatureClasses] = None
    ) -> FeatureClasses:
        """
        Create a new FeatureClasses object or update an existing one.

        Args:
            cluster_labels: Array of cluster labels for each point
            selected_cluster_ids: Set of cluster IDs to classify
            class_name: Class name to assign to the selected clusters
            existing_feature_classes: Existing FeatureClasses to update (or None)

        Returns:
            FeatureClasses: New or updated FeatureClasses object
        """
        # Start with existing mapping or create new one
        if existing_feature_classes is not None:
            class_mapping = existing_feature_classes.class_mapping.copy()
            class_colors = existing_feature_classes.class_colors.copy()
        else:
            class_mapping = {}
            class_colors = {}

        # Add/update classifications for selected clusters
        for cluster_id in selected_cluster_ids:
            class_mapping[cluster_id] = class_name

        # Ensure all classes in the mapping have colors
        for cls in set(class_mapping.values()):
            if cls not in class_colors:
                # Use default color if available, otherwise generate random color
                if cls in self.DEFAULT_CLASS_COLORS:
                    class_colors[cls] = self.DEFAULT_CLASS_COLORS[cls]
                else:
                    # Generate a random color
                    class_colors[cls] = np.random.rand(3).astype(np.float32)

        # Create new FeatureClasses object
        feature_classes = FeatureClasses(
            labels=cluster_labels,
            class_mapping=class_mapping,
            class_colors=class_colors
        )

        return feature_classes

    def _find_existing_feature_classes(self, data_nodes, branch_uid: str) -> tuple:
        """
        Search for an existing FeatureClasses as a child of the given branch.

        Args:
            data_nodes: DataNodes collection
            branch_uid: UID of the parent branch (Clusters)

        Returns:
            Tuple of (DataNode or None, FeatureClasses or None)
        """
        import uuid

        # Get all nodes
        all_nodes = data_nodes.data_nodes

        # Convert branch_uid to UUID for comparison
        parent_uuid = uuid.UUID(branch_uid) if isinstance(branch_uid, str) else branch_uid

        # Search for a child node with data_type="feature_classes" and parent_uid=branch_uid
        for uid, node in all_nodes.items():
            if node.parent_uid == parent_uuid and node.data_type == "feature_classes":
                return node, node.data

        return None, None
