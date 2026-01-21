"""
Plugin for manually classifying clusters.

Workflow:
1. User runs DBSCAN clustering
2. User makes clusters visible in viewer
3. User clicks on points in clusters to select them (Shift+Click)
4. User runs this plugin
5. Dialog asks for class label
6. Selected clusters are classified and stored in the Clusters DataNode
"""

import numpy as np
from typing import Dict, Any, Set, Optional, Tuple
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.clusters import Clusters


class ClassifyClusterPlugin(ActionPlugin):
    """
    Action plugin for manually classifying clusters.

    Uses the existing point selection mechanism (Shift+Click) to select clusters.
    Finds which clusters contain the selected points and assigns them a class label.
    Updates or creates cluster_names in the Clusters object.
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
        cluster_labels = point_cloud.get_attribute("cluster_labels")
        if cluster_labels is None:
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
            if idx < len(cluster_labels):
                cluster_id = cluster_labels[idx]
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

        # Check if Clusters with names already exists as a child of this branch
        existing_clusters_node, existing_clusters = self._find_existing_named_clusters(
            data_nodes, selected_uid
        )

        # Create or update Clusters with names
        clusters = self.create_or_update_clusters(
            cluster_labels,
            selected_cluster_ids,
            class_name,
            existing_clusters
        )

        # Create or update the DataNode
        if existing_clusters_node is not None:
            # Update existing node's data
            existing_clusters_node.data = clusters

            # Make the clusters branch visible to show updated classifications
            from PyQt5.QtCore import Qt
            clusters_item = tree_widget.branches_dict.get(str(existing_clusters_node.uid))
            if clusters_item:
                clusters_item.setCheckState(0, Qt.Checked)
                tree_widget.visibility_status[str(existing_clusters_node.uid)] = True

            # Manually trigger visibility update to refresh the visualization
            data_manager._render_visible_data(tree_widget.visibility_status, zoom_extent=False)
        else:
            # Create new DataNode for Clusters with names
            from core.data_node import DataNode
            import uuid

            # Convert selected_uid from string to UUID
            parent_uuid = uuid.UUID(selected_uid) if isinstance(selected_uid, str) else selected_uid

            clusters_node = DataNode(
                params="cluster_labels",
                data=clusters,
                data_type="cluster_labels",
                parent_uid=parent_uuid,
                depends_on=[parent_uuid],
                tags=["classification"]
            )

            # Add to data_nodes collection
            clusters_uid = data_nodes.add_node(clusters_node)

            # Add to tree widget and make it visible
            # CRITICAL: Block signals BEFORE adding branch to prevent auto-check from triggering render
            from PyQt5.QtCore import Qt
            tree_widget.blockSignals(True)

            try:
                # Add branch (it will be auto-checked inside add_branch, but signals are blocked)
                tree_widget.add_branch(str(clusters_uid), str(selected_uid), "cluster_labels")

                # Get the newly added item and set it to checked (visible)
                clusters_item = tree_widget.branches_dict.get(str(clusters_uid))
                if clusters_item:
                    clusters_item.setCheckState(0, Qt.Checked)
                    tree_widget.visibility_status[str(clusters_uid)] = True
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
            f"Total classified clusters: {len(clusters.cluster_names)}"
        )

    def create_or_update_clusters(
        self,
        cluster_labels: np.ndarray,
        selected_cluster_ids: Set[int],
        class_name: str,
        existing_clusters: Optional[Clusters] = None
    ) -> Clusters:
        """
        Create a new Clusters object or update an existing one with cluster names.

        Args:
            cluster_labels: Array of cluster labels for each point
            selected_cluster_ids: Set of cluster IDs to classify
            class_name: Class name to assign to the selected clusters
            existing_clusters: Existing Clusters to update (or None)

        Returns:
            Clusters: New or updated Clusters object with cluster_names
        """
        # Start with existing mapping or create new one
        if existing_clusters is not None:
            cluster_names = existing_clusters.cluster_names.copy()
            cluster_colors = existing_clusters.cluster_colors.copy()
        else:
            cluster_names = {}
            cluster_colors = {}

        # Add/update classifications for selected clusters
        for cluster_id in selected_cluster_ids:
            cluster_names[cluster_id] = class_name

        # Ensure all classes in the mapping have colors
        for cls in set(cluster_names.values()):
            if cls not in cluster_colors:
                # Use default color if available, otherwise generate random color
                if cls in self.DEFAULT_CLASS_COLORS:
                    cluster_colors[cls] = self.DEFAULT_CLASS_COLORS[cls]
                else:
                    # Generate a random color
                    cluster_colors[cls] = np.random.rand(3).astype(np.float32)

        # Create new Clusters object with names
        clusters = Clusters(
            labels=cluster_labels,
            cluster_names=cluster_names,
            cluster_colors=cluster_colors
        )

        return clusters

    def _find_existing_named_clusters(self, data_nodes, branch_uid: str) -> tuple:
        """
        Search for an existing Clusters with names as a child of the given branch.

        Args:
            data_nodes: DataNodes collection
            branch_uid: UID of the parent branch (Clusters)

        Returns:
            Tuple of (DataNode or None, Clusters or None)
        """
        import uuid

        # Get all nodes
        all_nodes = data_nodes.data_nodes

        # Convert branch_uid to UUID for comparison
        parent_uuid = uuid.UUID(branch_uid) if isinstance(branch_uid, str) else branch_uid

        # Search for a child node with data_type="cluster_labels" and has cluster_names
        for uid, node in all_nodes.items():
            if node.parent_uid == parent_uuid and node.data_type == "cluster_labels":
                if hasattr(node.data, 'has_names') and node.data.has_names():
                    return node, node.data

        return None, None
