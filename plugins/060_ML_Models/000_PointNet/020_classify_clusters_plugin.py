"""
PointNet Cluster Classification Plugin (PyTorch)

Uses a trained PointNet model to automatically classify clusters in a Clusters branch.

Workflow:
1. User selects a Clusters branch
2. (Optional) User selects specific clusters with Shift+Click
3. User runs this plugin
4. Dialog prompts for:
   - Model directory
   - Process mode (selected clusters or all)
   - Confidence threshold
   - Skip small clusters option
5. Plugin loads model and classifies clusters
6. Creates/updates Clusters DataNode with cluster_names for predictions
"""

import os
import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.clusters import Clusters
from models.pointnet.inference import (
    load_model_with_metadata,
    classify_clusters_batch
)
from gui.dialogs.classification_progress_dialog import ClassificationProgressDialog


class ClassifyClustersMLPlugin(ActionPlugin):
    """
    Action plugin for ML-based cluster classification using PointNet (PyTorch).
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
        "Unclassified": np.array([0.5, 0.5, 0.5]),  # Gray
        "Other": np.array([0.9, 0.9, 0.9]),         # Light gray
    }

    # Class variable for last used parameters
    last_params = {
        "model_directory": "models",
        "process_mode": "Selected Clusters Only",
        "confidence_threshold": 0.5,
        "skip_small_clusters": True,
        "max_points_per_cluster": 20000
    }

    def get_name(self) -> str:
        """Return the plugin name."""
        return "classify_clusters"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define parameters for the classification dialog.

        Returns:
            Dict[str, Any]: Parameter schema
        """
        return {
            "model_directory": {
                "type": "directory",
                "default": self.last_params["model_directory"],
                "label": "Model Directory",
                "description": "Directory containing trained PointNet model files (pointnet_best.pt, class_mapping.json, training_metadata.json)"
            },
            "process_mode": {
                "type": "choice",
                "options": ["Selected Clusters Only", "All Clusters"],
                "default": self.last_params["process_mode"],
                "label": "Processing Mode",
                "description": "Classify only selected clusters (Shift+Click) or all clusters in branch"
            },
            "confidence_threshold": {
                "type": "float",
                "default": self.last_params["confidence_threshold"],
                "min": 0.0,
                "max": 1.0,
                "decimals": 2,
                "label": "Confidence Threshold",
                "description": "Minimum prediction confidence (clusters below threshold marked as 'Unclassified')"
            },
            "skip_small_clusters": {
                "type": "bool",
                "default": self.last_params["skip_small_clusters"],
                "label": "Skip Small Clusters",
                "description": "Skip clusters with fewer points than model's training num_points"
            },
            "max_points_per_cluster": {
                "type": "int",
                "default": self.last_params.get("max_points_per_cluster", 20000),
                "min": 1024,
                "max": 100000,
                "label": "Max Points Per Cluster",
                "description": "Maximum points per cluster. Large clusters are subsampled to this limit before feature computation (must match training data export limit)"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the cluster classification.

        Args:
            main_window: The main application window
            params: Parameters from the dialog
        """
        # Store parameters for next time
        ClassifyClustersMLPlugin.last_params = params.copy()

        # Get global instances
        data_manager = global_variables.global_data_manager
        viewer_widget = global_variables.global_pcd_viewer_widget
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Get parameters
        model_dir = params["model_directory"].strip()
        process_mode = params["process_mode"]
        confidence_threshold = float(params["confidence_threshold"])
        skip_small_clusters = params["skip_small_clusters"]
        max_points_per_cluster = int(params["max_points_per_cluster"])

        # Validate branch selection
        selected_branches = data_manager.selected_branches

        if not selected_branches:
            QMessageBox.warning(
                main_window,
                "No Branch Selected",
                "Please select a Clusters branch before running classification."
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

        # Validate model directory
        if not os.path.exists(model_dir):
            QMessageBox.critical(
                main_window,
                "Invalid Directory",
                f"Model directory does not exist:\n{model_dir}"
            )
            return

        # Check required files (PyTorch format)
        required_files = ['pointnet_best.pt', 'class_mapping.json', 'training_metadata.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]

        if missing_files:
            QMessageBox.critical(
                main_window,
                "Missing Model Files",
                f"The following required files are missing from the model directory:\n" +
                "\n".join(missing_files) +
                "\n\nNote: PyTorch models use .pt extension (not .keras)"
            )
            return

        try:
            # Disable UI
            main_window.disable_menus()
            main_window.disable_tree()
            main_window.tree_overlay.show_processing("Loading model...")

            # Load model and metadata
            model, class_mapping, metadata = load_model_with_metadata(model_dir)

            print(f"\n{'='*80}")
            print(f"PointNet Cluster Classification (PyTorch)")
            print(f"{'='*80}")
            print(f"Model directory: {model_dir}")
            print(f"Model configuration:")
            print(f"  - Points per cluster: {metadata['num_points']}")
            print(f"  - Features: {metadata['num_features']}")
            print(f"  - Classes: {len(class_mapping)}")
            print(f"  - Class names: {', '.join(class_mapping.values())}")
            print(f"Confidence threshold: {confidence_threshold:.2f}")
            print(f"Skip small clusters: {skip_small_clusters}")
            print(f"Max points per cluster: {max_points_per_cluster}")

            # Reconstruct branch to get PointCloud
            main_window.tree_overlay.show_processing("Reconstructing branch...")

            point_cloud = data_manager.reconstruct_branch(selected_uid)

            # Check if point cloud has cluster labels
            cluster_labels = point_cloud.get_attribute("cluster_labels")
            if cluster_labels is None:
                raise ValueError(
                    "The selected branch has no cluster labels.\n\n"
                    "Please:\n"
                    "1. Run DBSCAN clustering\n"
                    "2. Select the cluster branch\n"
                    "3. Run this classification plugin"
                )

            # Determine which clusters to classify
            if process_mode == "Selected Clusters Only":
                # Get selected point indices from viewer
                selected_indices = viewer_widget.picked_points_indices

                if not selected_indices:
                    raise ValueError(
                        "No clusters are selected.\n\n"
                        "Please click on points in the clusters you want to classify.\n"
                        "Use Shift+Click to select points, or choose 'All Clusters' mode."
                    )

                # Find which clusters contain the selected points
                selected_cluster_ids = set()
                for idx in selected_indices:
                    if idx < len(cluster_labels):
                        cluster_id = cluster_labels[idx]
                        # Ignore noise points (cluster_id == -1)
                        if cluster_id != -1:
                            selected_cluster_ids.add(cluster_id)

                if not selected_cluster_ids:
                    raise ValueError(
                        "Selected points do not belong to any valid clusters (all noise points)."
                    )

                clusters_to_classify = list(selected_cluster_ids)
                print(f"\nProcessing mode: Selected clusters only")
                print(f"Selected clusters: {len(clusters_to_classify)}")

            else:  # All Clusters
                # Get all unique cluster IDs (excluding noise)
                unique_clusters = np.unique(cluster_labels)
                clusters_to_classify = [int(cid) for cid in unique_clusters if cid != -1]
                print(f"\nProcessing mode: All clusters")
                print(f"Total clusters: {len(clusters_to_classify)}")

            if len(clusters_to_classify) == 0:
                raise ValueError("No valid clusters to classify.")

            # Create progress dialog
            progress_dialog = ClassificationProgressDialog(
                main_window,
                total_clusters=len(clusters_to_classify)
            )
            progress_dialog.show()

            # Center the dialog
            from PyQt5.QtWidgets import QApplication
            screen_geometry = QApplication.desktop().screenGeometry()
            x = (screen_geometry.width() - progress_dialog.width()) // 2
            y = (screen_geometry.height() - progress_dialog.height()) // 2
            progress_dialog.move(x, y)

            # Progress callback
            def progress_callback(current, total, cluster_id, class_name, confidence):
                if progress_dialog.cancelled:
                    return
                progress_dialog.update_progress(current, cluster_id, class_name, confidence)

            # Classify clusters
            main_window.tree_overlay.show_processing("Classifying clusters...")

            class_ids, stats = classify_clusters_batch(
                point_cloud=point_cloud,
                model=model,
                metadata=metadata,
                clusters_to_classify=clusters_to_classify,
                confidence_threshold=confidence_threshold,
                skip_small_clusters=skip_small_clusters,
                max_points_per_cluster=max_points_per_cluster,
                progress_callback=progress_callback
            )

            # Check if cancelled
            if progress_dialog.cancelled:
                print("\nClassification cancelled by user.")
                progress_dialog.close()
                return

            # Update progress dialog with final statistics
            progress_dialog.update_statistics(stats)

            # Create cluster_names mapping with Unclassified
            class_mapping_extended = {int(k): v for k, v in class_mapping.items()}
            unclassified_id = max(class_mapping_extended.keys()) + 1 if class_mapping_extended else 0
            class_mapping_extended[unclassified_id] = "Unclassified"

            # Build cluster_names dict: maps cluster_id -> class_name
            cluster_names = {}
            for cluster_id in clusters_to_classify:
                predicted_class_id = class_ids.get(cluster_id, -1)
                if predicted_class_id == -1:
                    cluster_names[cluster_id] = "Unclassified"
                else:
                    cluster_names[cluster_id] = class_mapping_extended.get(
                        predicted_class_id, "Unclassified"
                    )

            # Create cluster_colors
            cluster_colors = {}
            for class_name in set(cluster_names.values()):
                if class_name in self.DEFAULT_CLASS_COLORS:
                    cluster_colors[class_name] = self.DEFAULT_CLASS_COLORS[class_name]
                else:
                    # Generate random color for unknown classes
                    cluster_colors[class_name] = np.random.rand(3).astype(np.float32)

            # Create Clusters object with cluster_names
            clusters = Clusters(
                labels=cluster_labels.copy(),
                cluster_names=cluster_names,
                cluster_colors=cluster_colors
            )

            # Check if Clusters with names already exists as a child
            existing_clusters_node = self._find_existing_named_clusters(data_nodes, selected_uid)

            # Create or update DataNode
            if existing_clusters_node is not None:
                # Update existing node
                existing_clusters_node.data = clusters

                # Make visible
                clusters_item = tree_widget.branches_dict.get(str(existing_clusters_node.uid))
                if clusters_item:
                    clusters_item.setCheckState(0, Qt.Checked)
                    tree_widget.visibility_status[str(existing_clusters_node.uid)] = True

                # Trigger visibility update
                data_manager._render_visible_data(tree_widget.visibility_status, zoom_extent=False)

            else:
                # Create new DataNode
                from core.data_node import DataNode
                import uuid

                parent_uuid = uuid.UUID(selected_uid) if isinstance(selected_uid, str) else selected_uid

                clusters_node = DataNode(
                    params="cluster_labels",
                    data=clusters,
                    data_type="cluster_labels",
                    parent_uid=parent_uuid,
                    depends_on=[parent_uuid],
                    tags=["classification", "ml", "pointnet"]
                )

                # Add to data_nodes
                clusters_uid = data_nodes.add_node(clusters_node)

                # Add to tree widget (block signals to prevent auto-render)
                tree_widget.blockSignals(True)
                try:
                    tree_widget.add_branch(str(clusters_uid), str(selected_uid), "cluster_labels")

                    # Set visible
                    clusters_item = tree_widget.branches_dict.get(str(clusters_uid))
                    if clusters_item:
                        clusters_item.setCheckState(0, Qt.Checked)
                        tree_widget.visibility_status[str(clusters_uid)] = True
                finally:
                    tree_widget.blockSignals(False)

                # Manually trigger visibility update
                data_manager._render_visible_data(tree_widget.visibility_status, zoom_extent=False)

            # Clear point selection
            viewer_widget.picked_points_indices.clear()
            viewer_widget.update()

            # Mark progress dialog as complete
            progress_dialog.classification_completed(stats)

            # Count clusters per class
            cluster_class_counts = {}
            for cluster_id, class_name in cluster_names.items():
                cluster_class_counts[class_name] = cluster_class_counts.get(class_name, 0) + 1

            # Print summary
            print(f"\n{'='*80}")
            print(f"Classification Complete!")
            print(f"{'='*80}")
            print(f"Classified: {stats['classified']} clusters")
            print(f"Skipped (too small): {stats['skipped_small']} clusters")
            print(f"Unclassified (low confidence): {stats['skipped_low_confidence']} clusters")
            print(f"\nCluster classification results:")
            for class_name, count in sorted(cluster_class_counts.items()):
                print(f"  {class_name}: {count} cluster(s)")
            print(f"{'='*80}")

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"\nERROR during classification:\n{error_msg}")

            QMessageBox.critical(
                main_window,
                "Classification Error",
                f"An error occurred during classification:\n\n{str(e)}"
            )

        finally:
            # Re-enable UI
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

    def _find_existing_named_clusters(self, data_nodes, branch_uid: str):
        """
        Search for an existing Clusters with names as a child of the given branch.

        Args:
            data_nodes: DataNodes collection
            branch_uid: UID of the parent branch

        Returns:
            DataNode or None
        """
        import uuid

        all_nodes = data_nodes.data_nodes
        parent_uuid = uuid.UUID(branch_uid) if isinstance(branch_uid, str) else branch_uid

        for uid, node in all_nodes.items():
            if node.parent_uid == parent_uuid and node.data_type == "cluster_labels":
                if hasattr(node.data, 'has_names') and node.data.has_names():
                    return node

        return None
