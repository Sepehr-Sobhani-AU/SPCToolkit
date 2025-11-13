# plugins/Training/label_clusters_plugin.py
"""
Plugin for labeling multiple selected clusters as training data.

Workflow:
1. User runs DBSCAN clustering
2. User computes eigenvalues
3. User makes clusters visible in viewer
4. User clicks on points in clusters to select them (Shift+Click)
5. User runs this plugin
6. Dialog asks for class label
7. All selected clusters are saved with that label
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.point_cloud import PointCloud


class LabelClustersPlugin(ActionPlugin):
    """
    Action plugin for labeling multiple selected clusters as training data.

    Uses the existing point selection mechanism (Shift+Click) to select clusters.
    Finds which clusters contain the selected points and saves them all with
    the same class label.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "label_clusters"

    def _get_existing_classes(self, training_folder: str) -> List[str]:
        """
        Scan the training folder for existing class subdirectories.

        Args:
            training_folder: Path to the training data folder

        Returns:
            List of existing class names (folder names)
        """
        existing_classes = []

        # Check if folder exists
        if os.path.exists(training_folder) and os.path.isdir(training_folder):
            try:
                # Get all subdirectories
                for item in os.listdir(training_folder):
                    item_path = os.path.join(training_folder, item)
                    if os.path.isdir(item_path):
                        existing_classes.append(item)
            except (PermissionError, OSError):
                # If we can't read the folder, return empty list
                pass

        return sorted(existing_classes)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define parameters for the labeling dialog.

        Returns:
            Dict[str, Any]: Parameter schema with class selection and save location
        """
        # Get the current training folder from global variables
        current_training_folder = global_variables.training_data_folder

        # Scan for existing class folders
        existing_classes = self._get_existing_classes(current_training_folder)

        # Default class options
        default_classes = ["Tree", "Pole", "Building", "Car", "Vegetation", "Ground", "Cable", "Sign", "Other"]

        # Combine existing classes with defaults (remove duplicates, keep existing first)
        all_classes = []
        for cls in existing_classes:
            if cls not in all_classes:
                all_classes.append(cls)
        for cls in default_classes:
            if cls not in all_classes:
                all_classes.append(cls)

        # If no classes found, use defaults
        if not all_classes:
            all_classes = default_classes

        return {
            "class_name": {
                "type": "choice",
                "options": all_classes,
                "default": all_classes[0] if all_classes else "Tree",
                "label": "Class Label",
                "description": "Select or type a class name for the selected clusters"
            },
            "save_directory": {
                "type": "directory",
                "default": current_training_folder,
                "label": "Training Data Folder",
                "description": "Directory where training data will be saved (persists across sessions)"
            },
            "max_points_per_cluster": {
                "type": "int",
                "default": 20000,
                "min": 1024,
                "max": 100000,
                "label": "Max Points Per Cluster",
                "description": "Maximum number of points to save per cluster (larger clusters will be randomly subsampled)"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the label clusters action.

        Args:
            main_window: The main application window
            params: Parameters from the dialog (class_name, save_directory)
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
                "Please select a cluster branch (with eigenvalues) before labeling."
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
                "2. Compute eigenvalues\n"
                "3. Select the eigenvalues branch\n"
                "4. Make it visible in viewer\n"
                "5. Click on clusters to select them (Shift+Click)"
            )
            return

        # Get selected point indices from viewer
        selected_indices = viewer_widget.picked_points_indices

        if not selected_indices:
            QMessageBox.warning(
                main_window,
                "No Points Selected",
                "No clusters are selected.\n\n"
                "Please click on points in the clusters you want to label.\n"
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

        # Extract parameters
        class_name = params["class_name"].strip()
        save_directory = Path(params["save_directory"])
        max_points_per_cluster = params["max_points_per_cluster"]

        # Update global training folder for persistence
        global_variables.training_data_folder = str(save_directory)

        # Create class directory
        class_dir = save_directory / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Process each selected cluster
        saved_clusters = []
        skipped_clusters = []

        for cluster_id in sorted(selected_cluster_ids):
            # Extract cluster points
            cluster_mask = point_cloud.cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)

            # Check minimum points
            if cluster_size < 1024:
                skipped_clusters.append((cluster_id, cluster_size))
                continue

            # Extract cluster data
            try:
                cluster_data, was_subsampled = self.extract_cluster_data(point_cloud, cluster_mask, max_points_per_cluster)
                saved_size = len(cluster_data)

                # Find next available cluster ID
                cluster_file_id = self.get_next_cluster_id(class_dir)

                # Save
                save_path = class_dir / f"cluster_{cluster_file_id:03d}.npy"
                np.save(save_path, cluster_data)

                saved_clusters.append((cluster_id, save_path.name, cluster_size, saved_size, was_subsampled))

            except Exception as e:
                QMessageBox.warning(
                    main_window,
                    "Save Error",
                    f"Failed to save cluster {cluster_id}:\n{str(e)}"
                )
                continue

        # Clear selection after saving
        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()

        # Show summary
        self.show_summary(main_window, class_name, saved_clusters, skipped_clusters)

    def extract_cluster_data(self, point_cloud: PointCloud, cluster_mask: np.ndarray, max_points: int) -> tuple:
        """
        Extract cluster data with XYZ + Normals + Eigenvalues.

        If the cluster has more points than max_points, randomly subsample to the limit.

        Args:
            point_cloud: The full point cloud
            cluster_mask: Boolean mask for the cluster
            max_points: Maximum number of points to extract (will subsample if exceeded)

        Returns:
            tuple: (numpy array of shape (N, 9): XYZ + Normals + Eigenvalues, was_subsampled: bool)
        """
        # Extract XYZ
        cluster_points = point_cloud.points[cluster_mask].copy()

        # Extract normals
        if point_cloud.normals is not None:
            cluster_normals = point_cloud.normals[cluster_mask]
        else:
            raise ValueError("Point cloud has no normals. Compute normals before labeling.")

        # Extract eigenvalues
        if 'eigenvalues' in point_cloud.attributes:
            eigenvalues = point_cloud.attributes['eigenvalues']
            cluster_eigenvalues = eigenvalues[cluster_mask]
        else:
            raise ValueError("Point cloud has no eigenvalues. Compute eigenvalues before labeling.")

        # Subsample if cluster exceeds max_points
        num_points = len(cluster_points)
        was_subsampled = False
        if num_points > max_points:
            # Randomly select max_points indices
            subsample_indices = np.random.choice(num_points, size=max_points, replace=False)
            cluster_points = cluster_points[subsample_indices]
            cluster_normals = cluster_normals[subsample_indices]
            cluster_eigenvalues = cluster_eigenvalues[subsample_indices]
            was_subsampled = True
            print(f"Subsampled cluster from {num_points} to {max_points} points")

        # Combine: XYZ (3) + Normals (3) + Eigenvalues (3) = 9 features
        combined = np.hstack([cluster_points, cluster_normals, cluster_eigenvalues]).astype(np.float32)

        return combined, was_subsampled

    def get_next_cluster_id(self, class_dir: Path) -> int:
        """
        Find the next available cluster ID in the class directory.

        Args:
            class_dir: Directory containing cluster files

        Returns:
            Next available cluster ID
        """
        existing_files = list(class_dir.glob("cluster_*.npy"))
        if not existing_files:
            return 1

        # Extract IDs from filenames
        ids = []
        for file in existing_files:
            try:
                id_str = file.stem.split('_')[1]
                ids.append(int(id_str))
            except (IndexError, ValueError):
                continue

        return max(ids) + 1 if ids else 1

    def show_summary(self, parent, class_name, saved_clusters, skipped_clusters):
        """Show summary of the labeling operation."""
        message = f"Labeling complete for class: {class_name}\n\n"

        if saved_clusters:
            message += f"✓ Saved {len(saved_clusters)} cluster(s):\n"
            for cluster_id, filename, original_size, saved_size, was_subsampled in saved_clusters:
                if was_subsampled:
                    message += f"  • Cluster {cluster_id}: {filename} ({original_size:,} → {saved_size:,} points, subsampled)\n"
                else:
                    message += f"  • Cluster {cluster_id}: {filename} ({saved_size:,} points)\n"

        if skipped_clusters:
            message += f"\n✗ Skipped {len(skipped_clusters)} cluster(s) (< 1024 points):\n"
            for cluster_id, size in skipped_clusters:
                message += f"  • Cluster {cluster_id}: {size:,} points\n"

        if saved_clusters:
            QMessageBox.information(parent, "Labeling Complete", message)
        else:
            QMessageBox.warning(parent, "No Clusters Saved", message)
