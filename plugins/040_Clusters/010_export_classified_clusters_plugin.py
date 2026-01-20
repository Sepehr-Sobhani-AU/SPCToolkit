"""
Export Classified Clusters Plugin

Exports clusters with defined cluster names as numpy arrays for ML training data preparation.
Each cluster is saved as [XYZ, RGB] data to a subfolder named after its class, with optional subsampling.
"""

from typing import Dict, Any
import os
import re
import numpy as np
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class ExportClassifiedClustersPlugin(ActionPlugin):
    """
    Export classified clusters as numpy arrays for training data preparation.

    This plugin exports clusters that have defined cluster names (classifications).
    Each cluster is saved as a .npy file containing [XYZ, RGB] data (n, 6 array) in a
    subfolder corresponding to its class label. Clusters with more points than the
    specified maximum are randomly subsampled.

    File naming uses incremental numbering per class (tree_1.npy, tree_2.npy, etc.)
    """

    # Class variable to store the last used export directory for the session
    last_export_dir = ""

    def get_name(self) -> str:
        return "export_classified_clusters"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "export_directory": {
                "type": "directory",
                "default": ExportClassifiedClustersPlugin.last_export_dir,
                "label": "Export Directory",
                "description": "Main directory where class subfolders will be created"
            },
            "max_points": {
                "type": "int",
                "default": 20000,
                "min": 10,
                "max": 100000,
                "label": "Maximum Points Per Cluster",
                "description": "Maximum number of points to save per cluster. Clusters with more points will be randomly subsampled."
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Export classified clusters to disk.

        Args:
            main_window: The main application window
            params: Dictionary containing 'export_directory' and 'max_points' parameters
        """
        # Get the data manager
        data_manager = global_variables.global_data_manager

        # Check if a branch is selected
        if not data_manager.selected_branches:
            QMessageBox.warning(
                main_window,
                "No Selection",
                "Please select a Clusters branch with names to export."
            )
            return

        # Get the selected branch
        selected_uid = data_manager.selected_branches[0]

        # Convert string UID to UUID if needed
        import uuid
        if isinstance(selected_uid, str):
            selected_uid_uuid = uuid.UUID(selected_uid)
        else:
            selected_uid_uuid = selected_uid

        data_node = global_variables.global_data_nodes.get_node(selected_uid_uuid)

        # Check if it's a Clusters node with names
        if data_node.data_type != "cluster_labels":
            QMessageBox.warning(
                main_window,
                "Invalid Selection",
                "Please select a Clusters branch with names to export.\n"
                f"Selected branch type: {data_node.data_type}"
            )
            return

        clusters = data_node.data

        # Check if there are any cluster names (classifications)
        if not hasattr(clusters, 'cluster_names') or not clusters.cluster_names:
            QMessageBox.warning(
                main_window,
                "No Classifications",
                "The selected Clusters branch has no cluster names defined.\n"
                "Please classify clusters first using 'Classify Cluster' or ML classification."
            )
            return

        # Get export directory from parameters
        export_dir = params['export_directory'].strip()

        # Validate export directory
        if not export_dir:
            QMessageBox.warning(
                main_window,
                "No Directory Selected",
                "Please select an export directory."
            )
            return

        if not os.path.exists(export_dir):
            QMessageBox.warning(
                main_window,
                "Invalid Directory",
                f"The selected directory does not exist:\n{export_dir}"
            )
            return

        # Reconstruct the branch to get the point cloud
        try:
            point_cloud = data_manager.reconstruct_branch(selected_uid)
        except Exception as e:
            QMessageBox.critical(
                main_window,
                "Reconstruction Error",
                f"Failed to reconstruct branch:\n{str(e)}"
            )
            return

        # Get parameters
        max_points = int(params['max_points'])

        # Export clusters
        exported_count = 0
        skipped_count = 0
        downsampled_count = 0
        class_counts = {}  # Track exports per class
        errors = []  # Track any errors

        try:
            for cluster_id, class_name in clusters.cluster_names.items():
                try:
                    # Create mask for this cluster
                    mask = clusters.labels == cluster_id
                    num_points_in_cluster = np.sum(mask)

                    # Skip if no points found
                    if num_points_in_cluster == 0:
                        skipped_count += 1
                        errors.append(f"Cluster {cluster_id} ({class_name}): No points found")
                        continue

                    # Extract points and colors for this cluster
                    cluster_points = point_cloud.points[mask]
                    cluster_colors = point_cloud.colors[mask]

                    # Subsample if necessary
                    original_size = len(cluster_points)
                    if original_size > max_points:
                        indices = np.random.choice(original_size, max_points, replace=False)
                        cluster_points = cluster_points[indices]
                        cluster_colors = cluster_colors[indices]
                        downsampled_count += 1

                    # Combine XYZ and RGB into single (n, 6) array
                    cluster_data = np.hstack([cluster_points, cluster_colors])

                    # Create class folder if needed
                    class_dir = os.path.join(export_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    # Determine next incremental number for this class
                    next_number = self._get_next_file_number(class_dir, class_name)

                    # Generate filename
                    filename = f"{class_name.lower()}_{next_number}.npy"
                    filepath = os.path.join(class_dir, filename)

                    # Save as numpy array
                    np.save(filepath, cluster_data)

                    exported_count += 1
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                except Exception as e:
                    skipped_count += 1
                    errors.append(f"Cluster {cluster_id} ({class_name}): {str(e)}")

        except Exception as e:
            QMessageBox.critical(
                main_window,
                "Export Error",
                f"Failed during export:\n{str(e)}"
            )
            return

        # Build summary message
        if exported_count == 0:
            error_msg = "No clusters were exported!\n\n"
            error_msg += f"Total classified clusters: {len(clusters.cluster_names)}\n"
            error_msg += f"Point cloud shape: {point_cloud.points.shape}\n"
            error_msg += f"Labels shape: {clusters.labels.shape}\n\n"
            if errors:
                error_msg += "Errors:\n" + "\n".join(errors[:5])  # Show first 5 errors
            QMessageBox.warning(
                main_window,
                "Export Failed",
                error_msg
            )
            return

        # Store the export directory for future use in this session
        ExportClassifiedClustersPlugin.last_export_dir = export_dir

        summary_lines = [f"Successfully exported {exported_count} clusters to:\n{export_dir}\n"]

        if class_counts:
            summary_lines.append("Breakdown by class:")
            for class_name, count in sorted(class_counts.items()):
                summary_lines.append(f"  - {class_name}: {count} clusters")

        if downsampled_count > 0:
            summary_lines.append(f"\nDownsampled {downsampled_count} clusters (exceeded {max_points} points)")

        if skipped_count > 0:
            summary_lines.append(f"\nSkipped {skipped_count} clusters (see console for details)")
            if errors:
                print("Export warnings:")
                for error in errors:
                    print(f"  - {error}")

        # Show success message
        QMessageBox.information(
            main_window,
            "Export Complete",
            "\n".join(summary_lines)
        )

    def _get_next_file_number(self, class_dir: str, class_name: str) -> int:
        """
        Scan the class directory for existing files and determine the next incremental number.

        Args:
            class_dir: Path to the class-specific directory
            class_name: Name of the class (e.g., "Tree", "Car")

        Returns:
            Next available number for file naming
        """
        # Pattern to match files like "tree_1.npy", "tree_2.npy", etc.
        pattern = re.compile(rf"^{re.escape(class_name.lower())}_(\d+)\.npy$")

        max_number = 0

        # Check if directory exists and has files
        if os.path.exists(class_dir):
            for filename in os.listdir(class_dir):
                match = pattern.match(filename)
                if match:
                    number = int(match.group(1))
                    max_number = max(max_number, number)

        # Return next number (starting at 1 if no files exist)
        return max_number + 1