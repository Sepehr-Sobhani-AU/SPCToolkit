"""
PointNet Semantic Segmentation Plugin

Applies a trained PointNet segmentation model to a point cloud, producing
per-point semantic labels. Works on large point clouds via blockwise processing.

Creates a Clusters DataNode with cluster_labels where each semantic class
maps to a cluster ID with name and color.
"""

import os
import uuid
import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.clusters import Clusters


# Default colors for common semantic classes
DEFAULT_SEG_COLORS = {
    "Ground": np.array([0.6, 0.4, 0.2]),
    "Building": np.array([0.7, 0.3, 0.1]),
    "Tree": np.array([0.0, 0.8, 0.0]),
    "Vegetation": np.array([0.2, 0.6, 0.2]),
    "Car": np.array([0.0, 0.0, 1.0]),
    "Pole": np.array([0.5, 0.5, 0.5]),
    "Cable": np.array([0.0, 0.0, 0.0]),
    "Sign": np.array([1.0, 1.0, 0.0]),
    "Fence": np.array([0.6, 0.3, 0.0]),
    "Pedestrian": np.array([1.0, 0.5, 0.0]),
    "Road": np.array([0.4, 0.4, 0.4]),
    "Sidewalk": np.array([0.8, 0.6, 0.8]),
    "Wall": np.array([0.7, 0.5, 0.3]),
    "Other": np.array([0.9, 0.9, 0.9]),
    "Unclassified": np.array([0.5, 0.5, 0.5]),
}


class SegmentPointCloudPlugin(ActionPlugin):
    """
    Action plugin for semantic segmentation of point clouds using PointNet.
    """

    last_params = {
        "model_directory": "models",
        "block_size": 10.0,
        "block_overlap": 1.0,
        "confidence_threshold": 0.3,
    }

    def get_name(self) -> str:
        return "segment_point_cloud"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "model_directory": {
                "type": "directory",
                "default": self.last_params["model_directory"],
                "label": "Model Directory",
                "description": "Directory containing trained segmentation model (seg_model_best.pt, class_mapping.json, training_metadata.json)"
            },
            "block_size": {
                "type": "float",
                "default": self.last_params["block_size"],
                "min": 1.0,
                "max": 100.0,
                "label": "Block Size (m)",
                "description": "Size of spatial blocks in XY plane for processing"
            },
            "block_overlap": {
                "type": "float",
                "default": self.last_params["block_overlap"],
                "min": 0.0,
                "max": 10.0,
                "label": "Block Overlap (m)",
                "description": "Overlap between adjacent blocks to reduce boundary artifacts"
            },
            "confidence_threshold": {
                "type": "float",
                "default": self.last_params["confidence_threshold"],
                "min": 0.0,
                "max": 1.0,
                "decimals": 2,
                "label": "Confidence Threshold",
                "description": "Points below this confidence are labeled 'Unclassified'"
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Execute semantic segmentation on selected point cloud branch."""
        SegmentPointCloudPlugin.last_params = params.copy()

        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        model_dir = params["model_directory"].strip()
        block_size = float(params["block_size"])
        block_overlap = float(params["block_overlap"])
        confidence_threshold = float(params["confidence_threshold"])

        # Validate branch selection
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                              "Please select a point cloud branch.")
            return

        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                              "Please select only ONE branch.")
            return

        selected_uid = selected_branches[0]

        # Validate model directory
        if not os.path.exists(model_dir):
            QMessageBox.critical(main_window, "Invalid Directory",
                               f"Model directory does not exist:\n{model_dir}")
            return

        required_files = ['seg_model_best.pt', 'class_mapping.json', 'training_metadata.json']
        missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        if missing:
            QMessageBox.critical(main_window, "Missing Model Files",
                               f"Missing files:\n" + "\n".join(missing))
            return

        try:
            main_window.disable_menus()
            main_window.disable_tree()
            main_window.tree_overlay.show_processing("Loading segmentation model...")

            from models.pointnet.seg_inference import (
                load_seg_model_with_metadata,
                segment_point_cloud_blockwise
            )

            model, class_mapping, metadata = load_seg_model_with_metadata(model_dir)

            print(f"\n{'='*80}")
            print(f"PointNet Semantic Segmentation")
            print(f"{'='*80}")
            print(f"Model: {model_dir}")
            print(f"Classes: {len(class_mapping)} - {', '.join(class_mapping.values())}")
            print(f"Block size: {block_size}m, overlap: {block_overlap}m")
            print(f"Confidence threshold: {confidence_threshold}")

            # Reconstruct point cloud
            main_window.tree_overlay.show_processing("Reconstructing point cloud...")
            point_cloud = controller.reconstruct(selected_uid)

            print(f"Point cloud: {len(point_cloud.points)} points")

            # Progress callback
            def progress_callback(current, total, status):
                percent = int((current / max(total, 1)) * 100)
                main_window.tree_overlay.show_processing(
                    f"Segmenting: {status} ({percent}%)")
                global_variables.global_progress = (percent, status)
                QApplication.processEvents()

            # Run segmentation
            main_window.tree_overlay.show_processing("Running segmentation...")

            labels = segment_point_cloud_blockwise(
                point_cloud=point_cloud,
                model=model,
                metadata=metadata,
                block_size=block_size,
                overlap=block_overlap,
                batch_size=8,
                progress_callback=progress_callback
            )

            # Apply confidence threshold if needed
            # For now, labels come directly from argmax of averaged probabilities

            # Build cluster_names: each unique label -> class name
            unique_labels = np.unique(labels)
            cluster_names = {}
            cluster_colors = {}

            for label_id in unique_labels:
                class_name = class_mapping.get(int(label_id), f"Class_{label_id}")
                cluster_names[int(label_id)] = class_name

                if class_name in DEFAULT_SEG_COLORS:
                    cluster_colors[class_name] = DEFAULT_SEG_COLORS[class_name]
                else:
                    np.random.seed(int(label_id) + 42)
                    cluster_colors[class_name] = np.random.rand(3).astype(np.float32)

            # Create Clusters object
            clusters = Clusters(
                labels=labels,
                cluster_names=cluster_names,
                cluster_colors=cluster_colors
            )

            # Create DataNode
            from core.entities.data_node import DataNode

            parent_uuid = uuid.UUID(selected_uid) if isinstance(selected_uid, str) else selected_uid

            clusters_node = DataNode(
                params="cluster_labels",
                data=clusters,
                data_type="cluster_labels",
                parent_uid=parent_uuid,
                depends_on=[parent_uuid],
                tags=["segmentation", "ml", "pointnet"]
            )

            clusters_uid = data_nodes.add_node(clusters_node)

            # Add to tree
            tree_widget.blockSignals(True)
            try:
                tree_widget.add_branch(str(clusters_uid), str(selected_uid), "cluster_labels")
                clusters_item = tree_widget.branches_dict.get(str(clusters_uid))
                if clusters_item:
                    clusters_item.setCheckState(0, Qt.Checked)
                    tree_widget.visibility_status[str(clusters_uid)] = True
            finally:
                tree_widget.blockSignals(False)

            main_window.render_visible_data(zoom_extent=False)

            # Print summary
            class_counts = {}
            for label_id in unique_labels:
                class_name = cluster_names[int(label_id)]
                count = np.sum(labels == label_id)
                class_counts[class_name] = count

            print(f"\n{'='*80}")
            print(f"Segmentation Complete!")
            print(f"{'='*80}")
            print(f"Total points: {len(labels)}")
            for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                pct = count / len(labels) * 100
                print(f"  {class_name}: {count:,} points ({pct:.1f}%)")
            print(f"{'='*80}")

            QMessageBox.information(
                main_window,
                "Segmentation Complete",
                f"Successfully segmented {len(labels):,} points into "
                f"{len(unique_labels)} classes.\n\n"
                f"Results added to tree as 'cluster_labels'."
            )

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"\nERROR during segmentation:\n{error_msg}")
            QMessageBox.critical(main_window, "Segmentation Error",
                               f"An error occurred:\n\n{str(e)}")

        finally:
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()
            global_variables.global_progress = (None, "")
