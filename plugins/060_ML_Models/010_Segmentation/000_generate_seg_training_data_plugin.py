"""
Generate Segmentation Training Data Plugin

Converts classified clusters (DBSCAN + classification results) into per-point
labeled training blocks for semantic segmentation.

Source: A branch with named cluster_labels (e.g., from classify_clusters plugin).
Output: .npz files with 'features' (N, F) and 'labels' (N,) arrays.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QApplication

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud


class GenerateSegTrainingDataPlugin(ActionPlugin):
    """
    Generate training data for PointNet segmentation from classified clusters.
    """

    last_params = {
        "output_directory": "training_data_seg",
        "block_size": 10.0,
        "points_per_block": 4096,
        "normalize": True,
        "compute_normals": True,
        "normals_knn": 30,
        "compute_eigenvalues": True,
        "eigenvalues_knn": 30,
        "eigenvalues_smooth": True,
        "augmentation_multiplier": 3,
    }

    def get_name(self) -> str:
        return "generate_seg_training_data"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "output_directory": {
                "type": "directory",
                "default": self.last_params["output_directory"],
                "label": "Output Directory",
                "description": "Directory to save training .npz files"
            },
            "block_size": {
                "type": "float",
                "default": self.last_params["block_size"],
                "min": 1.0,
                "max": 100.0,
                "label": "Block Size (m)",
                "description": "Spatial block size in XY plane (should match inference block size)"
            },
            "points_per_block": {
                "type": "int",
                "default": self.last_params["points_per_block"],
                "min": 512,
                "max": 16384,
                "label": "Points Per Block",
                "description": "Target points per training block (should match model num_points)"
            },
            "normalize": {
                "type": "bool",
                "default": self.last_params["normalize"],
                "label": "Normalize XYZ (center + unit sphere)"
            },
            "compute_normals": {
                "type": "bool",
                "default": self.last_params["compute_normals"],
                "label": "Compute Normals"
            },
            "normals_knn": {
                "type": "int",
                "default": self.last_params["normals_knn"],
                "min": 3,
                "max": 100,
                "label": "Normals KNN"
            },
            "compute_eigenvalues": {
                "type": "bool",
                "default": self.last_params["compute_eigenvalues"],
                "label": "Compute Eigenvalues"
            },
            "eigenvalues_knn": {
                "type": "int",
                "default": self.last_params["eigenvalues_knn"],
                "min": 3,
                "max": 100,
                "label": "Eigenvalues KNN"
            },
            "eigenvalues_smooth": {
                "type": "bool",
                "default": self.last_params["eigenvalues_smooth"],
                "label": "Smooth Eigenvalues"
            },
            "augmentation_multiplier": {
                "type": "int",
                "default": self.last_params["augmentation_multiplier"],
                "min": 1,
                "max": 10,
                "label": "Augmentation Multiplier",
                "description": "Number of augmented versions per block (1 = no augmentation)"
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Generate segmentation training data from classified clusters."""
        GenerateSegTrainingDataPlugin.last_params = params.copy()

        controller = global_variables.global_application_controller

        output_dir = params['output_directory'].strip()
        block_size = float(params['block_size'])
        points_per_block = int(params['points_per_block'])

        # Validate branch selection
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                              "Please select a branch with named cluster_labels.")
            return

        selected_uid = selected_branches[0]

        if not output_dir:
            QMessageBox.warning(main_window, "Invalid Directory",
                              "Please specify an output directory.")
            return

        os.makedirs(output_dir, exist_ok=True)

        try:
            main_window.disable_menus()
            main_window.disable_tree()
            main_window.tree_overlay.show_processing("Reconstructing point cloud...")

            # Reconstruct to get PointCloud with cluster_labels
            point_cloud = controller.reconstruct(selected_uid)

            # Find cluster_labels attribute
            cluster_labels = point_cloud.get_attribute("cluster_labels")
            if cluster_labels is None:
                QMessageBox.critical(main_window, "No Labels",
                    "Selected branch has no cluster_labels.\n\n"
                    "Please select a branch with named clusters\n"
                    "(run DBSCAN + classify_clusters first).")
                return

            # Find the Clusters object from data nodes to get cluster_names
            clusters_data = self._find_clusters_data(selected_uid)
            if clusters_data is None or not clusters_data.has_names():
                QMessageBox.critical(main_window, "No Named Clusters",
                    "The cluster_labels do not have semantic names.\n\n"
                    "Please run classify_clusters first to assign names.")
                return

            # Build class mapping: name -> class_id
            unique_names = sorted(set(clusters_data.cluster_names.values()))
            class_name_to_id = {name: idx for idx, name in enumerate(unique_names)}
            class_mapping = {idx: name for name, idx in class_name_to_id.items()}

            # Build per-point labels
            point_labels = np.full(len(point_cloud.points), -1, dtype=np.int32)
            for cluster_id, class_name in clusters_data.cluster_names.items():
                mask = cluster_labels == cluster_id
                point_labels[mask] = class_name_to_id[class_name]

            # Remove unlabeled points (label == -1)
            valid_mask = point_labels >= 0
            valid_points = point_cloud.points[valid_mask]
            valid_labels = point_labels[valid_mask]

            print(f"\n{'='*80}")
            print(f"Generating Segmentation Training Data")
            print(f"{'='*80}")
            print(f"Total points: {len(point_cloud.points)}")
            print(f"Labeled points: {len(valid_points)} ({len(valid_points)/len(point_cloud.points)*100:.1f}%)")
            print(f"Classes: {len(class_mapping)}")
            for cid, name in class_mapping.items():
                count = np.sum(valid_labels == cid)
                print(f"  {name}: {count:,} points")
            print(f"Block size: {block_size}m")
            print(f"Points per block: {points_per_block}")

            # Divide into spatial blocks
            main_window.tree_overlay.show_processing("Creating spatial blocks...")
            blocks = self._create_blocks(valid_points, valid_labels, block_size, points_per_block)

            if len(blocks) == 0:
                QMessageBox.warning(main_window, "No Blocks",
                    f"No blocks with >= {points_per_block} points were found.\n"
                    f"Try reducing block size or points per block.")
                return

            print(f"Created {len(blocks)} valid blocks")

            # Process blocks with features and augmentation
            augmentation_multiplier = int(params['augmentation_multiplier'])
            total_samples = len(blocks) * augmentation_multiplier
            saved_count = 0

            main_window.tree_overlay.show_processing("Computing features and saving...")

            for block_idx, (block_points, block_labels) in enumerate(blocks):
                for aug_idx in range(augmentation_multiplier):
                    # Apply augmentation (skip for aug_idx == 0)
                    if aug_idx > 0:
                        block_xyz = self._augment(block_points.copy(), aug_idx + block_idx * 1000)
                    else:
                        block_xyz = block_points.copy()

                    # Compute features
                    features = self._compute_features(block_xyz, params)
                    if features is None:
                        continue

                    # Subsample to exact points_per_block
                    if len(features) > points_per_block:
                        indices = np.random.choice(len(features), points_per_block, replace=False)
                        features = features[indices]
                        sample_labels = block_labels[indices]
                    elif len(features) < points_per_block:
                        deficit = points_per_block - len(features)
                        pad_idx = np.random.choice(len(features), deficit, replace=True)
                        features = np.vstack([features, features[pad_idx]])
                        sample_labels = np.concatenate([block_labels, block_labels[pad_idx]])
                    else:
                        sample_labels = block_labels

                    # Save as .npz
                    filename = f"block_{block_idx:05d}_aug_{aug_idx:02d}.npz"
                    filepath = os.path.join(output_dir, filename)
                    np.savez_compressed(filepath,
                                       features=features.astype(np.float32),
                                       labels=sample_labels.astype(np.int32))
                    saved_count += 1

                # Update progress
                percent = int(((block_idx + 1) / len(blocks)) * 100)
                main_window.tree_overlay.show_processing(
                    f"Processing block {block_idx+1}/{len(blocks)} ({percent}%)")
                global_variables.global_progress = (percent, f"Block {block_idx+1}/{len(blocks)}")
                QApplication.processEvents()

            # Save metadata
            feature_order = ["X_norm", "Y_norm", "Z_norm"] if params['normalize'] else ["X", "Y", "Z"]
            num_features = 3
            if params['compute_normals']:
                feature_order.extend(["Nx", "Ny", "Nz"])
                num_features += 3
            if params['compute_eigenvalues']:
                feature_order.extend(["E1", "E2", "E3"])
                num_features += 3

            metadata = {
                "dataset_info": {
                    "created_at": datetime.now().isoformat(),
                    "created_by": "SPCToolkit",
                    "plugin": "Generate Seg Training Data",
                    "task_type": "Semantic Segmentation",
                    "source": "classified_clusters"
                },
                "class_mapping": class_mapping,
                "num_classes": len(class_mapping),
                "num_features": num_features,
                "feature_order": feature_order,
                "block_size": block_size,
                "points_per_block": points_per_block,
                "total_blocks": len(blocks),
                "total_samples": saved_count,
                "augmentation_multiplier": augmentation_multiplier,
                "processing": {
                    "normalization": {"enabled": params['normalize']},
                    "features": {
                        "normals": {
                            "enabled": params['compute_normals'],
                            "knn": params['normals_knn']
                        },
                        "eigenvalues": {
                            "enabled": params['compute_eigenvalues'],
                            "knn": params['eigenvalues_knn'],
                            "smooth": params['eigenvalues_smooth']
                        }
                    }
                }
            }

            with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\n{'='*80}")
            print(f"Training Data Generation Complete!")
            print(f"{'='*80}")
            print(f"Saved {saved_count} samples to {output_dir}")
            print(f"{'='*80}")

            QMessageBox.information(main_window, "Training Data Generated",
                f"Successfully generated {saved_count} training samples\n"
                f"from {len(blocks)} spatial blocks.\n\n"
                f"Classes: {len(class_mapping)}\n"
                f"Features: {num_features}\n\n"
                f"Saved to:\n{output_dir}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(main_window, "Error",
                               f"An error occurred:\n\n{str(e)}")

        finally:
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()
            global_variables.global_progress = (None, "")

    def _find_clusters_data(self, branch_uid):
        """Find Clusters data object from data nodes tree."""
        import uuid
        data_nodes = global_variables.global_data_nodes
        parent_uuid = uuid.UUID(branch_uid) if isinstance(branch_uid, str) else branch_uid

        # Check the selected node itself
        node = data_nodes.data_nodes.get(parent_uuid)
        if node and hasattr(node.data, 'has_names'):
            return node.data

        # Check children
        for uid, child_node in data_nodes.data_nodes.items():
            if child_node.parent_uid == parent_uuid and child_node.data_type == "cluster_labels":
                if hasattr(child_node.data, 'has_names') and child_node.data.has_names():
                    return child_node.data

        return None

    def _create_blocks(self, points, labels, block_size, min_points):
        """
        Divide point cloud into spatial blocks on XY plane.

        Returns list of (block_points, block_labels) tuples.
        Only includes blocks with >= min_points points.
        """
        min_xy = np.min(points[:, :2], axis=0)
        max_xy = np.max(points[:, :2], axis=0)

        extent = max_xy - min_xy
        nx = max(1, int(np.ceil(extent[0] / block_size)))
        ny = max(1, int(np.ceil(extent[1] / block_size)))

        blocks = []

        for ix in range(nx):
            for iy in range(ny):
                x_min = min_xy[0] + ix * block_size
                x_max = min_xy[0] + (ix + 1) * block_size
                y_min = min_xy[1] + iy * block_size
                y_max = min_xy[1] + (iy + 1) * block_size

                mask = (
                    (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] < y_max)
                )

                block_points = points[mask]
                block_labels = labels[mask]

                if len(block_points) >= min_points:
                    blocks.append((block_points, block_labels))

        return blocks

    def _augment(self, xyz, seed):
        """Apply random Z-rotation and jitter."""
        np.random.seed(seed % (2**32))

        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        xyz = xyz @ rot.T

        jitter = np.random.normal(0, 0.01, xyz.shape).astype(np.float32)
        xyz += jitter

        return xyz

    def _compute_features(self, xyz, params):
        """Compute features for a block of points. Returns (N, F) array or None."""
        if len(xyz) < 3:
            return None

        pc = PointCloud(points=xyz)

        if params['normalize']:
            pc.normalise(
                apply_scaling=True,
                apply_centering=True,
                rotation_axes=(False, False, False)
            )

        features = [pc.points]

        if params['compute_normals']:
            knn = params['normals_knn']
            if len(pc.points) >= knn:
                pc.estimate_normals(k=knn)
                features.append(pc.normals)
            else:
                features.append(np.zeros((len(pc.points), 3), dtype=np.float32))

        if params['compute_eigenvalues']:
            knn = params['eigenvalues_knn']
            if len(pc.points) >= knn:
                eigenvalues = pc.get_eigenvalues(
                    k=knn, smooth=params['eigenvalues_smooth'])
                features.append(eigenvalues)
            else:
                features.append(np.zeros((len(pc.points), 3), dtype=np.float32))

        return np.hstack(features).astype(np.float32)
