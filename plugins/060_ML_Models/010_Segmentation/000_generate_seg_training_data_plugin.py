"""
Generate Segmentation Training Data Plugin

Converts classified clusters (DBSCAN + classification results) into per-point
labeled training blocks for semantic segmentation.

Source: A branch with named cluster_labels (e.g., from classify_clusters plugin).
Output: .npz files with 'features' (N, F) and 'labels' (N,) arrays.

Reads pre-computed normals and eigenvalues from existing branches instead of
computing them per-block, reducing generation time from hours to seconds.
"""

import os
import json
import time
import numpy as np
from typing import Dict, Any
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QApplication

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class GenerateSegTrainingDataPlugin(ActionPlugin):
    """
    Generate training data for PointNet segmentation from classified clusters.
    """

    last_params = {
        "output_directory": "training_data_seg",
        "block_size": 10.0,
        "stride": 2.0,
        "points_per_block": 4096,
        "normals_branch": "None",
        "eigenvalues_branch": "None",
        "normalize": True,
        "augmentation_multiplier": 3,
    }

    def get_name(self) -> str:
        return "generate_seg_training_data"

    def get_parameters(self) -> Dict[str, Any]:
        # Discover pre-computed normals and eigenvalues branches
        self._normals_map = {}      # {display_label: uuid}
        self._eigenvalues_map = {}  # {display_label: uuid}

        normals_options = ["None"]
        eigenvalues_options = ["None"]

        data_nodes = global_variables.global_data_nodes
        if data_nodes:
            for uid, node in data_nodes.data_nodes.items():
                if node.data_type == "normals" and hasattr(node.data, 'normals'):
                    n = len(node.data.normals)
                    label = f"{node.params} ({n:,} pts)"
                    normals_options.append(label)
                    self._normals_map[label] = uid
                elif node.data_type == "eigenvalues" and hasattr(node.data, 'eigenvalues'):
                    n = len(node.data.eigenvalues)
                    label = f"{node.params} ({n:,} pts)"
                    eigenvalues_options.append(label)
                    self._eigenvalues_map[label] = uid

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
            "stride": {
                "type": "float",
                "default": self.last_params["stride"],
                "min": 0.5,
                "max": 100.0,
                "label": "Stride (m)",
                "description": "Step size between blocks. Set smaller than block size for overlapping blocks (e.g., stride=1m with block_size=2m gives ~4x more blocks)."
            },
            "points_per_block": {
                "type": "int",
                "default": self.last_params["points_per_block"],
                "min": 512,
                "max": 16384,
                "label": "Points Per Block",
                "description": "Target points per training block (should match model num_points)"
            },
            "normals_branch": {
                "type": "choice",
                "options": normals_options,
                "default": self.last_params.get("normals_branch", "None"),
                "label": "Normals Branch",
                "description": "Pre-computed normals branch. 'None' to skip normals."
            },
            "eigenvalues_branch": {
                "type": "choice",
                "options": eigenvalues_options,
                "default": self.last_params.get("eigenvalues_branch", "None"),
                "label": "Eigenvalues Branch",
                "description": "Pre-computed eigenvalues branch. 'None' to skip eigenvalues."
            },
            "normalize": {
                "type": "bool",
                "default": self.last_params["normalize"],
                "label": "Normalize XYZ (center + unit sphere)"
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
        data_nodes = global_variables.global_data_nodes

        output_dir = params['output_directory'].strip()
        block_size = float(params['block_size'])
        points_per_block = int(params['points_per_block'])
        stride = float(params['stride'])
        augmentation_multiplier = int(params['augmentation_multiplier'])
        normalize = params['normalize']

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
            t_start = time.time()

            # --- Phase 1: Reconstruct and build labels ---
            main_window.tree_overlay.show_processing("Reconstructing point cloud...")

            point_cloud = controller.reconstruct(selected_uid)

            cluster_labels = point_cloud.get_attribute("cluster_labels")
            if cluster_labels is None:
                QMessageBox.critical(main_window, "No Labels",
                    "Selected branch has no cluster_labels.\n\n"
                    "Please select a branch with named clusters\n"
                    "(run DBSCAN + classify_clusters first).")
                return

            clusters_data = self._find_clusters_data(selected_uid)
            if clusters_data is None or not clusters_data.has_names():
                QMessageBox.critical(main_window, "No Named Clusters",
                    "The cluster_labels do not have semantic names.\n\n"
                    "Please run classify_clusters first to assign names.")
                return

            # Build class mapping
            unique_names = sorted(set(clusters_data.cluster_names.values()))
            class_name_to_id = {name: idx for idx, name in enumerate(unique_names)}
            class_mapping = {idx: name for name, idx in class_name_to_id.items()}

            # Build per-point labels
            point_labels = np.full(len(point_cloud.points), -1, dtype=np.int32)
            for cluster_id, class_name in clusters_data.cluster_names.items():
                mask = cluster_labels == cluster_id
                point_labels[mask] = class_name_to_id[class_name]

            # Filter to labeled points
            valid_mask = point_labels >= 0
            valid_points = point_cloud.points[valid_mask]
            valid_labels = point_labels[valid_mask]

            print(f"\n{'='*80}")
            print(f"Generating Segmentation Training Data")
            print(f"{'='*80}")
            print(f"Total points: {len(point_cloud.points):,}")
            print(f"Labeled points: {len(valid_points):,} ({len(valid_points)/len(point_cloud.points)*100:.1f}%)")
            print(f"Classes: {len(class_mapping)}")
            for cid, name in class_mapping.items():
                count = np.sum(valid_labels == cid)
                print(f"  {name}: {count:,} points")
            print(f"Block size: {block_size}m, Stride: {stride}m")
            print(f"Points per block: {points_per_block}")

            # --- Phase 2: Load pre-computed features ---
            main_window.tree_overlay.show_processing("Loading pre-computed features...")

            full_normals = None
            full_eigenvalues = None
            normals_name = params.get('normals_branch', 'None')
            eigenvalues_name = params.get('eigenvalues_branch', 'None')

            if normals_name != "None" and normals_name in self._normals_map:
                t0 = time.time()
                uid = self._normals_map[normals_name]
                normals_node = data_nodes.data_nodes[uid]
                full_normals = normals_node.data.normals
                if len(full_normals) != len(point_cloud.points):
                    QMessageBox.critical(main_window, "Normals Mismatch",
                        f"Normals branch has {len(full_normals):,} points but "
                        f"selected branch has {len(point_cloud.points):,} points.\n\n"
                        f"They must have the same number of points.")
                    return
                print(f"Loaded normals: {time.time() - t0:.1f}s ({len(full_normals):,} pts)")

            if eigenvalues_name != "None" and eigenvalues_name in self._eigenvalues_map:
                t0 = time.time()
                uid = self._eigenvalues_map[eigenvalues_name]
                eig_node = data_nodes.data_nodes[uid]
                full_eigenvalues = eig_node.data.eigenvalues
                if len(full_eigenvalues) != len(point_cloud.points):
                    QMessageBox.critical(main_window, "Eigenvalues Mismatch",
                        f"Eigenvalues branch has {len(full_eigenvalues):,} points but "
                        f"selected branch has {len(point_cloud.points):,} points.\n\n"
                        f"They must have the same number of points.")
                    return
                print(f"Loaded eigenvalues: {time.time() - t0:.1f}s ({len(full_eigenvalues):,} pts)")

            # Filter features by valid_mask
            valid_normals = full_normals[valid_mask] if full_normals is not None else None
            valid_eigenvalues = full_eigenvalues[valid_mask] if full_eigenvalues is not None else None

            use_normals = valid_normals is not None
            use_eigenvalues = valid_eigenvalues is not None

            # --- Phase 3: Create spatial blocks ---
            main_window.tree_overlay.show_processing("Creating spatial blocks...")
            t_blocks = time.time()

            block_indices_list = self._create_blocks(
                valid_points, block_size, points_per_block, stride)

            if len(block_indices_list) == 0:
                QMessageBox.warning(main_window, "No Blocks",
                    f"No blocks with >= {points_per_block} points were found.\n"
                    f"Try reducing block size or points per block.")
                return

            print(f"Block creation: {time.time() - t_blocks:.1f}s")

            # --- Phase 4: Assemble features and save ---
            total_samples = len(block_indices_list) * augmentation_multiplier
            saved_count = 0

            main_window.tree_overlay.show_processing("Assembling features and saving...")
            print(f"\nAssembling {total_samples} samples ({len(block_indices_list)} blocks x {augmentation_multiplier} augmentations)...")
            t_assemble = time.time()

            for block_idx, indices in enumerate(block_indices_list):
                block_points = valid_points[indices]
                block_labels = valid_labels[indices]
                block_normals = valid_normals[indices] if use_normals else None
                block_eigenvalues = valid_eigenvalues[indices] if use_eigenvalues else None

                # Assemble base features (normalize XYZ, scale eigenvalues)
                norm_xyz, base_normals, scaled_eigs = self._assemble_block_features(
                    block_points, block_normals, block_eigenvalues, normalize)

                if norm_xyz is None:
                    continue

                for aug_idx in range(augmentation_multiplier):
                    if aug_idx > 0:
                        # Rotate XYZ + normals, keep eigenvalues
                        aug_xyz, aug_normals, aug_eigs = self._apply_augmentation_to_features(
                            norm_xyz, base_normals, scaled_eigs,
                            seed=aug_idx + block_idx * 1000)
                    else:
                        aug_xyz = norm_xyz
                        aug_normals = base_normals
                        aug_eigs = scaled_eigs

                    # Stack features
                    feature_list = [aug_xyz]
                    if aug_normals is not None:
                        feature_list.append(aug_normals)
                    if aug_eigs is not None:
                        feature_list.append(aug_eigs)
                    features = np.hstack(feature_list).astype(np.float32)

                    # Subsample to exact points_per_block
                    if len(features) > points_per_block:
                        idx = np.random.choice(len(features), points_per_block, replace=False)
                        features = features[idx]
                        sample_labels = block_labels[idx]
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
                percent = int(((block_idx + 1) / len(block_indices_list)) * 100)
                main_window.tree_overlay.show_processing(
                    f"Block {block_idx+1}/{len(block_indices_list)} ({percent}%)")
                global_variables.global_progress = (percent, f"Block {block_idx+1}/{len(block_indices_list)}")
                QApplication.processEvents()

            print(f"Assembly + save: {time.time() - t_assemble:.1f}s")

            # Save metadata
            feature_order = ["X_norm", "Y_norm", "Z_norm"] if normalize else ["X", "Y", "Z"]
            num_features = 3
            if use_normals:
                feature_order.extend(["Nx", "Ny", "Nz"])
                num_features += 3
            if use_eigenvalues:
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
                "stride": stride,
                "points_per_block": points_per_block,
                "total_blocks": len(block_indices_list),
                "total_samples": saved_count,
                "augmentation_multiplier": augmentation_multiplier,
                "processing": {
                    "normalization": {"enabled": normalize},
                    "features": {
                        "normals": {
                            "enabled": use_normals,
                            "branch": normals_name if use_normals else None,
                        },
                        "eigenvalues": {
                            "enabled": use_eigenvalues,
                            "branch": eigenvalues_name if use_eigenvalues else None,
                        }
                    }
                }
            }

            with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            t_total = time.time() - t_start

            print(f"\n{'='*80}")
            print(f"Training Data Generation Complete!")
            print(f"{'='*80}")
            print(f"Saved {saved_count} samples to {output_dir}")
            print(f"Total time: {t_total:.1f}s")
            print(f"{'='*80}")

            QMessageBox.information(main_window, "Training Data Generated",
                f"Successfully generated {saved_count} training samples\n"
                f"from {len(block_indices_list)} spatial blocks.\n\n"
                f"Classes: {len(class_mapping)}\n"
                f"Features: {num_features}\n"
                f"Time: {t_total:.1f}s\n\n"
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

    def _create_blocks(self, points, block_size, min_points, stride):
        """
        Divide point cloud into spatial blocks on XY plane.

        Args:
            stride: Step size between block origins. Set < block_size for overlapping blocks.

        Returns list of index arrays (np.ndarray of int indices into points).
        Only includes blocks with >= min_points points.
        """
        min_xy = np.min(points[:, :2], axis=0)
        max_xy = np.max(points[:, :2], axis=0)

        extent = max_xy - min_xy
        nx = max(1, int(np.ceil(extent[0] / stride)))
        ny = max(1, int(np.ceil(extent[1] / stride)))

        blocks = []
        non_empty = 0
        filtered = 0

        for ix in range(nx):
            for iy in range(ny):
                x_min = min_xy[0] + ix * stride
                x_max = x_min + block_size
                y_min = min_xy[1] + iy * stride
                y_max = y_min + block_size

                mask = (
                    (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] < y_max)
                )

                block_indices = np.where(mask)[0]

                if len(block_indices) > 0:
                    non_empty += 1
                    if len(block_indices) >= min_points:
                        blocks.append(block_indices)
                    else:
                        filtered += 1

        total_cells = nx * ny
        print(f"Grid: {nx} x {ny} = {total_cells:,} cells (stride={stride}m, block={block_size}m)")
        print(f"  Non-empty cells: {non_empty:,}")
        print(f"  Passed min_points (>={min_points}): {len(blocks):,}")
        print(f"  Filtered (too sparse): {filtered:,}")

        return blocks

    def _assemble_block_features(self, block_points, block_normals, block_eigenvalues, normalize):
        """
        Assemble features for a single block from pre-computed data.

        Applies per-block XYZ normalization and scales eigenvalues by s²
        to match per-block normalized space.

        Returns (normalized_xyz, normals, scaled_eigenvalues) or (None, None, None).
        """
        if len(block_points) < 3:
            return None, None, None

        if normalize:
            centroid = np.mean(block_points, axis=0)
            centered = block_points - centroid
            max_distance = np.max(np.sqrt(np.sum(centered ** 2, axis=1)))
            if max_distance > 0:
                s = 1.0 / max_distance
                norm_xyz = centered * s
            else:
                s = 1.0
                norm_xyz = centered
        else:
            norm_xyz = block_points.copy()
            s = 1.0

        # Scale eigenvalues by s² to match per-block normalized space
        scaled_eigs = block_eigenvalues * (s ** 2) if block_eigenvalues is not None else None

        return norm_xyz, block_normals, scaled_eigs

    def _augment(self, xyz, seed):
        """Apply random Z-rotation and jitter. Returns (augmented_xyz, rotation_matrix)."""
        np.random.seed(seed % (2**32))

        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        xyz = xyz @ rot.T

        jitter = np.random.normal(0, 0.01, xyz.shape).astype(np.float32)
        xyz += jitter

        return xyz, rot

    def _apply_augmentation_to_features(self, norm_xyz, normals, eigenvalues, seed):
        """
        Apply augmentation to pre-assembled features.

        XYZ: Z-rotation + jitter (same as _augment)
        Normals: Z-rotation only (direction vectors transform covariantly)
        Eigenvalues: unchanged (rotation-invariant)
        """
        aug_xyz, rot = self._augment(norm_xyz.copy(), seed)

        aug_normals = None
        if normals is not None:
            aug_normals = normals @ rot.T

        return aug_xyz, aug_normals, eigenvalues
