"""
Import External Dataset Plugin

Imports labeled point cloud datasets from external sources (S3DIS, SemanticKITTI,
or custom XYZ+Label formats) and converts them to segmentation training data.

Output: .npz files with 'features' (N, F) and 'labels' (N,) arrays.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QApplication

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from plugins.dialogs.data_generation_progress_dialog import DataGenerationProgressDialog


class ImportExternalDatasetPlugin(ActionPlugin):
    """
    Import external labeled datasets for segmentation training.
    """

    last_params = {
        "input_directory": "",
        "output_directory": "training_data_seg",
        "dataset_format": "Custom (XYZ + Label columns)",
        "block_size": 10.0,
        "points_per_block": 4096,
        "normalize": True,
        "compute_normals": True,
        "normals_knn": 30,
        "compute_eigenvalues": True,
        "eigenvalues_knn": 30,
        "eigenvalues_smooth": True,
        "class_remap_json": "",
    }

    def get_name(self) -> str:
        return "import_external_dataset"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "input_directory": {
                "type": "directory",
                "default": self.last_params["input_directory"],
                "label": "Input Directory",
                "description": "Directory containing source dataset files"
            },
            "output_directory": {
                "type": "directory",
                "default": self.last_params["output_directory"],
                "label": "Output Directory",
                "description": "Directory to save training .npz files"
            },
            "dataset_format": {
                "type": "choice",
                "options": [
                    "Custom (XYZ + Label columns)",
                    "S3DIS (Stanford .txt)",
                    "SemanticKITTI (.bin + .label)"
                ],
                "default": self.last_params["dataset_format"],
                "label": "Dataset Format",
                "description": "Format of the input dataset"
            },
            "block_size": {
                "type": "float",
                "default": self.last_params["block_size"],
                "min": 1.0,
                "max": 100.0,
                "label": "Block Size (m)",
                "description": "Spatial block size for dividing scenes"
            },
            "points_per_block": {
                "type": "int",
                "default": self.last_params["points_per_block"],
                "min": 512,
                "max": 16384,
                "label": "Points Per Block",
                "description": "Target points per training block"
            },
            "normalize": {
                "type": "bool",
                "default": self.last_params["normalize"],
                "label": "Normalize XYZ"
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
            "class_remap_json": {
                "type": "str",
                "default": self.last_params["class_remap_json"],
                "label": "Class Remap JSON (optional)",
                "description": "Path to JSON file mapping original labels to new labels. Format: {\"original_id\": \"new_name\", ...}. Use \"__ignore__\" to skip classes."
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Import external dataset and convert to training format."""
        ImportExternalDatasetPlugin.last_params = params.copy()

        input_dir = params['input_directory'].strip()
        output_dir = params['output_directory'].strip()
        dataset_format = params['dataset_format']
        block_size = float(params['block_size'])
        points_per_block = int(params['points_per_block'])

        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(main_window, "Invalid Input",
                              "Please select a valid input directory.")
            return

        if not output_dir:
            QMessageBox.warning(main_window, "Invalid Output",
                              "Please specify an output directory.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Load class remap if provided
        class_remap = None
        remap_path = params.get('class_remap_json', '').strip()
        if remap_path and os.path.exists(remap_path):
            with open(remap_path, 'r') as f:
                class_remap = json.load(f)
            print(f"Loaded class remap with {len(class_remap)} entries")

        # Create progress dialog
        progress_dialog = DataGenerationProgressDialog(parent=main_window, total_steps=100)
        progress_dialog.setWindowTitle("Import External Dataset")
        progress_dialog.show()
        progress_dialog.set_operation(f"Reading {dataset_format} files...")
        QApplication.processEvents()

        cancel_event = global_variables.global_cancel_event
        cancel_event.clear()

        try:
            main_window.disable_menus()
            main_window.disable_tree()

            print(f"\n{'='*80}")
            print(f"Importing External Dataset: {dataset_format}")
            print(f"{'='*80}")
            print(f"Input: {input_dir}")
            print(f"Output: {output_dir}")

            # Load scenes based on format
            main_window.tree_overlay.show_processing(f"Reading {dataset_format} files...")

            if "S3DIS" in dataset_format:
                scenes = self._load_s3dis(input_dir)
            elif "SemanticKITTI" in dataset_format:
                scenes = self._load_semantickitti(input_dir)
            else:
                scenes = self._load_custom(input_dir)

            if progress_dialog.cancelled or cancel_event.is_set():
                print("Import cancelled during scene loading.")
                progress_dialog.mark_complete(success=False, message="Import cancelled.")
                return

            if not scenes:
                QMessageBox.warning(main_window, "No Data",
                    f"No valid scene files found in:\n{input_dir}")
                progress_dialog.mark_complete(success=False, message="No data found.")
                return

            print(f"Loaded {len(scenes)} scenes")

            # Apply class remapping
            all_class_names = {}
            if class_remap:
                scenes, all_class_names = self._apply_remap(scenes, class_remap)
            else:
                # Auto-generate class names from unique labels
                all_labels = set()
                for _, labels in scenes:
                    all_labels.update(np.unique(labels).tolist())
                all_class_names = {int(l): f"Class_{l}" for l in sorted(all_labels) if l >= 0}

            # Remap labels to contiguous 0..N-1
            unique_names = sorted(set(all_class_names.values()))
            name_to_new_id = {name: idx for idx, name in enumerate(unique_names)}
            class_mapping = {idx: name for idx, name in enumerate(unique_names)}

            print(f"Classes: {len(class_mapping)}")
            for cid, name in class_mapping.items():
                print(f"  {cid}: {name}")

            # Process scenes into blocks
            total_saved = 0
            scene_idx = 0
            total_blocks_estimated = len(scenes)  # rough estimate, updated per scene
            was_cancelled = False

            for scene_points, scene_labels in scenes:
                # Check cancel before each scene
                if progress_dialog.cancelled or cancel_event.is_set():
                    was_cancelled = True
                    break

                scene_idx += 1
                main_window.tree_overlay.show_processing(
                    f"Processing scene {scene_idx}/{len(scenes)}...")
                progress_dialog.set_operation(
                    f"Processing scene {scene_idx}/{len(scenes)}...")
                progress_dialog.update_progress(
                    scene_idx, len(scenes),
                    current_class=f"Scene {scene_idx}",
                    processed_count=total_saved)
                QApplication.processEvents()

                # Remap labels to contiguous IDs
                remapped = np.full(len(scene_labels), -1, dtype=np.int32)
                for orig_label, class_name in all_class_names.items():
                    if class_name in name_to_new_id:
                        mask = scene_labels == orig_label
                        remapped[mask] = name_to_new_id[class_name]

                # Keep only valid points
                valid = remapped >= 0
                valid_points = scene_points[valid]
                valid_labels = remapped[valid]

                if len(valid_points) < points_per_block:
                    continue

                # Divide into blocks
                blocks = self._create_blocks(valid_points, valid_labels,
                                           block_size, points_per_block)

                for block_idx, (block_pts, block_lbl) in enumerate(blocks):
                    # Check cancel before each block
                    if progress_dialog.cancelled or cancel_event.is_set():
                        was_cancelled = True
                        break

                    features = self._compute_features(block_pts, params)
                    if features is None:
                        continue

                    # Subsample/pad to target
                    if len(features) > points_per_block:
                        idx = np.random.choice(len(features), points_per_block, replace=False)
                        features = features[idx]
                        block_lbl = block_lbl[idx]
                    elif len(features) < points_per_block:
                        deficit = points_per_block - len(features)
                        pad = np.random.choice(len(features), deficit, replace=True)
                        features = np.vstack([features, features[pad]])
                        block_lbl = np.concatenate([block_lbl, block_lbl[pad]])

                    filename = f"scene_{scene_idx:04d}_block_{block_idx:05d}.npz"
                    np.savez_compressed(
                        os.path.join(output_dir, filename),
                        features=features.astype(np.float32),
                        labels=block_lbl.astype(np.int32)
                    )
                    total_saved += 1

                    # Keep UI responsive during block processing
                    if block_idx % 5 == 0:
                        QApplication.processEvents()

                if was_cancelled:
                    break

                print(f"  Scene {scene_idx}: {len(blocks)} blocks saved")

            if was_cancelled:
                print(f"\nImport cancelled. Saved {total_saved} blocks before cancellation.")
                progress_dialog.mark_complete(
                    success=False,
                    message=f"Import cancelled. {total_saved} blocks saved before cancellation.")
                return

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
                    "plugin": "Import External Dataset",
                    "task_type": "Semantic Segmentation",
                    "source_format": dataset_format,
                    "source_directory": input_dir
                },
                "class_mapping": class_mapping,
                "num_classes": len(class_mapping),
                "num_features": num_features,
                "feature_order": feature_order,
                "block_size": block_size,
                "points_per_block": points_per_block,
                "total_samples": total_saved,
                "num_scenes": len(scenes),
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
            print(f"Import Complete! Saved {total_saved} training samples.")
            print(f"{'='*80}")

            progress_dialog.mark_complete(
                success=True,
                message=f"Import complete! {total_saved} samples from {len(scenes)} scenes.")

            QMessageBox.information(main_window, "Import Complete",
                f"Successfully imported {total_saved} training samples\n"
                f"from {len(scenes)} scenes.\n\n"
                f"Classes: {len(class_mapping)}\n"
                f"Features: {num_features}\n\n"
                f"Saved to:\n{output_dir}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                progress_dialog.mark_complete(
                    success=False, message=f"Error: {str(e)}")
            except:
                pass
            QMessageBox.critical(main_window, "Import Error",
                               f"An error occurred:\n\n{str(e)}")

        finally:
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

    def _load_s3dis(self, input_dir) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load S3DIS format: each .txt file has XYZRGBL columns (space-separated).
        Directory structure: Area_N/room_name/room_name.txt or annotations/*.txt
        """
        scenes = []
        for root, dirs, files in os.walk(input_dir):
            for f in sorted(files):
                if not f.endswith('.txt'):
                    continue
                filepath = os.path.join(root, f)
                try:
                    data = np.loadtxt(filepath)
                    if data.ndim != 2 or data.shape[1] < 4:
                        continue
                    xyz = data[:, :3].astype(np.float32)
                    # Label is last column
                    labels = data[:, -1].astype(np.int32)
                    scenes.append((xyz, labels))
                    print(f"  Loaded {f}: {len(xyz)} points")
                except Exception as e:
                    print(f"  Skipping {f}: {e}")
        return scenes

    def _load_semantickitti(self, input_dir) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load SemanticKITTI format:
        velodyne/*.bin (float32, 4 columns: XYZI)
        labels/*.label (uint32, lower 16 bits = semantic label)
        """
        scenes = []
        velodyne_dir = os.path.join(input_dir, 'velodyne')
        labels_dir = os.path.join(input_dir, 'labels')

        if not os.path.exists(velodyne_dir):
            # Try sequences/XX/velodyne pattern
            for seq_dir in sorted(os.listdir(input_dir)):
                seq_path = os.path.join(input_dir, seq_dir)
                if os.path.isdir(seq_path):
                    vel_path = os.path.join(seq_path, 'velodyne')
                    lbl_path = os.path.join(seq_path, 'labels')
                    if os.path.exists(vel_path) and os.path.exists(lbl_path):
                        scenes.extend(self._load_kitti_sequence(vel_path, lbl_path))
        else:
            scenes.extend(self._load_kitti_sequence(velodyne_dir, labels_dir))

        return scenes

    def _load_kitti_sequence(self, velodyne_dir, labels_dir):
        """Load a single SemanticKITTI sequence."""
        scenes = []
        bin_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])

        for bin_file in bin_files:
            label_file = bin_file.replace('.bin', '.label')
            label_path = os.path.join(labels_dir, label_file)
            bin_path = os.path.join(velodyne_dir, bin_file)

            if not os.path.exists(label_path):
                continue

            try:
                scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                xyz = scan[:, :3]

                label_data = np.fromfile(label_path, dtype=np.uint32)
                # Lower 16 bits are semantic label
                labels = (label_data & 0xFFFF).astype(np.int32)

                scenes.append((xyz, labels))
                print(f"  Loaded {bin_file}: {len(xyz)} points")
            except Exception as e:
                print(f"  Skipping {bin_file}: {e}")

        return scenes

    def _load_custom(self, input_dir) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load custom format: CSV/TXT files with XYZ + Label columns.
        Tries common delimiters (space, comma, tab).
        Label is the last column (or 4th column if > 4 columns).
        """
        scenes = []
        for f in sorted(os.listdir(input_dir)):
            if not (f.endswith('.txt') or f.endswith('.csv') or f.endswith('.xyz')):
                continue

            filepath = os.path.join(input_dir, f)
            try:
                # Try different delimiters
                data = None
                for delimiter in [' ', ',', '\t', None]:
                    try:
                        data = np.loadtxt(filepath, delimiter=delimiter,
                                        comments='#', max_rows=5)
                        if data.ndim == 2 and data.shape[1] >= 4:
                            data = np.loadtxt(filepath, delimiter=delimiter,
                                            comments='#')
                            break
                        data = None
                    except:
                        data = None

                if data is None or data.ndim != 2 or data.shape[1] < 4:
                    continue

                xyz = data[:, :3].astype(np.float32)
                # Label column: last column
                labels = data[:, -1].astype(np.int32)
                scenes.append((xyz, labels))
                print(f"  Loaded {f}: {len(xyz)} points, {len(np.unique(labels))} classes")
            except Exception as e:
                print(f"  Skipping {f}: {e}")

        return scenes

    def _apply_remap(self, scenes, class_remap):
        """Apply class remapping. Returns (scenes, class_names_dict)."""
        # Convert remap keys to int
        remap = {}
        for k, v in class_remap.items():
            try:
                remap[int(k)] = v
            except ValueError:
                remap[k] = v

        # Find ignored classes
        ignored = {k for k, v in remap.items() if v == "__ignore__"}

        # Build name mapping
        class_names = {}
        for k, v in remap.items():
            if v != "__ignore__" and isinstance(k, int):
                class_names[k] = v

        # Filter scenes
        new_scenes = []
        for points, labels in scenes:
            # Remove ignored labels
            keep = np.ones(len(labels), dtype=bool)
            for ign_label in ignored:
                if isinstance(ign_label, int):
                    keep &= (labels != ign_label)

            new_scenes.append((points[keep], labels[keep]))

        return new_scenes, class_names

    def _create_blocks(self, points, labels, block_size, min_points):
        """Divide into spatial blocks on XY plane."""
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

                if np.sum(mask) >= min_points:
                    blocks.append((points[mask], labels[mask]))

        return blocks

    def _compute_features(self, xyz, params):
        """Compute features for a block."""
        if len(xyz) < 3:
            return None

        pc = PointCloud(points=xyz.copy())

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
