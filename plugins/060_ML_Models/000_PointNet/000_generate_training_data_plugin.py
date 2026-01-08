"""
PointNet Generate Training Data Plugin

Generates training data for PointNet feature classification from exported classified clusters.
Processes clusters with normalization, normals, and eigenvalues computation, then balances
classes by resampling to ensure equal representation.
"""

from typing import Dict, Any, List, Tuple
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.point_cloud import PointCloud
from gui.dialogs.training_data_preview_window import DataPreviewWindow
from gui.dialogs.data_generation_progress_dialog import DataGenerationProgressDialog


class GenerateTrainingDataPlugin(ActionPlugin):
    """
    Generate training data for PointNet feature classification.

    This plugin processes exported clusters and creates balanced training datasets
    with configurable features (normalized XYZ, normals, eigenvalues).
    """

    def get_name(self) -> str:
        return "generate_training_data"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define parameters for training data generation.

        Note: DynamicDialog doesn't support conditional parameters yet,
        so KNN parameters are always shown but only used when checkboxes are checked.
        """
        return {
            "input_directory": {
                "type": "directory",
                "default": "",
                "label": "Input Directory",
                "description": "Directory containing exported classified clusters"
            },
            "output_directory": {
                "type": "directory",
                "default": "",
                "label": "Output Directory",
                "description": "Directory where training data will be saved"
            },
            "n_multiplier": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "label": "Points Per Sample (n × 1024)",
                "description": "Each sample will contain n × 1024 points. Target = n × 1024."
            },
            "normalize": {
                "type": "bool",
                "default": True,
                "label": "Normalize XYZ (center + unit sphere)"
            },
            "compute_normals": {
                "type": "bool",
                "default": True,
                "label": "Compute Normals"
            },
            "normals_knn": {
                "type": "int",
                "default": 30,
                "min": 3,
                "max": 100,
                "label": "Normals KNN",
                "description": "Number of neighbors for normal estimation (used if Compute Normals is checked)"
            },
            "compute_eigenvalues": {
                "type": "bool",
                "default": True,
                "label": "Compute Eigenvalues"
            },
            "eigenvalues_knn": {
                "type": "int",
                "default": 30,
                "min": 3,
                "max": 100,
                "label": "Eigenvalues KNN",
                "description": "Number of neighbors for eigenvalue computation (used if Compute Eigenvalues is checked)"
            },
            "eigenvalues_smooth": {
                "type": "bool",
                "default": True,
                "label": "Smooth Eigenvalues",
                "description": "Apply smoothing to eigenvalues (used if Compute Eigenvalues is checked)"
            },
            "augmentation_multiplier": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 20,
                "label": "Augmentation Multiplier",
                "description": "Number of augmented versions per cluster (1 = no augmentation)"
            },
            "enable_z_rotation": {
                "type": "bool",
                "default": True,
                "label": "Enable Z-axis Rotation",
                "description": "Random rotation around vertical axis (0-360°)"
            },
            "enable_horizontal_mirror": {
                "type": "bool",
                "default": True,
                "label": "Enable Horizontal Mirroring",
                "description": "Random flip along X or Y axis"
            },
            "enable_xyz_jitter": {
                "type": "bool",
                "default": True,
                "label": "Enable XYZ Jitter",
                "description": "Add random noise to coordinates"
            },
            "jitter_sigma": {
                "type": "float",
                "default": 0.01,
                "min": 0.0,
                "max": 0.1,
                "label": "Jitter Sigma",
                "description": "Standard deviation for jitter noise (used if Enable XYZ Jitter is checked)"
            },
            "max_points_for_features": {
                "type": "int",
                "default": 20000,
                "min": 1024,
                "max": 100000,
                "label": "Max Points for Feature Computation",
                "description": "Maximum points per cluster before feature computation. Large clusters are subsampled to this limit (MUST match inference setting to prevent data leakage)"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute training data generation.

        Args:
            main_window: The main application window
            params: Dictionary containing all parameters
        """
        # Validate directories
        input_dir = params['input_directory'].strip()
        output_dir = params['output_directory'].strip()

        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(
                main_window,
                "Invalid Input Directory",
                "Please select a valid input directory."
            )
            return

        if not output_dir:
            QMessageBox.warning(
                main_window,
                "Invalid Output Directory",
                "Please select an output directory."
            )
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Calculate target points
        n = params['n_multiplier']
        target_points = n * 1024

        print("=" * 60)
        print("Starting PointNet Training Data Generation...")
        print("=" * 60)

        # Create and show progress dialog
        progress_dialog = DataGenerationProgressDialog(parent=main_window)
        progress_dialog.show()

        # Track cancellation
        cancelled = False

        def check_cancelled():
            """Check if user cancelled."""
            return progress_dialog.cancelled

        try:
            # Step 1: Scan input directory and organize clusters by class
            print("\n[1/6] Scanning input directory...")
            progress_dialog.set_operation("Step 1/6: Scanning input directory...")
            progress_dialog.set_status("Reading directory structure...")

            class_clusters = self._scan_input_directory(input_dir)

            print(f"DEBUG: Scanned input directory: {input_dir}")
            print(f"DEBUG: Found {len(class_clusters)} classes")
            for class_name, paths in class_clusters.items():
                print(f"  - {class_name}: {len(paths)} files")

            if not class_clusters:
                progress_dialog.mark_complete(success=False, message="No data found")
                QMessageBox.warning(
                    main_window,
                    "No Data Found",
                    f"No cluster files (.npy) found in:\n{input_dir}\n\n"
                    "Expected structure:\n"
                    "  input_dir/\n"
                    "    ├── ClassName1/\n"
                    "    │   ├── file_1.npy\n"
                    "    │   └── ...\n"
                    "    └── ClassName2/\n"
                    "        └── ..."
                )
                return

            # Step 2: Process all clusters with augmentation
            augmentation_multiplier = params.get('augmentation_multiplier', 1)
            print(f"\n[2/6] Processing clusters (target: {target_points} points, augmentation: {augmentation_multiplier}x)...")
            progress_dialog.set_operation(f"Step 2/6: Processing clusters (augmentation: {augmentation_multiplier}x)...")

            # Calculate total samples to process
            total_clusters = sum(len(paths) for paths in class_clusters.values())
            total_samples_to_process = total_clusters * augmentation_multiplier
            processed_samples = 0

            class_samples = {}  # {class_name: [(data, source_cluster_path, aug_idx), ...]}
            skipped_total = 0
            error_count = 0

            for class_name, cluster_paths in class_clusters.items():
                class_samples[class_name] = []
                skipped_class = 0

                for cluster_path in cluster_paths:
                    # Check for cancellation
                    if check_cancelled():
                        cancelled = True
                        break

                    # Create augmentation_multiplier versions of each cluster
                    for aug_idx in range(augmentation_multiplier):
                        # Check for cancellation
                        if check_cancelled():
                            cancelled = True
                            break

                        try:
                            # Process cluster with augmentation
                            sample_data = self._process_cluster(cluster_path, params, target_points, augmentation_index=aug_idx)

                            if sample_data is not None:
                                class_samples[class_name].append((sample_data, cluster_path, aug_idx))
                            else:
                                # Only count as skipped on first attempt (aug_idx == 0)
                                if aug_idx == 0:
                                    skipped_class += 1

                        except Exception as e:
                            error_count += 1
                            print(f"Error processing {cluster_path} (aug {aug_idx}): {str(e)}")

                        # Update progress
                        processed_samples += 1
                        progress_dialog.update_progress(
                            current=processed_samples,
                            total=total_samples_to_process,
                            current_class=class_name,
                            processed_count=processed_samples - error_count - skipped_total,
                            skipped_count=skipped_total
                        )

                    if cancelled:
                        break

                if cancelled:
                    break

                skipped_total += skipped_class
                unique_clusters = len(cluster_paths) - skipped_class
                print(f"Class '{class_name}': {len(class_samples[class_name])} samples from {unique_clusters} unique clusters, {skipped_class} skipped")

            # Check if cancelled
            if cancelled:
                progress_dialog.mark_complete(success=False, message="Generation cancelled by user")
                print("\nGeneration cancelled by user")
                return

            # Remove empty classes
            class_samples = {k: v for k, v in class_samples.items() if len(v) > 0}

            if not class_samples:
                progress_dialog.mark_complete(success=False, message="No valid samples found")
                QMessageBox.warning(
                    main_window,
                    "No Valid Samples",
                    f"No clusters had enough points (>= {target_points}).\n"
                    f"Skipped {skipped_total} clusters."
                )
                return

            # Step 3: Balance classes
            print(f"\n[3/6] Balancing classes...")
            progress_dialog.set_operation("Step 3/6: Balancing classes...")
            progress_dialog.set_status("Ensuring equal samples per class...")

            balanced_samples, balance_stats = self._balance_classes(class_samples, params, target_points)

            # Step 4: Save all samples
            print(f"\n[4/6] Saving training data...")
            progress_dialog.set_operation("Step 4/6: Saving training data...")
            progress_dialog.set_status("Writing files to disk...")

            total_saved = 0
            total_to_save = sum(len(samples) for samples in balanced_samples.values())

            for class_name, samples in balanced_samples.items():
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                for idx, sample_data in enumerate(samples, start=1):
                    filename = f"{class_name.lower()}_{idx}.npy"
                    filepath = os.path.join(class_dir, filename)
                    np.save(filepath, sample_data)
                    total_saved += 1

                    # Update progress every 10 files
                    if total_saved % 10 == 0 or total_saved == total_to_save:
                        progress_dialog.update_progress(
                            current=total_saved,
                            total=total_to_save,
                            current_class=class_name,
                            processed_count=total_saved,
                            skipped_count=0
                        )

            print(f"   Saved {total_saved} training samples")

            # Step 5: Generate and save metadata
            print(f"\n[5/6] Generating metadata...")
            progress_dialog.set_operation("Step 5/6: Generating metadata...")
            progress_dialog.set_status("Writing metadata.json...")

            self._save_metadata(output_dir, params, balanced_samples, balance_stats, skipped_total)

            # Step 6: Update session persistence and show success message
            print(f"\n[6/6] Complete!")
            print("=" * 60)
            progress_dialog.set_operation("Step 6/6: Complete!")
            progress_dialog.set_status("All done!")

            # Mark progress dialog as complete
            progress_dialog.mark_complete(success=True)

            # Store output directory for Preview Training Data plugin
            DataPreviewWindow.last_directories["Training Data Preview"] = output_dir

            self._show_summary(main_window, output_dir, balanced_samples, balance_stats, skipped_total, target_points)

        except Exception as e:
            # Mark dialog as complete with error
            if 'progress_dialog' in locals():
                progress_dialog.mark_complete(success=False, message=f"Error: {str(e)}")

            QMessageBox.critical(
                main_window,
                "Processing Error",
                f"An error occurred during processing:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _scan_input_directory(self, input_dir: str) -> Dict[str, List[str]]:
        """
        Scan input directory and organize cluster files by class.

        Args:
            input_dir: Path to input directory

        Returns:
            Dictionary mapping class names to lists of cluster file paths
        """
        class_clusters = {}

        # Iterate through subdirectories (class folders)
        for class_name in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            # Find all .npy files in this class folder
            npy_files = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.endswith('.npy')
            ]

            if npy_files:
                class_clusters[class_name] = npy_files

        return class_clusters

    def _apply_augmentation(self, xyz: np.ndarray, params: Dict[str, Any], seed: int) -> np.ndarray:
        """
        Apply augmentation transformations to point cloud coordinates.

        IMPORTANT: Augmentation is applied BEFORE normalization to unit sphere.
        This ensures that features (normals, eigenvalues) are computed on properly
        normalized data after augmentation.

        Augmentations applied (if enabled):
        1. Z-axis rotation (0-360 degrees)
        2. Horizontal mirroring (randomly X or Y axis, 50% chance)
        3. XYZ jitter (Gaussian noise)

        Args:
            xyz: Point cloud coordinates (n, 3)
            params: Augmentation parameters
            seed: Random seed for reproducibility

        Returns:
            Augmented coordinates (n, 3)
        """
        # Set random seed for reproducibility
        np.random.seed(seed)

        xyz = xyz.copy()

        # 1. Z-axis rotation (random angle 0-360 degrees)
        if params.get('enable_z_rotation', True):
            angle_deg = np.random.uniform(0, 360)
            angle_rad = np.radians(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Rotation matrix around Z-axis
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])

            xyz = xyz @ rotation_matrix.T

        # 2. Horizontal mirroring (50% chance, randomly X or Y axis)
        if params.get('enable_horizontal_mirror', True):
            if np.random.random() > 0.5:
                # Mirror along X or Y axis
                if np.random.random() > 0.5:
                    xyz[:, 0] *= -1  # Mirror X-axis (left-right)
                else:
                    xyz[:, 1] *= -1  # Mirror Y-axis (front-back)

        # 3. XYZ jitter (Gaussian noise)
        if params.get('enable_xyz_jitter', True):
            sigma = params.get('jitter_sigma', 0.01)
            jitter = np.random.normal(0, sigma, xyz.shape)
            xyz += jitter

        return xyz

    def _process_cluster(self, cluster_path: str, params: Dict[str, Any], target_points: int, augmentation_index: int = 0) -> np.ndarray:
        """
        Process a single cluster file with optional augmentation.

        IMPORTANT: Processing order:
        1. Load cluster
        2. Apply augmentation (if augmentation_index > 0)
        3. SUBSAMPLE if cluster > 20,000 points (for feature computation efficiency)
        4. Normalize to unit sphere
        5. Compute features (normals, eigenvalues) on subsampled+normalized cluster
        6. Return FULL-SIZE features (variable size up to 20K points)

        NOTE: Clusters must have >= target_points (n×1024) but are saved at full size.
        Random subsampling to n×1024 happens during training, not here!

        Args:
            cluster_path: Path to .npy cluster file
            params: Processing parameters
            target_points: Required number of points (n × 1024)
            augmentation_index: Index for augmentation (0 = no augmentation)

        Returns:
            Processed sample data as numpy array, or None if cluster should be skipped
        """
        # Load cluster (expecting shape (n, 6) with XYZ + RGB)
        cluster_data = np.load(cluster_path)

        # Check if cluster has enough points
        if len(cluster_data) < target_points:
            return None

        # Extract XYZ only (discard RGB)
        xyz = cluster_data[:, :3].copy()

        # Apply augmentation BEFORE normalization (if not the first version)
        if augmentation_index > 0:
            # Create unique seed from cluster path and augmentation index
            seed = hash(cluster_path + str(augmentation_index)) % (2**32)
            xyz = self._apply_augmentation(xyz, params, seed)

        # CRITICAL FIX: Subsample large clusters BEFORE feature computation
        # This matches the inference pipeline and prevents data leakage
        max_points_for_features = params.get('max_points_for_features', 20000)
        if len(xyz) > max_points_for_features:
            indices = np.random.choice(len(xyz), max_points_for_features, replace=False)
            xyz = xyz[indices]

        # Create PointCloud from subsampled cluster
        point_cloud = PointCloud(points=xyz)

        # Build feature list
        features = []

        # Apply normalization if requested
        if params['normalize']:
            point_cloud.normalise(
                apply_scaling=True,
                apply_centering=True,
                rotation_axes=(False, False, False)
            )

        # Add XYZ (normalized or not)
        features.append(point_cloud.points)

        # Compute normals if requested
        if params['compute_normals']:
            knn = params['normals_knn']

            # Check if we have enough points for KNN
            if len(point_cloud.points) < knn:
                print(f"Warning: Cluster has {len(point_cloud.points)} points, less than KNN={knn}. Skipping normals.")
            else:
                point_cloud.estimate_normals(k=knn)
                features.append(point_cloud.normals)

        # Compute eigenvalues if requested
        if params['compute_eigenvalues']:
            knn = params['eigenvalues_knn']
            smooth = params['eigenvalues_smooth']

            # Check if we have enough points for KNN
            if len(point_cloud.points) < knn:
                print(f"Warning: Cluster has {len(point_cloud.points)} points, less than KNN={knn}. Skipping eigenvalues.")
            else:
                eigenvalues = point_cloud.get_eigenvalues(k=knn, smooth=smooth)
                features.append(eigenvalues)

        # Stack features horizontally
        combined = np.hstack(features).astype(np.float32)

        # NOTE: We do NOT subsample to target_points (n×1024) here anymore!
        # Random subsampling will be done during training for better augmentation
        # This keeps full feature-rich data (up to max_points_for_features = 20K points)
        # But we still require minimum of target_points to ensure enough data

        return combined

    def _balance_classes(
        self,
        class_samples: Dict[str, List[Tuple[np.ndarray, str, int]]],
        params: Dict[str, Any],
        target_points: int
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Dict[str, int]]]:
        """
        Balance classes by resampling to match the maximum class count.

        With augmentation: samples already include multiple augmented versions per cluster,
        so we just need to ensure all classes have the same total number of samples.

        Args:
            class_samples: Dictionary of {class_name: [(sample_data, source_path, aug_idx), ...]}
            params: Processing parameters
            target_points: Target points per sample

        Returns:
            Tuple of (balanced_samples, balance_stats) where:
            - balanced_samples: {class_name: [sample_data, ...]}
            - balance_stats: {class_name: {'unique_clusters': int, 'total_samples': int}}
        """
        # Find maximum class count (total samples including augmented versions)
        max_count = max(len(samples) for samples in class_samples.values())

        balanced_samples = {}
        balance_stats = {}
        augmentation_multiplier = params.get('augmentation_multiplier', 1)

        for class_name, samples in class_samples.items():
            # Calculate number of unique clusters (divide by augmentation_multiplier)
            num_unique_clusters = len(samples) // augmentation_multiplier

            if len(samples) >= max_count:
                # Already has enough samples, just take the first max_count
                balanced_samples[class_name] = [data for data, _, _ in samples[:max_count]]
                balance_stats[class_name] = {
                    'unique_clusters': num_unique_clusters,
                    'total_samples': max_count
                }
            else:
                # Need to cycle through samples to reach max_count
                all_class_samples = []
                sample_idx = 0

                for _ in range(max_count):
                    data, _, _ = samples[sample_idx % len(samples)]
                    all_class_samples.append(data)
                    sample_idx += 1

                balanced_samples[class_name] = all_class_samples
                balance_stats[class_name] = {
                    'unique_clusters': num_unique_clusters,
                    'total_samples': max_count
                }

        return balanced_samples, balance_stats

    def _save_metadata(
        self,
        output_dir: str,
        params: Dict[str, Any],
        balanced_samples: Dict[str, List[np.ndarray]],
        balance_stats: Dict[str, Dict[str, int]],
        skipped_count: int
    ) -> None:
        """
        Generate and save metadata.json.

        Args:
            output_dir: Output directory path
            params: Processing parameters
            balanced_samples: Balanced samples by class
            balance_stats: Balance statistics
            skipped_count: Number of skipped clusters
        """
        # Determine feature order
        feature_order = ["X_norm", "Y_norm", "Z_norm"] if params['normalize'] else ["X", "Y", "Z"]
        num_features = 3

        if params['compute_normals']:
            feature_order.extend(["Nx", "Ny", "Nz"])
            num_features += 3

        if params['compute_eigenvalues']:
            feature_order.extend(["E1", "E2", "E3"])
            num_features += 3

        # Get target points
        target_points = params['n_multiplier'] * 1024

        # Build metadata
        augmentation_multiplier = params.get('augmentation_multiplier', 1)
        augmentation_enabled = augmentation_multiplier > 1

        metadata = {
            "dataset_info": {
                "created_at": datetime.now().isoformat(),
                "created_by": "SPCToolkit",
                "plugin": "PointNet Generate Training Data v2.0 (with augmentation)",
                "source_directory": params['input_directory'],
                "output_directory": output_dir,
                "model_target": "PointNet",
                "task_type": "Feature Classification"
            },
            "processing": {
                "normalization": {
                    "enabled": params['normalize'],
                    "method": "center_and_unit_sphere" if params['normalize'] else None,
                    "random_rotation": False
                },
                "augmentation": {
                    "enabled": augmentation_enabled,
                    "multiplier": augmentation_multiplier,
                    "z_rotation": params.get('enable_z_rotation', False),
                    "horizontal_mirror": params.get('enable_horizontal_mirror', False),
                    "xyz_jitter": params.get('enable_xyz_jitter', False),
                    "jitter_sigma": params.get('jitter_sigma', 0.0) if params.get('enable_xyz_jitter', False) else None,
                    "note": "Augmentation applied BEFORE normalization, features computed on augmented+normalized data" if augmentation_enabled else None
                },
                "features": {
                    "normals": {
                        "enabled": params['compute_normals'],
                        "knn": params['normals_knn'] if params['compute_normals'] else None
                    },
                    "eigenvalues": {
                        "enabled": params['compute_eigenvalues'],
                        "knn": params['eigenvalues_knn'] if params['compute_eigenvalues'] else None,
                        "smooth": params['eigenvalues_smooth'] if params['compute_eigenvalues'] else None
                    },
                    "distance_from_ground": {
                        "enabled": False
                    }
                },
                "sampling": {
                    "n_multiplier": params['n_multiplier'],
                    "target_points_per_sample": target_points
                }
            },
            "data_format": {
                "file_format": "numpy (.npy)",
                "array_shape": f"({target_points}, {num_features})",
                "feature_order": feature_order,
                "data_type": "float32"
            },
            "balance_info": balance_stats,
            "processing_summary": {
                "total_samples": sum(len(samples) for samples in balanced_samples.values()),
                "num_classes": len(balanced_samples),
                "skipped_clusters": skipped_count
            }
        }

        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _show_summary(
        self,
        main_window,
        output_dir: str,
        balanced_samples: Dict[str, List[np.ndarray]],
        balance_stats: Dict[str, Dict[str, int]],
        skipped_count: int,
        target_points: int
    ) -> None:
        """
        Show summary message dialog.

        Args:
            main_window: Main window
            output_dir: Output directory
            balanced_samples: Balanced samples by class
            balance_stats: Balance statistics
            skipped_count: Number of skipped clusters
            target_points: Target points per sample
        """
        total_samples = sum(len(samples) for samples in balanced_samples.values())

        summary_lines = [
            f"Successfully generated {total_samples} training samples",
            f"Output: {output_dir}\n"
        ]

        if balanced_samples:
            summary_lines.append("Breakdown by class:")
            for class_name in sorted(balanced_samples.keys()):
                unique_clusters = balance_stats[class_name]['unique_clusters']
                total_samples = balance_stats[class_name]['total_samples']
                summary_lines.append(
                    f"  - {class_name}: {total_samples} samples from {unique_clusters} unique clusters"
                )

        if skipped_count > 0:
            summary_lines.append(f"\nSkipped {skipped_count} clusters (< {target_points} points)")

        summary_lines.append(f"\nAll samples contain exactly {target_points} points")

        # Check if balanced
        counts = [len(samples) for samples in balanced_samples.values()]
        if len(set(counts)) == 1:
            summary_lines.append(f"Balanced to {counts[0]} samples per class")

        QMessageBox.information(
            main_window,
            "Training Data Generated",
            "\n".join(summary_lines)
        )