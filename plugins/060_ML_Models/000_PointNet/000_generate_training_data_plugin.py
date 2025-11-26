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

        try:
            # Step 1: Scan input directory and organize clusters by class
            print("\n[1/6] Scanning input directory...")
            class_clusters = self._scan_input_directory(input_dir)

            print(f"DEBUG: Scanned input directory: {input_dir}")
            print(f"DEBUG: Found {len(class_clusters)} classes")
            for class_name, paths in class_clusters.items():
                print(f"  - {class_name}: {len(paths)} files")

            if not class_clusters:
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

            # Step 2: Process all clusters (first pass)
            print(f"\n[2/6] Processing clusters (target: {target_points} points)...")
            class_samples = {}  # {class_name: [(data, source_cluster_path), ...]}
            skipped_total = 0
            error_count = 0

            for class_name, cluster_paths in class_clusters.items():
                class_samples[class_name] = []
                skipped_class = 0

                for cluster_path in cluster_paths:
                    try:
                        # Process cluster
                        sample_data = self._process_cluster(cluster_path, params, target_points)

                        if sample_data is not None:
                            class_samples[class_name].append((sample_data, cluster_path))
                        else:
                            skipped_class += 1

                    except Exception as e:
                        error_count += 1
                        print(f"Error processing {cluster_path}: {str(e)}")

                skipped_total += skipped_class
                print(f"Class '{class_name}': {len(class_samples[class_name])} samples, {skipped_class} skipped")

            # Remove empty classes
            class_samples = {k: v for k, v in class_samples.items() if len(v) > 0}

            if not class_samples:
                QMessageBox.warning(
                    main_window,
                    "No Valid Samples",
                    f"No clusters had enough points (>= {target_points}).\n"
                    f"Skipped {skipped_total} clusters."
                )
                return

            # Step 3: Balance classes
            print(f"\n[3/6] Balancing classes...")
            balanced_samples, balance_stats = self._balance_classes(class_samples, params, target_points)

            # Step 4: Save all samples
            print(f"\n[4/6] Saving training data...")
            total_saved = 0

            for class_name, samples in balanced_samples.items():
                class_dir = os.path.join(output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                for idx, sample_data in enumerate(samples, start=1):
                    filename = f"{class_name.lower()}_{idx}.npy"
                    filepath = os.path.join(class_dir, filename)
                    np.save(filepath, sample_data)
                    total_saved += 1

            print(f"   Saved {total_saved} training samples")

            # Step 5: Generate and save metadata
            print(f"\n[5/6] Generating metadata...")
            self._save_metadata(output_dir, params, balanced_samples, balance_stats, skipped_total)

            # Step 6: Update session persistence and show success message
            print(f"\n[6/6] Complete!")
            print("=" * 60)

            # Store output directory for Preview Training Data plugin
            DataPreviewWindow.last_directories["Training Data Preview"] = output_dir

            self._show_summary(main_window, output_dir, balanced_samples, balance_stats, skipped_total, target_points)

        except Exception as e:
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

    def _process_cluster(self, cluster_path: str, params: Dict[str, Any], target_points: int) -> np.ndarray:
        """
        Process a single cluster file.

        IMPORTANT: Features (normals, eigenvalues) must be computed on the FULL cluster
        before subsampling, as they depend on local neighborhoods.

        Args:
            cluster_path: Path to .npy cluster file
            params: Processing parameters
            target_points: Required number of points (n × 1024)

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

        # Create PointCloud from FULL cluster (DO NOT subsample yet!)
        point_cloud = PointCloud(points=xyz)

        # Build feature list by computing on FULL cluster
        features = []

        # Apply normalization if requested (on full cluster)
        if params['normalize']:
            point_cloud.normalise(
                apply_scaling=True,
                apply_centering=True,
                rotation_axes=(False, False, False)
            )

        # Add XYZ (normalized or not)
        features.append(point_cloud.points)

        # Compute normals if requested (on full cluster)
        if params['compute_normals']:
            knn = params['normals_knn']

            # Check if we have enough points for KNN
            if len(point_cloud.points) < knn:
                print(f"Warning: Cluster has {len(point_cloud.points)} points, less than KNN={knn}. Skipping normals.")
            else:
                point_cloud.estimate_normals(k=knn)
                features.append(point_cloud.normals)

        # Compute eigenvalues if requested (on full cluster)
        if params['compute_eigenvalues']:
            knn = params['eigenvalues_knn']
            smooth = params['eigenvalues_smooth']

            # Check if we have enough points for KNN
            if len(point_cloud.points) < knn:
                print(f"Warning: Cluster has {len(point_cloud.points)} points, less than KNN={knn}. Skipping eigenvalues.")
            else:
                eigenvalues = point_cloud.get_eigenvalues(k=knn, smooth=smooth)
                features.append(eigenvalues)

        # Stack features horizontally (full cluster with all computed features)
        combined = np.hstack(features).astype(np.float32)

        # NOW subsample to target_points (AFTER all features are computed)
        # This preserves feature correspondence and neighborhood-based computations
        if len(combined) > target_points:
            indices = np.random.choice(len(combined), target_points, replace=False)
            combined = combined[indices]

        return combined

    def _balance_classes(
        self,
        class_samples: Dict[str, List[Tuple[np.ndarray, str]]],
        params: Dict[str, Any],
        target_points: int
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Dict[str, int]]]:
        """
        Balance classes by resampling to match the maximum class count.

        Uses even distribution strategy: if 105 samples needed from 29 clusters,
        create floor(105/29)=3 samples from all clusters, then 4 samples from first 18 clusters.

        Args:
            class_samples: Dictionary of {class_name: [(sample_data, source_path), ...]}
            params: Processing parameters
            target_points: Target points per sample

        Returns:
            Tuple of (balanced_samples, balance_stats) where:
            - balanced_samples: {class_name: [sample_data, ...]}
            - balance_stats: {class_name: {'unique': int, 'resampled': int}}
        """
        # Find maximum class count
        max_count = max(len(samples) for samples in class_samples.values())

        balanced_samples = {}
        balance_stats = {}

        for class_name, samples in class_samples.items():
            num_unique_clusters = len(samples)

            if num_unique_clusters >= max_count:
                # Already has enough samples
                balanced_samples[class_name] = [data for data, _ in samples[:max_count]]
                balance_stats[class_name] = {
                    'unique': num_unique_clusters,
                    'resampled': 0
                }
            else:
                # Need to generate more samples through resampling
                base_samples_per_cluster = max_count // num_unique_clusters
                remainder = max_count - (base_samples_per_cluster * num_unique_clusters)

                all_class_samples = []

                for cluster_idx, (original_data, source_path) in enumerate(samples):
                    # Determine how many samples to create from this cluster
                    if cluster_idx < remainder:
                        samples_to_create = base_samples_per_cluster + 1
                    else:
                        samples_to_create = base_samples_per_cluster

                    # Create samples by reprocessing the source cluster
                    for _ in range(samples_to_create):
                        # Reprocess to get different random subsample
                        sample_data = self._process_cluster(source_path, params, target_points)
                        if sample_data is not None:
                            all_class_samples.append(sample_data)

                balanced_samples[class_name] = all_class_samples
                balance_stats[class_name] = {
                    'unique': num_unique_clusters,
                    'resampled': max_count - num_unique_clusters
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
        metadata = {
            "dataset_info": {
                "created_at": datetime.now().isoformat(),
                "created_by": "SPCToolkit",
                "plugin": "PointNet Generate Training Data v1.0",
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
                unique = balance_stats[class_name]['unique']
                resampled = balance_stats[class_name]['resampled']
                count = len(balanced_samples[class_name])
                summary_lines.append(
                    f"  - {class_name}: {count} samples ({unique} unique clusters, {resampled} resampled)"
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