# convert_ply_to_training_data.py
"""
Convert PLY files from dataset directory to PointNet training data format.

This script:
1. Loads PLY files from a directory structure (class_name/file.ply)
2. Processes each cluster: normalize, compute normals, eigenvalues
3. Saves as .npy files in training_data format (1024, 9)

Usage:
    python convert_ply_to_training_data.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Import project modules
from core.point_cloud import PointCloud


def load_ply_file(filepath):
    """
    Load a PLY file and return a PointCloud object.

    Args:
        filepath: Path to PLY file

    Returns:
        PointCloud object or None if error
    """
    try:
        import open3d as o3d

        # Load PLY file
        pcd = o3d.io.read_point_cloud(filepath)

        # Extract data
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) > 0 else None
        normals = np.asarray(pcd.normals, dtype=np.float32) if len(pcd.normals) > 0 else None

        if len(points) == 0:
            print(f"  WARNING: Empty point cloud in {filepath}")
            return None

        # Create PointCloud object
        point_cloud = PointCloud(points=points, colors=colors, normals=normals)
        return point_cloud

    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None


def normalize_point_cloud(point_cloud):
    """
    Normalize point cloud to unit sphere centered at origin.

    Args:
        point_cloud: PointCloud object

    Returns:
        Normalized PointCloud object
    """
    # Center at origin
    centroid = np.mean(point_cloud.points, axis=0)
    centered_points = point_cloud.points - centroid

    # Scale to unit sphere
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    if max_distance > 0:
        normalized_points = centered_points / max_distance
    else:
        normalized_points = centered_points

    # Create new point cloud with normalized points
    normalized_pc = PointCloud(
        points=normalized_points.astype(np.float32),
        colors=point_cloud.colors,
        normals=point_cloud.normals
    )

    return normalized_pc


def process_cluster(point_cloud, target_points=1024, compute_normals=True, compute_eigenvalues=True, knn=30):
    """
    Process a cluster into training data format.

    CRITICAL PIPELINE (must match inference exactly):
    1. Check if enough points
    2. NORMALIZE to unit sphere (ALL points)
    3. Compute normals and eigenvalues on ALL NORMALIZED points
    4. Sample to target_points (sampling XYZ, normals, eigenvalues together)

    Args:
        point_cloud: PointCloud object
        target_points: Number of points to sample (default: 1024)
        compute_normals: Whether to compute normals
        compute_eigenvalues: Whether to compute eigenvalues
        knn: KNN parameter for normals/eigenvalues

    Returns:
        numpy array of shape (target_points, num_features) or None
    """
    # Check if enough points
    if point_cloud.size < target_points:
        print(f"    Skipping: only {point_cloud.size} points (need {target_points})")
        return None

    # CRITICAL: Normalize FIRST (on ALL points)
    normalized_pc = normalize_point_cloud(point_cloud)

    # CRITICAL: Compute normals and eigenvalues on ALL normalized points (BEFORE downsampling)
    # This ensures high-quality features computed from full neighborhood context
    if compute_normals:
        try:
            normalized_pc.estimate_normals(k=knn)
            normals_full = normalized_pc.normals  # (n_points, 3)
        except Exception as e:
            print(f"    WARNING: Normal computation failed: {e}")
            normals_full = np.zeros((normalized_pc.size, 3), dtype=np.float32)
    else:
        normals_full = np.zeros((normalized_pc.size, 3), dtype=np.float32)

    if compute_eigenvalues:
        try:
            eigenvalues_full = normalized_pc.get_eigenvalues(k=knn, smooth=True)  # (n_points, 3)
        except Exception as e:
            print(f"    WARNING: Eigenvalue computation failed: {e}")
            eigenvalues_full = np.zeros((normalized_pc.size, 3), dtype=np.float32)
    else:
        eigenvalues_full = np.zeros((normalized_pc.size, 3), dtype=np.float32)

    # Combine ALL features: [X_norm, Y_norm, Z_norm, Nx, Ny, Nz, E1, E2, E3] (n_points, 9)
    features_full = np.hstack([normalized_pc.points, normals_full, eigenvalues_full])

    # NOW sample to target_points (after computing features on full cluster)
    if features_full.shape[0] > target_points:
        # Random sampling
        indices = np.random.choice(features_full.shape[0], target_points, replace=False)
        features = features_full[indices]
    else:
        # Use all points (pad if needed)
        features = features_full
        if features.shape[0] < target_points:
            # Random duplication to reach target_points
            indices = np.random.choice(features.shape[0], target_points - features.shape[0])
            features = np.vstack([features, features[indices]])

    return features.astype(np.float32)


def convert_dataset(input_dir, output_dir, target_points=1024, min_points_per_class=3):
    """
    Convert entire PLY dataset to training data format.

    Args:
        input_dir: Directory containing class subdirectories with PLY files
        output_dir: Directory to save training data
        target_points: Number of points per sample
        min_points_per_class: Minimum samples required per class

    Returns:
        Dictionary with conversion statistics
    """
    print("=" * 80)
    print("PLY Dataset to PointNet Training Data Conversion")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target points per sample: {target_points}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Scan input directory for class folders
    print("[1/3] Scanning input directory...")
    class_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('_'):
            class_dirs.append(item)

    class_dirs = sorted(class_dirs)
    print(f"Found {len(class_dirs)} class directories")

    # Process each class
    print(f"\n[2/3] Processing PLY files...")
    conversion_stats = {}
    total_processed = 0
    total_skipped = 0

    for class_name in class_dirs:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)

        # Find PLY files
        ply_files = [f for f in os.listdir(class_input_dir) if f.endswith('.ply')]

        if len(ply_files) == 0:
            print(f"  {class_name}: No PLY files found, skipping")
            continue

        print(f"  {class_name}: Processing {len(ply_files)} files...")

        # Create output directory
        os.makedirs(class_output_dir, exist_ok=True)

        processed = 0
        skipped = 0

        for ply_file in ply_files:
            ply_path = os.path.join(class_input_dir, ply_file)

            # Load PLY
            point_cloud = load_ply_file(ply_path)
            if point_cloud is None:
                skipped += 1
                continue

            # Process cluster
            features = process_cluster(
                point_cloud,
                target_points=target_points,
                compute_normals=True,
                compute_eigenvalues=True,
                knn=30
            )

            if features is None:
                skipped += 1
                continue

            # Save as .npy
            output_filename = f"{class_name.lower()}_{processed + 1}.npy"
            output_path = os.path.join(class_output_dir, output_filename)
            np.save(output_path, features)

            processed += 1

        total_processed += processed
        total_skipped += skipped

        conversion_stats[class_name] = {
            'processed': processed,
            'skipped': skipped
        }

        print(f"    [OK] Saved {processed} samples, skipped {skipped}")

    # Filter out classes with too few samples
    print(f"\n[3/3] Filtering classes (minimum {min_points_per_class} samples)...")
    filtered_stats = {}
    removed_classes = []

    for class_name, stats in conversion_stats.items():
        if stats['processed'] >= min_points_per_class:
            filtered_stats[class_name] = stats
        else:
            removed_classes.append(class_name)
            # Remove directory
            class_output_dir = os.path.join(output_dir, class_name)
            if os.path.exists(class_output_dir):
                import shutil
                shutil.rmtree(class_output_dir)

    if removed_classes:
        print(f"  Removed {len(removed_classes)} classes with insufficient samples:")
        for cls in removed_classes:
            print(f"    - {cls} ({conversion_stats[cls]['processed']} samples)")

    # Save metadata
    metadata = {
        "dataset_info": {
            "created_at": datetime.now().isoformat(),
            "created_by": "SPCToolkit PLY Converter",
            "source_directory": input_dir,
            "output_directory": output_dir,
            "model_target": "PointNet",
            "task_type": "Feature Classification"
        },
        "processing": {
            "normalization": {
                "enabled": True,
                "method": "center_and_unit_sphere"
            },
            "features": {
                "normals": {
                    "enabled": True,
                    "knn": 30
                },
                "eigenvalues": {
                    "enabled": True,
                    "knn": 30,
                    "smooth": True
                }
            },
            "sampling": {
                "target_points_per_sample": target_points
            }
        },
        "data_format": {
            "file_format": "numpy (.npy)",
            "array_shape": f"({target_points}, 9)",
            "feature_order": [
                "X_norm", "Y_norm", "Z_norm",
                "Nx", "Ny", "Nz",
                "E1", "E2", "E3"
            ],
            "data_type": "float32"
        },
        "conversion_stats": filtered_stats,
        "processing_summary": {
            "total_samples": total_processed,
            "num_classes": len(filtered_stats),
            "skipped_files": total_skipped,
            "removed_classes": removed_classes
        }
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"Total samples processed: {total_processed}")
    print(f"Total classes: {len(filtered_stats)}")
    print(f"Skipped files: {total_skipped}")
    print(f"\nClass breakdown:")
    for class_name, stats in filtered_stats.items():
        print(f"  {class_name}: {stats['processed']} samples")
    print(f"\nOutput directory: {output_dir}")
    print(f"Metadata saved: {metadata_path}")
    print("=" * 80)

    return metadata


def main():
    """Main conversion function."""
    # Configuration
    INPUT_DIR = r"C:\Users\Sepeh\OneDrive\AI\Point Cloud\_Data\Dataset"
    OUTPUT_DIR = r"training_data"
    TARGET_POINTS = 1024
    MIN_SAMPLES_PER_CLASS = 3

    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    # Run conversion
    try:
        metadata = convert_dataset(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            target_points=TARGET_POINTS,
            min_points_per_class=MIN_SAMPLES_PER_CLASS
        )
    except Exception as e:
        print(f"\nERROR during conversion:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
