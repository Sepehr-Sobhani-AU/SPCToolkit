"""
CRITICAL DIAGNOSTIC: Find preprocessing mismatch between training and inference.

Since the model fails on the SAME data used for training, there must be a
difference in how features are computed.
"""

import numpy as np
import open3d as o3d
from core.point_cloud import PointCloud

def load_and_process_for_training(ply_path):
    """
    Replicate EXACTLY what convert_ply_to_training_data.py does.
    """
    print("\n" + "="*80)
    print("TRAINING PIPELINE")
    print("="*80)

    # Load PLY
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    print(f"1. Loaded PLY: {len(points)} points")
    print(f"   XYZ range: [{points.min():.3f}, {points.max():.3f}]")

    # Create PointCloud
    point_cloud = PointCloud(points=points)
    print(f"2. Created PointCloud")

    # NORMALIZE (custom function from convert script)
    centroid = np.mean(point_cloud.points, axis=0)
    centered_points = point_cloud.points - centroid
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    if max_distance > 0:
        normalized_points = centered_points / max_distance
    else:
        normalized_points = centered_points

    normalized_pc = PointCloud(
        points=normalized_points.astype(np.float32),
        colors=point_cloud.colors,
        normals=point_cloud.normals
    )
    print(f"3. Normalized (custom)")
    print(f"   Centroid: {centroid}")
    print(f"   Max distance: {max_distance:.6f}")
    print(f"   Normalized range: [{normalized_pc.points.min():.6f}, {normalized_pc.points.max():.6f}]")

    # Compute normals on ALL points
    knn = 30
    normalized_pc.estimate_normals(k=knn)
    normals_full = normalized_pc.normals
    print(f"4. Computed normals (k={knn})")
    print(f"   Normals shape: {normals_full.shape}")
    print(f"   Normals range: [{normals_full.min():.6f}, {normals_full.max():.6f}]")
    print(f"   Mean magnitude: {np.linalg.norm(normals_full, axis=1).mean():.6f}")

    # Compute eigenvalues on ALL points
    eigenvalues_full = normalized_pc.get_eigenvalues(k=knn, smooth=True)
    print(f"5. Computed eigenvalues (k={knn}, smooth=True)")
    print(f"   Eigenvalues shape: {eigenvalues_full.shape}")
    print(f"   Eigenvalues range: [{eigenvalues_full.min():.9f}, {eigenvalues_full.max():.9f}]")
    print(f"   Mean: {eigenvalues_full.mean(axis=0)}")

    # Stack features
    features_full = np.hstack([normalized_pc.points, normals_full, eigenvalues_full])
    print(f"6. Stacked features: {features_full.shape}")

    # Sample to 1024
    target_points = 1024
    if features_full.shape[0] > target_points:
        indices = np.random.choice(features_full.shape[0], target_points, replace=False)
        features = features_full[indices]
    else:
        features = features_full
        if features.shape[0] < target_points:
            num_needed = target_points - features.shape[0]
            indices = np.random.choice(features.shape[0], num_needed, replace=True)
            features = np.vstack([features, features[indices]])

    print(f"7. Sampled to {target_points} points")
    print(f"   Final shape: {features.shape}")

    return features


def load_and_process_for_inference(ply_path):
    """
    Replicate EXACTLY what inference.py does.
    """
    print("\n" + "="*80)
    print("INFERENCE PIPELINE")
    print("="*80)

    # Load PLY
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    print(f"1. Loaded PLY: {len(points)} points")
    print(f"   XYZ range: [{points.min():.3f}, {points.max():.3f}]")

    # Create PointCloud from cluster
    point_cloud = PointCloud(points=points.copy())
    print(f"2. Created PointCloud")

    # Normalize using normalise() method
    point_cloud.normalise(
        apply_scaling=True,
        apply_centering=True,
        rotation_axes=(False, False, False)
    )
    print(f"3. Normalized (normalise method)")
    print(f"   Normalized range: [{point_cloud.points.min():.6f}, {point_cloud.points.max():.6f}]")

    # Compute normals
    knn = 30
    if len(point_cloud.points) >= knn:
        point_cloud.estimate_normals(k=knn)
        normals = point_cloud.normals
    else:
        normals = np.zeros((len(point_cloud.points), 3), dtype=np.float32)

    print(f"4. Computed normals (k={knn})")
    print(f"   Normals shape: {normals.shape}")
    print(f"   Normals range: [{normals.min():.6f}, {normals.max():.6f}]")
    print(f"   Mean magnitude: {np.linalg.norm(normals, axis=1).mean():.6f}")

    # Compute eigenvalues
    if len(point_cloud.points) >= knn:
        eigenvalues = point_cloud.get_eigenvalues(k=knn, smooth=True)
    else:
        eigenvalues = np.zeros((len(point_cloud.points), 3), dtype=np.float32)

    print(f"5. Computed eigenvalues (k={knn}, smooth=True)")
    print(f"   Eigenvalues shape: {eigenvalues.shape}")
    print(f"   Eigenvalues range: [{eigenvalues.min():.9f}, {eigenvalues.max():.9f}]")
    print(f"   Mean: {eigenvalues.mean(axis=0)}")

    # Stack features
    features = [point_cloud.points, normals, eigenvalues]
    combined = np.hstack(features).astype(np.float32)
    print(f"6. Stacked features: {combined.shape}")

    # Sample to 1024
    num_points = 1024
    if len(combined) > num_points:
        indices = np.random.choice(len(combined), num_points, replace=False)
        combined = combined[indices]
    elif len(combined) < num_points:
        deficit = num_points - len(combined)
        if len(combined) > 0:
            pad_indices = np.random.choice(len(combined), deficit, replace=True)
            pad_data = combined[pad_indices]
            combined = np.vstack([combined, pad_data])

    print(f"7. Sampled to {num_points} points")
    print(f"   Final shape: {combined.shape}")

    return combined


def compare_in_detail(train_features, infer_features):
    """
    Detailed comparison of features.
    """
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)

    # Compare each feature column
    for i, name in enumerate(['X', 'Y', 'Z', 'Nx', 'Ny', 'Nz', 'E1', 'E2', 'E3']):
        train_col = train_features[:, i]
        infer_col = infer_features[:, i]

        print(f"\n{name}:")
        print(f"  Train - mean: {train_col.mean():10.6f}, std: {train_col.std():10.6f}, range: [{train_col.min():10.6f}, {train_col.max():10.6f}]")
        print(f"  Infer - mean: {infer_col.mean():10.6f}, std: {infer_col.std():10.6f}, range: [{infer_col.min():10.6f}, {infer_col.max():10.6f}]")

        # Check if values are identical
        if np.allclose(train_col, infer_col, rtol=1e-5, atol=1e-7):
            print(f"  ✅ IDENTICAL (within tolerance)")
        else:
            mean_diff = abs(train_col.mean() - infer_col.mean())
            std_diff = abs(train_col.std() - infer_col.std())
            print(f"  ❌ DIFFERENT - mean diff: {mean_diff:.6e}, std diff: {std_diff:.6e}")

    # Check data types
    print(f"\nData types:")
    print(f"  Train: {train_features.dtype}")
    print(f"  Infer: {infer_features.dtype}")

    # Check for NaN/Inf
    print(f"\nNaN/Inf check:")
    print(f"  Train NaN: {np.isnan(train_features).sum()}, Inf: {np.isinf(train_features).sum()}")
    print(f"  Infer NaN: {np.isnan(infer_features).sum()}, Inf: {np.isinf(infer_features).sum()}")

    # Overall similarity
    mse = np.mean((train_features - infer_features) ** 2)
    print(f"\nMean Squared Error: {mse:.6e}")

    if mse < 1e-10:
        print("✅ Features are IDENTICAL - preprocessing is consistent")
        print("   → Problem must be elsewhere (data quality, labels, model)")
    elif mse < 1e-5:
        print("⚠️  Features are SIMILAR but not identical")
        print("   → Small numerical differences (likely acceptable)")
    else:
        print("❌ Features are SIGNIFICANTLY DIFFERENT")
        print("   → PREPROCESSING MISMATCH CONFIRMED")


if __name__ == "__main__":
    import sys

    print("\n" + "="*80)
    print("PREPROCESSING MISMATCH DIAGNOSTIC")
    print("="*80)

    # Check if PLY file provided
    if len(sys.argv) < 2:
        print("\nUsage: python diagnose_mismatch.py <path_to_ply_file>")
        print("\nExample:")
        print("  python diagnose_mismatch.py training_data/Car/car_1.ply")
        sys.exit(1)

    ply_path = sys.argv[1]

    try:
        # Process using training pipeline
        train_features = load_and_process_for_training(ply_path)

        # Process using inference pipeline
        infer_features = load_and_process_for_inference(ply_path)

        # Compare
        compare_in_detail(train_features, infer_features)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()