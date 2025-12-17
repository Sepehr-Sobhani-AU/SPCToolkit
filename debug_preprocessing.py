"""
Diagnostic script to compare training vs inference preprocessing.

This will help identify why inference produces low confidence (unclassified) predictions.
"""

import numpy as np
from core.point_cloud import PointCloud
from models.pointnet.inference import _process_cluster_for_inference
import glob

def analyze_training_sample(filepath):
    """Analyze a training data sample."""
    data = np.load(filepath)

    print(f"\n{'='*80}")
    print(f"TRAINING DATA: {filepath}")
    print(f"{'='*80}")
    print(f"Shape: {data.shape}")
    print(f"\nXYZ (columns 0-2):")
    print(f"  Range: [{data[:, 0:3].min():.6f}, {data[:, 0:3].max():.6f}]")
    print(f"  Mean: {data[:, 0:3].mean(axis=0)}")
    print(f"  Std: {data[:, 0:3].std(axis=0)}")

    print(f"\nNormals (columns 3-5):")
    print(f"  Range: [{data[:, 3:6].min():.6f}, {data[:, 3:6].max():.6f}]")
    print(f"  Mean magnitude: {np.linalg.norm(data[:, 3:6], axis=1).mean():.6f}")
    print(f"  Min magnitude: {np.linalg.norm(data[:, 3:6], axis=1).min():.6f}")
    print(f"  Max magnitude: {np.linalg.norm(data[:, 3:6], axis=1).max():.6f}")

    print(f"\nEigenvalues (columns 6-8):")
    print(f"  Range: [{data[:, 6:9].min():.6f}, {data[:, 6:9].max():.6f}]")
    print(f"  Mean: {data[:, 6:9].mean(axis=0)}")
    print(f"  Sum mean: {data[:, 6:9].sum(axis=1).mean():.6f}")

    return data


def simulate_inference_preprocessing(ply_filepath):
    """Simulate what happens during inference."""
    # Load the PLY file
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_filepath)
    points = np.asarray(pcd.points).astype(np.float32)

    print(f"\n{'='*80}")
    print(f"INFERENCE SIMULATION: {ply_filepath}")
    print(f"{'='*80}")
    print(f"Original points: {len(points)}")

    # Process using inference pipeline
    features = _process_cluster_for_inference(
        cluster_xyz=points,
        num_points=1024,
        use_normals=True,
        use_eigenvalues=True
    )

    print(f"Processed shape: {features.shape}")
    print(f"\nXYZ (columns 0-2):")
    print(f"  Range: [{features[:, 0:3].min():.6f}, {features[:, 0:3].max():.6f}]")
    print(f"  Mean: {features[:, 0:3].mean(axis=0)}")
    print(f"  Std: {features[:, 0:3].std(axis=0)}")

    print(f"\nNormals (columns 3-5):")
    print(f"  Range: [{features[:, 3:6].min():.6f}, {features[:, 3:6].max():.6f}]")
    print(f"  Mean magnitude: {np.linalg.norm(features[:, 3:6], axis=1).mean():.6f}")
    print(f"  Min magnitude: {np.linalg.norm(features[:, 3:6], axis=1).min():.6f}")
    print(f"  Max magnitude: {np.linalg.norm(features[:, 3:6], axis=1).max():.6f}")

    print(f"\nEigenvalues (columns 6-8):")
    print(f"  Range: [{features[:, 6:9].min():.6f}, {features[:, 6:9].max():.6f}]")
    print(f"  Mean: {features[:, 6:9].mean(axis=0)}")
    print(f"  Sum mean: {features[:, 6:9].sum(axis=1).mean():.6f}")

    return features


def compare_features(train_data, inference_data):
    """Compare training vs inference features."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: Training vs Inference")
    print(f"{'='*80}")

    print(f"\nXYZ Difference:")
    print(f"  Mean abs diff: {np.abs(train_data[:, 0:3].mean(axis=0) - inference_data[:, 0:3].mean(axis=0))}")
    print(f"  Std ratio: {train_data[:, 0:3].std(axis=0) / (inference_data[:, 0:3].std(axis=0) + 1e-10)}")

    print(f"\nNormals Difference:")
    train_norm_mag = np.linalg.norm(train_data[:, 3:6], axis=1).mean()
    infer_norm_mag = np.linalg.norm(inference_data[:, 3:6], axis=1).mean()
    print(f"  Magnitude: train={train_norm_mag:.6f}, inference={infer_norm_mag:.6f}, ratio={train_norm_mag/infer_norm_mag:.6f}")

    print(f"\nEigenvalues Difference:")
    train_eigen_mean = train_data[:, 6:9].mean(axis=0)
    infer_eigen_mean = inference_data[:, 6:9].mean(axis=0)
    print(f"  Train mean: {train_eigen_mean}")
    print(f"  Inference mean: {infer_eigen_mean}")
    print(f"  Ratio: {train_eigen_mean / (infer_eigen_mean + 1e-10)}")

    # Check if eigenvalues are ordered
    print(f"\nEigenvalue Ordering:")
    train_sorted = np.all(train_data[:, 6:9] >= np.roll(train_data[:, 6:9], -1, axis=1)[:, :-1], axis=1).mean()
    infer_sorted = np.all(inference_data[:, 6:9] >= np.roll(inference_data[:, 6:9], -1, axis=1)[:, :-1], axis=1).mean()
    print(f"  Training sorted (desc): {train_sorted*100:.1f}% of samples")
    print(f"  Inference sorted (desc): {infer_sorted*100:.1f}% of samples")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PREPROCESSING DIAGNOSTIC TOOL")
    print("="*80)

    # Find a training sample (Car class)
    car_samples = glob.glob("training_data/Car/*.npy")
    if not car_samples:
        print("ERROR: No training data found!")
        exit(1)

    # Find corresponding PLY file
    ply_samples = glob.glob("training_data/Car/*.ply")

    if car_samples and ply_samples:
        # Analyze training data
        train_data = analyze_training_sample(car_samples[0])

        # Simulate inference on same PLY
        inference_data = simulate_inference_preprocessing(ply_samples[0])

        # Compare
        compare_features(train_data, inference_data)

        print(f"\n{'='*80}")
        print("VERDICT:")
        print(f"{'='*80}")

        # Check for major differences
        norm_diff = abs(np.linalg.norm(train_data[:, 3:6], axis=1).mean() -
                       np.linalg.norm(inference_data[:, 3:6], axis=1).mean())

        eigen_diff = np.abs(train_data[:, 6:9].mean() - inference_data[:, 6:9].mean())

        if norm_diff > 0.1:
            print("⚠️  PROBLEM: Normal vectors have different magnitudes!")
            print("   → Check normal computation/normalization")

        if eigen_diff > 0.0001:
            print("⚠️  PROBLEM: Eigenvalues have significant differences!")
            print("   → Check eigenvalue computation or smoothing")

        if norm_diff < 0.1 and eigen_diff < 0.0001:
            print("✅ Preprocessing appears consistent")
            print("   → Problem may be data distribution mismatch")
    else:
        print("ERROR: Need both .npy and .ply files for same cluster to compare")
        print(f"Found {len(car_samples)} .npy files and {len(ply_samples)} .ply files")