"""
Unit test for label_clusters_plugin (simplified version).

Tests that:
1. Plugin saves only raw XYZ coordinates (shape: N, 3)
2. No prerequisites required (eigenvalues, normals)
3. Subsampling works correctly when cluster exceeds max_points
4. Saved data is in correct format (float32)
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plugins.Training.label_clusters_plugin import LabelClustersPlugin
from core.entities.point_cloud import PointCloud


def test_extract_cluster_data():
    """Test the extract_cluster_data method."""
    print("=" * 60)
    print("Testing Label Clusters Plugin (Simplified)")
    print("=" * 60)

    # Create plugin instance
    plugin = LabelClustersPlugin()

    # Test 1: Basic cluster extraction (only XYZ)
    print("\n[TEST 1] Extract cluster data - only XYZ")
    points = np.random.rand(1000, 3).astype(np.float32) * 10  # Random points
    point_cloud = PointCloud(points=points)

    # Add cluster labels as attribute
    cluster_labels = np.zeros(1000, dtype=int)
    cluster_labels[:500] = 0  # First cluster
    cluster_labels[500:] = 1  # Second cluster
    point_cloud.add_attribute("cluster_labels", cluster_labels)

    cluster_mask = point_cloud.get_attribute("cluster_labels") == 0
    max_points = 10000

    try:
        cluster_data, was_subsampled = plugin.extract_cluster_data(
            point_cloud, cluster_mask, max_points
        )

        # Verify shape - should be (N, 3) for XYZ only
        assert cluster_data.shape[1] == 3, f"Expected 3 columns (XYZ), got {cluster_data.shape[1]}"
        assert cluster_data.shape[0] == 500, f"Expected 500 points, got {cluster_data.shape[0]}"
        assert not was_subsampled, "Should not be subsampled"
        assert cluster_data.dtype == np.float32, f"Expected float32, got {cluster_data.dtype}"

        print(f"[PASS] Cluster data shape: {cluster_data.shape}, dtype: {cluster_data.dtype}")
    except Exception as e:
        print(f"[FAIL] {e}")
        raise

    # Test 2: Verify no eigenvalues/normals requirement
    print("\n[TEST 2] No eigenvalues/normals required")
    points_simple = np.random.rand(100, 3).astype(np.float32)
    point_cloud_simple = PointCloud(points=points_simple)  # No normals, no eigenvalues
    point_cloud_simple.add_attribute("cluster_labels", np.zeros(100, dtype=int))

    cluster_mask_simple = point_cloud_simple.get_attribute("cluster_labels") == 0

    try:
        cluster_data_simple, _ = plugin.extract_cluster_data(
            point_cloud_simple, cluster_mask_simple, max_points
        )
        assert cluster_data_simple.shape == (100, 3), "Should extract XYZ even without normals/eigenvalues"
        print("[PASS] Extraction works without eigenvalues or normals")
    except Exception as e:
        print(f"[FAIL] {e}")
        raise

    # Test 3: Subsampling when cluster exceeds max_points
    print("\n[TEST 3] Subsampling large clusters")
    large_points = np.random.rand(5000, 3).astype(np.float32)
    point_cloud_large = PointCloud(points=large_points)
    point_cloud_large.add_attribute("cluster_labels", np.zeros(5000, dtype=int))

    cluster_mask_large = point_cloud_large.get_attribute("cluster_labels") == 0
    max_points_limit = 2000

    try:
        cluster_data_large, was_subsampled_large = plugin.extract_cluster_data(
            point_cloud_large, cluster_mask_large, max_points_limit
        )

        assert was_subsampled_large, "Should be subsampled"
        assert cluster_data_large.shape[0] == max_points_limit, f"Expected {max_points_limit} points, got {cluster_data_large.shape[0]}"
        assert cluster_data_large.shape[1] == 3, "Should still have 3 columns (XYZ)"

        print(f"[PASS] Subsampled from 5000 to {cluster_data_large.shape[0]} points")
    except Exception as e:
        print(f"[FAIL] {e}")
        raise

    # Test 4: Verify XYZ values are preserved
    print("\n[TEST 4] Verify XYZ values preserved")
    known_points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)
    point_cloud_known = PointCloud(points=known_points)
    point_cloud_known.add_attribute("cluster_labels", np.zeros(3, dtype=int))

    cluster_mask_known = point_cloud_known.get_attribute("cluster_labels") == 0

    try:
        cluster_data_known, _ = plugin.extract_cluster_data(
            point_cloud_known, cluster_mask_known, max_points
        )

        # Verify all XYZ values are present
        np.testing.assert_array_almost_equal(cluster_data_known, known_points)
        print("[PASS] XYZ values correctly preserved")
    except Exception as e:
        print(f"[FAIL] {e}")
        raise

    # Test 5: Save and load verification
    print("\n[TEST 5] Save and load verification")
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "test_cluster.npy"

        try:
            # Save the data
            np.save(save_path, cluster_data_known)

            # Load it back
            loaded_data = np.load(save_path)

            assert loaded_data.shape == (3, 3), f"Loaded shape mismatch: {loaded_data.shape}"
            assert loaded_data.dtype == np.float32, f"Loaded dtype mismatch: {loaded_data.dtype}"
            np.testing.assert_array_almost_equal(loaded_data, known_points)

            print(f"[PASS] Save/load successful: {save_path.name}")
        except Exception as e:
            print(f"[FAIL] {e}")
            raise

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_extract_cluster_data()
        sys.exit(0 if success else 1)
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
