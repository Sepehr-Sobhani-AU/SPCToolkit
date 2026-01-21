"""
Unit test for classify_cluster_plugin.

Tests that:
1. Plugin is correctly recognized as ActionPlugin
2. Plugin can classify selected clusters
3. Creates Clusters object with correct cluster_names
4. Updates existing Clusters when classifying more clusters
5. Assigns colors to classes automatically
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import importlib.util

# Import plugin with numeric prefix using importlib
plugin_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plugins", "040_Clusters", "000_classify_cluster_plugin.py"
)
spec = importlib.util.spec_from_file_location("classify_cluster_plugin", plugin_path)
classify_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classify_module)
ClassifyClusterPlugin = classify_module.ClassifyClusterPlugin

from plugins.interfaces import ActionPlugin
from core.point_cloud import PointCloud
from core.clusters import Clusters


def test_classify_cluster_plugin():
    """Test the classify_cluster_plugin."""
    print("=" * 60)
    print("Testing Classify Cluster Plugin")
    print("=" * 60)

    # Test 1: Plugin instantiation
    print("\n[TEST 1] Plugin instantiation")
    plugin = ClassifyClusterPlugin()
    assert isinstance(plugin, ActionPlugin), "Plugin should be an ActionPlugin"
    assert plugin.get_name() == "classify_cluster", "Plugin name should be 'classify_cluster'"
    print("[PASS] Plugin instantiated correctly")

    # Test 2: Get parameters
    print("\n[TEST 2] Get parameters")
    params = plugin.get_parameters()
    assert "class_name" in params, "Should have 'class_name' parameter"
    assert params["class_name"]["type"] == "choice", "class_name should be a choice parameter"
    assert "options" in params["class_name"], "class_name should have options"
    print(f"[PASS] Parameters: {list(params.keys())}")
    print(f"      Available classes: {params['class_name']['options'][:5]}...")  # Show first 5

    # Test 3: Create Clusters from cluster classification
    print("\n[TEST 3] Create Clusters object with cluster_names")

    # Create a point cloud with cluster labels
    points = np.random.rand(1000, 3).astype(np.float32)
    point_cloud = PointCloud(points=points)
    cluster_labels_data = np.zeros(1000, dtype=int)
    cluster_labels_data[:300] = 0    # Cluster 0
    cluster_labels_data[300:600] = 1  # Cluster 1
    cluster_labels_data[600:] = 2     # Cluster 2
    point_cloud.add_attribute("cluster_labels", cluster_labels_data)

    # Simulate classifying cluster 0 as "Tree"
    selected_cluster_ids = {0}
    class_name = "Tree"

    # Create initial Clusters with cluster_names
    clusters = plugin.create_or_update_clusters(
        point_cloud.get_attribute("cluster_labels"),
        selected_cluster_ids,
        class_name,
        existing_clusters=None
    )

    assert isinstance(clusters, Clusters), "Should return Clusters object"
    assert 0 in clusters.cluster_names, "Cluster 0 should be in cluster_names"
    assert clusters.cluster_names[0] == "Tree", "Cluster 0 should be classified as Tree"
    assert "Tree" in clusters.cluster_colors, "Tree should have a color assigned"
    print(f"[PASS] Created Clusters with cluster_names: {clusters.cluster_names}")

    # Test 4: Update existing Clusters
    print("\n[TEST 4] Update existing Clusters")

    # Classify cluster 1 as "Car"
    selected_cluster_ids_2 = {1}
    class_name_2 = "Car"

    updated_clusters = plugin.create_or_update_clusters(
        point_cloud.get_attribute("cluster_labels"),
        selected_cluster_ids_2,
        class_name_2,
        existing_clusters=clusters
    )

    assert 0 in updated_clusters.cluster_names, "Cluster 0 should still be in cluster_names"
    assert 1 in updated_clusters.cluster_names, "Cluster 1 should be in cluster_names"
    assert updated_clusters.cluster_names[0] == "Tree", "Cluster 0 should still be Tree"
    assert updated_clusters.cluster_names[1] == "Car", "Cluster 1 should be Car"
    assert "Tree" in updated_clusters.cluster_colors, "Tree color should persist"
    assert "Car" in updated_clusters.cluster_colors, "Car should have a color"
    print(f"[PASS] Updated cluster_names: {updated_clusters.cluster_names}")

    # Test 5: Reclassify existing cluster
    print("\n[TEST 5] Reclassify existing cluster")

    # Reclassify cluster 0 from "Tree" to "Building"
    selected_cluster_ids_3 = {0}
    class_name_3 = "Building"

    reclassified_clusters = plugin.create_or_update_clusters(
        point_cloud.get_attribute("cluster_labels"),
        selected_cluster_ids_3,
        class_name_3,
        existing_clusters=updated_clusters
    )

    assert reclassified_clusters.cluster_names[0] == "Building", "Cluster 0 should be reclassified to Building"
    assert reclassified_clusters.cluster_names[1] == "Car", "Cluster 1 should still be Car"
    print(f"[PASS] Reclassified cluster_names: {reclassified_clusters.cluster_names}")

    # Test 6: Classify multiple clusters at once
    print("\n[TEST 6] Classify multiple clusters at once")

    # Create fresh point cloud with 5 clusters
    points_multi = np.random.rand(500, 3).astype(np.float32)
    point_cloud_multi = PointCloud(points=points_multi)
    point_cloud_multi.add_attribute("cluster_labels", np.repeat([0, 1, 2, 3, 4], 100))

    # Classify clusters 2, 3, 4 as "Pole"
    selected_cluster_ids_multi = {2, 3, 4}
    class_name_multi = "Pole"

    multi_clusters = plugin.create_or_update_clusters(
        point_cloud_multi.get_attribute("cluster_labels"),
        selected_cluster_ids_multi,
        class_name_multi,
        existing_clusters=None
    )

    assert multi_clusters.cluster_names[2] == "Pole", "Cluster 2 should be Pole"
    assert multi_clusters.cluster_names[3] == "Pole", "Cluster 3 should be Pole"
    assert multi_clusters.cluster_names[4] == "Pole", "Cluster 4 should be Pole"
    assert len(multi_clusters.cluster_names) == 3, "Should have 3 clusters classified"
    print(f"[PASS] Multiple clusters classified: {multi_clusters.cluster_names}")

    # Test 7: Color consistency
    print("\n[TEST 7] Color consistency")

    # Colors should be RGB arrays
    for class_name, color in updated_clusters.cluster_colors.items():
        assert isinstance(color, np.ndarray), f"{class_name} color should be numpy array"
        assert color.shape == (3,), f"{class_name} color should have shape (3,)"
        assert np.all(color >= 0) and np.all(color <= 1), f"{class_name} color should be in [0, 1]"

    print(f"[PASS] All colors valid")
    print(f"      Colors: {dict(updated_clusters.cluster_colors)}")

    # Test 8: Get named colors
    print("\n[TEST 8] Get named colors")

    # Use Clusters with names to get colors
    named_colors = reclassified_clusters.get_named_colors()
    assert named_colors.shape == (1000, 3), "Named colors should match number of points"
    assert named_colors.dtype == np.float32, "Colors should be float32"

    # Check that cluster 0 points have Building color
    cluster_0_mask = point_cloud.get_attribute("cluster_labels") == 0
    building_color = reclassified_clusters.cluster_colors["Building"]
    np.testing.assert_array_almost_equal(
        named_colors[cluster_0_mask][0],
        building_color,
        decimal=5,
        err_msg="Points in cluster 0 should have Building color"
    )

    print("[PASS] Named colors assigned correctly")

    # Test 9: has_names() method
    print("\n[TEST 9] has_names() method")

    # Clusters with names should return True
    assert reclassified_clusters.has_names(), "Clusters with cluster_names should return True"

    # Clusters without names should return False
    plain_clusters = Clusters(labels=np.array([0, 1, 2, 0, 1, 2]))
    assert not plain_clusters.has_names(), "Clusters without cluster_names should return False"

    print("[PASS] has_names() works correctly")

    # Test 10: get_unique_names() method
    print("\n[TEST 10] get_unique_names() method")

    unique_names = reclassified_clusters.get_unique_names()
    assert "Building" in unique_names, "Building should be in unique names"
    assert "Car" in unique_names, "Car should be in unique names"
    assert len(unique_names) == 2, "Should have 2 unique class names"

    print(f"[PASS] Unique names: {unique_names}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_classify_cluster_plugin()
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
