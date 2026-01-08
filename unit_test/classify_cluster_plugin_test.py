"""
Unit test for classify_cluster_plugin.

Tests that:
1. Plugin is correctly recognized as ActionPlugin
2. Plugin can classify selected clusters
3. Creates FeatureClasses object with correct class_mapping
4. Updates existing FeatureClasses when classifying more clusters
5. Assigns colors to classes automatically
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plugins.Clusters.classify_cluster_plugin import ClassifyClusterPlugin
from plugins.interfaces import ActionPlugin
from core.point_cloud import PointCloud
from core.feature_classes import FeatureClasses


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

    # Test 3: Create FeatureClasses from cluster classification
    print("\n[TEST 3] Create FeatureClasses object")

    # Create a point cloud with cluster labels
    points = np.random.rand(1000, 3).astype(np.float32)
    point_cloud = PointCloud(points=points)
    point_cloud.cluster_labels = np.zeros(1000, dtype=int)
    point_cloud.cluster_labels[:300] = 0    # Cluster 0
    point_cloud.cluster_labels[300:600] = 1  # Cluster 1
    point_cloud.cluster_labels[600:] = 2     # Cluster 2

    # Simulate classifying cluster 0 as "Tree"
    selected_cluster_ids = {0}
    class_name = "Tree"

    # Create initial FeatureClasses
    feature_classes = plugin.create_or_update_feature_classes(
        point_cloud.cluster_labels,
        selected_cluster_ids,
        class_name,
        existing_feature_classes=None
    )

    assert isinstance(feature_classes, FeatureClasses), "Should return FeatureClasses object"
    assert 0 in feature_classes.class_mapping, "Cluster 0 should be in mapping"
    assert feature_classes.class_mapping[0] == "Tree", "Cluster 0 should be classified as Tree"
    assert "Tree" in feature_classes.class_colors, "Tree should have a color assigned"
    print(f"[PASS] Created FeatureClasses with mapping: {feature_classes.class_mapping}")

    # Test 4: Update existing FeatureClasses
    print("\n[TEST 4] Update existing FeatureClasses")

    # Classify cluster 1 as "Car"
    selected_cluster_ids_2 = {1}
    class_name_2 = "Car"

    updated_feature_classes = plugin.create_or_update_feature_classes(
        point_cloud.cluster_labels,
        selected_cluster_ids_2,
        class_name_2,
        existing_feature_classes=feature_classes
    )

    assert 0 in updated_feature_classes.class_mapping, "Cluster 0 should still be in mapping"
    assert 1 in updated_feature_classes.class_mapping, "Cluster 1 should be in mapping"
    assert updated_feature_classes.class_mapping[0] == "Tree", "Cluster 0 should still be Tree"
    assert updated_feature_classes.class_mapping[1] == "Car", "Cluster 1 should be Car"
    assert "Tree" in updated_feature_classes.class_colors, "Tree color should persist"
    assert "Car" in updated_feature_classes.class_colors, "Car should have a color"
    print(f"[PASS] Updated mapping: {updated_feature_classes.class_mapping}")

    # Test 5: Reclassify existing cluster
    print("\n[TEST 5] Reclassify existing cluster")

    # Reclassify cluster 0 from "Tree" to "Building"
    selected_cluster_ids_3 = {0}
    class_name_3 = "Building"

    reclassified_feature_classes = plugin.create_or_update_feature_classes(
        point_cloud.cluster_labels,
        selected_cluster_ids_3,
        class_name_3,
        existing_feature_classes=updated_feature_classes
    )

    assert reclassified_feature_classes.class_mapping[0] == "Building", "Cluster 0 should be reclassified to Building"
    assert reclassified_feature_classes.class_mapping[1] == "Car", "Cluster 1 should still be Car"
    print(f"[PASS] Reclassified mapping: {reclassified_feature_classes.class_mapping}")

    # Test 6: Classify multiple clusters at once
    print("\n[TEST 6] Classify multiple clusters at once")

    # Create fresh point cloud with 5 clusters
    points_multi = np.random.rand(500, 3).astype(np.float32)
    point_cloud_multi = PointCloud(points=points_multi)
    point_cloud_multi.cluster_labels = np.repeat([0, 1, 2, 3, 4], 100)

    # Classify clusters 2, 3, 4 as "Pole"
    selected_cluster_ids_multi = {2, 3, 4}
    class_name_multi = "Pole"

    multi_feature_classes = plugin.create_or_update_feature_classes(
        point_cloud_multi.cluster_labels,
        selected_cluster_ids_multi,
        class_name_multi,
        existing_feature_classes=None
    )

    assert multi_feature_classes.class_mapping[2] == "Pole", "Cluster 2 should be Pole"
    assert multi_feature_classes.class_mapping[3] == "Pole", "Cluster 3 should be Pole"
    assert multi_feature_classes.class_mapping[4] == "Pole", "Cluster 4 should be Pole"
    assert len(multi_feature_classes.class_mapping) == 3, "Should have 3 clusters classified"
    print(f"[PASS] Multiple clusters classified: {multi_feature_classes.class_mapping}")

    # Test 7: Color consistency
    print("\n[TEST 7] Color consistency")

    # Colors should be RGB arrays
    for class_name, color in updated_feature_classes.class_colors.items():
        assert isinstance(color, np.ndarray), f"{class_name} color should be numpy array"
        assert color.shape == (3,), f"{class_name} color should have shape (3,)"
        assert np.all(color >= 0) and np.all(color <= 1), f"{class_name} color should be in [0, 1]"

    print(f"[PASS] All colors valid")
    print(f"      Colors: {dict(updated_feature_classes.class_colors)}")

    # Test 8: Get point colors
    print("\n[TEST 8] Get point colors")

    point_colors = updated_feature_classes.get_point_colors()
    assert point_colors.shape == (1000, 3), "Point colors should match number of points"
    assert point_colors.dtype == np.float32, "Colors should be float32"

    # Check that Tree points have Tree color
    tree_mask = point_cloud.cluster_labels == 0
    tree_color = updated_feature_classes.class_colors["Building"]  # Cluster 0 was reclassified to Building
    np.testing.assert_array_almost_equal(
        point_colors[tree_mask][0],
        tree_color,
        decimal=5,
        err_msg="Points in cluster 0 should have Building color"
    )

    print("[PASS] Point colors assigned correctly")

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
