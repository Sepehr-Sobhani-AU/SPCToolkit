# unit_test/knn_analysis_plugin_test.py
"""
Unit test for the KNN Analysis Plugin.

This test verifies that the KNN analysis plugin correctly computes
various distance statistics from k-nearest neighbors.
"""

import numpy as np
from core.data_node import DataNode
from core.point_cloud import PointCloud
from plugins.plugin_manager import PluginManager


def test_knn_analysis_plugin():
    """Test the KNN analysis plugin with all available statistics."""

    print("=" * 60)
    print("Testing KNN Analysis Plugin")
    print("=" * 60)

    # Create a plugin manager
    plugin_manager = PluginManager()

    # Get the KNN analysis plugin
    plugins = plugin_manager.get_analysis_plugins()

    if "knn_analysis" not in plugins:
        print("✗ KNN Analysis plugin not found!")
        print("Available plugins:", list(plugins.keys()))
        return False

    print("✓ KNN Analysis plugin found")

    # Create a simple test point cloud
    # Create a grid of points for predictable KNN results
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    z = np.zeros(20)
    points = np.column_stack([x, y, z]).astype(np.float32)

    print(f"\nCreated test point cloud with {len(points)} points")

    # Create PointCloud and DataNode
    point_cloud = PointCloud(
        points=points,
        normals=None,
        colors=None,
        intensities=None,
        labels=None
    )

    data_node = DataNode(
        params="test",
        data=point_cloud,
        data_type="point_cloud"
    )

    # Get plugin instance
    plugin_class = plugins["knn_analysis"]
    plugin = plugin_class()

    # Get available parameters
    param_schema = plugin.get_parameters()
    print(f"\nAvailable parameters:")
    for param_name, param_info in param_schema.items():
        print(f"  - {param_name}: {param_info}")

    # Test all available statistics
    statistics = param_schema["statistic"]["options"]
    k_neighbors = 5

    print(f"\nTesting with k={k_neighbors} neighbors:")
    print("-" * 60)

    for statistic in statistics:
        try:
            # Create parameters
            params = {
                "k_neighbors": k_neighbors,
                "statistic": statistic
            }

            # Execute the plugin
            result, result_type, dependencies = plugin.execute(data_node, params)

            # Verify results
            assert result_type == "values", f"Expected result type 'values', got '{result_type}'"
            assert len(result.values) == len(points), \
                f"Result length {len(result.values)} doesn't match points {len(points)}"
            assert dependencies == [data_node.uid], "Dependencies should contain the input data node"

            # Print statistics about the results
            values = result.values
            print(f"\n{statistic}:")
            print(f"  Min value:  {np.min(values):.4f}")
            print(f"  Max value:  {np.max(values):.4f}")
            print(f"  Mean value: {np.mean(values):.4f}")
            print(f"  Std value:  {np.std(values):.4f}")
            print(f"  ✓ Success")

        except Exception as e:
            print(f"\n{statistic}:")
            print(f"  ✗ Error: {str(e)}")
            return False

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_knn_analysis_plugin()
    exit(0 if success else 1)