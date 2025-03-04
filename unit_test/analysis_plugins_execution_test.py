# test_analysis_plugins.py
from plugins.plugin_manager import PluginManager
from core.anaysis_manager import AnalysisManager
from core.data_node import DataNode
from core.point_cloud import PointCloud
import numpy as np


def test_analysis_plugins():
    """Test that analysis plugins can be executed correctly."""
    # Create a plugin manager and analysis manager
    plugin_manager = PluginManager()
    analysis_manager = AnalysisManager(plugin_manager)

    # Create a simple test point cloud with random points
    points = np.random.rand(100, 3).astype(np.float32)
    point_cloud = PointCloud(points=points, normals=None, colors=None, intensities=None, labels=None)

    # Create a DataNode with the point cloud
    data_node = DataNode(params="test", data=point_cloud, data_type="point_cloud")

    # Test each available plugin
    for plugin_name in plugin_manager.get_analysis_plugins().keys():
        print(f"\nTesting {plugin_name} plugin:")

        # Get a plugin instance to get its parameters
        plugin_class = plugin_manager.get_analysis_plugins()[plugin_name]
        plugin = plugin_class()
        parameters = plugin.get_parameters()

        # Use default parameter values
        params = {name: param_info["default"] for name, param_info in parameters.items()}
        print(f"Using parameters: {params}")

        try:
            # Execute the plugin using the analysis manager
            plugin.execute(data_node, params)
            print(f"✓ Successfully executed {plugin_name}")
        except Exception as e:
            print(f"✗ Error executing {plugin_name}: {str(e)}")


if __name__ == "__main__":
    test_analysis_plugins()