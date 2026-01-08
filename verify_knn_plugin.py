# verify_knn_plugin.py
"""
Simple verification script to check if the KNN plugin is properly structured.
"""

import sys
import importlib.util

def verify_plugin():
    """Verify the KNN plugin structure without requiring full dependencies."""

    print("=" * 60)
    print("Verifying KNN Analysis Plugin")
    print("=" * 60)

    # Load the plugin module directly
    plugin_path = "plugins/020_Points/030_Analysis/010_knn_analysis_plugin.py"
    spec = importlib.util.spec_from_file_location("knn_plugin", plugin_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
        print("✓ Plugin module loaded successfully")
    except Exception as e:
        print(f"✗ Error loading plugin: {e}")
        return False

    # Check if the plugin class exists
    if not hasattr(module, 'KNNAnalysisPlugin'):
        print("✗ KNNAnalysisPlugin class not found")
        return False
    print("✓ KNNAnalysisPlugin class found")

    # Get the plugin class
    plugin_class = module.KNNAnalysisPlugin

    # Check required methods
    required_methods = ['get_name', 'get_parameters', 'execute']
    for method in required_methods:
        if not hasattr(plugin_class, method):
            print(f"✗ Missing required method: {method}")
            return False
        print(f"✓ Method '{method}' found")

    # Try to instantiate the plugin
    try:
        plugin = plugin_class()
        print("✓ Plugin instantiated successfully")
    except Exception as e:
        print(f"✗ Error instantiating plugin: {e}")
        return False

    # Get plugin name
    try:
        name = plugin.get_name()
        print(f"✓ Plugin name: '{name}'")
        if name != "knn_analysis":
            print(f"  ⚠ Warning: Expected 'knn_analysis', got '{name}'")
    except Exception as e:
        print(f"✗ Error getting plugin name: {e}")
        return False

    # Get parameters
    try:
        params = plugin.get_parameters()
        print(f"✓ Plugin parameters retrieved:")
        for param_name, param_info in params.items():
            print(f"  - {param_name}:")
            print(f"      type: {param_info.get('type')}")
            print(f"      default: {param_info.get('default')}")
            if param_info.get('type') == 'choice':
                print(f"      options: {param_info.get('options')}")
    except Exception as e:
        print(f"✗ Error getting parameters: {e}")
        return False

    # Verify parameter structure
    if 'k_neighbors' not in params:
        print("✗ Missing required parameter: k_neighbors")
        return False
    print("✓ Required parameter 'k_neighbors' found")

    if 'statistic' not in params:
        print("✗ Missing required parameter: statistic")
        return False
    print("✓ Required parameter 'statistic' found")

    # Check statistic options
    expected_stats = [
        "Average Distance",
        "Maximum Distance",
        "Minimum Distance",
        "Std Deviation",
        "Distance to Kth Neighbor",
        "Sum of Distances"
    ]
    actual_stats = params['statistic'].get('options', [])
    for stat in expected_stats:
        if stat in actual_stats:
            print(f"✓ Statistic option '{stat}' available")
        else:
            print(f"✗ Missing statistic option: {stat}")
            return False

    print("\n" + "=" * 60)
    print("Plugin verification completed successfully!")
    print("=" * 60)
    print("\nThe plugin is properly structured and ready to use.")
    print("It will appear in the menu: Points > Analysis > KNN Analysis")
    print("\nAvailable statistics:")
    for stat in actual_stats:
        print(f"  - {stat}")

    return True


if __name__ == "__main__":
    success = verify_plugin()
    sys.exit(0 if success else 1)