import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.plugin_manager import PluginManager


def test_plugin_loading():
    """Test that plugins are correctly discovered and loaded."""
    # Create the plugin manager
    plugin_manager = PluginManager()

    # Print discovered data processing plugins
    print("\nData Processing Plugins:")
    for name, plugin_class in plugin_manager.get_analysis_plugins().items():
        print(f"  - {name}")

    # Print discovered action plugins
    print("\nAction Plugins:")
    for name, plugin_class in plugin_manager.action_plugins.items():
        print(f"  - {name}")

    # Print menu structure (derived from folder hierarchy)
    print("\nMenu Structure:")
    menu_structure = plugin_manager.get_menu_structure()
    for menu_path, plugin_names in sorted(menu_structure.items()):
        print(f"  {menu_path}:")
        for plugin_name in plugin_names:
            print(f"    - {plugin_name}")

    # Test creating an instance of an analysis plugin
    analysis_plugins = plugin_manager.get_analysis_plugins()
    if "dbscan" in analysis_plugins:
        dbscan_class = analysis_plugins["dbscan"]
        dbscan_plugin = dbscan_class()

        print("\nDBSCAN Plugin Parameters:")
        params = dbscan_plugin.get_parameters()
        for param_name, param_info in params.items():
            print(f"  - {param_name}: {param_info}")
    else:
        print("\nDBSCAN plugin not found")


if __name__ == "__main__":
    test_plugin_loading()