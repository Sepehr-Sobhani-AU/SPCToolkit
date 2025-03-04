
from plugins.plugin_manager import PluginManager


def test_plugin_loading():
    """Test that plugins are correctly discovered and loaded."""
    # Create the plugin manager
    plugin_manager = PluginManager()

    # Print discovered plugins
    print("\nAnalysis Plugins:")
    for name, plugin_class in plugin_manager.get_analysis_plugins().items():
        print(f"  - {name}")

    print("\nMenu Plugins:")
    for plugin in plugin_manager.get_menu_plugins():
        print(f"  - {plugin.__class__.__name__}")
        menu_location = plugin.get_menu_location()
        menu_items = plugin.get_menu_items()
        print(f"    Location: {menu_location}")
        print(f"    Items: {', '.join(item['name'] for item in menu_items)}")

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