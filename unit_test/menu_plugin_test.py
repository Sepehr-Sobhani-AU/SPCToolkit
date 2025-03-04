# test_menu_plugins.py
from plugins.plugin_manager import PluginManager


def test_menu_plugins():
    """Test that menu plugins are correctly discovered and loaded."""
    # Create the plugin manager
    plugin_manager = PluginManager()

    # Print discovered menu plugins
    print("\nMenu Plugins:")
    for plugin in plugin_manager.get_menu_plugins():
        print(f"  - {plugin.__class__.__name__}")
        menu_location = plugin.get_menu_location()
        menu_items = plugin.get_menu_items()
        print(f"    Location: {menu_location}")
        print(f"    Items:")
        for item in menu_items:
            print(f"      - {item['name']} ({item['action']})")
            if 'tooltip' in item:
                print(f"        Tooltip: {item['tooltip']}")


if __name__ == "__main__":
    test_menu_plugins()