"""
Test script for the new folder-based plugin system.
Tests plugin loading and menu structure generation.
"""

from plugins.plugin_manager import PluginManager

def test_plugin_system():
    print("=" * 70)
    print("TESTING FOLDER-BASED PLUGIN SYSTEM")
    print("=" * 70)
    print()

    # Initialize plugin manager
    plugin_manager = PluginManager()
    print()

    # Get menu structure
    menu_structure = plugin_manager.get_menu_structure()

    print("=" * 70)
    print("MENU STRUCTURE")
    print("=" * 70)
    print()

    if not menu_structure:
        print("[ERROR] No plugins found in folder structure!")
        return False

    # Display menu hierarchy
    for menu_path in sorted(menu_structure.keys()):
        plugins = menu_structure[menu_path]
        print(f"[MENU] {menu_path}/")
        for plugin_name in plugins:
            print(f"   + {plugin_name}")
        print()

    # Test statistics
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    total_plugins = sum(len(plugins) for plugins in menu_structure.values())
    print(f"Total menu categories: {len(menu_structure)}")
    print(f"Total plugins: {total_plugins}")
    print()

    # Verify plugin classes can be retrieved
    print("=" * 70)
    print("PLUGIN CLASS VERIFICATION")
    print("=" * 70)
    all_plugins = plugin_manager.get_analysis_plugins()

    success_count = 0
    for plugin_name, plugin_class in all_plugins.items():
        try:
            # Try to instantiate
            instance = plugin_class()
            name = instance.get_name()
            params = instance.get_parameters()
            print(f"[OK] {plugin_name}: {plugin_class.__name__}")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {plugin_name}: ERROR - {e}")

    print()
    print(f"Successfully verified {success_count}/{len(all_plugins)} plugins")
    print()

    # Check for system plugins
    print("=" * 70)
    print("SYSTEM PLUGINS (root level - not in menus)")
    print("=" * 70)
    system_plugins = [name for name, (cls, path) in plugin_manager.plugins.items() if path is None]
    if system_plugins:
        for sp in system_plugins:
            print(f"   [SYSTEM] {sp}")
    else:
        print("   (none)")
    print()

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return success_count == len(all_plugins)


if __name__ == "__main__":
    success = test_plugin_system()
    exit(0 if success else 1)
