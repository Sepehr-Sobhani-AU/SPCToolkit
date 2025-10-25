"""
Unit test for ActionPlugin architecture.

Tests that:
1. ActionPlugin interface is properly defined
2. Plugins are registered with correct type (action vs data)
3. Plugin routing works correctly
4. Import point cloud plugin is recognized as an action plugin
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.plugin_manager import PluginManager
from plugins.interfaces import ActionPlugin, Plugin


def test_plugin_manager():
    """Test that PluginManager correctly loads and categorizes plugins."""
    print("=" * 60)
    print("Testing Plugin Manager with ActionPlugin architecture")
    print("=" * 60)

    # Create plugin manager
    pm = PluginManager()

    # Test 1: Verify plugins are loaded
    print("\n[TEST 1] Plugin Loading")
    assert len(pm.plugins) > 0, "No plugins loaded"
    print(f"[PASS] Loaded {len(pm.plugins)} plugins")

    # Test 2: Verify import_point_cloud is registered as action plugin
    print("\n[TEST 2] Import Point Cloud Registration")
    assert 'import_point_cloud' in pm.plugins, "import_point_cloud not found"
    assert pm.is_action_plugin('import_point_cloud'), "import_point_cloud not recognized as action plugin"
    assert pm.get_plugin_type('import_point_cloud') == 'action', "import_point_cloud has wrong type"
    print("[PASS] import_point_cloud correctly registered as action plugin")

    # Test 3: Verify data processing plugins are registered correctly
    print("\n[TEST 3] Data Processing Plugin Registration")
    if 'subtract' in pm.plugins:
        assert not pm.is_action_plugin('subtract'), "subtract incorrectly identified as action plugin"
        assert pm.get_plugin_type('subtract') == 'data', "subtract has wrong type"
        print("[PASS] subtract correctly registered as data processing plugin")
    else:
        print("[WARN] subtract plugin not loaded (likely missing dependencies)")

    # Test 4: Verify plugin retrieval
    print("\n[TEST 4] Plugin Retrieval")
    ipc_class = pm.get_plugin('import_point_cloud')
    assert ipc_class is not None, "Could not retrieve import_point_cloud plugin class"
    assert issubclass(ipc_class, ActionPlugin), "import_point_cloud is not an ActionPlugin subclass"
    print("[PASS] Plugin retrieval works correctly")

    # Test 5: Verify plugin instance creation
    print("\n[TEST 5] Plugin Instance Creation")
    ipc_instance = ipc_class()
    assert ipc_instance.get_name() == 'import_point_cloud', "Plugin name mismatch"
    assert ipc_instance.get_parameters() == {}, "Action plugin should have no parameters"
    print("[PASS] Plugin instance created successfully")

    # Test 6: Verify menu structure
    print("\n[TEST 6] Menu Structure")
    menu_structure = pm.get_menu_structure()
    assert 'File' in menu_structure, "File menu not in structure"
    assert 'import_point_cloud' in menu_structure['File'], "import_point_cloud not in File menu"
    print(f"[PASS] Menu structure correct: {dict(menu_structure)}")

    # Test 7: Verify action and data plugin counts
    print("\n[TEST 7] Plugin Categorization")
    print(f"  Total plugins: {len(pm.plugins)}")
    print(f"  Action plugins: {len(pm.action_plugins)}")
    print(f"  Data plugins: {len(pm.analysis_plugins)}")
    assert len(pm.action_plugins) > 0, "No action plugins found"
    print("[PASS] Plugins correctly categorized")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_plugin_manager()
        sys.exit(0 if success else 1)
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
