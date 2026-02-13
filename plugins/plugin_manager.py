
import os
import importlib
import inspect
import sys
import re
from typing import Dict, List, Type, Any, Tuple
from collections import defaultdict

from plugins.interfaces import Plugin, AnalysisPlugin, ActionPlugin


class PluginManager:
    """
    Manages the discovery, loading, and access to plugins using folder-based menu structure.

    NEW ARCHITECTURE:
    - Folder structure defines menu hierarchy automatically
    - Plugins in subdirectories appear in menus based on their location
    - Plugins in root directory are system plugins (not in menus)

    Examples:
        plugins/Points/cluster.py           -> Menu: Points > Cluster
        plugins/Analysis/Advanced/pca.py    -> Menu: Analysis > Advanced > PCA
        plugins/system_plugin.py            -> Not in menu (system plugin)
    """

    def __init__(self, plugin_root=None):
        """
        Initialize the PluginManager with root directory to scan for plugins.

        Args:
            plugin_root (str, optional): Root directory path to scan for plugins.
                If None, defaults to 'plugins/' directory.
        """
        if plugin_root is None:
            # Default plugin root directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.plugin_root = os.path.join(base_dir, 'plugins')
        else:
            self.plugin_root = plugin_root

        # Store all discovered plugins: {plugin_name: (plugin_class, menu_path, plugin_type)}
        # menu_path is None for root-level system plugins
        # plugin_type is either "data" or "action"
        self.plugins: Dict[str, Tuple[Type[Plugin], str, str]] = {}

        # Menu structure derived from folder hierarchy
        # Structure: {menu_path: [plugin_names]}
        # Example: {"Points": ["Cluster", "Subsample"], "Analysis/Advanced": ["PCA"]}
        self.menu_structure: Dict[str, List[str]] = defaultdict(list)

        # Legacy support
        self.analysis_plugins: Dict[str, Type[Plugin]] = {}  # For backward compatibility

        # Store action plugins separately for easy access
        self.action_plugins: Dict[str, Type[ActionPlugin]] = {}

        # Load plugins on initialization
        self.load_plugins()

    def load_plugins(self):
        """
        Discover and load all plugins from the plugin root directory recursively.

        This method scans the plugin root folder and all subdirectories:
        - Plugins in subdirectories are added to menus based on folder structure
        - Plugins in root directory are system plugins (not in menus)
        """
        print(f"Loading plugins from: {self.plugin_root}")

        if not os.path.exists(self.plugin_root):
            os.makedirs(self.plugin_root, exist_ok=True)
            print("Plugin directory created.")
            return

        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(self.plugin_root):
            # Skip __pycache__ and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('__') and not d.startswith('.')]

            # Calculate menu path from folder structure
            rel_path = os.path.relpath(root, self.plugin_root)

            # Determine if this is root directory or subdirectory
            if rel_path == '.':
                menu_path = None  # Root level - system plugins
                is_system_plugin = True
            else:
                # Convert path to menu structure: "Points/Advanced" -> "Points/Advanced"
                menu_path = rel_path.replace(os.sep, '/')
                is_system_plugin = False

            # Process Python files in this directory
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    self._load_plugin_file(root, file, menu_path, is_system_plugin)

        # Print summary
        total_plugins = len(self.plugins)
        menu_plugins = len([p for p, (_, path, _) in self.plugins.items() if path is not None])
        system_plugins = total_plugins - menu_plugins
        action_plugin_count = len(self.action_plugins)
        data_plugin_count = len(self.analysis_plugins)

        print(f"Loaded {total_plugins} total plugins:")
        print(f"  - {menu_plugins} menu plugins")
        print(f"  - {system_plugins} system plugins")
        print(f"  - {action_plugin_count} action plugins")
        print(f"  - {data_plugin_count} data processing plugins")
        print(f"  - {len(self.menu_structure)} menu categories")

    def _load_plugin_file(self, directory: str, filename: str, menu_path: str, is_system_plugin: bool):
        """
        Load a single plugin file and register any Plugin classes found.

        Args:
            directory: The directory containing the plugin file
            filename: The Python file name
            menu_path: The menu path for this plugin (None for system plugins)
            is_system_plugin: Whether this is a system plugin (in root directory)
        """
        # Construct module import path
        rel_dir = os.path.relpath(directory, os.path.dirname(self.plugin_root))
        package_path = rel_dir.replace(os.sep, '.')
        module_name = f"{package_path}.{filename[:-3]}"

        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Look for Plugin or ActionPlugin classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if the class is an ActionPlugin first (more specific)
                if issubclass(obj, ActionPlugin) and obj != ActionPlugin:
                    self._register_plugin(obj, menu_path, is_system_plugin, plugin_type="action")
                # Then check if the class is a Plugin (but not the interface itself)
                elif issubclass(obj, Plugin) and obj not in (Plugin, AnalysisPlugin):
                    self._register_plugin(obj, menu_path, is_system_plugin, plugin_type="data")

        except Exception as e:
            print(f"Error loading plugin module {module_name}: {str(e)}")

    def _register_plugin(self, plugin_class, menu_path: str, is_system_plugin: bool, plugin_type: str):
        """
        Register a plugin class.

        Args:
            plugin_class: The plugin class to register (Plugin or ActionPlugin)
            menu_path: The menu path for this plugin (None for system plugins)
            is_system_plugin: Whether this is a system plugin
            plugin_type: Either "data" or "action"
        """
        try:
            # Create an instance to get the name
            plugin_instance = plugin_class()
            plugin_name = plugin_instance.get_name()

            # Check for duplicate plugin names
            if plugin_name in self.plugins:
                print(f"Warning: Plugin '{plugin_name}' is already registered. Overwriting.")

            # Register the plugin with its menu path and type
            self.plugins[plugin_name] = (plugin_class, menu_path, plugin_type)

            # Add to type-specific registries
            if plugin_type == "action":
                self.action_plugins[plugin_name] = plugin_class
            else:
                # Add to legacy analysis_plugins dict for backward compatibility
                self.analysis_plugins[plugin_name] = plugin_class

            # Add to menu structure if not a system plugin
            if not is_system_plugin and menu_path is not None:
                self.menu_structure[menu_path].append(plugin_name)
                print(f"Registered {plugin_type} plugin: '{plugin_name}' -> Menu: {menu_path}")
            else:
                print(f"Registered system {plugin_type} plugin: '{plugin_name}'")

        except Exception as e:
            print(f"Error registering plugin {plugin_class.__name__}: {str(e)}")

    def get_plugin(self, plugin_name: str):
        """
        Get a plugin class by name.

        Args:
            plugin_name: The name of the plugin

        Returns:
            The plugin class, or None if not found
        """
        if plugin_name in self.plugins:
            return self.plugins[plugin_name][0]
        return None

    def get_plugin_type(self, plugin_name: str) -> str:
        """
        Get the type of a plugin ("data" or "action").

        Args:
            plugin_name: The name of the plugin

        Returns:
            "data", "action", or None if plugin not found
        """
        if plugin_name in self.plugins:
            return self.plugins[plugin_name][2]
        return None

    def is_action_plugin(self, plugin_name: str) -> bool:
        """
        Check if a plugin is an action plugin.

        Args:
            plugin_name: The name of the plugin

        Returns:
            True if the plugin is an action plugin, False otherwise
        """
        return self.get_plugin_type(plugin_name) == "action"

    def get_menu_structure(self) -> Dict[str, List[str]]:
        """
        Get the menu structure derived from folder hierarchy.

        Returns:
            Dictionary mapping menu paths to lists of plugin names
            Example: {"Points": ["Cluster", "Subsample"], "Analysis/Advanced": ["PCA"]}
        """
        return dict(self.menu_structure)

    def get_analysis_plugins(self) -> Dict[str, Type[Plugin]]:
        """
        Get all registered plugins (legacy method for backward compatibility).

        Returns:
            Dict[str, Type[Plugin]]: Dictionary mapping plugin names to plugin classes
        """
        return self.analysis_plugins

    def get_all_plugin_info(self):
        """
        Return info for every loaded plugin.

        Returns:
            List of dicts with keys: name, module, menu_path, type
        """
        info_list = []
        for plugin_name, (plugin_class, menu_path, plugin_type) in self.plugins.items():
            info_list.append({
                'name': plugin_name,
                'module': plugin_class.__module__,
                'menu_path': menu_path,
                'type': plugin_type,
            })
        return info_list

    def reload_plugin(self, plugin_name):
        """
        Reload a plugin from disk using importlib.reload().

        Args:
            plugin_name: Name of the plugin to reload.

        Returns:
            (success: bool, message: str)
        """
        if plugin_name not in self.plugins:
            return False, f"Plugin '{plugin_name}' not found."

        old_class, menu_path, plugin_type = self.plugins[plugin_name]
        module_name = old_class.__module__

        try:
            module = sys.modules.get(module_name)
            if module is None:
                return False, f"Module '{module_name}' not found in sys.modules."

            # Reload the module from disk
            module = importlib.reload(module)

            # Find the new plugin class in the reloaded module
            new_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, ActionPlugin) and obj != ActionPlugin:
                    new_class = obj
                    break
                elif issubclass(obj, Plugin) and obj not in (Plugin, AnalysisPlugin):
                    new_class = obj
                    break

            if new_class is None:
                return False, f"No plugin class found in reloaded module '{module_name}'."

            # Re-register with existing slot
            self.plugins[plugin_name] = (new_class, menu_path, plugin_type)
            if plugin_type == "action":
                self.action_plugins[plugin_name] = new_class
            else:
                self.analysis_plugins[plugin_name] = new_class

            return True, f"Plugin '{plugin_name}' reloaded successfully."

        except Exception as e:
            return False, f"Error reloading '{plugin_name}': {e}"

    def unload_plugin(self, plugin_name):
        """
        Unload a plugin, removing it from all registries and menu structure.

        Args:
            plugin_name: Name of the plugin to unload.

        Returns:
            (success: bool, message: str)
        """
        if plugin_name not in self.plugins:
            return False, f"Plugin '{plugin_name}' not found."

        _, menu_path, plugin_type = self.plugins.pop(plugin_name)

        # Remove from type-specific registries
        self.action_plugins.pop(plugin_name, None)
        self.analysis_plugins.pop(plugin_name, None)

        # Remove from menu structure
        if menu_path and menu_path in self.menu_structure:
            if plugin_name in self.menu_structure[menu_path]:
                self.menu_structure[menu_path].remove(plugin_name)
            # Clean up empty menu paths
            if not self.menu_structure[menu_path]:
                del self.menu_structure[menu_path]

        return True, f"Plugin '{plugin_name}' unloaded."

    def scan_and_load_new_plugins(self):
        """
        Walk the plugin filesystem and load any new plugin files not already loaded.

        Returns:
            List of newly loaded plugin names.
        """
        already_loaded_modules = set()
        for plugin_name, (plugin_class, _, _) in self.plugins.items():
            already_loaded_modules.add(plugin_class.__module__)

        newly_loaded = []
        before_names = set(self.plugins.keys())

        if not os.path.exists(self.plugin_root):
            return newly_loaded

        for root, dirs, files in os.walk(self.plugin_root):
            dirs[:] = [d for d in dirs if not d.startswith('__') and not d.startswith('.')]

            rel_path = os.path.relpath(root, self.plugin_root)

            if rel_path == '.':
                menu_path = None
                is_system_plugin = True
            else:
                menu_path = rel_path.replace(os.sep, '/')
                is_system_plugin = False

            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    # Build module name to check if already loaded
                    rel_dir = os.path.relpath(root, os.path.dirname(self.plugin_root))
                    package_path = rel_dir.replace(os.sep, '.')
                    module_name = f"{package_path}.{file[:-3]}"

                    if module_name not in already_loaded_modules:
                        self._load_plugin_file(root, file, menu_path, is_system_plugin)

        newly_loaded = [name for name in self.plugins.keys() if name not in before_names]
        return newly_loaded

    @staticmethod
    def _strip_prefix(name: str) -> str:
        """
        Strip numeric prefix from folder or file names.

        Removes patterns like "000_", "010_", etc. from the beginning of names.
        This allows folders/files to be ordered using prefixes while displaying clean names.

        Examples:
            "000_File" -> "File"
            "010_Points" -> "Points"
            "NoPrefix" -> "NoPrefix"

        Args:
            name: The folder or file name (possibly with numeric prefix)

        Returns:
            Name with prefix removed, or original name if no prefix found
        """
        return re.sub(r'^\d{3}_', '', name)
