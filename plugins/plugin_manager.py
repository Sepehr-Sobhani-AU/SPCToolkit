
import os
import importlib
import inspect
import sys
from typing import Dict, List, Type, Any, Tuple
from collections import defaultdict

from plugins.interfaces import Plugin, AnalysisPlugin


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

        # Store all discovered plugins: {plugin_name: (plugin_class, menu_path)}
        # menu_path is None for root-level system plugins
        self.plugins: Dict[str, Tuple[Type[Plugin], str]] = {}

        # Menu structure derived from folder hierarchy
        # Structure: {menu_path: [plugin_names]}
        # Example: {"Points": ["Cluster", "Subsample"], "Analysis/Advanced": ["PCA"]}
        self.menu_structure: Dict[str, List[str]] = defaultdict(list)

        # Legacy support
        self.analysis_plugins: Dict[str, Type[Plugin]] = {}  # For backward compatibility
        self.menu_plugins: List = []  # Empty list for backward compatibility

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
        menu_plugins = len([p for p, (_, path) in self.plugins.items() if path is not None])
        system_plugins = total_plugins - menu_plugins

        print(f"Loaded {total_plugins} total plugins:")
        print(f"  - {menu_plugins} menu plugins")
        print(f"  - {system_plugins} system plugins")
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

            # Look for Plugin classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if the class is a Plugin (but not the interface itself)
                if issubclass(obj, Plugin) and obj not in (Plugin, AnalysisPlugin):
                    self._register_plugin(obj, menu_path, is_system_plugin)

        except Exception as e:
            print(f"Error loading plugin module {module_name}: {str(e)}")

    def _register_plugin(self, plugin_class: Type[Plugin], menu_path: str, is_system_plugin: bool):
        """
        Register a plugin class.

        Args:
            plugin_class: The plugin class to register
            menu_path: The menu path for this plugin (None for system plugins)
            is_system_plugin: Whether this is a system plugin
        """
        try:
            # Create an instance to get the name
            plugin_instance = plugin_class()
            plugin_name = plugin_instance.get_name()

            # Check for duplicate plugin names
            if plugin_name in self.plugins:
                print(f"Warning: Plugin '{plugin_name}' is already registered. Overwriting.")

            # Register the plugin with its menu path
            self.plugins[plugin_name] = (plugin_class, menu_path)

            # Add to legacy analysis_plugins dict for backward compatibility
            self.analysis_plugins[plugin_name] = plugin_class

            # Add to menu structure if not a system plugin
            if not is_system_plugin and menu_path is not None:
                self.menu_structure[menu_path].append(plugin_name)
                print(f"Registered plugin: '{plugin_name}' -> Menu: {menu_path}")
            else:
                print(f"Registered system plugin: '{plugin_name}'")

        except Exception as e:
            print(f"Error registering plugin {plugin_class.__name__}: {str(e)}")

    def get_plugin(self, plugin_name: str) -> Type[Plugin]:
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

    def get_menu_plugins(self) -> List:
        """
        Get menu plugins (legacy method - returns empty list in new architecture).

        Returns:
            Empty list (menus are now built from folder structure)
        """
        return self.menu_plugins
