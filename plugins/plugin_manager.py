
import os
import importlib
import inspect
import sys
from typing import Dict, List, Type, Any

from plugins.interfaces import AnalysisPlugin, MenuPlugin


class PluginManager:
    """
    Manages the discovery, loading, and access to plugins.

    The PluginManager scans specified directories for Python modules,
    loads those modules, and identifies classes that implement plugin
    interfaces. These plugins are then registered and made available
    to the rest of the application.
    """

    def __init__(self, plugin_dirs=None):
        """
        Initialize the PluginManager with directories to scan for plugins.

        Args:
            plugin_dirs (List[str], optional): List of directory paths to scan for plugins.
                If None, defaults to ['plugins/analysis', 'plugins/menus'].
        """
        if plugin_dirs is None:
            # Default plugin directories relative to the project root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.plugin_dirs = [
                os.path.join(base_dir, 'plugins', 'analysis'),
                os.path.join(base_dir, 'plugins', 'menus')
            ]
        else:
            self.plugin_dirs = plugin_dirs

        # Dictionaries to store discovered plugins
        self.analysis_plugins: Dict[str, Type[AnalysisPlugin]] = {}
        self.menu_plugins: List[MenuPlugin] = []

        # Load plugins on initialization
        self.load_plugins()

    def load_plugins(self):
        """
        Discover and load all plugins from the specified directories.

        This method scans the plugin directories for Python files, imports them,
        and identifies classes that implement plugin interfaces. Those classes
        are then registered as available plugins.
        """
        print("Loading plugins...")

        for plugin_dir in self.plugin_dirs:
            # Create the directory if it doesn't exist
            if not os.path.exists(plugin_dir):
                os.makedirs(plugin_dir, exist_ok=True)
                continue

            # Get Python files in the directory
            for file in os.listdir(plugin_dir):
                if file.endswith('.py') and not file.startswith('__'):
                    # Construct the module import path
                    # Convert directory path to package path (e.g., 'plugins/analysis' -> 'plugins.analysis')
                    package_path = os.path.relpath(plugin_dir, os.path.dirname(os.path.dirname(plugin_dir)))
                    package_path = package_path.replace(os.sep, '.')
                    module_name = f"{package_path}.{file[:-3]}"

                    try:
                        # Import the module
                        module = importlib.import_module(module_name)

                        # Look for plugin classes in the module
                        for _, obj in inspect.getmembers(module, inspect.isclass):
                            # Check if the class is a plugin (but not the interface itself)
                            if issubclass(obj, AnalysisPlugin) and obj != AnalysisPlugin:
                                self._register_analysis_plugin(obj)
                            elif issubclass(obj, MenuPlugin) and obj != MenuPlugin:
                                self._register_menu_plugin(obj)
                    except Exception as e:
                        print(f"Error loading plugin module {module_name}: {str(e)}")

        # Print summary of loaded plugins
        print(f"Loaded {len(self.analysis_plugins)} analysis plugins and {len(self.menu_plugins)} menu plugins.")

    def _register_analysis_plugin(self, plugin_class):
        """
        Register an analysis plugin class.

        Args:
            plugin_class (Type[AnalysisPlugin]): The plugin class to register
        """
        try:
            # Create an instance to get the name
            plugin_instance = plugin_class()
            plugin_name = plugin_instance.get_name()

            # Check for duplicate plugin names
            if plugin_name in self.analysis_plugins:
                print(f"Warning: Analysis plugin '{plugin_name}' is already registered. Overwriting.")

            # Register the plugin class (not the instance)
            self.analysis_plugins[plugin_name] = plugin_class
            print(f"Registered analysis plugin: {plugin_name}")
        except Exception as e:
            print(f"Error registering analysis plugin {plugin_class.__name__}: {str(e)}")

    def _register_menu_plugin(self, plugin_class):
        """
        Register a menu plugin class.

        Args:
            plugin_class (Type[MenuPlugin]): The plugin class to register
        """
        try:
            # Create an instance of the plugin directly
            plugin_instance = plugin_class()
            self.menu_plugins.append(plugin_instance)
            print(f"Registered menu plugin: {plugin_class.__name__}")
        except Exception as e:
            print(f"Error registering menu plugin {plugin_class.__name__}: {str(e)}")

    def get_analysis_plugins(self) -> Dict[str, Type[AnalysisPlugin]]:
        """
        Get all registered analysis plugins.

        Returns:
            Dict[str, Type[AnalysisPlugin]]: Dictionary mapping plugin names to plugin classes
        """
        return self.analysis_plugins

    def get_menu_plugins(self) -> List[MenuPlugin]:
        """
        Get all registered menu plugins.

        Returns:
            List[MenuPlugin]: List of menu plugin instances
        """
        return self.menu_plugins