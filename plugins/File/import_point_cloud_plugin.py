# plugins/File/import_point_cloud_plugin.py
"""
Plugin for importing point cloud files into the application.

This plugin provides the core file import functionality, allowing users
to load point cloud files in various formats (PLY, PCD, etc.).
"""

from typing import Dict, Any, List, Tuple
from plugins.interfaces import Plugin
from core.data_node import DataNode
from config.config import global_variables


class ImportPointCloudPlugin(Plugin):
    """
    Plugin for importing point cloud files.

    This plugin triggers the file dialog to allow users to select
    and load point cloud files into the application.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "import_point_cloud"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - this plugin directly opens a file dialog.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the import point cloud operation.

        This plugin doesn't follow the typical execute pattern since it needs
        to trigger a file dialog. The actual file loading is handled by FileManager.

        Args:
            data_node: Not used for this plugin
            params: Not used for this plugin

        Returns:
            Tuple of (None, "none", []) - actual loading handled by FileManager
        """
        # Get the main window from global variables
        main_window = global_variables.global_data_manager.tree_widget.window()

        # Trigger the file manager's open dialog
        file_manager = global_variables.global_file_manager
        file_manager.open_point_cloud_file(main_window)

        # Return empty result - file loading is handled by FileManager
        return None, "none", []
