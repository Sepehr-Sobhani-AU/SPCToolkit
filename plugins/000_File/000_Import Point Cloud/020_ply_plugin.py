# plugins/File/import_point_cloud_plugin.py
"""
Plugin for importing point cloud files into the application.

This plugin provides the core file import functionality, allowing users
to load point cloud files in various formats (PLY, PCD, etc.).
"""

from typing import Dict, Any
from plugins.interfaces import ActionPlugin
from config.config import global_variables


class ImportPointCloudPlugin(ActionPlugin):
    """
    Action plugin for importing point cloud files.

    This plugin triggers the file dialog to allow users to select
    and load point cloud files into the application.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "import_ply"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - this plugin directly opens a file dialog.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the import point cloud action.

        Opens a file dialog to allow the user to select and load point cloud files.

        Args:
            main_window: The main application window
            params: Not used for this plugin (empty dict)
        """
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.show_progress("Importing PLY file...")
        try:
            file_manager = global_variables.global_file_manager
            file_manager.open_point_cloud_file(main_window)
        finally:
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()
