# plugins/File/save_project_as_plugin.py
"""
Plugin for saving the current project to a new file location.

Always prompts for a new filename, allowing users to save a copy
of the current project with a different name or location.

Saved as .pcdtk file using pickle serialization.
"""

from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class SaveProjectAsPlugin(ActionPlugin):
    """
    Action plugin for saving the current project to a new location.

    Workflow:
    1. User clicks File → Save Project As
    2. Plugin gets current data_nodes from global variables
    3. Calls file_manager.save_project() with new_file=True
    4. Shows error message if save fails (no success message shown)
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "save_project_as"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - directly opens save dialog.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the save project as action.

        Always prompts for a new filename, regardless of whether
        the project has been saved before. This is the standard
        "Save As" behavior (Ctrl+Shift+S).

        Args:
            main_window: The main application window
            params: Not used for this plugin (empty dict)
        """
        # Get global instances
        file_manager = global_variables.global_file_manager
        data_nodes = global_variables.global_data_nodes

        # Check if there's anything to save
        if data_nodes is None or len(data_nodes.data_nodes) == 0:
            QMessageBox.warning(
                main_window,
                "Nothing to Save",
                "There is no data to save. Please load a point cloud first."
            )
            return

        # Call file_manager to save project
        # new_file=True means always prompt for a new location
        success, message = file_manager.save_project(
            data_nodes=data_nodes,
            parent=main_window,
            new_file=True
        )

        # Only show error messages, not success messages
        if not success and "cancelled" not in message.lower():
            QMessageBox.warning(
                main_window,
                "Save Failed",
                message
            )
