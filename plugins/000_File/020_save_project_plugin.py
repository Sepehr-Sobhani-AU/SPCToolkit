# plugins/File/save_project_plugin.py
"""
Plugin for saving the current project to a file.

Saves the entire project state including:
- All point clouds
- All analysis results (clusters, masks, eigenvalues, etc.)
- Data hierarchy and dependencies

Saved as .pcdtk file using pickle serialization.
"""

from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class SaveProjectPlugin(ActionPlugin):
    """
    Action plugin for saving the current project.

    Workflow:
    1. User clicks File → Save Project
    2. Plugin gets current data_nodes from global variables
    3. Calls file_manager.save_project()
    4. Shows success/error message
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "save_project"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - directly opens save dialog.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the save project action.

        Saves to the current project path if one exists, otherwise prompts for a new filename.
        This is the standard "Save" behavior (Ctrl+S).

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
        # new_file=False means use current path if available
        success, message = file_manager.save_project(
            data_nodes=data_nodes,
            parent=main_window,
            new_file=False
        )

        # Only show error messages, not success messages
        if not success and "cancelled" not in message.lower():
            QMessageBox.warning(
                main_window,
                "Save Failed",
                message
            )
