# plugins/analysis/save_project_as_plugin.py
from typing import Dict, Any, List, Tuple

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode


class SaveProjectAsPlugin(AnalysisPlugin):
    """
    Plugin for saving the current project state to a new file.

    This plugin is similar to SaveProjectPlugin but always prompts for
    a new filename, regardless of whether the project has been saved before.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "save_project_as"
        """
        return "save_project_as"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for saving a project.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {}  # No parameters needed, we'll always show a file dialog

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the project saving operation with a new filename.

        Args:
            data_node (DataNode): Not used in this plugin
            params (Dict[str, Any]): Not used in this plugin

        Returns:
            Tuple[str, str, List]:
                - Success message
                - Result type identifier "message"
                - Empty list (no dependencies)
        """
        # Import here to avoid circular imports
        from config.config import global_variables

        # Get the global data_nodes instance and file manager
        data_nodes = global_variables.global_data_nodes
        file_manager = global_variables.global_file_manager
        main_window = global_variables.main_window

        # Save using the file manager, forcing a new file
        success, message = file_manager.save_project(data_nodes, parent=main_window, new_file=True)

        # Return the result message
        return message, "message", []