# plugins/analysis/save_project_plugin.py
from typing import Dict, Any, List, Tuple
from PyQt5.QtWidgets import QFileDialog

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode


class SaveProjectPlugin(AnalysisPlugin):
    """
    Plugin for saving the current project state.

    This plugin serializes the entire DataNodes structure, preserving all
    point clouds, analysis results, and their relationships.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "save_project"
        """
        return "save_project"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for saving a project.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {}  # No parameters needed, we'll handle this in execute

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the project saving operation.

        Args:
            data_node (DataNode): Not used in this plugin
            params (Dict[str, Any]): Parameters for saving (not used)

        Returns:
            Tuple[str, str, List]:
                - Success message
                - Result type identifier "message"
                - Empty list (no dependencies)
        """
        try:
            # Import here to avoid circular imports
            from config.config import global_variables

            # Get the global data_nodes instance
            data_nodes = global_variables.global_data_nodes
            main_window = global_variables.main_window

            # Check if we can access the file_manager from global_variables
            if hasattr(global_variables, 'global_file_manager'):
                file_manager = global_variables.global_file_manager

                # Use the file manager to save the project
                success, message = file_manager.save_project(data_nodes, parent=main_window, new_file=False)
            else:
                # If file_manager isn't in global_variables, use a direct approach
                # This is a fallback in case the file_manager isn't properly set up
                options = QFileDialog.Options()
                filename, _ = QFileDialog.getSaveFileName(
                    main_window,
                    "Save Project",
                    "",
                    "PCD Toolkit Project Files (*.pcdtk);;All Files (*)",
                    options=options
                )

                if not filename:
                    return "Save cancelled", "message", []

                # Ensure the file has the correct extension
                if not filename.endswith(".pcdtk"):
                    filename += ".pcdtk"

                # Import pickle here
                import pickle

                # Create project data with version info
                project_data = {
                    'version': '1.0.0',
                    'data_nodes': data_nodes
                }

                # Save directly using pickle
                with open(filename, 'wb') as file:
                    pickle.dump(project_data, file)

                success = True
                message = f"Project saved successfully to {filename}"

            # Return the result message
            return message, "message", []

        except Exception as e:
            # Return detailed error information
            import traceback
            error_details = traceback.format_exc()
            return f"Error saving project: {str(e)}\n\nDetails: {error_details}", "message", []