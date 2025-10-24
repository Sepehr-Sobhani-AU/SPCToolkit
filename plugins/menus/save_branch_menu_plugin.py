# plugins/menus/save_branch_menu_plugin.py
"""
Save Branch menu plugin for exporting selected branches to PLY files.

This plugin adds a "Save Branch" option to the File menu, allowing users to
export a single selected branch from the tree structure as a PLY file.
"""

from typing import Dict, Any, List
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from plugins.interfaces import MenuPlugin
from config.config import global_variables


class SaveBranchMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds branch export functionality.

    This plugin allows users to save a selected branch (DataNode) as a PLY file.
    The branch is first reconstructed to obtain the complete PointCloud with all
    derived data applied, then saved using the PointCloud's built-in save method.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "File"
        """
        return "File"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for save operations.

        Returns:
            List[Dict[str, Any]]: List containing the "Save Branch" menu item
        """
        return [
            {
                "name": "Save Branch",
                "action": "save_branch",
                "tooltip": "Save selected branch to PLY file",
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when the menu item is clicked.

        This method validates the selection, reconstructs the branch, prompts
        for a save location, and saves the branch as a PLY file.

        Args:
            action_name (str): The action identifier ("save_branch")
            main_window: Reference to the main application window

        Workflow:
            1. Validate that exactly one branch is selected
            2. Reconstruct the branch to get full PointCloud
            3. Prompt user for save file location
            4. Save using PointCloud.save() method
            5. Display success/error message
        """
        if action_name == "save_branch":
            try:
                # Get the tree widget to check selection
                tree_widget = global_variables.global_tree_structure_widget

                # Get selected branches
                selected_items = tree_widget.selectedItems()

                # Validate selection
                if len(selected_items) == 0:
                    QMessageBox.information(
                        main_window,
                        "No Selection",
                        "Please select a branch to save.",
                        QMessageBox.Ok
                    )
                    return

                if len(selected_items) > 1:
                    QMessageBox.information(
                        main_window,
                        "Multiple Selection",
                        "Please select only one branch to save.",
                        QMessageBox.Ok
                    )
                    return

                # Get the selected branch UID
                selected_item = selected_items[0]
                branch_uid = selected_item.data(0, Qt.UserRole)

                if not branch_uid:
                    QMessageBox.warning(
                        main_window,
                        "Invalid Selection",
                        "Selected item does not have a valid UID.",
                        QMessageBox.Ok
                    )
                    return

                # Get the data manager for branch reconstruction
                data_manager = global_variables.global_data_manager

                # Reconstruct the branch to get the full PointCloud
                print(f"Reconstructing branch with UID: {branch_uid}")
                point_cloud = data_manager.reconstruct_branch(branch_uid)

                # Check if reconstruction was successful
                if point_cloud is None or point_cloud.size == 0:
                    QMessageBox.warning(
                        main_window,
                        "Reconstruction Failed",
                        "Failed to reconstruct the selected branch.",
                        QMessageBox.Ok
                    )
                    return

                # Open save file dialog
                file_path, _ = QFileDialog.getSaveFileName(
                    main_window,
                    "Save Branch as PLY",
                    "",
                    "PLY Files (*.ply);;All Files (*)",
                )

                # Check if user cancelled the dialog
                if not file_path:
                    return

                # Ensure the file has .ply extension
                if not file_path.endswith('.ply'):
                    file_path += '.ply'

                # Save the point cloud using its built-in save method
                print(f"Saving branch to: {file_path}")
                point_cloud.save(file_path)

                # Show success message
                QMessageBox.information(
                    main_window,
                    "Save Successful",
                    f"Branch saved successfully to:\n{file_path}",
                    QMessageBox.Ok
                )

            except Exception as e:
                # Handle any errors that occur during the save process
                error_message = f"Error saving branch: {str(e)}"
                print(error_message)
                QMessageBox.critical(
                    main_window,
                    "Save Error",
                    error_message,
                    QMessageBox.Ok
                )
