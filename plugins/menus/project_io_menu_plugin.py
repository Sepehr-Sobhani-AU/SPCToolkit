# plugins/menus/project_io_menu_plugin.py
from typing import Dict, Any, List
import pickle
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from plugins.interfaces import MenuPlugin
from config.config import global_variables


class ProjectIOMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds project save and load functionality to the File menu.

    This plugin adds "Save Project", "Save Project As...", and "Open Project"
    menu items to the File menu, allowing users to save their work and restore
    it later.
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
        Return the menu items for project I/O operations.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Open Project...",
                "action": "open_project",
                "tooltip": "Open a saved project file"
            },
            {
                "name": "Save Project",
                "action": "save_project",
                "tooltip": "Save the current project"
            },
            {
                "name": "Save Project As...",
                "action": "save_project_as",
                "tooltip": "Save the current project to a new file"
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        # Get global variables
        data_nodes = global_variables.global_data_nodes
        data_manager = global_variables.global_data_manager

        if action_name == "open_project":
            self._load_project(main_window, data_nodes, data_manager)
        elif action_name == "save_project":
            self._save_project(main_window, data_nodes, False)
        elif action_name == "save_project_as":
            self._save_project(main_window, data_nodes, True)

    def _save_project(self, main_window, data_nodes, new_file=False):
        """
        Save the current project to a file.

        Args:
            main_window: The main application window
            data_nodes: The DataNodes instance to save
            new_file (bool): If True, always prompt for a new filename
        """
        # Static variable to remember the last save path
        if not hasattr(self, 'current_project_path'):
            self.current_project_path = None

        filename = self.current_project_path

        # If no current path or new_file is True, prompt for a filename
        if filename is None or new_file:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                main_window,
                "Save Project",
                "",
                "PCD Toolkit Project Files (*.pcdtk);;All Files (*)",
                options=options
            )

            if not filename:
                main_window.statusBar().showMessage("Save cancelled", 3000)
                return

            # Ensure the file has the correct extension
            if not filename.endswith(".pcdtk"):
                filename += ".pcdtk"

        # Add version information to the saved data
        project_data = {
            'version': '1.0.0',
            'data_nodes': data_nodes
        }

        try:
            # Save the data_nodes instance using pickle
            with open(filename, 'wb') as file:
                pickle.dump(project_data, file)

            # Store the path for future saves
            self.current_project_path = filename

            # Extract just the filename for the success message
            base_filename = filename.split("/")[-1].split("\\")[-1]
            main_window.statusBar().showMessage(f"Project saved to {base_filename}", 3000)

        except Exception as e:
            QMessageBox.critical(main_window, "Error Saving Project", f"Failed to save project: {str(e)}")

    def _load_project(self, main_window, data_nodes, data_manager):
        """
        Load a project from a file.

        Args:
            main_window: The main application window
            data_nodes: The DataNodes instance to update
            data_manager: The DataManager instance
        """
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            main_window,
            "Open Project",
            "",
            "PCD Toolkit Project Files (*.pcdtk);;All Files (*)",
            options=options
        )

        if not filename:
            main_window.statusBar().showMessage("Open cancelled", 3000)
            return

        # Confirm before replacing current project if it has data
        if data_nodes.data_nodes and len(data_nodes.data_nodes) > 0:
            reply = QMessageBox.question(
                main_window,
                'Confirm Load',
                'Loading a project will replace your current work. Continue?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                main_window.statusBar().showMessage("Load cancelled", 3000)
                return

        try:
            # Load the project data using pickle
            with open(filename, 'rb') as file:
                project_data = pickle.load(file)

            # Check version information if available
            if isinstance(project_data, dict) and 'version' in project_data:
                version = project_data['version']
                loaded_data_nodes = project_data['data_nodes']
            else:
                # Handle older project files without version info
                loaded_data_nodes = project_data

            # Store the path for future saves
            self.current_project_path = filename

            # Replace the current data_nodes with the loaded one
            # We need to update the global reference in a way that preserves it
            # First, clear the current data
            data_nodes.data_nodes.clear()

            # Then add all nodes from the loaded data
            for uid, node in loaded_data_nodes.data_nodes.items():
                data_nodes.data_nodes[uid] = node

            # Clear the tree widget and rebuild it
            tree_widget = global_variables.global_tree_structure_widget
            tree_widget.clear()

            # Rebuild the tree structure
            # First add top-level nodes (nodes without parents)
            top_level_nodes = {uid: node for uid, node in data_nodes.data_nodes.items()
                               if node.parent_uid is None}

            for uid, node in top_level_nodes.items():
                tree_widget.add_branch(str(uid), "", node.params)

            # Then add child nodes level by level
            remaining_nodes = {uid: node for uid, node in data_nodes.data_nodes.items()
                               if node.parent_uid is not None}

            # Keep processing until all nodes are added
            while remaining_nodes:
                added_this_round = []

                for uid, node in remaining_nodes.items():
                    # If the parent is already in the tree, we can add this node
                    if str(node.parent_uid) in tree_widget.branches_dict:
                        tree_widget.add_branch(str(uid), str(node.parent_uid), node.params)
                        added_this_round.append(uid)

                # Remove the nodes we've added
                for uid in added_this_round:
                    remaining_nodes.pop(uid)

                # If we didn't add any nodes this round, we might have orphaned nodes
                if not added_this_round and remaining_nodes:
                    # Add remaining nodes at the top level with a warning
                    for uid, node in remaining_nodes.items():
                        tree_widget.add_branch(str(uid), "", f"{node.params} (Orphaned)")
                    break

            # Extract just the filename for the success message
            base_filename = filename.split("/")[-1].split("\\")[-1]
            main_window.statusBar().showMessage(f"Project loaded from {base_filename}", 3000)

        except Exception as e:
            QMessageBox.critical(main_window, "Error Loading Project", f"Failed to load project: {str(e)}")