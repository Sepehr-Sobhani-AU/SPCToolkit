# plugins/analysis/load_project_plugin.py
from typing import Dict, Any, List, Tuple

from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode


class LoadProjectPlugin(AnalysisPlugin):
    """
    Plugin for loading a saved project.

    This plugin deserializes a previously saved project file, restoring
    all point clouds, analysis results, and their relationships.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "load_project"
        """
        return "load_project"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for loading a project.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {}  # No parameters needed, we'll handle this in execute

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the project loading operation.

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
        tree_widget = global_variables.global_tree_structure_widget

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
                return "Load cancelled", "message", []

        # Load using the file manager
        loaded_data_nodes, message = file_manager.load_project(parent=main_window)

        if loaded_data_nodes is None:
            # Loading failed or was cancelled
            return message, "message", []

        # Replace the current data_nodes with the loaded one
        # First, clear the current data
        data_nodes.data_nodes.clear()

        # Then add all nodes from the loaded data
        for uid, node in loaded_data_nodes.data_nodes.items():
            data_nodes.data_nodes[uid] = node

        # Clear the tree widget and rebuild it
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

        # Return the result message
        return message, "message", []