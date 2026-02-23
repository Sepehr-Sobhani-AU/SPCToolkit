# plugins/File/load_project_plugin.py
"""
Plugin for loading a saved project from a file.

Loads the entire project state including:
- All point clouds
- All analysis results (clusters, masks, eigenvalues, etc.)
- Data hierarchy and dependencies

Loads from .pcdtk file using pickle deserialization.
"""

from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class LoadProjectPlugin(ActionPlugin):
    """
    Action plugin for loading a saved project.

    Workflow:
    1. User clicks File → Load Project
    2. Plugin opens file dialog
    3. Loads data_nodes from .pcdtk file
    4. Replaces current project data
    5. Rebuilds tree widget
    6. Clears viewer
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "load_project"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - directly opens load dialog.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the load project action.

        Args:
            main_window: The main application window
            params: Not used for this plugin (empty dict)
        """
        # Get global instances
        file_manager = global_variables.global_file_manager
        controller = global_variables.global_application_controller
        tree_widget = global_variables.global_tree_structure_widget
        viewer_widget = global_variables.global_pcd_viewer_widget

        # Warn user if there's unsaved data
        if global_variables.global_data_nodes and len(global_variables.global_data_nodes.data_nodes) > 0:
            reply = QMessageBox.question(
                main_window,
                "Load Project",
                "Loading a project will replace the current data.\n"
                "Any unsaved changes will be lost.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        # Call file_manager to load project
        main_window.show_progress("Loading project file...")
        loaded_data_nodes, message = file_manager.load_project(parent=main_window)

        # Check if loading was successful
        if loaded_data_nodes is None:
            main_window.clear_progress()
            if "cancelled" not in message.lower():
                QMessageBox.warning(
                    main_window,
                    "Load Failed",
                    message
                )
            return

        main_window.disable_menus()
        main_window.disable_tree()
        try:
            # Extract tree visibility state if available
            tree_visibility = None
            if hasattr(file_manager, '_last_loaded_tree_visibility'):
                tree_visibility = file_manager._last_loaded_tree_visibility

            main_window.show_progress("Replacing project data...", 20)

            # Replace data_nodes across all services via ApplicationController
            controller.load_project(loaded_data_nodes)

            main_window.show_progress("Clearing tree...", 30)

            # Clear the tree widget
            tree_widget.clear()

            # Clear the viewer
            viewer_widget.set_points(None)
            viewer_widget.update()

            # Block signals during tree rebuild to prevent automatic reconstruction
            tree_widget.blockSignals(True)

            # Rebuild tree from loaded data_nodes (with per-node progress)
            self.rebuild_tree(tree_widget, loaded_data_nodes, main_window)

            # Restore visibility state or set all to invisible
            if tree_visibility is not None:
                self.restore_tree_visibility(tree_widget, tree_visibility)
            else:
                # Set all branches to invisible (unchecked) for older project files
                self.set_all_branches_invisible(tree_widget)

            # Unblock signals
            tree_widget.blockSignals(False)

            main_window.show_progress("Updating memory labels...", 80)

            # Update memory labels for all loaded branches
            memory_labels = controller.update_all_branch_memory_labels()
            for uid_str, memory_size in memory_labels.items():
                if uid_str in tree_widget.branches_dict:
                    tree_widget.update_cache_tooltip(uid_str, memory_size)

            main_window.show_progress("Rendering...", 90)

            # Update viewer to show visible branches and zoom to extent
            main_window.render_visible_data(zoom_extent=True)
        finally:
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()

    def rebuild_tree(self, tree_widget, data_nodes, main_window=None):
        """
        Rebuild the tree widget from loaded data_nodes.

        Iterates through data_nodes in hierarchical order and adds branches.

        Args:
            tree_widget: The TreeStructureWidget instance
            data_nodes: The loaded DataNodes instance
            main_window: Optional main window for progress reporting
        """
        # Get all nodes
        all_nodes = data_nodes.data_nodes
        total_nodes = len(all_nodes)
        node_counter = [0]  # Mutable counter for recursive tracking

        # Find root nodes (nodes with no parent)
        root_nodes = [node for node in all_nodes.values() if node.parent_uid is None]

        # Add root nodes first
        for node in root_nodes:
            node_counter[0] += 1
            if main_window and total_nodes > 0:
                percent = 40 + int((node_counter[0] / total_nodes) * 30)
                main_window.show_progress(
                    f"Rebuilding tree ({node_counter[0]}/{total_nodes} nodes)...", percent
                )
            # Detect if this is a root PointCloud node
            is_root = (node.data_type == "point_cloud" or node.data_type == "PointCloud")

            display_name = node.alias or node.params or node.data_type
            tooltip = f"{node.tags[0]},{node.tags[1]}" if len(node.tags) > 1 else None
            tree_widget.add_branch(
                str(node.uid),
                "",  # No parent
                display_name,
                is_root=is_root,
                tooltip=tooltip
            )

        # Add child nodes recursively
        for root_node in root_nodes:
            self._add_children_recursive(
                tree_widget, data_nodes, root_node.uid,
                main_window, total_nodes, node_counter
            )

    def _add_children_recursive(self, tree_widget, data_nodes, parent_uid,
                                main_window=None, total_nodes=0, node_counter=None):
        """
        Recursively add child nodes to the tree.

        Args:
            tree_widget: The TreeStructureWidget instance
            data_nodes: The DataNodes instance
            parent_uid: The UID of the parent node
            main_window: Optional main window for progress reporting
            total_nodes: Total number of nodes for progress calculation
            node_counter: Mutable list [count] for tracking progress across recursion
        """
        # Find all children of this parent
        children = [
            node for node in data_nodes.data_nodes.values()
            if node.parent_uid == parent_uid
        ]

        # Add each child
        for child in children:
            if node_counter is not None:
                node_counter[0] += 1
                if main_window and total_nodes > 0:
                    percent = 40 + int((node_counter[0] / total_nodes) * 30)
                    main_window.show_progress(
                        f"Rebuilding tree ({node_counter[0]}/{total_nodes} nodes)...", percent
                    )
            display_name = child.alias or child.params or child.data_type
            tooltip = f"{child.tags[0]},{child.tags[1]}" if len(child.tags) > 1 else None
            tree_widget.add_branch(
                str(child.uid),
                str(parent_uid),
                display_name,
                tooltip=tooltip
            )

            # Recursively add this child's children
            self._add_children_recursive(
                tree_widget, data_nodes, child.uid,
                main_window, total_nodes, node_counter
            )

    def set_all_branches_invisible(self, tree_widget):
        """
        Set all branches in the tree to invisible (unchecked).

        This prevents automatic reconstruction of all branches when loading a project.

        Args:
            tree_widget: The TreeStructureWidget instance
        """
        from PyQt5.QtCore import Qt

        # Iterate through all branches and uncheck them
        for uuid, item in tree_widget.branches_dict.items():
            item.setCheckState(0, Qt.Unchecked)
            tree_widget.visibility_status[uuid] = False

    def restore_tree_visibility(self, tree_widget, tree_visibility):
        """
        Restore the visibility state of all branches from saved state.

        Args:
            tree_widget: The TreeStructureWidget instance
            tree_visibility: Dictionary mapping UUID strings to visibility boolean values
        """
        from PyQt5.QtCore import Qt

        # Iterate through the saved visibility state and restore each branch
        for uuid_str, is_visible in tree_visibility.items():
            # Check if this branch exists in the tree
            if uuid_str in tree_widget.branches_dict:
                item = tree_widget.branches_dict[uuid_str]
                # Set check state based on visibility
                check_state = Qt.Checked if is_visible else Qt.Unchecked
                item.setCheckState(0, check_state)
                # Update visibility status dictionary
                tree_widget.visibility_status[uuid_str] = is_visible
