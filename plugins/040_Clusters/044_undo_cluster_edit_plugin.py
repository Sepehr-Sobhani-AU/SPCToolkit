"""
Plugin for undoing the last cluster edit (remove, cut, or merge) on the selected branch.

Restores the previous Clusters object stored before the last destructive operation.
Single-level undo: only the most recent edit per node can be undone.
"""

from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class UndoClusterEditPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "undo_cluster_edit"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller

        # Validate: exactly one branch selected
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a cluster branch first.")
            return
        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch at a time.")
            return

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)

        # Validate: must be cluster_labels
        if node is None or node.data_type != "cluster_labels":
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Please select a cluster_labels branch.")
            return

        # Check if undo is available
        uid_key = str(node.uid)
        if uid_key not in controller._cluster_undo:
            QMessageBox.information(main_window, "Nothing to Undo",
                                    "No previous cluster edit to undo on this branch.")
            return

        # Restore previous Clusters object
        node.data = controller._cluster_undo.pop(uid_key)

        # Invalidate cache and re-render
        controller.cache_service.invalidate(uid_key)
        controller.cache_service.invalidate_descendants(uid_key)
        main_window.render_visible_data(zoom_extent=False)
