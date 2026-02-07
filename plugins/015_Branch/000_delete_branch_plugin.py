from typing import Dict, Any, List, Set
import uuid

from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class DeleteBranchPlugin(ActionPlugin):
    """Plugin to delete selected branches and all their children from the tree."""

    def get_name(self) -> str:
        return "delete_branch"

    def get_parameters(self) -> Dict[str, Any]:
        return {}  # Uses selected branch(es) from tree

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Get selected branches
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(
                main_window,
                "No Selection",
                "Please select one or more branches to delete."
            )
            return

        # Build complete deletion set (selected + all descendants)
        uids_to_delete: Set[uuid.UUID] = set()
        for uid_str in selected_branches:
            uid = uuid.UUID(uid_str)
            uids_to_delete.add(uid)
            descendants = self._get_all_descendants(uid, data_nodes)
            uids_to_delete.update(descendants)

        # Check for external dependencies
        external_deps = self._check_external_dependencies(uids_to_delete, data_nodes)
        if external_deps:
            dep_list = "\n".join(external_deps[:5])
            if len(external_deps) > 5:
                dep_list += f"\n... and {len(external_deps) - 5} more"
            QMessageBox.warning(
                main_window,
                "Cannot Delete",
                f"Cannot delete selected branches because other nodes depend on them:\n\n{dep_list}"
            )
            return

        # Build confirmation message
        node_names = []
        for uid_str in selected_branches:
            uid = uuid.UUID(uid_str)
            node = data_nodes.get_node(uid)
            if node:
                node_names.append(node.data_name or str(uid)[:8])

        child_count = len(uids_to_delete) - len(selected_branches)
        message = f"Delete the following branch(es)?\n\n"
        message += "\n".join(f"- {name}" for name in node_names[:5])
        if len(node_names) > 5:
            message += f"\n... and {len(node_names) - 5} more"
        if child_count > 0:
            message += f"\n\nThis will also delete {child_count} child node(s)."
        message += "\n\nThis action cannot be undone."

        reply = QMessageBox.question(
            main_window,
            "Confirm Delete",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Perform deletion
        tree_widget.blockSignals(True)
        try:
            # Sort UIDs by depth (deepest first) for bottom-up deletion
            sorted_uids = self._sort_by_depth(uids_to_delete, data_nodes)

            for uid in sorted_uids:
                # Remove from data nodes
                data_nodes.remove_node(uid)
                # Remove from tree widget UI
                tree_widget.remove_branch(str(uid))

            # Clear selection
            controller.selected_branches = []

        finally:
            tree_widget.blockSignals(False)

        # Re-render visible data
        main_window.render_visible_data(zoom_extent=False)

    def _get_all_descendants(self, uid: uuid.UUID, data_nodes) -> List[uuid.UUID]:
        """Get all descendant UIDs (children, grandchildren, etc.)"""
        descendants = []
        for node_uid, node in data_nodes.data_nodes.items():
            if node.parent_uid == uid:
                descendants.append(node_uid)
                descendants.extend(self._get_all_descendants(node_uid, data_nodes))
        return descendants

    def _check_external_dependencies(self, uids_to_delete: Set[uuid.UUID], data_nodes) -> List[str]:
        """Find nodes outside deletion set that depend on nodes being deleted."""
        external_deps = []
        for node_uid, node in data_nodes.data_nodes.items():
            if node_uid not in uids_to_delete:
                for dep_uid in node.depends_on:
                    if dep_uid in uids_to_delete:
                        dep_node = data_nodes.get_node(dep_uid)
                        dep_name = dep_node.data_name if dep_node else str(dep_uid)[:8]
                        external_deps.append(f"'{node.data_name}' depends on '{dep_name}'")
        return external_deps

    def _sort_by_depth(self, uids: Set[uuid.UUID], data_nodes) -> List[uuid.UUID]:
        """Sort UIDs by depth in tree (deepest first for bottom-up deletion)."""
        def get_depth(uid: uuid.UUID) -> int:
            depth = 0
            node = data_nodes.get_node(uid)
            while node and node.parent_uid:
                depth += 1
                node = data_nodes.get_node(node.parent_uid)
            return depth

        return sorted(uids, key=get_depth, reverse=True)
