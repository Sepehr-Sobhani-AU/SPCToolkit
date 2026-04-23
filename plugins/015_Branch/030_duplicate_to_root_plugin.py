"""
Duplicate to Root Plugin

Reconstructs the selected branch and adds the result as a new independent
root-level point cloud in the tree.
"""

import uuid
import logging
from typing import Dict, Any

from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.data_node import DataNode

logger = logging.getLogger(__name__)


class DuplicateToRootPlugin(ActionPlugin):
    """Promote a reconstructed branch to an independent root point cloud."""

    def get_name(self) -> str:
        return "duplicate_to_root"

    def get_parameters(self) -> Dict[str, Any]:
        default_name = "Duplicated Branch"
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        if controller and data_nodes and len(controller.selected_branches) == 1:
            try:
                src = data_nodes.get_node(uuid.UUID(controller.selected_branches[0]))
                if src is not None:
                    base = src.alias or src.params or "Branch"
                    default_name = f"{base} (copy)"
            except (ValueError, AttributeError):
                pass

        return {
            "new_name": {
                "type": "string",
                "default": default_name,
                "label": "Name for New Root Branch",
                "description": "Name for the duplicated point cloud placed at the root level"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        logger.info("DuplicateToRootPlugin.execute() called")

        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        selected_branches = controller.selected_branches
        if len(selected_branches) != 1:
            QMessageBox.warning(
                main_window,
                "Select One Branch",
                "Please select exactly one branch to duplicate to root."
            )
            return

        source_uid = selected_branches[0]
        new_name = params.get("new_name") or "Duplicated Branch"

        main_window.tree_overlay.position_over(tree_widget)
        main_window.tree_overlay.show_processing(f"Duplicating branch to root...")
        main_window.disable_menus()
        main_window.disable_tree()

        try:
            pc = controller.reconstruct(source_uid)
            logger.info(f"Reconstructed {pc.size} points from {source_uid}")

            new_node = DataNode(
                params=new_name,
                data=pc,
                data_type="point_cloud",
                parent_uid=None,
                depends_on=[],
                tags=["duplicated"]
            )
            new_node.memory_size = controller._calculate_point_cloud_memory(pc)

            new_uid = data_nodes.add_node(new_node)

            tree_widget.add_branch(str(new_uid), None, new_name, is_root=True)
            tree_widget.update_cache_tooltip(str(new_uid), new_node.memory_size)

            logger.info(f"Duplicated branch added at root: {new_uid}")

        except Exception as e:
            logger.error(f"Duplicate to root failed: {e}", exc_info=True)
            QMessageBox.critical(
                main_window,
                "Duplicate Error",
                f"Failed to duplicate branch:\n{str(e)}"
            )
        finally:
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()
