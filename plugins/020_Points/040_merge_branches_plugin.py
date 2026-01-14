"""
Merge Branches Plugin

This plugin merges multiple selected branches into a single PointCloud.
The result is placed at root level in the tree.

Workflow:
1. User selects multiple branches using Ctrl+click in tree
2. User runs Points > Merge Branches
3. All selected branches are reconstructed and merged
4. Result is placed at root level in tree
5. Attributes from all branches are preserved (NaN fill for missing)
"""

import uuid
import logging
import numpy as np
from typing import Dict, Any, List

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.point_cloud import PointCloud
from core.data_node import DataNode

logger = logging.getLogger(__name__)


class MergeBranchesPlugin(ActionPlugin):
    """
    Action plugin for merging multiple selected branches into one PointCloud.

    The merged result is placed at the root level of the tree.
    All attributes from all branches are preserved, with NaN values
    for points that don't have a particular attribute.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "merge_branches"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define parameters for the merge operation.

        Returns:
            Dict[str, Any]: Parameter schema with name input
        """
        # Count currently selected branches for default name
        data_manager = global_variables.global_data_manager
        num_selected = len(data_manager.selected_branches) if data_manager else 0

        return {
            "merged_name": {
                "type": "string",
                "default": f"Merged ({num_selected} branches)",
                "label": "Name for Merged Branch",
                "description": "Name for the new merged point cloud"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the merge branches action.

        Args:
            main_window: The main application window
            params: Parameters from the dialog (merged_name)
        """
        logger.info("MergeBranchesPlugin.execute() called")
        logger.debug(f"Params: {params}")

        # Get global instances via singleton pattern
        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Get selected branches
        selected_branches = data_manager.selected_branches
        logger.info(f"Selected branches: {len(selected_branches)} - {selected_branches}")

        # Validation: minimum 2 branches
        if len(selected_branches) < 2:
            logger.warning("Not enough branches selected for merge (need >= 2)")
            QMessageBox.warning(
                main_window,
                "Not Enough Branches",
                "Please select at least 2 branches to merge.\n\n"
                "Use Ctrl+Click to select multiple branches."
            )
            return

        # Show processing overlay
        logger.debug("Showing processing overlay")
        main_window.tree_overlay.position_over(tree_widget)
        main_window.tree_overlay.show_processing(f"Merging {len(selected_branches)} branches...")
        main_window.disable_menus()
        main_window.disable_tree()

        try:
            # Collect all point clouds by reconstructing each branch
            point_clouds: List[PointCloud] = []

            for i, uid_str in enumerate(selected_branches):
                logger.debug(f"Reconstructing branch {i+1}/{len(selected_branches)}: {uid_str}")
                try:
                    pc = data_manager.reconstruct_branch(uid_str)
                    logger.debug(f"  Reconstructed: {pc.size} points")
                    point_clouds.append(pc)
                except Exception as e:
                    logger.error(f"Failed to reconstruct branch {uid_str}: {e}", exc_info=True)
                    QMessageBox.warning(
                        main_window,
                        "Reconstruction Error",
                        f"Failed to reconstruct branch {uid_str}:\n{str(e)}"
                    )
                    return

            # Merge the point clouds using PointCloud.merge()
            merged_name = params.get("merged_name", f"Merged ({len(point_clouds)} branches)")
            logger.info(f"Merging {len(point_clouds)} point clouds...")
            merged_pc = PointCloud.merge(point_clouds, name=merged_name)
            logger.info(f"Merge complete: {merged_pc.size} total points")

            # Create DataNode for merged result at root level
            logger.debug("Creating DataNode for merged result")
            merged_node = DataNode(
                params=merged_name,
                data=merged_pc,
                data_type="point_cloud",
                parent_uid=None,  # Root level
                depends_on=[],  # No dependencies - merged result is independent
                tags=["merged"]
            )

            # Calculate memory size
            logger.debug("Calculating memory size")
            merged_node.memory_size = data_manager._calculate_point_cloud_memory(merged_pc)
            logger.debug(f"Memory size: {merged_node.memory_size}")

            # Add to data nodes collection
            logger.debug("Adding node to data_nodes collection")
            new_uid = data_nodes.add_node(merged_node)
            logger.debug(f"New UID: {new_uid}")

            # Hide source branches BEFORE adding new branch
            # This way, when add_branch triggers render, only merged branch is visible
            logger.debug("Hiding source branches before adding merged branch")
            self._hide_source_branches(tree_widget, selected_branches)

            # Add to tree widget at root level
            # The branch_added signal will trigger _on_branch_added → render
            # Since sources are already hidden, render will show only merged branch
            logger.debug("Adding branch to tree widget")
            tree_widget.add_branch(str(new_uid), None, merged_name, is_root=True)
            logger.info("Branch added and rendered via standard workflow")

            # Update memory tooltip
            logger.debug("Updating cache tooltip")
            tree_widget.update_cache_tooltip(str(new_uid), merged_node.memory_size)

            # Show success message
            QMessageBox.information(
                main_window,
                "Merge Complete",
                f"Successfully merged {len(point_clouds)} branches.\n\n"
                f"Total points: {merged_pc.size:,}\n"
                f"Memory: {merged_node.memory_size}"
            )
            logger.info("Merge operation completed successfully")

        except Exception as e:
            logger.error(f"Merge failed with exception: {e}", exc_info=True)
            QMessageBox.critical(
                main_window,
                "Merge Error",
                f"Failed to merge branches:\n{str(e)}"
            )
        finally:
            # Hide overlay and re-enable UI
            logger.debug("Hiding overlay and re-enabling UI")
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()
            logger.debug("UI restored")

    def _hide_source_branches(
        self,
        tree_widget,
        source_uids: List[str]
    ) -> None:
        """
        Hide source branches before adding merged branch.

        This is called BEFORE add_branch so that when the automatic render
        happens, only the new merged branch will be visible.

        Args:
            tree_widget: TreeStructureWidget instance
            source_uids: List of source branch UIDs to hide
        """
        logger.debug(f"_hide_source_branches: hiding {len(source_uids)} sources")

        # Block signals to prevent triggering visibility_changed render
        # We want the render to happen only once when add_branch is called
        tree_widget.blockSignals(True)

        try:
            for uid in source_uids:
                logger.debug(f"  Hiding source branch: {uid}")
                tree_widget.visibility_status[uid] = False
                item = tree_widget.branches_dict.get(uid)
                if item:
                    item.setCheckState(0, Qt.Unchecked)
                else:
                    logger.warning(f"  Source branch {uid} not found in tree")

        finally:
            # Always restore signals
            tree_widget.blockSignals(False)

        logger.debug("_hide_source_branches complete")
