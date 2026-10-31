"""
Union Branches Plugin

Combines multiple selected branches into their set union (A ∪ B ∪ ...).

Two paths:

1. Fast lineage path -- when every selected branch descends from a common
   ancestor through pure subset chains, the union is exactly the OR of each
   branch's membership mask over that ancestor. The result is a ``masks`` node
   attached under the common ancestor: no reconstruction, no coordinate
   matching, and exact. (A union is not a subset of any single selected branch,
   so it can't be an Analysis-plugin result -- those attach under the selected
   branch. Hence this is an Action plugin that parents the node itself.)

2. Fallback -- when the branches don't share a compatible lineage (e.g.
   independently imported clouds), reconstruct them, concatenate, and drop
   duplicate coordinate rows. The deduplicated result is placed at root.
"""

import uuid
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode
from core.entities.masks import Masks
from core.services.branch_lineage import (
    lowest_common_ancestor,
    membership_mask_over_ancestor,
)

logger = logging.getLogger(__name__)


class UnionBranchesPlugin(ActionPlugin):
    """Action plugin for unioning multiple selected branches into one branch."""

    def get_name(self) -> str:
        return "union_branches"

    def get_parameters(self) -> Dict[str, Any]:
        controller = global_variables.global_application_controller
        num_selected = len(controller.selected_branches) if controller else 0
        return {
            "union_name": {
                "type": "string",
                "default": f"Union ({num_selected} branches)",
                "label": "Name for Union Branch",
                "description": "Name for the new unioned branch"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        selected_branches = list(controller.selected_branches)
        logger.info(f"UnionBranchesPlugin: {len(selected_branches)} selected")

        if len(selected_branches) < 2:
            QMessageBox.warning(
                main_window,
                "Not Enough Branches",
                "Please select at least 2 branches to union.\n\n"
                "Use Ctrl+Click to select multiple branches."
            )
            return

        union_name = params.get("union_name", f"Union ({len(selected_branches)} branches)")

        main_window.tree_overlay.position_over(tree_widget)
        main_window.tree_overlay.show_processing(f"Unioning {len(selected_branches)} branches...")
        main_window.disable_menus()
        main_window.disable_tree()

        try:
            nodes = [data_nodes.get_node(uuid.UUID(uid)) for uid in selected_branches]
            if any(n is None for n in nodes):
                raise ValueError("One or more selected branches could not be found.")

            # --- Fast path: lineage membership masks over a common ancestor ---
            fast = self._lineage_union_mask(nodes, controller, data_nodes)
            if fast is not None:
                ancestor, union_mask = fast
                self._add_mask_node(
                    union_mask, ancestor, selected_branches, union_name,
                    controller, data_nodes, tree_widget,
                )
                kept = int(np.count_nonzero(union_mask))
                ancestor_name = getattr(ancestor, "alias", None) or ancestor.params
                QMessageBox.information(
                    main_window,
                    "Union Complete",
                    f"Unioned {len(nodes)} branches via shared lineage of "
                    f"“{ancestor_name}”.\n\n"
                    f"Points kept: {kept:,} of {union_mask.shape[0]:,}"
                )
                logger.info("Union completed via lineage fast path")
                return

            # --- Fallback: reconstruct, merge, deduplicate, place at root ------
            point_clouds: List[PointCloud] = []
            for i, uid_str in enumerate(selected_branches):
                main_window.tree_overlay.show_processing(
                    f"Reconstructing branch {i + 1}/{len(selected_branches)}..."
                )
                point_clouds.append(controller.reconstruct(uid_str))

            main_window.tree_overlay.show_processing("Deduplicating union...")
            union_pc = PointCloud.merge(point_clouds, name=union_name)
            union_pc, n_dups = self._dedup_rows(union_pc)

            union_node = DataNode(
                params=union_name,
                data=union_pc,
                data_type="point_cloud",
                parent_uid=None,  # Root level -- independent of the sources.
                depends_on=[],
                tags=["union"],
            )
            union_node.memory_size = controller._calculate_point_cloud_memory(union_pc)
            new_uid = data_nodes.add_node(union_node)

            self._hide_source_branches(tree_widget, selected_branches)
            tree_widget.add_branch(str(new_uid), None, union_name, is_root=True)
            tree_widget.update_cache_tooltip(str(new_uid), union_node.memory_size)

            QMessageBox.information(
                main_window,
                "Union Complete",
                f"Unioned {len(point_clouds)} branches by coordinate match.\n\n"
                f"Total points: {union_pc.size:,}\n"
                f"Duplicates removed: {n_dups:,}\n"
                f"Memory: {union_node.memory_size}"
            )
            logger.info("Union completed via coordinate-match fallback")

        except Exception as e:
            logger.error(f"Union failed: {e}", exc_info=True)
            QMessageBox.critical(main_window, "Union Error", f"Failed to union branches:\n{str(e)}")
        finally:
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

    @staticmethod
    def _lineage_union_mask(nodes, controller, data_nodes):
        """
        Try to express the union as a boolean mask over a common ancestor.

        Returns ``(ancestor_node, union_mask)`` where ``union_mask`` is the OR of
        each branch's membership mask over the ancestor, or None when the
        branches don't share a pure subset lineage.
        """
        ancestor = lowest_common_ancestor(nodes, data_nodes)
        if ancestor is None:
            return None

        n_ancestor = controller.get_node_reconstructed_count(ancestor)
        if n_ancestor is None:
            return None

        union_mask = np.zeros(n_ancestor, dtype=bool)
        for node in nodes:
            membership = membership_mask_over_ancestor(node, ancestor, data_nodes, n_ancestor)
            if membership is None:
                return None  # Not a pure subset chain -> fall back.
            union_mask |= membership
        return ancestor, union_mask

    @staticmethod
    def _add_mask_node(union_mask, ancestor, source_uids, name,
                       controller, data_nodes, tree_widget):
        """Create a ``masks`` node under the common ancestor and add it to the tree."""
        union_node = DataNode(
            params=name,
            data=Masks(union_mask),
            data_type="masks",
            parent_uid=ancestor.uid,
            depends_on=[uuid.UUID(uid) for uid in source_uids],
            tags=["union"],
        )
        union_node.memory_size = controller._estimate_derived_memory(union_node.data)
        new_uid = data_nodes.add_node(union_node)

        UnionBranchesPlugin._hide_source_branches(tree_widget, source_uids)
        tree_widget.add_branch(str(new_uid), str(ancestor.uid), name, is_root=False)
        tree_widget.update_cache_tooltip(str(new_uid), union_node.memory_size)

    @staticmethod
    def _dedup_rows(point_cloud: PointCloud):
        """
        Drop duplicate coordinate rows, keeping each row's first occurrence.

        Returns ``(deduplicated_point_cloud, num_duplicates_removed)``. When
        there are no duplicates the original point cloud is returned unchanged.
        """
        pts = np.ascontiguousarray(point_cloud.points)
        if pts.shape[0] == 0:
            return point_cloud, 0
        void_dtype = np.dtype((np.void, pts.dtype.itemsize * pts.shape[1]))
        _, first_idx = np.unique(pts.reshape(-1).view(void_dtype), return_index=True)
        n_dups = pts.shape[0] - first_idx.shape[0]
        if n_dups == 0:
            return point_cloud, 0
        keep = np.zeros(pts.shape[0], dtype=bool)
        keep[first_idx] = True
        return point_cloud.get_subset(mask=keep), n_dups

    @staticmethod
    def _hide_source_branches(tree_widget, source_uids: List[str]) -> None:
        """
        Hide source branches before adding the union branch, so the automatic
        render triggered by ``add_branch`` shows only the new union branch.
        """
        tree_widget.blockSignals(True)
        try:
            for uid in source_uids:
                tree_widget.visibility_status[uid] = False
                item = tree_widget.branches_dict.get(uid)
                if item:
                    item.setCheckState(0, Qt.Unchecked)
        finally:
            tree_widget.blockSignals(False)
