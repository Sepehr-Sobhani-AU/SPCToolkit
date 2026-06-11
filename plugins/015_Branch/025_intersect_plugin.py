# plugins/015_Branch/025_intersect_plugin.py
from typing import Dict, Any, List, Tuple
import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.masks import Masks
from core.services.branch_lineage import (
    lowest_common_ancestor,
    membership_mask_over_ancestor,
    row_membership,
)


class IntersectPlugin(Plugin):
    """
    Plugin for intersecting two branches.

    Produces a mask over the selected (target) branch that keeps only the
    points also present in a second branch -- i.e. A ∩ B, expressed as a subset
    of A. The result therefore attaches under the selected branch like subtract.

    Fast path: when both branches descend from a common ancestor through pure
    subset chains, the result is composed from their membership masks over that
    ancestor (``inB`` restricted to A's points) -- no reconstruction, no
    coordinate matching, and exact. Falls back to exact coordinate matching when
    the lineage isn't a pure subset chain.
    """

    def get_name(self) -> str:
        return "intersect"

    def get_parameters(self) -> Dict[str, Any]:
        """Define the parameters for the intersection operation."""
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes

        node_options = {}
        for node_uid, node in data_nodes.data_nodes.items():
            node_options[str(node_uid)] = node.alias or node.params

        default_value = ""
        if node_options:
            default_value = next(iter(node_options))

        return {
            "intersect_node": {
                "type": "dropdown",
                "options": node_options,
                "default": default_value,
                "label": "Branch to Intersect With",
                "description": "Keep only points shared with this branch"
            }
        }

    def confirm_before_execute(self, data_node, params):
        """
        Warn when intersect can't use the fast lineage path and must compare
        points by coordinate (potentially very slow). Runs on the main thread
        before the worker starts; reuses ``_fast_keep_mask`` which only inspects
        lineage and mask arrays, so it's cheap.
        """
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes
        controller = global_variables.global_application_controller
        if data_nodes is None or controller is None:
            return None

        try:
            other_node = data_nodes.get_node(uuid.UUID(params["intersect_node"]))
        except Exception:
            return None
        if other_node is None:
            return None

        n_target = controller.get_node_reconstructed_count(data_node)
        if n_target is None:
            return None  # Can't reason cheaply; let execute() handle it.

        if self._fast_keep_mask(data_node, other_node, controller, data_nodes) is not None:
            return None  # Fast lineage path available -- no prompt needed.

        n_other = controller.get_node_reconstructed_count(other_node) or 0
        target_name = getattr(data_node, "alias", None) or "the selected branch"
        other_name = getattr(other_node, "alias", None) or "the other branch"
        return (
            f"“{other_name}” doesn't share a compatible lineage with “{target_name}”, "
            f"so intersect must match points by coordinate "
            f"(~{n_target:,} vs {n_other:,} points). This can be slow.\n\n"
            f"Proceed with the slow exact match?"
        )

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the intersection between the selected branch and another.

        Returns a ``Masks`` over the selected branch's points (True = keep,
        i.e. the point is also present in the other branch).
        """
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes
        controller = global_variables.global_application_controller

        target_pc = data_node.data
        target_points = target_pc.points

        try:
            other_uid = uuid.UUID(params["intersect_node"])
            other_node = data_nodes.get_node(other_uid)
            if other_node is None:
                raise ValueError(f"Branch with UUID {other_uid} not found")
        except Exception as e:
            raise ValueError(f"Error processing branch to intersect: {str(e)}")

        # Fast path: compose the result from lineage membership masks.
        keep_mask = self._fast_keep_mask(data_node, other_node, controller, data_nodes)
        if keep_mask is not None:
            return Masks(keep_mask), "masks", [data_node.uid, other_uid]

        # Slow path: reconstruct the other branch and match points by coordinate.
        try:
            global_variables.global_progress = (None, "Reconstructing other branch...")
            other_pc = controller.reconstruct(other_uid)
            other_points = other_pc.points
        except Exception as e:
            raise ValueError(f"Error processing branch to intersect: {str(e)}")

        if target_points.shape[1] != other_points.shape[1]:
            raise ValueError("Point dimensions do not match between the two point clouds")

        if global_variables.global_cancel_event.is_set():
            raise RuntimeError("Intersect cancelled.")

        global_variables.global_progress = (
            50, f"Comparing {len(target_points):,} vs {len(other_points):,} points..."
        )

        # row_membership returns True where the target point IS present in the
        # other branch -- exactly the intersect keep-mask.
        mask = row_membership(target_points, other_points)

        if global_variables.global_cancel_event.is_set():
            raise RuntimeError("Intersect cancelled.")

        return Masks(mask), "masks", [data_node.uid, other_uid]

    @staticmethod
    def _fast_keep_mask(target_node, other_node, controller, data_nodes):
        """
        Try to compute the intersect keep-mask over the target's points using
        lineage alone. Returns a boolean array of length == target point count
        (True = keep), or None when the lineage isn't a pure subset chain and
        the caller must fall back to coordinate matching.
        """
        # During execute() the AnalysisExecutor hands us a *temporary*
        # reconstructed node whose parent_uid is None (data_type "point_cloud"),
        # which would sever the target's lineage walk. Resolve the real tree node
        # by uid to recover its ancestry. (In confirm_before_execute the node is
        # already the real one, so this is a no-op there.)
        target_node = data_nodes.get_node(target_node.uid) or target_node

        lca = lowest_common_ancestor([target_node, other_node], data_nodes)
        if lca is None:
            return None  # No shared ancestor.

        n_lca = controller.get_node_reconstructed_count(lca)
        if n_lca is None:
            return None  # Can't size the ancestor cheaply.

        in_target = membership_mask_over_ancestor(target_node, lca, data_nodes, n_lca)
        in_other = membership_mask_over_ancestor(other_node, lca, data_nodes, n_lca)
        if in_target is None or in_other is None:
            return None  # Lineage isn't a pure subset chain.

        # Ancestor indices retained by the target, in the target's point order.
        idx_target = np.flatnonzero(in_target)
        # For each target point, keep it iff that ancestor point is also in the
        # other branch.
        return in_other[idx_target]
