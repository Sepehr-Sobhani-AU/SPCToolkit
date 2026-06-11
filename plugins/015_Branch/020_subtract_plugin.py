# plugins/analysis/subtract_plugin.py
from typing import Dict, Any, List, Tuple
import logging
import numpy as np
import uuid

logger = logging.getLogger(__name__)

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.masks import Masks

try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:  # pragma: no cover - CuPy is optional
    _cp = None
    _HAS_CUPY = False


class SubtractPlugin(Plugin):
    """
    Plugin for subtracting one point cloud from another.

    Creates a mask that identifies points in the first point cloud
    that are not present in the second point cloud.
    Only supports exact point matching.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The name "subtract"
        """
        return "subtract"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for the subtraction operation.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        # Get the data nodes manager from global variables
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes

        # Get all node UUIDs and names for the dropdown
        node_options = {}
        for node_uid, node in data_nodes.data_nodes.items():
            node_options[str(node_uid)] = node.alias or node.params

        # Set default if options exist
        default_value = ""
        if node_options:
            default_value = next(iter(node_options))

        return {
            "subtract_node": {
                "type": "dropdown",
                "options": node_options,
                "default": default_value,
                "label": "Branch to Subtract",
                "description": "Branch to subtract from the selected branch"
            }
        }

    def confirm_before_execute(self, data_node, params):
        """
        Warn the user when this subtract can't use the fast lineage path and
        will have to compare points by coordinate (potentially very slow).

        Runs on the main thread before the worker starts. It reuses
        ``_lineage_keep_mask`` -- which only inspects lineage and mask arrays,
        never reconstructs or matches coordinates -- so it's cheap. Returns a
        confirmation message only when the fast path is unavailable.
        """
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes
        controller = global_variables.global_application_controller
        if data_nodes is None or controller is None:
            return None

        try:
            subtract_node = data_nodes.get_node(uuid.UUID(params["subtract_node"]))
        except Exception:
            return None
        if subtract_node is None:
            return None

        n_target = controller.get_node_reconstructed_count(data_node)
        if n_target is None:
            return None  # Can't reason cheaply; let execute() handle it.

        keep_mask = self._lineage_keep_mask(data_node, subtract_node, data_nodes, n_target)
        if keep_mask is not None:
            return None  # Fast lineage path available -- no prompt needed.

        n_sub = controller.get_node_reconstructed_count(subtract_node) or 0
        target_name = getattr(data_node, "alias", None) or "the selected branch"
        sub_name = getattr(subtract_node, "alias", None) or "the subtract branch"
        return (
            f"“{sub_name}” doesn't share a compatible lineage with “{target_name}”, "
            f"so subtract must match points by coordinate "
            f"(~{n_target:,} vs {n_sub:,} points). This can be slow.\n\n"
            f"Proceed with the slow exact match?"
        )

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the subtraction between two point clouds.

        Args:
            data_node (DataNode): The target data node (branch to subtract from)
            params (Dict[str, Any]): Parameters for the operation

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the result of subtraction
                - Result type identifier "masks"
                - List containing the data_node UIDs as dependencies
        """
        # Get the data nodes manager from global variables
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes
        controller = global_variables.global_application_controller

        # Get the target point cloud
        target_pc = data_node.data
        target_points = target_pc.points

        try:
            subtract_uid = uuid.UUID(params["subtract_node"])
            subtract_node = data_nodes.get_node(subtract_uid)

            if subtract_node is None:
                raise ValueError(f"Branch with UUID {subtract_uid} not found")
        except Exception as e:
            raise ValueError(f"Error processing branch to subtract: {str(e)}")

        # Fast path: when the subtract branch shares lineage with the target and
        # was carved out by boolean masks (e.g. ground extracted from the same
        # normal-estimated branch), the result is just the inverse of those masks.
        # This composes a boolean mask directly -- no reconstruction, no
        # coordinate matching -- and is exact. Returns None if the lineage isn't
        # a pure subset chain, in which case we fall back to exact matching.
        keep_mask = self._lineage_keep_mask(data_node, subtract_node, data_nodes, len(target_points))
        if keep_mask is not None:
            dependencies = [data_node.uid, subtract_uid]
            return Masks(keep_mask), "masks", dependencies

        # Slow path: reconstruct the subtract branch and match points by coordinate.
        try:
            # Use the controller to reconstruct the point cloud (thread-safe: read-only)
            global_variables.global_progress = (None, "Reconstructing subtract branch...")
            subtract_pc = controller.reconstruct(subtract_uid)
            subtract_points = subtract_pc.points
        except Exception as e:
            raise ValueError(f"Error processing branch to subtract: {str(e)}")

        # Ensure the point clouds have the same dimensionality
        if target_points.shape[1] != subtract_points.shape[1]:
            raise ValueError("Point dimensions do not match between the two point clouds")

        if global_variables.global_cancel_event.is_set():
            raise RuntimeError("Subtract cancelled.")

        global_variables.global_progress = (50, f"Comparing {len(target_points):,} vs {len(subtract_points):,} points...")

        # Ensure C-contiguous and a common dtype so view-based row hashing is safe.
        common_dtype = np.promote_types(target_points.dtype, subtract_points.dtype)
        A = np.ascontiguousarray(target_points, dtype=common_dtype)
        B = np.ascontiguousarray(subtract_points, dtype=common_dtype)

        mask = self._row_isin_not(A, B)

        if global_variables.global_cancel_event.is_set():
            raise RuntimeError("Subtract cancelled.")

        # Create a Masks object with the result
        result_mask = Masks(mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid, subtract_uid]
        return result_mask, "masks", dependencies

    @classmethod
    def _lineage_keep_mask(cls, target_node, subtract_node, data_nodes, n_target):
        """
        Try to compute the subtraction result as a boolean keep-mask over the
        target's points using lineage alone -- no reconstruction, no coordinate
        matching.

        Both the target and the subtract branch are expressed as membership
        masks over their lowest common ancestor (``inX`` = which ancestor points
        survive into branch X), composed from pure subset chains (``masks`` /
        ``class_reference`` selections plus attribute-only identity transforms,
        see ``branch_lineage.IDENTITY_TYPES``). The result is then
        ``~in_subtract`` restricted to the target's points::

            keep_over_target = ~in_subtract[flatnonzero(in_target)]

        Crucially this does NOT require the target to be the ancestor itself --
        it also works when the target is a *subset* of the shared ancestor (e.g.
        subtracting "vertical features" from a "non-ground" branch when both are
        masks descending from the same cloud). The older formulation bailed in
        that case because the ancestor -> target path wasn't pure identity.

        Returns:
            np.ndarray | None: Boolean keep-mask over the target points (True =
            keep, i.e. not present in the subtract branch), or None when the
            lineage isn't a pure subset chain and the caller must fall back to
            exact coordinate matching.
        """
        from config.config import global_variables
        from core.services.branch_lineage import (
            lowest_common_ancestor, membership_mask_over_ancestor,
        )

        controller = global_variables.global_application_controller
        if controller is None:
            return None

        # During execute() the AnalysisExecutor hands us a *temporary*
        # reconstructed node whose parent_uid is None (data_type "point_cloud"),
        # which would sever the target's lineage walk. Resolve the real tree node
        # by uid to recover its ancestry. (In confirm_before_execute the node is
        # already the real one, so this is a no-op there.)
        target_node = data_nodes.get_node(target_node.uid) or target_node

        ancestor = lowest_common_ancestor([target_node, subtract_node], data_nodes)
        if ancestor is None:
            return None  # No shared ancestor -> can't reason about lineage.

        n_ancestor = controller.get_node_reconstructed_count(ancestor)
        if n_ancestor is None:
            return None  # Can't size the ancestor cheaply -> fall back.

        in_target = membership_mask_over_ancestor(target_node, ancestor, data_nodes, n_ancestor)
        in_subtract = membership_mask_over_ancestor(subtract_node, ancestor, data_nodes, n_ancestor)
        if in_target is None or in_subtract is None:
            return None  # Lineage isn't a pure subset chain -> fall back.

        # Ancestor indices retained by the target, in the target's point order.
        idx_target = np.flatnonzero(in_target)
        if idx_target.shape[0] != n_target:
            return None  # Safety: membership must line up with the target's points.

        # Keep target points that are NOT present in the subtract branch.
        return ~in_subtract[idx_target]

    @staticmethod
    def _row_isin_not(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Boolean mask of rows in A that are NOT present in B (exact match).

        On GPU (CuPy): runs the full pipeline -- transfer, row-wise unique,
        and isin -- on the device. On CPU: reduces rows to int64 ids via
        np.unique on a stacked void-view, then np.isin(kind='table').
        """
        n = len(A)
        if n == 0:
            return np.zeros(0, dtype=bool)
        if len(B) == 0:
            return np.ones(n, dtype=bool)

        if _HAS_CUPY:
            try:
                return SubtractPlugin._row_isin_not_gpu(A, B)
            except Exception as exc:
                logger.warning(
                    "Subtract: CuPy path failed (%s); falling back to NumPy.", exc
                )

        return SubtractPlugin._row_isin_not_cpu(A, B)

    # Rows processed per cancel-checked chunk. Bounds how long a Cancel click
    # takes to take effect on the exact-match path (the heavy fallback).
    _MATCH_CHUNK = 2_000_000

    @staticmethod
    def _row_isin_not_gpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        from config.config import global_variables

        cancel_event = global_variables.global_cancel_event
        n = len(A)

        # Deduplicate B once, then test A in bounded chunks so a Cancel click is
        # honoured between chunks instead of only after one giant unique() call.
        B_unique = _cp.unique(_cp.asarray(B), axis=0)
        if cancel_event.is_set():
            raise RuntimeError("Subtract cancelled.")

        mask = np.empty(n, dtype=bool)
        for start in range(0, n, SubtractPlugin._MATCH_CHUNK):
            if cancel_event.is_set():
                raise RuntimeError("Subtract cancelled.")
            end = min(start + SubtractPlugin._MATCH_CHUNK, n)
            chunk = _cp.asarray(A[start:end])
            stacked = _cp.concatenate([chunk, B_unique], axis=0)
            # cp.unique(axis=0) deduplicates whole rows; return_inverse gives a
            # shared id space for the chunk's rows and B's rows.
            _, inv = _cp.unique(stacked, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            c_ids = inv[: end - start]
            b_ids = inv[end - start:]
            mask[start:end] = _cp.asnumpy(~_cp.isin(c_ids, b_ids))
        return mask

    @staticmethod
    def _row_isin_not_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        from config.config import global_variables

        cancel_event = global_variables.global_cancel_event
        n = len(A)

        row_bytes = A.dtype.itemsize * A.shape[1]
        void_dtype = np.dtype((np.void, row_bytes))

        # Reduce each row to a single void scalar so we can use np.isin. Sort B's
        # unique row-keys once, then test A in bounded chunks with cancel checks.
        B_keys = np.unique(np.ascontiguousarray(B).reshape(-1).view(void_dtype))
        if cancel_event.is_set():
            raise RuntimeError("Subtract cancelled.")

        A_void = np.ascontiguousarray(A).reshape(-1).view(void_dtype)
        mask = np.empty(n, dtype=bool)
        for start in range(0, n, SubtractPlugin._MATCH_CHUNK):
            if cancel_event.is_set():
                raise RuntimeError("Subtract cancelled.")
            end = min(start + SubtractPlugin._MATCH_CHUNK, n)
            mask[start:end] = ~np.isin(A_void[start:end], B_keys)
        return mask