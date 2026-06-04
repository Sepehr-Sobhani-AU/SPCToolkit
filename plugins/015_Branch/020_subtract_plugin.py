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

    # Node types whose transformer preserves point identity, order and count
    # (they only attach per-point attributes / colors or pass the cloud
    # through). cluster_labels is included because ClustersTransformer keeps the
    # parent's points unchanged and merely adds a per-point "cluster_labels"
    # attribute -- so it does NOT subset, even though clustering feels like it
    # should. Missing it here was forcing the slow coordinate-match fallback for
    # any subtract whose lineage crosses a clustering step.
    _IDENTITY_TYPES = frozenset({
        "values", "eigenvalues", "colors", "normals", "dist_to_ground",
        "cluster_labels", "container", "vector_feature", "cad_object",
    })

    @classmethod
    def _lineage_keep_mask(cls, target_node, subtract_node, data_nodes, n_target):
        """
        Try to compute the subtraction result as a boolean mask over the
        target's points using lineage alone, without reconstructing or matching
        coordinates.

        This succeeds only when the subtract branch is reachable from a common
        ancestor of the target through a pure subset chain (``masks`` nodes that
        select a boolean subset of their parent, plus identity nodes that merely
        attach attributes), AND the common-ancestor-to-target path is all
        identity (so the ancestor's points equal the target's points in order
        and count). Under those conditions the points selected by the subtract
        branch map directly to indices into the target's points.

        Returns:
            np.ndarray | None: Boolean keep-mask over the target points (True =
            keep, i.e. not present in the subtract branch), or None when the
            lineage isn't a pure subset chain and the caller must fall back to
            exact coordinate matching.
        """
        # Ancestor chain of the target, from target up to root.
        target_chain = []
        node = target_node
        while node is not None:
            target_chain.append(node)
            node = data_nodes.get_node(node.parent_uid) if node.parent_uid else None
        target_index = {n.uid: i for i, n in enumerate(target_chain)}

        # Walk the subtract branch up until we reach a node shared with the
        # target's ancestry (the common ancestor), collecting the subset chain.
        sub_path = []  # subtract -> ... -> (child of common ancestor)
        node = subtract_node
        common_pos = None
        while node is not None:
            if node.uid in target_index:
                common_pos = target_index[node.uid]
                break
            sub_path.append(node)
            node = data_nodes.get_node(node.parent_uid) if node.parent_uid else None
        if common_pos is None:
            return None  # No shared ancestor -> can't reason about lineage.

        # The common-ancestor -> target path must preserve point identity so the
        # composed mask (built over the ancestor) is valid over the target.
        for node in target_chain[:common_pos]:
            if node.data_type not in cls._IDENTITY_TYPES:
                return None

        # Compose the subtract subset chain into indices over the ancestor's
        # points (== target's points). Process top-down: ancestor -> subtract.
        idx = np.arange(n_target)
        for node in reversed(sub_path):
            data_type = node.data_type
            if data_type in cls._IDENTITY_TYPES:
                continue  # Count and order preserved; nothing to select.
            if data_type == "masks":
                mask = node.data.mask
                if mask.shape[0] != idx.shape[0]:
                    return None  # Lineage assumption violated; bail out safely.
                idx = idx[mask]
            elif data_type == "class_reference":
                # A class_reference selects points whose cluster label is in
                # cluster_ids. The labels come from the nearest cluster_labels
                # ancestor and line up positionally with the current selection.
                labels = cls._find_cluster_labels(node, data_nodes)
                if labels is None or labels.shape[0] != idx.shape[0]:
                    return None  # Can't align labels safely; fall back.
                sel = np.isin(labels, node.data.cluster_ids)
                idx = idx[sel]
            else:
                return None  # Unknown / reordering transform -> fall back.

        keep = np.ones(n_target, dtype=bool)
        keep[idx] = False  # Points belonging to the subtract branch are removed.
        return keep

    @staticmethod
    def _find_cluster_labels(node, data_nodes):
        """
        Walk up from ``node`` to the nearest ``cluster_labels`` ancestor and
        return its per-point integer labels (the array a class_reference filters
        on), or None if there's no such ancestor / no usable labels.
        """
        current = data_nodes.get_node(node.parent_uid) if node.parent_uid else None
        while current is not None:
            if current.data_type == "cluster_labels":
                return getattr(current.data, "labels", None)
            current = data_nodes.get_node(current.parent_uid) if current.parent_uid else None
        return None

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

    @staticmethod
    def _row_isin_not_gpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        from config.config import global_variables

        cancel_event = global_variables.global_cancel_event
        n = len(A)

        A_g = _cp.asarray(A)
        B_g = _cp.asarray(B)
        if cancel_event.is_set():
            raise RuntimeError("Subtract cancelled.")

        stacked = _cp.concatenate([A_g, B_g], axis=0)
        # cp.unique with axis=0 deduplicates whole rows on the device.
        _, inv = _cp.unique(stacked, axis=0, return_inverse=True)
        if cancel_event.is_set():
            raise RuntimeError("Subtract cancelled.")

        tgt_ids = inv[:n]
        sub_ids = inv[n:]
        mask_g = ~_cp.isin(tgt_ids, sub_ids)
        return _cp.asnumpy(mask_g)

    @staticmethod
    def _row_isin_not_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        from config.config import global_variables

        cancel_event = global_variables.global_cancel_event
        n = len(A)

        row_bytes = A.dtype.itemsize * A.shape[1]
        void_dtype = np.dtype((np.void, row_bytes))

        stacked = np.concatenate([A, B], axis=0)
        stacked_v = stacked.reshape(-1).view(void_dtype)
        if cancel_event.is_set():
            raise RuntimeError("Subtract cancelled.")

        _, inv = np.unique(stacked_v, return_inverse=True)
        if cancel_event.is_set():
            raise RuntimeError("Subtract cancelled.")

        tgt_ids = inv[:n]
        sub_ids = inv[n:]
        return ~np.isin(tgt_ids, sub_ids, kind='table')