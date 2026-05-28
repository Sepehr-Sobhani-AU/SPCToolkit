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

        # Get the subtract node and reconstruct it to a point cloud
        try:
            subtract_uid = uuid.UUID(params["subtract_node"])
            subtract_node = data_nodes.get_node(subtract_uid)

            if subtract_node is None:
                raise ValueError(f"Branch with UUID {subtract_uid} not found")

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