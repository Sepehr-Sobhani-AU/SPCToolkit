"""
Coordinate Service

GPU-accelerated coordinate translation and shifting for import/export plugins.
"""

import logging
import uuid as _uuid
import numpy as np

logger = logging.getLogger(__name__)


def translate_and_convert(points_xyz, min_bound, colors):
    """
    Translate points to origin and convert to float32.
    Uses GPU (CuPy) when available for large arrays, falls back to CPU.
    """
    from infrastructure.hardware_detector import HardwareDetector

    if HardwareDetector.can_use_cupy():
        try:
            import cupy as cp
            pts_gpu = cp.asarray(points_xyz)
            mb_gpu = cp.asarray(min_bound)
            result_gpu = (pts_gpu - mb_gpu).astype(cp.float32)
            points_out = cp.asnumpy(result_gpu)
            del pts_gpu, mb_gpu, result_gpu

            if colors is not None:
                c_gpu = cp.asarray(colors).astype(cp.float32)
                colors_out = cp.asnumpy(c_gpu)
                del c_gpu
            else:
                colors_out = None

            logger.info(f"GPU-accelerated coordinate translation ({len(points_out)} pts)")
            return points_out, colors_out

        except Exception as e:
            logger.warning(f"GPU translation failed, falling back to CPU: {e}")

    points_out = (points_xyz - min_bound).astype(np.float32)
    colors_out = colors.astype(np.float32) if colors is not None else None
    return points_out, colors_out


def apply_shift(points_f32, shift):
    """
    Apply coordinate shift and convert to float64.
    Uses GPU (CuPy) when available, falls back to CPU.
    """
    from infrastructure.hardware_detector import HardwareDetector

    do_shift = shift is not None and np.any(shift != 0)

    if HardwareDetector.can_use_cupy():
        try:
            import cupy as cp
            pts_gpu = cp.asarray(points_f32).astype(cp.float64)
            if do_shift:
                pts_gpu += cp.asarray(shift, dtype=cp.float64)
            result = cp.asnumpy(pts_gpu)
            del pts_gpu
            logger.info(f"GPU-accelerated coordinate shift ({len(result)} pts)")
            return result
        except Exception as e:
            logger.warning(f"GPU shift failed, falling back to CPU: {e}")

    points = points_f32.astype(np.float64)
    if do_shift:
        points += shift.astype(np.float64)
    return points


def find_root_translation(data_nodes, uid_str: str) -> np.ndarray:
    """Walk up to the root PointCloud and return its translation."""
    node = data_nodes.get_node(_uuid.UUID(uid_str))
    visited = set()
    while node is not None and node.uid not in visited:
        visited.add(node.uid)
        if node.data_type == "point_cloud" and node.parent_uid is None:
            return getattr(node.data, 'translation', np.zeros(3))
        if node.parent_uid is None:
            break
        node = data_nodes.get_node(node.parent_uid)
    return np.zeros(3)
