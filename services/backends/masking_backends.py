"""
Masking Backend Implementations

Provides GPU (CuPy) and CPU (NumPy) backends for point cloud masking/filtering.
"""

import logging
import numpy as np
from .base import MaskingBackend

logger = logging.getLogger(__name__)


class CuPyMasking(MaskingBackend):
    """GPU-accelerated masking using CuPy."""

    @property
    def name(self) -> str:
        return "CuPy"

    @property
    def is_gpu(self) -> bool:
        return True

    def _estimate_gpu_memory_mb(self, *arrays) -> int:
        """Estimate GPU memory needed for arrays (with 4x overhead for operations)."""
        total_mb = 0
        for arr in arrays:
            if arr is not None:
                total_mb += arr.nbytes // (1024 * 1024) + 1
        return total_mb * 4  # 4x overhead for GPU operations

    def apply_mask(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply boolean mask to filter points using CuPy on GPU."""
        from services.memory_manager import MemoryManager

        # Check GPU memory using centralized manager
        required_mb = self._estimate_gpu_memory_mb(points, mask)
        if not MemoryManager.can_use_gpu(required_mb):
            return points[mask]  # CPU fallback

        self.log_execution("Masking")
        import cupy as cp

        try:
            points_gpu = cp.asarray(points)
            mask_gpu = cp.asarray(mask)
            result_gpu = points_gpu[mask_gpu]
            result = cp.asnumpy(result_gpu)

            del points_gpu, mask_gpu, result_gpu
            # Note: cleanup() removed - caller is responsible for cleanup after batch operations
            return result

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"GPU OOM during masking: {e}, falling back to CPU")
            MemoryManager.cleanup()  # Keep cleanup on OOM to recover memory
            return points[mask]

    def apply_mask_to_array(self, array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply boolean mask to any array using CuPy on GPU."""
        from services.memory_manager import MemoryManager

        # Check GPU memory using centralized manager
        required_mb = self._estimate_gpu_memory_mb(array, mask)
        if not MemoryManager.can_use_gpu(required_mb):
            return array[mask]  # CPU fallback

        self.log_execution("Masking (array)")
        import cupy as cp

        try:
            array_gpu = cp.asarray(array)
            mask_gpu = cp.asarray(mask)
            result_gpu = array_gpu[mask_gpu]
            result = cp.asnumpy(result_gpu)

            del array_gpu, mask_gpu, result_gpu
            # Note: cleanup() removed - caller is responsible for cleanup after batch operations
            return result

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"GPU OOM during array masking: {e}, falling back to CPU")
            MemoryManager.cleanup()  # Keep cleanup on OOM to recover memory
            return array[mask]


class NumpyMasking(MaskingBackend):
    """CPU masking using NumPy."""

    @property
    def name(self) -> str:
        return "NumPy"

    @property
    def is_gpu(self) -> bool:
        return False

    def apply_mask(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply boolean mask to filter points using NumPy on CPU."""
        self.log_execution("Masking")
        return points[mask]

    def apply_mask_to_array(self, array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply boolean mask to any array using NumPy on CPU."""
        self.log_execution("Masking (array)")
        return array[mask]
