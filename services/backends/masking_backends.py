"""
Masking Backend Implementations

Provides GPU (CuPy) and CPU (NumPy) backends for point cloud masking/filtering.
"""

import numpy as np
from .base import MaskingBackend


class CuPyMasking(MaskingBackend):
    """GPU-accelerated masking using CuPy."""

    @property
    def name(self) -> str:
        return "CuPy"

    @property
    def is_gpu(self) -> bool:
        return True

    def apply_mask(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply boolean mask to filter points using CuPy on GPU."""
        self.log_execution("Masking")

        import cupy as cp

        # Transfer to GPU
        points_gpu = cp.asarray(points)
        mask_gpu = cp.asarray(mask)

        # Apply mask on GPU
        result_gpu = points_gpu[mask_gpu]

        # Transfer back to CPU
        result = cp.asnumpy(result_gpu)

        # Clean up GPU memory
        del points_gpu, mask_gpu, result_gpu
        cp.get_default_memory_pool().free_all_blocks()

        return result

    def apply_mask_to_array(self, array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply boolean mask to any array using CuPy on GPU."""
        self.log_execution("Masking (array)")

        import cupy as cp

        # Transfer to GPU
        array_gpu = cp.asarray(array)
        mask_gpu = cp.asarray(mask)

        # Apply mask on GPU
        result_gpu = array_gpu[mask_gpu]

        # Transfer back to CPU
        result = cp.asnumpy(result_gpu)

        # Clean up GPU memory
        del array_gpu, mask_gpu, result_gpu
        cp.get_default_memory_pool().free_all_blocks()

        return result


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
