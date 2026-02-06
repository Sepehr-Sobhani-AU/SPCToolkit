"""
LOD (Level-of-Detail) Manager for dynamic point cloud subsampling.

This module provides stateless utilities for computing optimal sample rates
and generating subsampled indices for large point cloud rendering.
"""

import numpy as np
import logging

from config.config import global_variables

logger = logging.getLogger(__name__)


class LODManager:
    """Stateless LOD computation utility using dynamic point budget."""

    # Default point budget for smooth interaction (tunable)
    DEFAULT_POINT_BUDGET = 50_000_000  # 50M points

    @staticmethod
    def compute_dynamic_point_budget() -> int:
        """
        Calculate max renderable points based on available RAM and VRAM.

        Uses unified budget from MemoryManager which considers BOTH:
        - RAM: 32 bytes/point (combined array + KDTree)
        - VRAM: 24 bytes/point (VBO)

        The more constrained resource determines the budget, preventing
        OOM crashes in either RAM or VRAM.

        Returns:
            Max points that fit in current available memory with safety margin.
        """
        from infrastructure.memory_manager import MemoryManager

        max_points, limiting, details = MemoryManager.compute_unified_point_budget()

        logger.debug(
            f"Dynamic point budget: {max_points:,} (limited by {limiting})"
        )

        return max_points

    @staticmethod
    def compute_sample_rate(total_points: int,
                            camera_distance: float,
                            zoom_factor: float,
                            max_extent: float,
                            point_budget: int = None) -> float:
        """
        Dynamically compute optimal sample rate.

        Args:
            total_points: Total number of points in full dataset
            camera_distance: Current camera distance
            zoom_factor: Current zoom factor
            max_extent: Maximum extent of point cloud bounding box
            point_budget: Target rendered points (default: 50M)

        Returns:
            sample_rate between 0.01 and 1.0
        """
        if total_points <= 0:
            return 1.0

        budget = point_budget or LODManager.DEFAULT_POINT_BUDGET

        # Base rate from point budget
        base_rate = min(1.0, budget / total_points)

        # If already at full resolution, no need to adjust
        if base_rate >= 1.0:
            return 1.0

        # Scale by zoom (closer = more detail allowed)
        if max_extent > 0:
            normalized_distance = (camera_distance * zoom_factor) / (max_extent * 2)
            # Close zoom (< 0.3): allow up to 2x budget
            # Far zoom (> 1.0): use base budget
            zoom_multiplier = max(0.5, min(2.0, 1.5 - normalized_distance))
            base_rate = min(1.0, base_rate * zoom_multiplier)

        return max(0.01, base_rate)  # Never go below 1%

    @staticmethod
    def _is_gpu_available() -> bool:
        """Check if GPU (CuPy) is available via backend registry."""
        backend_registry = global_variables.global_backend_registry
        if backend_registry is None:
            return False
        # Check if masking backend is GPU (CuPy)
        masking_backend = backend_registry.get_masking()
        return masking_backend.is_gpu

    @staticmethod
    def subsample_indices(n_points: int, sample_rate: float) -> np.ndarray:
        """
        Generate random indices for subsampling.

        Args:
            n_points: Total number of points
            sample_rate: Fraction of points to keep (0.01 to 1.0)

        Returns:
            Sorted array of indices to keep, or None if no subsampling needed
        """
        if sample_rate >= 1.0:
            return None  # No subsampling needed

        target_count = max(int(n_points * sample_rate), 1000)

        if target_count >= n_points:
            return None

        # Use GPU if available via backend registry
        if LODManager._is_gpu_available():
            return LODManager._subsample_indices_gpu(n_points, target_count)
        else:
            return LODManager._subsample_indices_cpu(n_points, target_count)

    @staticmethod
    def _subsample_indices_gpu(n_points: int, target_count: int) -> np.ndarray:
        """GPU-based random index generation using CuPy."""
        from infrastructure.memory_manager import MemoryManager
        import cupy as cp

        logger.debug(f"LOD GPU subsampling: {n_points:,} -> {target_count:,}")

        try:
            indices = cp.random.choice(n_points, size=target_count, replace=False)
            indices = cp.sort(indices)  # Better memory access pattern
            result = cp.asnumpy(indices)

            # Cleanup GPU memory
            del indices
            MemoryManager.cleanup()

            return result

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.warning(f"GPU OOM during LOD subsampling: {e}, falling back to CPU")
            MemoryManager.cleanup()
            return LODManager._subsample_indices_cpu(n_points, target_count)

    @staticmethod
    def _subsample_indices_cpu(n_points: int, target_count: int) -> np.ndarray:
        """CPU fallback using numpy."""
        logger.debug(f"LOD CPU subsampling: {n_points:,} -> {target_count:,}")
        indices = np.random.choice(n_points, size=target_count, replace=False)
        indices.sort()
        return indices
