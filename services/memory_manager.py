"""
Centralized Memory Management for Point Cloud Operations

This module provides a single source of truth for all memory-related operations,
including RAM and GPU memory checks, estimations, and cleanup.
"""

import gc
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """Centralized memory management for point cloud operations."""

    @staticmethod
    def get_available_ram_mb() -> int:
        """
        Get available system RAM in MB.

        Returns:
            Available RAM in MB, or 0 if unable to determine.
        """
        try:
            import psutil
            available = psutil.virtual_memory().available
            return int(available / (1024 * 1024))
        except ImportError:
            logger.warning("psutil not available, cannot check RAM")
            return 0
        except Exception as e:
            logger.warning(f"Error checking RAM: {e}")
            return 0

    @staticmethod
    def get_available_gpu_mb() -> int:
        """
        Get available GPU memory in MB.

        Returns:
            Available GPU memory in MB, or 0 if unable to determine.
        """
        try:
            from services.hardware_detector import HardwareDetector
            return HardwareDetector.get_free_gpu_memory_mb()
        except ImportError:
            logger.warning("HardwareDetector not available")
            return 0
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}")
            return 0

    @staticmethod
    def estimate_render_memory(num_points: int, cached: bool = False) -> int:
        """
        Estimate RAM needed to render a given number of points.

        Memory breakdown:
        - If cached: only combined array needed (6 floats * 4 bytes per point)
        - If not cached: points + colors + combined + 20% overhead

        Args:
            num_points: Number of points to render
            cached: Whether all data is already cached in memory

        Returns:
            Estimated memory requirement in MB
        """
        bytes_per_float = 4
        floats_per_point_xyz = 3

        if cached:
            # Only need the combined array (6 floats per point: xyz + rgb)
            combined_bytes = num_points * 6 * bytes_per_float
            # Add 20% overhead for safety
            total_bytes = combined_bytes * 1.2
        else:
            # Need: points array + colors array + combined array
            points_bytes = num_points * floats_per_point_xyz * bytes_per_float
            colors_bytes = num_points * floats_per_point_xyz * bytes_per_float
            combined_bytes = num_points * 6 * bytes_per_float
            # Add 20% overhead for intermediate operations
            total_bytes = (points_bytes + colors_bytes + combined_bytes) * 1.2

        return int(total_bytes / (1024 * 1024))

    @staticmethod
    def can_render(num_points: int, cached: bool = False) -> tuple:
        """
        Check if rendering is possible with available memory.

        This is a BLOCKING check - if memory is insufficient, the operation
        should not proceed.

        Args:
            num_points: Number of points to render
            cached: Whether all data is already cached in memory

        Returns:
            Tuple of (can_render: bool, message: str)
            - If can_render is True, message is empty
            - If can_render is False, message explains why
        """
        required_mb = MemoryManager.estimate_render_memory(num_points, cached)
        available_mb = MemoryManager.get_available_ram_mb()

        logger.debug(f"Memory check: {num_points:,} points, cached={cached}")
        logger.debug(f"  Required: {required_mb:,} MB, Available: {available_mb:,} MB")

        if available_mb == 0:
            # Cannot determine memory - allow operation with warning
            logger.warning("Cannot determine available RAM - proceeding anyway")
            return (True, "")

        if required_mb > available_mb:
            message = f"Need {required_mb:,} MB, only {available_mb:,} MB available"
            logger.warning(f"Insufficient memory: {message}")
            return (False, message)

        return (True, "")

    @staticmethod
    def can_use_gpu(required_mb: int) -> bool:
        """
        Check if GPU has enough memory for an operation.

        Args:
            required_mb: Required GPU memory in MB

        Returns:
            True if GPU can be used, False to fall back to CPU
        """
        available_mb = MemoryManager.get_available_gpu_mb()

        if available_mb == 0:
            # Cannot determine - try GPU anyway
            return True

        if required_mb > available_mb:
            logger.info(
                f"GPU memory low: need {required_mb:,} MB, have {available_mb:,} MB. "
                "Falling back to CPU."
            )
            return False

        return True

    @staticmethod
    def cleanup():
        """
        Force memory cleanup.

        This includes:
        - Python garbage collection
        - CuPy memory pool cleanup (if available)
        """
        # Python garbage collection
        gc.collect()

        # CuPy memory pool cleanup
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            logger.debug("CuPy memory pools cleared")
        except ImportError:
            pass  # CuPy not available
        except Exception as e:
            logger.debug(f"Error clearing CuPy memory: {e}")

        logger.debug("Memory cleanup completed")

    @staticmethod
    def log_memory_status(context: str = ""):
        """
        Log current memory status for debugging.

        Args:
            context: Optional context string to include in log
        """
        ram_mb = MemoryManager.get_available_ram_mb()
        gpu_mb = MemoryManager.get_available_gpu_mb()

        prefix = f"[{context}] " if context else ""
        logger.info(f"{prefix}Memory status: RAM={ram_mb:,} MB free, GPU={gpu_mb:,} MB free")
