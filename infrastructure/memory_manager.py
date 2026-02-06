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

    # Memory constants (bytes per point)
    BYTES_PER_POINT_VBO = 24         # 6 floats * 4 bytes (VRAM)
    BYTES_PER_POINT_KDTREE = 8       # Estimated cKDTree overhead (RAM)
    BYTES_PER_POINT_COMBINED = 24    # self.points array (RAM)
    BYTES_PER_POINT_TOTAL_RAM = 32   # Combined + KDTree (RAM)
    BYTES_PER_POINT_TOTAL_VRAM = 24  # VBO only (VRAM)

    # Safety margins
    RAM_SAFETY_MARGIN = 0.7   # Use 70% of available RAM
    VRAM_SAFETY_MARGIN = 0.7  # Use 70% of available VRAM

    # Point budget clamps
    MIN_POINT_BUDGET = 1_000_000     # At least 1M points
    MAX_POINT_BUDGET = 200_000_000   # Cap at 200M points

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
            from infrastructure.hardware_detector import HardwareDetector
            return HardwareDetector.get_free_gpu_memory_mb()
        except ImportError:
            logger.warning("HardwareDetector not available")
            return 0
        except Exception as e:
            logger.warning(f"Error checking GPU memory: {e}")
            return 0

    @staticmethod
    def compute_unified_point_budget() -> tuple:
        """
        Compute point budget considering both RAM and VRAM constraints.

        The budget is determined by the more constrained resource:
        - RAM: 32 bytes/point (combined array + KDTree)
        - VRAM: 24 bytes/point (VBO)

        Returns:
            Tuple of (max_points: int, limiting_resource: str, details: dict)
            - max_points: Maximum number of points that can be safely rendered
            - limiting_resource: "RAM" or "VRAM" indicating the bottleneck
            - details: Dict with ram_budget, vram_budget, ram_mb, vram_mb
        """
        ram_mb = MemoryManager.get_available_ram_mb()
        vram_mb = MemoryManager.get_available_gpu_mb()

        # Calculate RAM budget (32 bytes/point with safety margin)
        if ram_mb > 0:
            ram_bytes = ram_mb * 1024 * 1024
            ram_budget = int(
                (ram_bytes * MemoryManager.RAM_SAFETY_MARGIN)
                / MemoryManager.BYTES_PER_POINT_TOTAL_RAM
            )
        else:
            ram_budget = float('inf')  # Unknown, don't constrain

        # Calculate VRAM budget (24 bytes/point with safety margin)
        if vram_mb > 0:
            vram_bytes = vram_mb * 1024 * 1024
            vram_budget = int(
                (vram_bytes * MemoryManager.VRAM_SAFETY_MARGIN)
                / MemoryManager.BYTES_PER_POINT_TOTAL_VRAM
            )
        else:
            vram_budget = float('inf')  # Unknown, don't constrain

        # Details for logging/debugging
        details = {
            'ram_mb': ram_mb,
            'vram_mb': vram_mb,
            'ram_budget': ram_budget if ram_budget != float('inf') else None,
            'vram_budget': vram_budget if vram_budget != float('inf') else None,
        }

        # Choose the more constrained resource
        if ram_budget <= vram_budget:
            limiting = "RAM"
            max_points = ram_budget
        else:
            limiting = "VRAM"
            max_points = vram_budget

        # Handle case where both are unknown
        if max_points == float('inf'):
            max_points = MemoryManager.MAX_POINT_BUDGET
            limiting = "default"

        # Clamp to reasonable range
        max_points = max(
            MemoryManager.MIN_POINT_BUDGET,
            min(int(max_points), MemoryManager.MAX_POINT_BUDGET)
        )

        logger.debug(
            f"Unified point budget: {max_points:,} (limited by {limiting}, "
            f"RAM={ram_mb:,} MB, VRAM={vram_mb:,} MB)"
        )

        return (max_points, limiting, details)

    @staticmethod
    def estimate_render_memory(num_points: int, cached: bool = False) -> dict:
        """
        Estimate memory needed to render a given number of points.

        Memory breakdown:
        - RAM: combined array (24 bytes) + KDTree (8 bytes) = 32 bytes/point
        - VRAM: VBO only = 24 bytes/point
        - Overhead: 10% if cached, 30% if not cached (reconstruction temps)

        Args:
            num_points: Number of points to render
            cached: Whether all data is already cached in memory

        Returns:
            Dict with ram_mb, vram_mb, and breakdown details
        """
        # RAM: combined array + KDTree
        ram_base = num_points * MemoryManager.BYTES_PER_POINT_TOTAL_RAM

        # VRAM: VBO only
        vram_base = num_points * MemoryManager.BYTES_PER_POINT_TOTAL_VRAM

        if cached:
            # Cached: minimal overhead (direct assignment to viewer)
            ram_overhead = 1.1
        else:
            # Not cached: additional overhead for reconstruction operations
            # Reconstruction may create temporary PointCloud objects
            ram_overhead = 1.3

        ram_bytes = int(ram_base * ram_overhead)
        vram_bytes = vram_base  # VRAM doesn't have reconstruction overhead

        return {
            'ram_mb': int(ram_bytes / (1024 * 1024)),
            'vram_mb': int(vram_bytes / (1024 * 1024)),
            'ram_bytes': ram_bytes,
            'vram_bytes': vram_bytes,
            'num_points': num_points,
            'cached': cached,
        }

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
