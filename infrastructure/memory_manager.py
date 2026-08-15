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
    #
    # These decide how many points the app allows itself to draw, so they have
    # to match what it really allocates. Getting them wrong is worse than a
    # conservative guess: too high and the budget permits a cloud that does not
    # fit, and the machine swaps instead of subsampling.
    BYTES_PER_POINT_VBO = 24         # 6 floats * 4 bytes (VRAM)
    BYTES_PER_POINT_BRANCH = 24      # the branch's own Nx6 render slice (RAM)
    BYTES_PER_POINT_COMBINED = 24    # the concatenated self.points copy (RAM)
    BYTES_PER_POINT_PICK_GRID = 1    # one byte of cell id per point (RAM)

    # One visible branch: self.points aliases the branch slice instead of
    # copying it (see PCDViewerWidget._build_combined), so the points are held
    # once, plus the pick grid.
    BYTES_PER_POINT_TOTAL_RAM = BYTES_PER_POINT_BRANCH + BYTES_PER_POINT_PICK_GRID  # 25

    # Several visible branches: they genuinely have to be concatenated into one
    # array for global indexing, so the points are held twice.
    BYTES_PER_POINT_TOTAL_RAM_MULTI = (BYTES_PER_POINT_BRANCH
                                       + BYTES_PER_POINT_COMBINED
                                       + BYTES_PER_POINT_PICK_GRID)  # 49

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
    def compute_unified_point_budget(bytes_per_point_ram: int = None) -> tuple:
        """
        Compute point budget considering both RAM and VRAM constraints.

        The budget is determined by the more constrained resource:
        - RAM: 25 bytes/point with one visible branch, 49 with several
        - VRAM: 24 bytes/point (VBO)

        Args:
            bytes_per_point_ram: Override the RAM cost per point. Callers that
                know how many branches are visible should pass
                ``BYTES_PER_POINT_TOTAL_RAM`` (one branch, where self.points
                aliases the branch slice) or ``BYTES_PER_POINT_TOTAL_RAM_MULTI``
                (several, where it is a genuine second copy). Defaults to the
                single-branch figure, which is the normal case.

        Returns:
            Tuple of (max_points: int, limiting_resource: str, details: dict)
            - max_points: Maximum number of points that can be safely rendered
            - limiting_resource: "RAM" or "VRAM" indicating the bottleneck
            - details: Dict with ram_budget, vram_budget, ram_mb, vram_mb
        """
        ram_mb = MemoryManager.get_available_ram_mb()
        vram_mb = MemoryManager.get_available_gpu_mb()

        if bytes_per_point_ram is None:
            bytes_per_point_ram = MemoryManager.BYTES_PER_POINT_TOTAL_RAM

        # Calculate RAM budget with safety margin
        if ram_mb > 0:
            ram_bytes = ram_mb * 1024 * 1024
            ram_budget = int(
                (ram_bytes * MemoryManager.RAM_SAFETY_MARGIN)
                / bytes_per_point_ram
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
            'bytes_per_point_ram': bytes_per_point_ram,
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
    def estimate_render_memory(num_points: int, cached: bool = False,
                               bytes_per_point_ram: int = None) -> dict:
        """
        Estimate memory needed to render a given number of points.

        Memory breakdown:
        - RAM: branch slice (24 bytes) + pick grid (1 byte) = 25 bytes/point,
          or 49 when several branches force a concatenated second copy
        - VRAM: VBO only = 24 bytes/point
        - Overhead: 10% if cached, 30% if not cached (reconstruction temps)

        Args:
            num_points: Number of points to render
            cached: Whether all data is already cached in memory
            bytes_per_point_ram: Override the RAM cost per point, as in
                ``compute_unified_point_budget``.

        Returns:
            Dict with ram_mb, vram_mb, and breakdown details
        """
        if bytes_per_point_ram is None:
            bytes_per_point_ram = MemoryManager.BYTES_PER_POINT_TOTAL_RAM

        # RAM: branch slice (+ the combined copy when several are visible)
        ram_base = num_points * bytes_per_point_ram

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
