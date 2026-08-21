"""
Base Backend Classes

Abstract base classes for all backend implementations.
Each backend type (DBSCAN, KNN, Masking, Eigenvalue, Normal Estimation) has its own
abstract class that defines the interface all implementations must follow.
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


class BaseBackend(ABC):
    """Base class for all backends with common logging functionality."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging (e.g., 'cuML', 'sklearn')."""
        pass

    @property
    def is_gpu(self) -> bool:
        """Whether this backend uses GPU acceleration."""
        return False

    def log_execution(self, operation: str) -> None:
        """Log that this backend is running an operation."""
        device = "GPU" if self.is_gpu else "CPU"
        logger.info(f"{operation} running on {self.name} ({device})")


class DBSCANBackend(BaseBackend):
    """Abstract base class for DBSCAN clustering backends."""

    @abstractmethod
    def run(self, points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """
        Run DBSCAN clustering on point cloud.

        Args:
            points: (N, 3) array of XYZ coordinates
            eps: Maximum distance between points in a cluster
            min_samples: Minimum points to form a cluster

        Returns:
            np.ndarray: (N,) array of cluster labels (-1 for noise)
        """
        pass


class HDBSCANBackend(BaseBackend):
    """Abstract base class for HDBSCAN clustering backends."""

    @abstractmethod
    def run(self, points: np.ndarray, min_cluster_size: int, min_samples: int,
            cluster_selection_epsilon: float = 0.0, alpha: float = 1.0) -> np.ndarray:
        """
        Run HDBSCAN clustering on point cloud.

        Args:
            points: (N, 3) array of XYZ coordinates
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples for core point
            cluster_selection_epsilon: Distance threshold for cluster merging
            alpha: Distance scaling parameter

        Returns:
            np.ndarray: (N,) array of cluster labels (-1 for noise)
        """
        pass


class KNNBackend(BaseBackend):
    """Abstract base class for K-Nearest Neighbors backends."""

    @abstractmethod
    def query(self, points: np.ndarray, k: int, batch_size: int = 100_000,
              reference: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for each query point.

        Args:
            points: (N, 3) array of XYZ query coordinates
            k: Number of neighbors to find
            batch_size: Number of query points to process per batch
            reference: optional (M, 3) array to build the index on. When given,
                neighbors are searched in `reference` and the returned indices
                index into it (cloud-to-cloud query). When None (default), the
                index is built on `points` itself (self-query).

        Returns:
            Tuple of:
                - distances: (N, k) array of distances to neighbors
                - indices: (N, k) array of neighbor indices (into `reference`
                  when given, else into `points`)
        """
        pass


class MaskingBackend(BaseBackend):
    """Abstract base class for point cloud masking/filtering backends."""

    @abstractmethod
    def apply_mask(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply boolean mask to filter points.

        Args:
            points: (N, 3) array of XYZ coordinates
            mask: (N,) boolean array

        Returns:
            np.ndarray: Filtered points array
        """
        pass

    @abstractmethod
    def apply_mask_to_array(self, array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply boolean mask to any array (colors, normals, attributes).

        Args:
            array: (N, ...) array to filter
            mask: (N,) boolean array

        Returns:
            np.ndarray: Filtered array
        """
        pass


class ScreenSelectionBackend(BaseBackend):
    """Abstract base class for screen-space selection backends."""

    @abstractmethod
    def points_in_polygon(self, block: np.ndarray, coeffs: np.ndarray,
                          polygon: np.ndarray, bounds: tuple,
                          out: np.ndarray) -> np.ndarray:
        """
        Mark which points of one block land inside a screen polygon.

        Backends receive a block, never the whole cloud: the caller
        (``core.services.screen_selection``) owns the splitting, so that the
        memory behaviour is the same whichever backend runs.

        Args:
            block: (N, >=3) float32 world coordinates. Only xyz is read, so an
                interleaved xyz+rgb render buffer can be passed unchanged.
            coeffs: (14,) float32 from ``screen_selection.screen_coeffs`` —
                model-view, projection and viewport already folded together.
            polygon: (M, 2) float32 screen vertices, Qt widget coordinates.
            bounds: (min_x, max_x, min_y, max_y) of *polygon*, for the cheap
                reject before the edge loop.
            out: (N,) boolean array to write into, a view on the caller's
                full-size answer so no per-block allocation is needed.

        Returns:
            *out*, filled in. Points behind the camera are always False.
        """
        pass


class SpatialGridBackend(BaseBackend):
    """Abstract base class for spatial grid (cell numbering) backends.

    Deliberately separate from ``ScreenSelectionBackend``. Numbering points by
    the box they fall in has nothing to do with a screen, a camera or a lasso —
    it is what ``core.services.spatial_grid`` uses to index a cloud, and any
    service that needs a cheap spatial index uses it through there.
    """

    @abstractmethod
    def cell_ids(self, block: np.ndarray, lo: np.ndarray, inv_step: np.ndarray,
                 shape: tuple, out: np.ndarray) -> np.ndarray:
        """
        Number each point of one block by the grid cell it falls in.

        Backends receive a block, never the whole cloud: the caller
        (``core.services.spatial_grid``) owns the splitting, so the memory
        behaviour is the same whichever backend runs.

        Args:
            block: (N, >=3) float32 world coordinates. Only xyz is read, so an
                interleaved xyz+rgb render buffer can be passed unchanged.
            lo: (3,) float32 low corner of the cloud's bounding box.
            inv_step: (3,) float32 reciprocal of the cell size per axis, so the
                per-point maths is a multiply rather than a divide.
            shape: (nx, ny, nz) cells per axis. The cell number is
                ``(ix * ny + iy) * nz + iz``.
            out: (N,) integer array to write into. Its dtype sets the width —
                ``uint8`` when the grid has at most 256 cells, ``int32``
                otherwise. The caller chooses; backends must honour it.

        Returns:
            *out*, filled in. Points on the far boundary are clamped into the
            last cell rather than overflowing it, and non-finite coordinates go
            to cell 0.
        """
        pass

    @abstractmethod
    def block_bounds(self, block: np.ndarray) -> tuple:
        """
        Low and high corner of one block's xyz bounding box.

        Non-finite coordinates must be ignored rather than propagated: one NaN
        would otherwise make the box NaN on that axis, every cell number on it
        garbage, and the grid collapse to a handful of cells.

        Args:
            block: (N, >=3) float32 world coordinates.

        Returns:
            ``(lo, hi)``, each a (3,) float32 array. An axis whose every value
            in this block was non-finite comes back as ``+inf`` / ``-inf`` so
            the caller can combine blocks with plain min/max and notice at the
            end.
        """
        pass

    @abstractmethod
    def argsort(self, cell_ids: np.ndarray) -> np.ndarray:
        """
        Row indices of *cell_ids* ordered by cell — a counting sort in effect.

        Used to bucket a grid so that fetching one cell is a slice rather than a
        scan over the whole index. Must be stable, so that two grids built from
        the same points are byte-identical whichever backend ran.

        Args:
            cell_ids: (N,) uint8, uint16 or int32 cell number per point.

        Returns:
            (N,) int32 row indices. int32 rather than the platform default
            because it halves the index: 0.63 GB at 168M points, not 1.26 GB.
        """
        pass


class EigenvalueBackend(BaseBackend):
    """Abstract base class for eigenvalue computation backends."""

    @abstractmethod
    def compute_eigenvalues(
        self, points: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues for local point neighborhoods.

        Args:
            points: (N, 3) array of XYZ coordinates
            k: Number of neighbors for local covariance computation

        Returns:
            Tuple of:
                - eigenvalues: (N, 3) array of eigenvalues (sorted descending)
                - eigenvectors: (N, 3, 3) array of eigenvectors
        """
        pass


class NormalEstimationBackend(BaseBackend):
    """Abstract base class for normal estimation backends."""

    @abstractmethod
    def estimate_normals(
        self, points: np.ndarray, k: int, max_radius: float, batch_size: int = 50000
    ) -> np.ndarray:
        """
        Estimate normals for each point using hybrid KNN + radius search.

        Args:
            points: (N, 3) array of XYZ coordinates
            k: Maximum number of neighbors for KNN
            max_radius: Maximum search radius (inf for pure KNN)
            batch_size: Points per processing batch

        Returns:
            np.ndarray: (N, 3) array of unit normal vectors
        """
        pass
