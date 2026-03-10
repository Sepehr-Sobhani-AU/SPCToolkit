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


class KNNBackend(BaseBackend):
    """Abstract base class for K-Nearest Neighbors backends."""

    @abstractmethod
    def query(self, points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for each point.

        Args:
            points: (N, 3) array of XYZ coordinates
            k: Number of neighbors to find

        Returns:
            Tuple of:
                - distances: (N, k) array of distances to neighbors
                - indices: (N, k) array of neighbor indices
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
