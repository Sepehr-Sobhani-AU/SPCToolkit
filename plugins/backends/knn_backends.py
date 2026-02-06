"""
KNN Backend Implementations

Provides GPU (cuML) and CPU (scipy) backends for K-Nearest Neighbors queries.
"""

import numpy as np
import time
from typing import Tuple
from .base import KNNBackend


class CuMLKNN(KNNBackend):
    """GPU-accelerated KNN using NVIDIA cuML (RAPIDS)."""

    @property
    def name(self) -> str:
        return "cuML"

    @property
    def is_gpu(self) -> bool:
        return True

    def query(self, points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using cuML on GPU."""
        self.log_execution("KNN")

        from cuml.neighbors import NearestNeighbors
        import cupy as cp
        import cuml

        start_time = time.time()
        print(f"Using GPU-accelerated cuML KNN (version {cuml.__version__})")
        print(f"Processing {len(points):,} points with k={k}")

        # Transfer data to GPU
        points_gpu = cp.asarray(points, dtype=cp.float32)

        # Create and fit the model
        model = NearestNeighbors(n_neighbors=k)
        model.fit(points_gpu)

        # Query neighbors
        distances_gpu, indices_gpu = model.kneighbors(points_gpu)

        # Transfer results back to CPU
        distances = cp.asnumpy(distances_gpu)
        indices = cp.asnumpy(indices_gpu)

        elapsed_time = time.time() - start_time
        print(f"KNN completed in {elapsed_time:.2f} seconds (cuML GPU)")

        return distances, indices


class ScipyKNN(KNNBackend):
    """CPU KNN using scipy KDTree."""

    @property
    def name(self) -> str:
        return "scipy"

    @property
    def is_gpu(self) -> bool:
        return False

    def query(self, points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using scipy KDTree on CPU."""
        self.log_execution("KNN")

        from scipy.spatial import KDTree

        start_time = time.time()
        print(f"Using scipy KDTree for KNN")
        print(f"Processing {len(points):,} points with k={k}")

        # Build KDTree and query
        tree = KDTree(points)
        distances, indices = tree.query(points, k=k)

        elapsed_time = time.time() - start_time
        print(f"KNN completed in {elapsed_time:.2f} seconds (scipy CPU)")

        return distances, indices
