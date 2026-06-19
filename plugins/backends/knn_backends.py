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

    def query(self, points: np.ndarray, k: int, batch_size: int = 500_000,
              reference: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using cuML on GPU.

        Builds the index on `reference` (or on `points` when None), then queries
        `points` in batches to avoid OOM on large point clouds.
        """
        self.log_execution("KNN")

        from cuml.neighbors import NearestNeighbors
        import cupy as cp
        import cuml

        start_time = time.time()
        n_points = len(points)
        print(f"Using GPU-accelerated cuML KNN (version {cuml.__version__})")
        print(f"Processing {n_points:,} points with k={k}")

        # Build the index on the reference set (self-query when reference is None)
        index_gpu = cp.asarray(points if reference is None else reference, dtype=cp.float32)
        model = NearestNeighbors(n_neighbors=k)
        model.fit(index_gpu)

        # Reuse the index array for self-query to avoid duplicating VRAM
        query_gpu = index_gpu if reference is None else cp.asarray(points, dtype=cp.float32)

        # Allocate output on CPU
        distances = np.empty((n_points, k), dtype=np.float32)
        indices = np.empty((n_points, k), dtype=np.int64)

        # Query in batches to keep GPU memory bounded
        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            query_batch = query_gpu[start:end]

            dist_gpu, idx_gpu = model.kneighbors(query_batch)

            distances[start:end] = cp.asnumpy(dist_gpu)
            indices[start:end] = cp.asnumpy(idx_gpu)

            cp.get_default_memory_pool().free_all_blocks()

        # Free GPU resources before returning
        del model, index_gpu, query_gpu
        cp.get_default_memory_pool().free_all_blocks()

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

    def query(self, points: np.ndarray, k: int, batch_size: int = 100_000,
              reference: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using scipy KDTree on CPU.

        Builds the index on `reference` (or on `points` when None), then queries
        `points` in batches to keep peak memory bounded for large point clouds.
        """
        self.log_execution("KNN")

        from scipy.spatial import KDTree

        start_time = time.time()
        n_points = len(points)
        print(f"Using scipy KDTree for KNN")
        print(f"Processing {n_points:,} points with k={k}, batch_size={batch_size:,}")

        tree = KDTree(points if reference is None else reference)

        distances = np.empty((n_points, k), dtype=np.float32)
        indices = np.empty((n_points, k), dtype=np.int64)

        for start in range(0, n_points, batch_size):
            end = min(start + batch_size, n_points)
            d, idx = tree.query(points[start:end], k=k)
            if k == 1:
                # scipy returns 1-D arrays for k=1; keep the (N, k) contract
                d = d[:, None]
                idx = idx[:, None]
            distances[start:end] = d.astype(np.float32)
            indices[start:end] = idx.astype(np.int64)

        elapsed_time = time.time() - start_time
        print(f"KNN completed in {elapsed_time:.2f} seconds (scipy CPU)")

        return distances, indices
