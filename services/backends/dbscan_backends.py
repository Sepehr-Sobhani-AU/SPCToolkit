"""
DBSCAN Backend Implementations

Provides GPU (cuML) and CPU (sklearn, Open3D) backends for DBSCAN clustering.
"""

import logging
import numpy as np
import time
from .base import DBSCANBackend

logger = logging.getLogger(__name__)


class CuMLDBSCAN(DBSCANBackend):
    """GPU-accelerated DBSCAN using NVIDIA cuML (RAPIDS)."""

    @property
    def name(self) -> str:
        return "cuML"

    @property
    def is_gpu(self) -> bool:
        return True

    def _check_gpu_memory(self, points: np.ndarray) -> bool:
        """
        Check if GPU has enough memory for DBSCAN.

        DBSCAN on GPU requires significant memory:
        - Input points array
        - Distance matrix (can be O(n^2) in worst case, but cuML is smarter)
        - Labels array
        - Internal data structures

        Args:
            points: Input points array

        Returns:
            bool: True if enough memory, False otherwise
        """
        from services.hardware_detector import HardwareDetector

        # Estimate required memory
        # cuML DBSCAN is memory efficient but still needs:
        # - Points: n * 3 * 4 bytes (float32)
        # - Labels: n * 4 bytes (int32)
        # - Internal: roughly 5-10x points size for KD-tree and neighbors
        n_points = len(points)
        points_mb = (n_points * 3 * 4) / (1024 * 1024)
        labels_mb = (n_points * 4) / (1024 * 1024)
        internal_mb = points_mb * 10  # Conservative estimate

        total_required_mb = points_mb + labels_mb + internal_mb

        free_mb = HardwareDetector.get_free_gpu_memory_mb()

        if free_mb == 0:
            logger.warning("Could not determine GPU memory - proceeding with caution")
            return True

        if total_required_mb > free_mb:
            logger.warning(
                f"Insufficient GPU memory for DBSCAN: need ~{total_required_mb:.0f} MB, "
                f"have {free_mb} MB free. Falling back to CPU."
            )
            return False

        logger.debug(f"GPU memory check passed for DBSCAN: need ~{total_required_mb:.0f} MB, have {free_mb} MB free")
        return True

    def run(self, points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """Run DBSCAN clustering using cuML on GPU."""
        # Check GPU memory before proceeding
        if not self._check_gpu_memory(points):
            # Fall back to sklearn CPU implementation
            logger.info("DBSCAN falling back to CPU (scikit-learn) due to memory constraints")
            fallback = SklearnDBSCAN()
            return fallback.run(points, eps, min_samples)

        self.log_execution("DBSCAN")

        from cuml.cluster import DBSCAN as cumlDBSCAN
        import cupy as cp
        import cuml

        start_time = time.time()
        print(f"Using GPU-accelerated cuML DBSCAN (version {cuml.__version__})")
        print(f"Processing {len(points):,} points with eps={eps}, min_samples={min_samples}")

        try:
            # Transfer data to GPU
            points_gpu = cp.asarray(points, dtype=cp.float32)

            # Create and run GPU DBSCAN
            db = cumlDBSCAN(eps=eps, min_samples=min_samples)
            labels_gpu = db.fit_predict(points_gpu)

            # Transfer results back to CPU as numpy array
            labels = cp.asnumpy(labels_gpu).astype(np.int32)

            # Clean up GPU memory
            del points_gpu, labels_gpu
            cp.get_default_memory_pool().free_all_blocks()

            # Report results
            elapsed_time = time.time() - start_time
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)

            print(f"\n{'='*60}")
            print(f"DBSCAN COMPLETED (cuML GPU)")
            print(f"{'='*60}")
            print(f"  Total points:     {len(points):,}")
            print(f"  Clusters found:   {n_clusters}")
            print(f"  Noise points:     {n_noise:,} ({100*n_noise/len(points):.1f}%)")
            print(f"  Processing time:  {elapsed_time:.2f} seconds")
            print(f"{'='*60}\n")

            return labels

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.error(f"GPU out of memory during DBSCAN: {e}")
            logger.info("Falling back to CPU (scikit-learn)")
            # Clean up any partial GPU allocations
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            # Fall back to CPU
            fallback = SklearnDBSCAN()
            return fallback.run(points, eps, min_samples)


class SklearnDBSCAN(DBSCANBackend):
    """CPU DBSCAN using scikit-learn (multi-threaded)."""

    @property
    def name(self) -> str:
        return "scikit-learn"

    @property
    def is_gpu(self) -> bool:
        return False

    def run(self, points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """Run DBSCAN clustering using scikit-learn on CPU."""
        self.log_execution("DBSCAN")

        from sklearn.cluster import DBSCAN
        import sklearn

        start_time = time.time()
        print(f"Using scikit-learn DBSCAN (version {sklearn.__version__})")
        print(f"Processing {len(points):,} points with eps={eps}, min_samples={min_samples}")

        # Create and run the DBSCAN algorithm with all CPU cores
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(points)

        # Report results
        elapsed_time = time.time() - start_time
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        print(f"\n{'='*60}")
        print(f"DBSCAN COMPLETED (scikit-learn CPU)")
        print(f"{'='*60}")
        print(f"  Total points:     {len(points):,}")
        print(f"  Clusters found:   {n_clusters}")
        print(f"  Noise points:     {n_noise:,} ({100*n_noise/len(points):.1f}%)")
        print(f"  Processing time:  {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")

        return labels


class Open3DDBSCAN(DBSCANBackend):
    """CPU DBSCAN using Open3D."""

    @property
    def name(self) -> str:
        return "Open3D"

    @property
    def is_gpu(self) -> bool:
        return False

    def run(self, points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """Run DBSCAN clustering using Open3D on CPU."""
        self.log_execution("DBSCAN")

        import open3d as o3d

        start_time = time.time()
        print(f"Using Open3D DBSCAN")
        print(f"Processing {len(points):,} points with eps={eps}, min_points={min_samples}")

        # Convert points to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Run DBSCAN
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=True))

        # Report results
        elapsed_time = time.time() - start_time
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        print(f"\n{'='*60}")
        print(f"DBSCAN COMPLETED (Open3D CPU)")
        print(f"{'='*60}")
        print(f"  Total points:     {len(points):,}")
        print(f"  Clusters found:   {n_clusters}")
        print(f"  Noise points:     {n_noise:,} ({100*n_noise/len(points):.1f}%)")
        print(f"  Processing time:  {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")

        return labels
