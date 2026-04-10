"""
HDBSCAN Backend Implementations

Provides GPU (cuML) and CPU (sklearn) backends for HDBSCAN clustering.
"""

import logging
import numpy as np
import time
from .base import HDBSCANBackend

logger = logging.getLogger(__name__)


class CuMLHDBSCAN(HDBSCANBackend):
    """GPU-accelerated HDBSCAN using NVIDIA cuML (RAPIDS)."""

    @property
    def name(self) -> str:
        return "cuML"

    @property
    def is_gpu(self) -> bool:
        return True

    def _check_gpu_memory(self, points: np.ndarray) -> bool:
        """
        Check if GPU has enough memory for HDBSCAN.

        Args:
            points: Input points array

        Returns:
            bool: True if enough memory, False otherwise
        """
        from infrastructure.hardware_detector import HardwareDetector

        n_points = len(points)
        points_mb = (n_points * 3 * 4) / (1024 * 1024)
        labels_mb = (n_points * 4) / (1024 * 1024)
        internal_mb = points_mb * 10

        total_required_mb = points_mb + labels_mb + internal_mb

        free_mb = HardwareDetector.get_free_gpu_memory_mb()

        if free_mb == 0:
            logger.warning("Could not determine GPU memory - proceeding with caution")
            return True

        if total_required_mb > free_mb:
            logger.warning(
                f"Insufficient GPU memory for HDBSCAN: need ~{total_required_mb:.0f} MB, "
                f"have {free_mb} MB free. Falling back to CPU."
            )
            return False

        logger.debug(f"GPU memory check passed for HDBSCAN: need ~{total_required_mb:.0f} MB, have {free_mb} MB free")
        return True

    def run(self, points: np.ndarray, min_cluster_size: int, min_samples: int,
            cluster_selection_epsilon: float = 0.0, alpha: float = 1.0) -> np.ndarray:
        """Run HDBSCAN clustering using cuML on GPU."""
        # Too few points for clustering — label all as noise without touching GPU
        if len(points) < min_cluster_size:
            return np.full(len(points), -1, dtype=np.int32)

        if not self._check_gpu_memory(points):
            logger.info("HDBSCAN falling back to CPU (scikit-learn) due to memory constraints")
            fallback = SklearnHDBSCAN()
            return fallback.run(points, min_cluster_size, min_samples,
                                cluster_selection_epsilon, alpha)

        self.log_execution("HDBSCAN")

        from cuml.cluster import HDBSCAN as cumlHDBSCAN
        import cupy as cp
        import cuml

        start_time = time.time()
        print(f"Using GPU-accelerated cuML HDBSCAN (version {cuml.__version__})")
        print(f"Processing {len(points):,} points with min_cluster_size={min_cluster_size}, "
              f"min_samples={min_samples}")

        try:
            points_gpu = cp.asarray(points, dtype=cp.float32)

            clusterer = cumlHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                alpha=alpha
            )
            labels_gpu = clusterer.fit_predict(points_gpu)

            labels_gpu_int32 = labels_gpu.astype(cp.int32)
            labels = cp.asnumpy(labels_gpu_int32)

            del points_gpu, labels_gpu, labels_gpu_int32
            cp.get_default_memory_pool().free_all_blocks()

            elapsed_time = time.time() - start_time
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)

            print(f"\n{'='*60}")
            print(f"HDBSCAN COMPLETED (cuML GPU)")
            print(f"{'='*60}")
            print(f"  Total points:     {len(points):,}")
            print(f"  Clusters found:   {n_clusters}")
            print(f"  Noise points:     {n_noise:,} ({100*n_noise/len(points):.1f}%)")
            print(f"  Processing time:  {elapsed_time:.2f} seconds")
            print(f"{'='*60}\n")

            return labels

        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
            logger.error(f"GPU out of memory during HDBSCAN: {e}")
            logger.info("Falling back to CPU (scikit-learn)")
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            fallback = SklearnHDBSCAN()
            return fallback.run(points, min_cluster_size, min_samples,
                                cluster_selection_epsilon, alpha)


class SklearnHDBSCAN(HDBSCANBackend):
    """CPU HDBSCAN using scikit-learn (multi-threaded)."""

    @property
    def name(self) -> str:
        return "scikit-learn"

    @property
    def is_gpu(self) -> bool:
        return False

    def run(self, points: np.ndarray, min_cluster_size: int, min_samples: int,
            cluster_selection_epsilon: float = 0.0, alpha: float = 1.0) -> np.ndarray:
        """Run HDBSCAN clustering using scikit-learn on CPU."""
        if len(points) < min_cluster_size:
            return np.full(len(points), -1, dtype=np.int32)

        self.log_execution("HDBSCAN")

        from sklearn.cluster import HDBSCAN
        import sklearn

        start_time = time.time()
        print(f"Using scikit-learn HDBSCAN (version {sklearn.__version__})")
        print(f"Processing {len(points):,} points with min_cluster_size={min_cluster_size}, "
              f"min_samples={min_samples}")

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            alpha=alpha,
            n_jobs=-1
        )
        labels = clusterer.fit_predict(points)

        elapsed_time = time.time() - start_time
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        print(f"\n{'='*60}")
        print(f"HDBSCAN COMPLETED (scikit-learn CPU)")
        print(f"{'='*60}")
        print(f"  Total points:     {len(points):,}")
        print(f"  Clusters found:   {n_clusters}")
        print(f"  Noise points:     {n_noise:,} ({100*n_noise/len(points):.1f}%)")
        print(f"  Processing time:  {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")

        return labels
