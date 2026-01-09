"""
DBSCAN Backend Implementations

Provides GPU (cuML) and CPU (sklearn, Open3D) backends for DBSCAN clustering.
"""

import numpy as np
import time
from .base import DBSCANBackend


class CuMLDBSCAN(DBSCANBackend):
    """GPU-accelerated DBSCAN using NVIDIA cuML (RAPIDS)."""

    @property
    def name(self) -> str:
        return "cuML"

    @property
    def is_gpu(self) -> bool:
        return True

    def run(self, points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
        """Run DBSCAN clustering using cuML on GPU."""
        self.log_execution("DBSCAN")

        from cuml.cluster import DBSCAN as cumlDBSCAN
        import cupy as cp
        import cuml

        start_time = time.time()
        print(f"Using GPU-accelerated cuML DBSCAN (version {cuml.__version__})")
        print(f"Processing {len(points):,} points with eps={eps}, min_samples={min_samples}")

        # Transfer data to GPU
        points_gpu = cp.asarray(points, dtype=cp.float32)

        # Create and run GPU DBSCAN
        db = cumlDBSCAN(eps=eps, min_samples=min_samples)
        labels_gpu = db.fit_predict(points_gpu)

        # Transfer results back to CPU as numpy array
        labels = cp.asnumpy(labels_gpu).astype(np.int32)

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
