"""
Normal Estimation Backend Implementations

Provides three backends for normal estimation:
- PyTorch CUDA: Primary GPU backend using KNN + batched covariance eigendecomposition
- Open3D CUDA: Fallback GPU backend using Open3D tensor pipeline
- Open3D CPU: CPU backend using Open3D legacy pipeline
"""

import numpy as np
import torch
import logging
from .base import NormalEstimationBackend

logger = logging.getLogger(__name__)


class PyTorchCUDANormals(NormalEstimationBackend):
    """GPU-accelerated normal estimation using PyTorch CUDA.

    Algorithm: KNN search via backend registry, then batched covariance
    eigendecomposition on GPU. The smallest eigenvector = surface normal.
    """

    def __init__(self):
        self.device = torch.device('cuda')

    @property
    def name(self) -> str:
        return "PyTorch CUDA"

    @property
    def is_gpu(self) -> bool:
        return True

    def estimate_normals(
        self, points: np.ndarray, k: int, max_radius: float, batch_size: int = 50000
    ) -> np.ndarray:
        self.log_execution("Normal estimation")
        from config.config import global_variables

        n_points = len(points)
        points = points.astype(np.float32)

        # Step 1: KNN query using backend registry
        global_variables.global_progress = (None, "Building KNN index...")
        registry = global_variables.global_backend_registry
        knn_backend = registry.get_knn()
        distances, indices = knn_backend.query(points, k=k)

        # Step 2: Compute normals in batches on GPU
        normals = np.zeros((n_points, 3), dtype=np.float32)
        use_radius = max_radius != float('inf')

        with torch.no_grad():
            points_torch = torch.from_numpy(points).to(self.device)

            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                bs = end - start

                batch_indices = torch.from_numpy(
                    indices[start:end].astype(np.int64)
                ).to(self.device)

                # Gather neighbor points: (batch, k, 3)
                neighbors = points_torch[batch_indices]

                if use_radius:
                    batch_distances = torch.from_numpy(
                        distances[start:end].astype(np.float32)
                    ).to(self.device)

                    # Hybrid radius mask
                    radius_mask = batch_distances < max_radius  # (batch, k)
                    valid_counts = radius_mask.sum(dim=1)  # (batch,)

                    # Fall back to all k neighbors if fewer than 3 within radius
                    fallback = valid_counts < 3
                    radius_mask[fallback] = True
                    valid_counts[fallback] = k
                    valid_counts = valid_counts.float()

                    weights = radius_mask.float().unsqueeze(-1)  # (batch, k, 1)
                else:
                    valid_counts = torch.full(
                        (bs,), float(k), device=self.device
                    )
                    weights = torch.ones(
                        (bs, k, 1), device=self.device
                    )

                # Weighted centroid
                weighted_neighbors = neighbors * weights
                centroids = weighted_neighbors.sum(dim=1, keepdim=True) / \
                    valid_counts.unsqueeze(-1).unsqueeze(-1)

                # Center and apply weights
                centered = (neighbors - centroids) * weights  # (batch, k, 3)

                # Covariance: (batch, 3, 3)
                cov = torch.bmm(centered.transpose(1, 2), centered) / \
                    valid_counts.unsqueeze(-1).unsqueeze(-1).clamp(min=1)

                # Eigendecomposition — eigh returns ascending eigenvalues
                _, evecs = torch.linalg.eigh(cov)

                # Normal = eigenvector of smallest eigenvalue (column 0)
                batch_normals = evecs[:, :, 0]  # (batch, 3)

                # Normalize to unit length
                norms = torch.linalg.norm(batch_normals, dim=1, keepdim=True).clamp(min=1e-8)
                batch_normals = batch_normals / norms

                normals[start:end] = batch_normals.cpu().numpy()

                percent = int((end / n_points) * 60)
                global_variables.global_progress = (
                    percent, f"Normal estimation: {end:,}/{n_points:,} points"
                )

        torch.cuda.empty_cache()
        return normals


class Open3DCUDANormals(NormalEstimationBackend):
    """GPU normal estimation using Open3D CUDA tensor pipeline.

    Fallback when PyTorch CUDA is unavailable but Open3D has CUDA support.
    Note: Open3D recommends specifying both radius and max_nn for CUDA devices.
    """

    @property
    def name(self) -> str:
        return "Open3D CUDA"

    @property
    def is_gpu(self) -> bool:
        return True

    def estimate_normals(
        self, points: np.ndarray, k: int, max_radius: float, batch_size: int = 50000
    ) -> np.ndarray:
        self.log_execution("Normal estimation")
        import open3d.core as o3c
        import open3d.t.geometry as tg

        device = o3c.Device("CUDA:0")
        points_tensor = o3c.Tensor(points.astype(np.float32), device=device)
        pcd = tg.PointCloud(points_tensor)

        if max_radius == float('inf'):
            pcd.estimate_normals(max_nn=k)
        else:
            pcd.estimate_normals(radius=max_radius, max_nn=k)

        return pcd.point.normals.cpu().numpy().astype(np.float32)


class Open3DNormals(NormalEstimationBackend):
    """CPU normal estimation using Open3D legacy pipeline."""

    @property
    def name(self) -> str:
        return "Open3D"

    def estimate_normals(
        self, points: np.ndarray, k: int, max_radius: float, batch_size: int = 50000
    ) -> np.ndarray:
        self.log_execution("Normal estimation")
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if max_radius == float('inf'):
            search_param = o3d.geometry.KDTreeSearchParamKNN(knn=k)
        else:
            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius=max_radius, max_nn=k
            )

        pcd.estimate_normals(search_param=search_param)
        return np.asarray(pcd.normals, dtype=np.float32)
