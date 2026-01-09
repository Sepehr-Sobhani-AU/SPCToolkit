"""
Eigenvalue Backend Implementations

Provides GPU (PyTorch CUDA) and CPU (PyTorch CPU) backends for eigenvalue computation.
These backends provide the core tensor operations; the full eigenvalue pipeline
is handled by EigenvalueUtils which uses these backends for device selection.
"""

import numpy as np
import torch
from typing import Tuple
from .base import EigenvalueBackend


class PyTorchCUDAEigen(EigenvalueBackend):
    """GPU-accelerated eigenvalue computation using PyTorch CUDA."""

    def __init__(self):
        self.device = torch.device('cuda')

    @property
    def name(self) -> str:
        return "PyTorch CUDA"

    @property
    def is_gpu(self) -> bool:
        return True

    def get_device(self) -> torch.device:
        """Get the torch device for this backend."""
        return self.device

    def compute_eigenvalues(
        self, points: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues for local point neighborhoods using GPU.

        This is a simplified implementation. For full functionality with
        batch processing, smoothing, and progress callbacks, use EigenvalueUtils.
        """
        self.log_execution("Eigenvalue computation")

        from scipy.spatial import KDTree

        print(f"Computing eigenvalues on GPU: {torch.cuda.get_device_name(0)}")
        print(f"Processing {len(points):,} points with k={k}")

        # Build KDTree for neighbor queries (CPU)
        tree = KDTree(points)
        _, indices = tree.query(points, k=k)

        # Convert to tensors and move to GPU
        points_tensor = torch.from_numpy(points).float().to(self.device)

        n_points = len(points)
        eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
        eigenvectors = np.zeros((n_points, 3, 3), dtype=np.float32)

        # Process in batches to manage GPU memory
        batch_size = 10000
        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)

            # Get neighbors for this batch
            batch_indices = indices[start_idx:end_idx]

            # Gather neighbor points
            neighbor_points = points_tensor[batch_indices]  # (batch, k, 3)

            # Compute centroids
            centroids = neighbor_points.mean(dim=1, keepdim=True)  # (batch, 1, 3)

            # Center the points
            centered = neighbor_points - centroids  # (batch, k, 3)

            # Compute covariance matrices: (batch, 3, 3)
            # cov = (1/k) * X^T @ X
            cov = torch.bmm(centered.transpose(1, 2), centered) / k

            # Compute eigenvalues and eigenvectors
            evals, evecs = torch.linalg.eigh(cov)

            # Sort eigenvalues in descending order
            sorted_indices = torch.argsort(evals, dim=1, descending=True)
            batch_size_actual = evals.shape[0]

            for i in range(batch_size_actual):
                idx = sorted_indices[i]
                eigenvalues[start_idx + i] = evals[i, idx].cpu().numpy()
                eigenvectors[start_idx + i] = evecs[i, :, idx].cpu().numpy()

        return eigenvalues, eigenvectors


class PyTorchCPUEigen(EigenvalueBackend):
    """CPU eigenvalue computation using PyTorch."""

    def __init__(self):
        self.device = torch.device('cpu')

    @property
    def name(self) -> str:
        return "PyTorch CPU"

    @property
    def is_gpu(self) -> bool:
        return False

    def get_device(self) -> torch.device:
        """Get the torch device for this backend."""
        return self.device

    def compute_eigenvalues(
        self, points: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues for local point neighborhoods using CPU.

        This is a simplified implementation. For full functionality with
        batch processing, smoothing, and progress callbacks, use EigenvalueUtils.
        """
        self.log_execution("Eigenvalue computation")

        from scipy.spatial import KDTree

        print(f"Computing eigenvalues on CPU")
        print(f"Processing {len(points):,} points with k={k}")

        # Build KDTree for neighbor queries
        tree = KDTree(points)
        _, indices = tree.query(points, k=k)

        # Convert to tensors
        points_tensor = torch.from_numpy(points).float()

        n_points = len(points)
        eigenvalues = np.zeros((n_points, 3), dtype=np.float32)
        eigenvectors = np.zeros((n_points, 3, 3), dtype=np.float32)

        # Process in batches
        batch_size = 10000
        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)

            # Get neighbors for this batch
            batch_indices = indices[start_idx:end_idx]

            # Gather neighbor points
            neighbor_points = points_tensor[batch_indices]  # (batch, k, 3)

            # Compute centroids
            centroids = neighbor_points.mean(dim=1, keepdim=True)  # (batch, 1, 3)

            # Center the points
            centered = neighbor_points - centroids  # (batch, k, 3)

            # Compute covariance matrices: (batch, 3, 3)
            cov = torch.bmm(centered.transpose(1, 2), centered) / k

            # Compute eigenvalues and eigenvectors
            evals, evecs = torch.linalg.eigh(cov)

            # Sort eigenvalues in descending order
            sorted_indices = torch.argsort(evals, dim=1, descending=True)
            batch_size_actual = evals.shape[0]

            for i in range(batch_size_actual):
                idx = sorted_indices[i]
                eigenvalues[start_idx + i] = evals[i, idx].numpy()
                eigenvectors[start_idx + i] = evecs[i, :, idx].numpy()

        return eigenvalues, eigenvectors
