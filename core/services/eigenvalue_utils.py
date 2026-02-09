# services/eigenvalue_utils.py
"""
Utility functions for computing eigenvalues from point clouds.

This module provides functionality for calculating and manipulating eigenvalues
derived from the covariance matrix of local neighborhoods in point clouds.

Uses PyTorch for GPU-accelerated batch eigenvalue computation.
"""

import numpy as np
import torch
from scipy.spatial import KDTree
from typing import Dict, Tuple, Optional, Union, List, Callable

from config.config import global_variables
from core.services.batch_processor import BatchProcessor


class EigenvalueUtils:
    """
    Utility class for computing and processing eigenvalues from point clouds.

    This class provides methods to compute eigenvalues of local neighborhoods,
    derive geometric features from eigenvalues, and visualize eigenvalue-based
    properties through color mappings.
    """

    def __init__(self, use_cpu: bool = False):
        """
        Initialize the EigenvalueUtils.

        Args:
            use_cpu (bool, optional): Flag to force CPU usage instead of GPU. Defaults to False.
        """
        self.use_cpu = use_cpu

        # Configure device
        if use_cpu:
            self.device = torch.device('cpu')
            print("EigenvalueUtils configured to use CPU only")
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"EigenvalueUtils using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("No GPU found, EigenvalueUtils will use CPU")

        # Caching for optimization
        self._last_computed_eigenvalues = None
        self._last_tree = None
        self._last_indices = None
        self._last_point_count = 0
        self._last_k = 0

    def clear_cache(self) -> None:
        """
        Clear cached data to free memory.

        Call this method when:
        - Processing a different point cloud
        - Parameters (k) have changed
        - Memory needs to be freed
        """
        self._last_computed_eigenvalues = None
        self._last_tree = None
        self._last_indices = None
        self._last_point_count = 0
        self._last_k = 0

    @staticmethod
    def _create_batches(total_size: int, batch_size: int) -> List[Tuple[int, int]]:
        """
        Create batch indices for processing.

        Args:
            total_size (int): Total number of points to process
            batch_size (int): Maximum size of each batch

        Returns:
            list: List of (start_idx, end_idx) tuples for each batch
        """
        num_batches = (total_size + batch_size - 1) // batch_size
        batches = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_size)
            batches.append((start_idx, end_idx))

        return batches

    def get_eigenvalues(
            self,
            data,
            k: int,
            smooth: bool = True,
            batch_size: Optional[int] = None,
            progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Compute the eigenvalues of the covariance matrix of k-nearest neighbor points for each point.

        This function supports input data in two forms:
        - A Clusters object, where it uses `data.points` for the points.
        - A numpy array directly representing 3D points.

        The computation is performed using a spatial grid-based batch processor for efficient
        handling of large point clouds.

        Parameters:
        - data (Clusters or np.ndarray): A Clusters object or a numpy array representing points.
        - k (int): The number of nearest neighbors to consider for each point.
        - smooth (bool, optional): If True, averages the eigenvalues across neighbors. Defaults to True.
        - batch_size (int, optional): The number of points to process in each batch. If None,
                                      the batch size is determined automatically.
        - progress_callback (callable, optional): A function to call with progress updates.

        Returns:
        - np.ndarray: A 2D array of eigenvalues for each point's neighborhood.
        """
        # Extract points based on input type
        if isinstance(data, np.ndarray):
            points = data
        elif hasattr(data, 'points'):
            points = data.points
        else:
            raise ValueError("The first parameter must be a Clusters object or a numpy array.")

        # Ensure points are float32 for better performance
        points = points.astype(np.float32)

        # Create batch processor with appropriate batch size
        point_count = len(points)

        # If batch_size is None, default to 250K for real spatial batching
        if batch_size is None:
            actual_batch_size = min(250000, point_count)
        else:
            actual_batch_size = min(batch_size, point_count)

        # Create a batch processor with 10% overlap to ensure smooth transitions between batches
        batch_processor = BatchProcessor(
            points=points,
            batch_size=actual_batch_size,
            overlap_percent=0.1
        )

        # Process the point cloud in spatial batches
        print(f"Computing eigenvalues for {point_count} points with k={k} using spatial batch processing...")

        eigenvalues = batch_processor.process_in_batches(
            processing_func=self._compute_eigenvalues,
            callback=progress_callback,
            k=k,
            smooth=smooth
        )
        return eigenvalues

    def compute_eigenvalues(
            self,
            points: np.ndarray,
            k: int = 20,
            smooth: bool = True,
            batch_size: int = 10000,
            progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Compute eigenvalues for each point based on its local neighborhood.

        Args:
            points (np.ndarray): Point cloud data as Nx3 array of (x,y,z) coordinates.
            k (int, optional): Number of nearest neighbors to consider. Defaults to 20.
            smooth (bool, optional): Whether to smooth eigenvalues. Defaults to True.
            batch_size (int, optional): Maximum points per batch for processing.
            progress_callback (Optional[Callable], optional): Function to report progress.

        Returns:
            np.ndarray: Nx3 array of eigenvalues for each point, sorted in ascending order.
        """
        # Check if points array is empty
        if len(points) == 0:
            return np.array([])

        # Validate parameters
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        # Adjust batch size if it's larger than the point cloud
        if batch_size > len(points):
            batch_size = len(points)

        # Ensure points are in the correct format
        points_float32 = points.astype(np.float32, copy=False)

        # Check if cache is valid (same point count and k value)
        cache_valid = (
            self._last_tree is not None and
            self._last_indices is not None and
            self._last_point_count == len(points) and
            self._last_k == k
        )

        if cache_valid:
            # Reuse cached indices
            indices = self._last_indices
        else:
            # Cache is invalid - clear old cache BEFORE creating new one
            # This avoids temporary memory duplication
            self.clear_cache()

            print("Building KD-tree and finding neighbors...")
            tree = KDTree(points_float32)
            self._last_tree = tree

            # Find k nearest neighbors for each point (including the point itself)
            distances, indices = tree.query(points_float32, k=k + 1)
            self._last_indices = indices
            self._last_point_count = len(points)
            self._last_k = k
            print("KD-tree built and neighbors found.")

        # Create batches for processing
        batches = self._create_batches(len(points), batch_size)

        # Initialize array to store all eigenvalues
        eigenvalues = np.zeros((len(points), 3), dtype=np.float32)

        # Choose computation method based on GPU availability and user preference
        if self.use_cpu:
            # Use optimized NumPy version for CPU
            self._compute_eigenvalues_numpy(points_float32, indices, batches, eigenvalues, progress_callback)
        else:
            # Use PyTorch with GPU acceleration
            self._compute_eigenvalues_pytorch(points_float32, indices, batches, eigenvalues, progress_callback)

        # Smooth eigenvalues if requested
        if smooth:
            if self.use_cpu:
                self._smooth_eigenvalues_numpy(eigenvalues, indices, batch_size)
            else:
                self._smooth_eigenvalues_pytorch(eigenvalues, indices, batch_size)

        # Cache the computed eigenvalues
        self._last_computed_eigenvalues = eigenvalues

        return eigenvalues

    def _compute_eigenvalues_pytorch(
            self,
            points: np.ndarray,
            neighbor_indices: np.ndarray,
            batches: List[Tuple[int, int]],
            eigenvalues: np.ndarray,
            progress_callback: Optional[Callable[[int, int], None]]
    ) -> None:
        """
        Compute eigenvalues using PyTorch operations in batches with GPU acceleration.

        Args:
            points (np.ndarray): Point cloud data as Nx3 array.
            neighbor_indices (np.ndarray): Indices of k+1 nearest neighbors for each point.
            batches (List[Tuple[int, int]]): List of (start_idx, end_idx) tuples for each batch.
            eigenvalues (np.ndarray): Pre-allocated array to store results.
            progress_callback (Optional[Callable]): Function to report progress.
        """
        # Convert points to PyTorch tensor on device
        points_torch = torch.from_numpy(points).to(self.device)

        # Total number of batches for progress reporting
        num_batches = len(batches)

        # Process batches
        for batch_idx, (start_idx, end_idx) in enumerate(batches):
            # Get neighbor indices for this batch (exclude the point itself at index 0)
            batch_neighbor_indices = neighbor_indices[start_idx:end_idx, 1:]

            try:
                # Convert indices to torch tensor
                indices_torch = torch.from_numpy(batch_neighbor_indices.astype(np.int64)).to(self.device)

                # Gather neighbors: (batch_size, k, 3)
                neighbors = points_torch[indices_torch]

                # Calculate centroid
                centroids = neighbors.mean(dim=1, keepdim=True)

                # Center the neighborhoods
                centered_neighbors = neighbors - centroids

                # Transpose for matrix multiplication: (batch_size, 3, k)
                centered_t = centered_neighbors.transpose(1, 2)

                # Calculate covariance matrices: (batch_size, 3, 3)
                k_float = float(centered_neighbors.shape[1])
                cov_matrices = torch.matmul(centered_t, centered_neighbors) / k_float

                # Compute eigenvalues using PyTorch
                batch_eigenvalues, _ = torch.linalg.eigh(cov_matrices)

                # Transfer results back to CPU
                eigenvalues[start_idx:end_idx] = batch_eigenvalues.cpu().numpy()

            except Exception as e:
                print(f"PyTorch error in batch {batch_idx + 1}/{num_batches}: {e}")
                print("Falling back to NumPy for this batch")

                # Fall back to NumPy for this batch
                for i in range(end_idx - start_idx):
                    point_idx = start_idx + i
                    neighbors = points[batch_neighbor_indices[i]]

                    centroid = np.mean(neighbors, axis=0)
                    centered_neighbors = neighbors - centroid

                    covariance = np.dot(centered_neighbors.T, centered_neighbors) / len(neighbors)

                    try:
                        evals = np.linalg.eigvalsh(covariance)
                        evals.sort()
                        eigenvalues[point_idx] = evals
                    except Exception as inner_e:
                        print(f"Error computing eigenvalues for point {point_idx}: {inner_e}")
                        eigenvalues[point_idx] = np.zeros(3)

            # Report progress if callback provided
            if progress_callback is not None and batch_idx % 5 == 0:
                try:
                    progress_callback(batch_idx + 1, num_batches)
                except Exception as e:
                    print(f"Error in progress callback: {e}")

    def _smooth_eigenvalues_pytorch(
            self,
            eigenvalues: np.ndarray,
            neighbor_indices: np.ndarray,
            batch_size: int = 10000
    ) -> None:
        """
        Smooth eigenvalues using PyTorch operations in batches.

        Args:
            eigenvalues (np.ndarray): Original eigenvalues to smooth (modified in-place).
            neighbor_indices (np.ndarray): Indices of neighbors for each point.
            batch_size (int): Maximum points per batch for processing.
        """
        # Create a copy of eigenvalues for safe smoothing
        original_eigenvalues = eigenvalues.copy()
        original_eigenvalues_torch = torch.from_numpy(original_eigenvalues).to(self.device)

        # Create batches for processing
        num_points = len(eigenvalues)
        batches = self._create_batches(num_points, batch_size)

        # Process each batch
        for start_idx, end_idx in batches:
            # Get the batch indices
            batch_indices = neighbor_indices[start_idx:end_idx]

            try:
                # Convert to PyTorch tensors
                batch_indices_torch = torch.from_numpy(batch_indices.astype(np.int64)).to(self.device)

                # Gather eigenvalues for all neighbors of each point
                neighbor_eigenvalues = original_eigenvalues_torch[batch_indices_torch]

                # Calculate mean eigenvalues for each neighborhood
                smoothed_batch = neighbor_eigenvalues.mean(dim=1)

                # Store results back in the original array
                eigenvalues[start_idx:end_idx] = smoothed_batch.cpu().numpy()

            except Exception as e:
                print(f"PyTorch error during smoothing: {e}")
                print("Falling back to NumPy for this batch")

                # Fall back to NumPy for this batch
                for i in range(start_idx, end_idx):
                    indices = neighbor_indices[i]
                    eigenvalues[i] = np.mean(original_eigenvalues[indices], axis=0)

    def _compute_eigenvalues(self, data, k: int, smooth: bool = True) -> np.ndarray:
        """
        Compute the eigenvalues of the covariance matrix of k-nearest neighbor points for each point.

        This is the internal method used by the BatchProcessor. Uses GPU acceleration via PyTorch
        for neighbor gathering, covariance computation, eigenvalue decomposition, and smoothing.
        Uses the backend registry for KNN queries (cuML on GPU when available).

        Parameters:
        - data (Clusters or np.ndarray): A Clusters object or a numpy array representing points.
        - k (int): The number of nearest neighbors to consider for each point.
        - smooth (bool, optional): If True, averages the eigenvalues across neighbors. Defaults to True.

        Returns:
        - np.ndarray: A 2D array of eigenvalues for each point's neighborhood.
        """
        # Extract points based on input type
        if isinstance(data, np.ndarray):
            point_cloud = data
        elif hasattr(data, 'points'):
            point_cloud = data.points
        else:
            raise ValueError("The first parameter must be a Clusters object or a numpy array.")

        point_cloud = point_cloud.astype(np.float32)

        # Ensure k is smaller than the number of points
        if k > len(point_cloud):
            eigenvalues = np.ones((len(point_cloud), 3))
            return eigenvalues

        # KNN query — use backend registry (cuML GPU when available, scipy CPU fallback)
        registry = global_variables.global_backend_registry
        if registry is not None:
            knn_backend = registry.get_knn()
            distances, indices = knn_backend.query(point_cloud, k=k)
        else:
            tree = KDTree(point_cloud)
            distances, indices = tree.query(point_cloud, k=k)

        # Move points and indices to GPU via PyTorch
        points_torch = torch.from_numpy(point_cloud).to(self.device)
        indices_torch = torch.from_numpy(indices.astype(np.int64)).to(self.device)

        # Gather KNN points on GPU: (num_points, k, 3)
        neighbors = points_torch[indices_torch]

        # Compute centroids on GPU: (num_points, 1, 3)
        centroids = neighbors.mean(dim=1, keepdim=True)

        # Center neighborhoods on GPU
        centered = neighbors - centroids

        # Transpose for batch matmul: (num_points, 3, k)
        centered_t = centered.transpose(1, 2)

        # Batch covariance matrices on GPU: (num_points, 3, 3)
        k_float = float(k - 1) if k > 1 else 1.0
        cov_matrices = torch.matmul(centered_t, centered) / k_float

        # Batch eigenvalue decomposition on GPU
        eigenvalues_torch, _ = torch.linalg.eigh(cov_matrices)

        if smooth:
            # Smooth eigenvalues on GPU using neighbor averaging
            neighbor_eigenvalues = eigenvalues_torch[indices_torch]  # (num_points, k, 3)
            avg_eigenvalues = neighbor_eigenvalues.mean(dim=1)  # (num_points, 3)
            return avg_eigenvalues.cpu().numpy()
        else:
            return eigenvalues_torch.cpu().numpy()

    def _compute_eigenvalues_numpy(
            self,
            points: np.ndarray,
            neighbor_indices: np.ndarray,
            batches: List[Tuple[int, int]],
            eigenvalues: np.ndarray,
            progress_callback: Optional[Callable[[int, int], None]]
    ) -> None:
        """
        Compute eigenvalues using optimized NumPy operations in batches.

        This method avoids GPU overhead for small point clouds.

        Args:
            points (np.ndarray): Point cloud data as Nx3 array.
            neighbor_indices (np.ndarray): Indices of k+1 nearest neighbors for each point.
            batches (List[Tuple[int, int]]): List of (start_idx, end_idx) tuples for each batch.
            eigenvalues (np.ndarray): Pre-allocated array to store results.
            progress_callback (Optional[Callable]): Function to report progress.
        """
        # Total number of batches for progress reporting
        num_batches = len(batches)

        # Process each batch
        for batch_idx, (start_idx, end_idx) in enumerate(batches):
            # Get neighbor indices for this batch (exclude the point itself at index 0)
            batch_neighbor_indices = neighbor_indices[start_idx:end_idx, 1:]

            # Vectorized computation for the batch
            # Gather neighbors: (batch_size, k, 3)
            neighbors = points[batch_neighbor_indices]

            # Compute centroids: (batch_size, 1, 3)
            centroids = np.mean(neighbors, axis=1, keepdims=True)

            # Center neighborhoods: (batch_size, k, 3)
            centered = neighbors - centroids

            # Transpose: (batch_size, 3, k)
            centered_t = centered.transpose(0, 2, 1)

            # Covariance matrices: (batch_size, 3, 3)
            k_points = centered.shape[1]
            cov_matrices = np.matmul(centered_t, centered) / k_points

            # Compute eigenvalues for each 3x3 matrix
            for i in range(end_idx - start_idx):
                point_idx = start_idx + i
                try:
                    evals = np.linalg.eigvalsh(cov_matrices[i])
                    evals.sort()
                    eigenvalues[point_idx] = evals
                except Exception as e:
                    print(f"Error computing eigenvalues for point {point_idx}: {e}")
                    eigenvalues[point_idx] = np.zeros(3)

            # Report progress if callback provided
            if progress_callback is not None and batch_idx % 10 == 0:
                try:
                    progress_callback(batch_idx + 1, num_batches)
                except Exception as e:
                    print(f"Error in progress callback: {e}")

    def _smooth_eigenvalues_numpy(
            self,
            eigenvalues: np.ndarray,
            neighbor_indices: np.ndarray,
            batch_size: int = 10000
    ) -> None:
        """
        Smooth eigenvalues by averaging with neighbors, processed in batches.
        Uses in-place NumPy operations for efficiency.

        Args:
            eigenvalues (np.ndarray): Original eigenvalues to smooth (modified in-place).
            neighbor_indices (np.ndarray): Indices of neighbors for each point.
            batch_size (int): Maximum points per batch for processing.
        """
        # Create a copy of eigenvalues for safe smoothing
        original_eigenvalues = eigenvalues.copy()

        # Create batches for processing
        num_points = len(eigenvalues)
        batches = self._create_batches(num_points, batch_size)

        # Process each batch
        for start_idx, end_idx in batches:
            for i in range(start_idx, end_idx):
                indices = neighbor_indices[i]
                eigenvalues[i] = np.mean(original_eigenvalues[indices], axis=0)

    def compute_geometric_features(
            self,
            eigenvalues: np.ndarray,
            batch_size: int = 100000
    ) -> Dict[str, np.ndarray]:
        """
        Compute geometric features from eigenvalues.

        Given eigenvalues λ₁ ≤ λ₂ ≤ λ₃, this method calculates metrics that
        describe the local geometric properties of the point cloud.

        Args:
            eigenvalues (np.ndarray): Nx3 array of eigenvalues for each point.
            batch_size (int, optional): Batch size for processing large arrays.

        Returns:
            Dict[str, np.ndarray]: Dictionary of geometric features.
        """
        # Ensure we have valid eigenvalues
        if eigenvalues is None or len(eigenvalues) == 0:
            return {
                "planarity": np.array([]),
                "linearity": np.array([]),
                "sphericity": np.array([]),
                "anisotropy": np.array([]),
                "total_variance": np.array([])
            }

        # Extract eigenvalues (λ₁ ≤ λ₂ ≤ λ₃)
        lambda1 = eigenvalues[:, 0]  # Smallest eigenvalue
        lambda2 = eigenvalues[:, 1]  # Middle eigenvalue
        lambda3 = eigenvalues[:, 2]  # Largest eigenvalue

        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        lambda3_safe = np.maximum(lambda3, epsilon)

        # Calculate geometric features - fully vectorized
        planarity = (lambda2 - lambda1) / lambda3_safe
        linearity = (lambda3 - lambda2) / lambda3_safe
        sphericity = lambda1 / lambda3_safe
        anisotropy = (lambda3 - lambda1) / lambda3_safe
        total_variance = lambda1 + lambda2 + lambda3

        # Handle invalid values
        planarity = np.nan_to_num(planarity)
        linearity = np.nan_to_num(linearity)
        sphericity = np.nan_to_num(sphericity)
        anisotropy = np.nan_to_num(anisotropy)

        # Return all features in a dictionary
        return {
            "planarity": planarity,
            "linearity": linearity,
            "sphericity": sphericity,
            "anisotropy": anisotropy,
            "total_variance": total_variance
        }

    def eigenvalues_to_colors(
            self,
            eigenvalues: np.ndarray
    ) -> np.ndarray:
        """
        Convert eigenvalues to RGB colors for visualization.

        This method maps eigenvalues to colors to visualize geometric properties.

        Args:
            eigenvalues (np.ndarray): Nx3 array of eigenvalues for each point.

        Returns:
            np.ndarray: Nx3 array of RGB colors in range [0,1].
        """
        # Check input
        if eigenvalues is None or len(eigenvalues) == 0:
            return np.array([])

        # Calculate geometric features
        features = self.compute_geometric_features(eigenvalues)

        # Initialize color array
        colors = np.zeros((len(eigenvalues), 3), dtype=np.float32)

        # Extract relevant features for coloring
        planarity = features["planarity"]
        linearity = features["linearity"]
        sphericity = features["sphericity"]

        # Map features to RGB values vectorized
        colors[:, 0] = planarity  # Red channel: planarity
        colors[:, 1] = linearity  # Green channel: linearity
        colors[:, 2] = sphericity * 3  # Blue channel: sphericity (scaled)

        # Normalize colors to [0,1] range
        max_val = np.max(colors)
        if max_val > 0:
            colors /= max_val

        # Ensure colors are in valid range
        np.clip(colors, 0, 1, out=colors)

        return colors
