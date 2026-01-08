# services/eigenvalue_utils2.py
"""
DEPRECATED: This file is a backup/older version.
Please use eigenvalue_utils.py instead, which has been updated to use PyTorch.

Utility functions for computing eigenvalues from point clouds.
"""

import numpy as np
import torch  # Replaced TensorFlow - but this file is deprecated
from scipy.spatial import KDTree
from typing import Dict, Tuple, Optional, Union, List, Callable

# NOTE: This file still contains TensorFlow-style code patterns but is not used.
# Use eigenvalue_utils.py for the updated PyTorch implementation.


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

        # Configure TensorFlow
        if use_cpu:
            # Force TensorFlow to use CPU
            try:
                tf.config.set_visible_devices([], 'GPU')
                print("TensorFlow configured to use CPU only")
            except Exception as e:
                print(f"Could not configure TensorFlow devices: {e}")
        else:
            # Check if GPU is available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Configure TensorFlow to use first GPU
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    print(f"TensorFlow using GPU: {gpus[0].name}")

                    # Enable mixed precision for better performance
                    self._enable_mixed_precision()
                except Exception as e:
                    print(f"Could not configure GPU memory growth: {e}")
            else:
                print("No GPU found, TensorFlow will use CPU")

        # Pre-compile TensorFlow functions
        self._compile_tf_functions()

        # Caching for optimization
        self._last_computed_eigenvalues = None
        self._last_tree = None
        self._last_indices = None

    @staticmethod
    def _enable_mixed_precision():
        """Enable mixed precision for TensorFlow operations if supported."""
        try:
            # Use mixed precision for improved performance on compatible GPUs
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            return True
        except (AttributeError, ValueError) as e:
            print(f"Mixed precision not supported: {e}")
            return False

    def _compile_tf_functions(self):
        """Pre-compile TensorFlow functions for better performance."""

        # Define and compile the function for computing covariance matrices
        @tf.function(jit_compile=True)
        def gather_and_process(points_tf, indices_tf):
            # Gather neighbors
            neighbors = tf.gather(points_tf, indices_tf)

            # Calculate centroid
            centroids = tf.reduce_mean(neighbors, axis=1, keepdims=True)

            # Center the neighborhoods
            centered_neighbors = neighbors - centroids

            # Transpose for matrix multiplication
            centered_t = tf.transpose(centered_neighbors, perm=[0, 2, 1])

            # Calculate covariance matrices
            k_float = tf.cast(tf.shape(centered_neighbors)[1], tf.float32)
            return tf.matmul(centered_t, centered_neighbors) / k_float

        # Define and compile function for eigenvalue computation
        @tf.function(jit_compile=True)
        def compute_eigenvalues(covariance_matrices):
            eigenvalues, _ = tf.linalg.eigh(covariance_matrices)
            return eigenvalues

        # Define and compile function for neighborhood smoothing
        @tf.function(jit_compile=True)
        def smooth_neighborhoods(eigenvalues_tf, indices_tf):
            # Gather eigenvalues for all neighbors of each point
            neighbor_eigenvalues = tf.gather(eigenvalues_tf, indices_tf)

            # Calculate mean eigenvalues for each neighborhood
            return tf.reduce_mean(neighbor_eigenvalues, axis=1)

        # Store the compiled functions
        self._gather_and_process = gather_and_process
        self._compute_eigenvalues_tf = compute_eigenvalues
        self._smooth_neighborhoods = smooth_neighborhoods

    @staticmethod
    def _create_batches(total_size, batch_size):
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

        # Memory optimization - only create the indices and tree once
        if self._last_tree is None or len(self._last_indices) != len(points) or self._last_indices.shape[1] != k + 1:
            print("Building KD-tree and finding neighbors...")
            tree = KDTree(points_float32)
            self._last_tree = tree

            # Find k nearest neighbors for each point (including the point itself)
            distances, indices = tree.query(points_float32, k=k + 1)
            self._last_indices = indices
            print("KD-tree built and neighbors found.")
        else:
            # Reuse indices if we're analyzing the same point cloud with the same k
            indices = self._last_indices

        # Create batches for processing
        batches = self._create_batches(len(points), batch_size)

        # Initialize array to store all eigenvalues
        eigenvalues = np.zeros((len(points), 3), dtype=np.float32)

        # Choose computation method based on GPU availability and user preference
        if self.use_cpu:
            # Use optimized NumPy version for CPU
            self._compute_eigenvalues_numpy(points_float32, indices, batches, eigenvalues, progress_callback)
        else:
            # Use TensorFlow with GPU acceleration
            self._compute_eigenvalues_tf_optimized(points_float32, indices, batches, eigenvalues, progress_callback)

        # Smooth eigenvalues if requested
        if smooth:
            if self.use_cpu:
                self._smooth_eigenvalues_numpy(eigenvalues, indices, batch_size)
            else:
                self._smooth_eigenvalues_tf(eigenvalues, indices, batch_size)

        # Cache the computed eigenvalues
        self._last_computed_eigenvalues = eigenvalues

        return eigenvalues

    def _compute_eigenvalues_tf_optimized(
            self,
            points: np.ndarray,
            neighbor_indices: np.ndarray,
            batches: List[Tuple[int, int]],
            eigenvalues: np.ndarray,
            progress_callback: Optional[Callable[[int, int], None]]
    ) -> None:
        """
        Compute eigenvalues using optimized TensorFlow operations in batches.

        This implementation uses pre-compiled TensorFlow functions and minimizes
        data transfer between CPU and GPU for better performance.

        Args:
            points (np.ndarray): Point cloud data as Nx3 array.
            neighbor_indices (np.ndarray): Indices of k+1 nearest neighbors for each point.
            batches (List[Tuple[int, int]]): List of (start_idx, end_idx) tuples for each batch.
            eigenvalues (np.ndarray): Pre-allocated array to store results.
            progress_callback (Optional[Callable]): Function to report progress.
        """
        # Convert points to TensorFlow tensor once (stays on GPU)
        points_tf = tf.convert_to_tensor(points, dtype=tf.float32)

        # Total number of batches for progress reporting
        num_batches = len(batches)

        # Process batches with larger sizes for GPU efficiency
        for batch_idx, (start_idx, end_idx) in enumerate(batches):
            # Get neighbor indices for this batch (exclude the point itself at index 0)
            batch_neighbor_indices = neighbor_indices[start_idx:end_idx, 1:]

            # Convert to TensorFlow tensor
            batch_neighbor_indices_tf = tf.convert_to_tensor(batch_neighbor_indices, dtype=tf.int32)

            try:
                # Use pre-compiled functions for better performance
                # Calculate covariance matrices
                covariance_matrices = self._gather_and_process(points_tf, batch_neighbor_indices_tf)

                # Calculate eigenvalues
                batch_eigenvalues_tf = self._compute_eigenvalues_tf(covariance_matrices)

                # Transfer results back to CPU only once per batch
                batch_eigenvalues_np = batch_eigenvalues_tf.numpy()

                # Store in the pre-allocated array
                eigenvalues[start_idx:end_idx] = batch_eigenvalues_np

            except tf.errors.InvalidArgumentError as e:
                print(f"TensorFlow error in batch {batch_idx + 1}/{num_batches}: {e}")
                print("Falling back to NumPy for this batch")

                # Fall back to NumPy for this batch
                for i in range(end_idx - start_idx):
                    point_idx = start_idx + i
                    # Get neighbors of the current point
                    neighbors = points[batch_neighbor_indices[i]]

                    # Center the neighborhood
                    centroid = np.mean(neighbors, axis=0)
                    centered_neighbors = neighbors - centroid

                    # Compute the covariance matrix
                    covariance = np.dot(centered_neighbors.T, centered_neighbors) / len(neighbors)

                    try:
                        # Compute eigenvalues
                        evals = np.linalg.eigvalsh(covariance)
                        evals.sort()  # Sort in ascending order
                        eigenvalues[point_idx] = evals
                    except Exception as inner_e:
                        print(f"Error computing eigenvalues for point {point_idx}: {inner_e}")
                        eigenvalues[point_idx] = np.zeros(3)

            # Report progress if callback provided
            if progress_callback is not None and batch_idx % 5 == 0:  # Report every 5 batches
                try:
                    progress_callback(batch_idx + 1, num_batches)
                except Exception as e:
                    print(f"Error in progress callback: {e}")

    def _smooth_eigenvalues_tf(
            self,
            eigenvalues: np.ndarray,
            neighbor_indices: np.ndarray,
            batch_size: int = 10000
    ) -> None:
        """
        Smooth eigenvalues using TensorFlow operations in batches.

        Args:
            eigenvalues (np.ndarray): Original eigenvalues to smooth (modified in-place).
            neighbor_indices (np.ndarray): Indices of neighbors for each point.
            batch_size (int): Maximum points per batch for processing.
        """
        # Create a copy of eigenvalues for safe smoothing
        original_eigenvalues = eigenvalues.copy()
        original_eigenvalues_tf = tf.convert_to_tensor(original_eigenvalues, dtype=tf.float32)

        # Create batches for processing
        num_points = len(eigenvalues)
        batches = self._create_batches(num_points, batch_size)

        # Process each batch
        for start_idx, end_idx in batches:
            # Get the batch indices
            batch_indices = neighbor_indices[start_idx:end_idx]

            # Convert to TensorFlow tensors
            batch_indices_tf = tf.convert_to_tensor(batch_indices, dtype=tf.int32)

            try:
                # Use pre-compiled function for smoothing
                smoothed_batch = self._smooth_neighborhoods(original_eigenvalues_tf, batch_indices_tf)

                # Store results back in the original array
                eigenvalues[start_idx:end_idx] = smoothed_batch.numpy()
            except tf.errors.InvalidArgumentError as e:
                print(f"TensorFlow error during smoothing: {e}")
                print("Falling back to NumPy for this batch")

                # Fall back to NumPy for this batch
                for i in range(start_idx, end_idx):
                    # Get indices of neighbors (including the point itself)
                    indices = neighbor_indices[i]

                    # Average eigenvalues of the point and its neighbors
                    eigenvalues[i] = np.mean(original_eigenvalues[indices], axis=0)

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
                # Get indices of neighbors (including the point itself)
                indices = neighbor_indices[i]

                # Average eigenvalues of the point and its neighbors
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

        This method avoids TensorFlow overhead for small 3×3 matrices.

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

            # Process each point in the batch efficiently
            for i in range(end_idx - start_idx):
                point_idx = start_idx + i
                # Get neighbors of the current point
                neighbors = points[batch_neighbor_indices[i]]

                # Center the neighborhood
                centroid = np.mean(neighbors, axis=0)
                centered_neighbors = neighbors - centroid

                # Compute the covariance matrix (3×3) - efficient for small matrices
                covariance = np.dot(centered_neighbors.T, centered_neighbors) / len(neighbors)

                try:
                    # Compute eigenvalues - very fast for 3×3 matrices
                    evals = np.linalg.eigvalsh(covariance)
                    evals.sort()  # Sort in ascending order
                    eigenvalues[point_idx] = evals
                except Exception as e:
                    # Handle numerical issues gracefully
                    print(f"Error computing eigenvalues for point {point_idx}: {e}")
                    eigenvalues[point_idx] = np.zeros(3)

            # Report progress if callback provided
            if progress_callback is not None and batch_idx % 10 == 0:  # Report every 10 batches
                try:
                    progress_callback(batch_idx + 1, num_batches)
                except Exception as e:
                    print(f"Error in progress callback: {e}")  # services/eigenvalue_utils.py