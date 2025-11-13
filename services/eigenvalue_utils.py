# services/eigenvalue_utils.py
from typing import Dict, Optional, Callable
import tensorflow as tf
import numpy as np
from scipy.spatial import KDTree

from services.batch_processor import BatchProcessor


class EigenvalueUtils:
    """
    A class for analysing and visualizing eigenvalues of local neighborhoods in point clouds.

    This class provides methods to compute eigenvalues of the covariance matrix for
    k-nearest neighbors of each point in a point cloud, as well as utility methods
    to convert these eigenvalues into meaningful visualizations.
    """

    def __init__(self, use_cpu=True):
        """
        Initialize the EigenvalueAnalyzer.

        Args:
            use_cpu (bool): Whether to force TensorFlow operations to run on CPU.
                            Default is True to avoid GPU memory issues with large point clouds.
        """
        self.use_cpu = use_cpu
        # Keep track of the last computed results for memory optimization
        self._last_computed_eigenvalues = None
        self._last_tree = None
        self._last_indices = None

    def get_eigenvalues(
            self,
            data,
            k: int,
            smooth: bool = True,
            batch_size: Optional[int] = None,
            progress_callback: Optional[Callable] = None
    ):
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

        # If batch_size is None, use the entire point cloud as one batch
        if batch_size is None:
            actual_batch_size = point_count
        else:
            actual_batch_size = min(batch_size, point_count)

        # Create a batch processor with 10% overlap to ensure smooth transitions between batches
        batch_processor = BatchProcessor(
            points=points,
            batch_size=actual_batch_size,
            overlap_percent=0.1
        )

        # Define the processing function for each batch

        # Process the point cloud in spatial batches
        print(f"Computing eigenvalues for {point_count} points with k={k} using spatial batch processing...")

        eigenvalues = batch_processor.process_in_batches(
            processing_func=self._compute_eigenvalues,
            callback=progress_callback,
            k=k,
            smooth=smooth
        )
        return eigenvalues

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

    def eigenvalues_to_colors(self, eigenvalues):
        """
        Convert eigenvalues to RGB colors for visualization.

        Args:
            eigenvalues (np.ndarray): Nx3 array of eigenvalues for each point.

        Returns:
            np.ndarray: Nx3 array of RGB color values.
        """
        # Normalize the eigenvalues row-wise
        normalized_eigenvalues = eigenvalues / np.sum(eigenvalues, axis=1, keepdims=True)

        # Calculate the differences
        differences = np.zeros_like(normalized_eigenvalues)
        differences[:, 0] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 1]  # First minus Second
        differences[:, 1] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 2]  # First minus Third
        differences[:, 2] = normalized_eigenvalues[:, 2] - normalized_eigenvalues[:, 1]  # Third minus Second

        # Normalize the differences to [0, 1]
        min_vals = np.min(differences, axis=0)
        max_vals = np.max(differences, axis=0)
        scaled_differences = (differences - min_vals) / (max_vals - min_vals)
        return scaled_differences

    def _compute_eigenvalues(self, data, k, smooth=True):
        """
        Compute the eigenvalues of the covariance matrix of k-nearest neighbor points for each point.

        This function supports input data in two forms:
        - A Clusters object, where it uses `data.points` for the points.
        - A numpy array directly representing 3D points.

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

        # Create a k-d tree
        tree = KDTree(point_cloud)

        # Ensure k is smaller than the number of points
        if k > len(point_cloud):
            eigenvalues = np.ones((len(point_cloud), 3))
            return eigenvalues

        # Query the k-d tree for KNN
        distances, indices = tree.query(point_cloud,
                                        k=k)  # 'indices' now contains the indices of the k-nearest neighbors for each point

        # Gather KNN points
        knn_points = point_cloud[indices]  # Shape: [num_points, k, 3]

        # Compute means and center the points
        mean_knn_points = np.mean(knn_points, axis=1, keepdims=True)
        centered_knn_points = knn_points - mean_knn_points

        # Reshape for batch matrix multiplication
        centered_knn_points_reshaped = centered_knn_points.transpose(0, 2, 1)

        # Batch covariance matrix computation
        cov_matrices = np.matmul(centered_knn_points_reshaped, centered_knn_points) / (
                k - 1)  # cov_matrices.shape is [num_points, 3, 3]

        # Use CPU for TensorFlow operations
        with tf.device('/CPU:0'):
            # Convert the NumPy array to a TensorFlow tensor
            cov_matrices_tf = tf.convert_to_tensor(cov_matrices, dtype=tf.float32)

            # Compute the eigenvalues and eigenvectors in batch
            eigenvalues_tf, eigenvectors_tf = tf.linalg.eigh(cov_matrices_tf)

        # Convert the eigenvalues back to a NumPy array if needed
        eigenvalues = eigenvalues_tf.numpy()

        if smooth:
            # Use advanced indexing to compute the mean eigenvalues across neighbors
            neighbor_eigenvalues = eigenvalues[indices]  # Use indices to gather neighbor eigenvalues
            avg_eigenvalues = np.mean(neighbor_eigenvalues, axis=1)  # Average over the neighbor axis

            return avg_eigenvalues
        else:
            return eigenvalues
