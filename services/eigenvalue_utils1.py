import numpy as np
import torch
from scipy.spatial import KDTree


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
        self._last_computed_eigenvalues = None
        self._last_tree = None
        self._last_indices = None

    def compute_eigenvalues(self, data, k, smooth=True, batch_size=None):
        """
        Compute the eigenvalues of the covariance matrix of k-nearest neighbor points for each point.

        This method supports input data in two forms:
        - An object with a 'points' attribute (like PointCloud or Clusters)
        - A numpy array directly representing 3D points

        Args:
            data: An object with a 'points' attribute or a numpy array of points with shape (n_points, 3).
            k (int): The number of nearest neighbors to consider for each point.
            smooth (bool): If True, averages the eigenvalues across neighbors. Defaults to True.
            batch_size (int, optional): If provided, processes points in batches of this size to
                                        reduce memory usage. Useful for very large point clouds.

        Returns:
            np.ndarray: A 2D array of eigenvalues for each point's neighborhood with shape (n_points, 3).
                        Eigenvalues are sorted in ascending order (λ₁ ≤ λ₂ ≤ λ₃).

        Raises:
            ValueError: If the input data type is not supported or if k is invalid.
        """
        # Extract points based on input type
        if isinstance(data, np.ndarray):
            point_cloud = data
        elif hasattr(data, 'points'):
            point_cloud = data.points
        else:
            raise ValueError("The first parameter must be an object with a 'points' attribute or a numpy array.")

        num_points = point_cloud.shape[0]

        # Validate k
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if k > num_points:
            raise ValueError(f"k ({k}) cannot be larger than the number of points ({num_points})")

        # Process in batches if requested
        if batch_size is not None and batch_size < num_points:
            return self._compute_eigenvalues_batched(point_cloud, k, smooth, batch_size)

        # Create a k-d tree for efficient nearest neighbor search
        self._last_tree = KDTree(point_cloud)

        # Query the k-d tree for KNN
        distances, indices = self._last_tree.query(point_cloud, k=k)
        self._last_indices = indices

        # Gather KNN points
        knn_points = point_cloud[indices]  # Shape: [num_points, k, 3]

        # Compute means and center the points
        mean_knn_points = np.mean(knn_points, axis=1, keepdims=True)
        centered_knn_points = knn_points - mean_knn_points

        # Reshape for batch matrix multiplication
        centered_knn_points_reshaped = centered_knn_points.transpose(0, 2, 1)

        # Batch covariance matrix computation
        cov_matrices = np.matmul(centered_knn_points_reshaped, centered_knn_points) / (k - 1)

        # Define the compute device based on configuration
        device = torch.device('cpu' if self.use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Use PyTorch for batch eigenvalue computation
        cov_matrices_torch = torch.from_numpy(cov_matrices.astype(np.float32)).to(device)

        # Compute the eigenvalues in batch
        eigenvalues_torch, _ = torch.linalg.eigh(cov_matrices_torch)

        # Convert the eigenvalues back to a NumPy array
        eigenvalues = eigenvalues_torch.cpu().numpy()

        # Apply smoothing if requested
        if smooth:
            # Use advanced indexing to compute the mean eigenvalues across neighbors
            neighbor_eigenvalues = eigenvalues[indices]  # Use indices to gather neighbor eigenvalues
            avg_eigenvalues = np.mean(neighbor_eigenvalues, axis=1)  # Average over the neighbor axis
            self._last_computed_eigenvalues = avg_eigenvalues
            return avg_eigenvalues
        else:
            self._last_computed_eigenvalues = eigenvalues
            return eigenvalues

    def _compute_eigenvalues_batched(self, point_cloud, k, smooth, batch_size):
        """
        Compute eigenvalues in batches to reduce memory usage.

        This internal method breaks down eigenvalue computation into smaller batches
        to handle very large point clouds without exhausting GPU memory.

        Args:
            point_cloud (np.ndarray): The point cloud data as a numpy array.
            k (int): The number of nearest neighbors.
            smooth (bool): Whether to smooth eigenvalues by averaging across neighbors.
            batch_size (int): The number of points to process in each batch.

        Returns:
            np.ndarray: Eigenvalues for each point.
        """
        num_points = point_cloud.shape[0]
        all_eigenvalues = np.zeros((num_points, 3), dtype=np.float32)

        # Create a k-d tree once for the entire point cloud
        self._last_tree = KDTree(point_cloud)
        all_indices = []

        # Process points in batches
        for start_idx in range(0, num_points, batch_size):
            end_idx = min(start_idx + batch_size, num_points)
            batch_points = point_cloud[start_idx:end_idx]

            # Query the k-d tree for KNN for this batch
            distances, indices = self._last_tree.query(batch_points, k=k)
            all_indices.append(indices)

            # Gather KNN points
            knn_points = point_cloud[indices]  # Shape: [batch_size, k, 3]

            # Compute means and center the points
            mean_knn_points = np.mean(knn_points, axis=1, keepdims=True)
            centered_knn_points = knn_points - mean_knn_points

            # Reshape for batch matrix multiplication
            centered_knn_points_reshaped = centered_knn_points.transpose(0, 2, 1)

            # Batch covariance matrix computation
            cov_matrices = np.matmul(centered_knn_points_reshaped, centered_knn_points) / (k - 1)

            # Use PyTorch for batch eigenvalue computation
            device = torch.device('cpu' if self.use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
            cov_matrices_torch = torch.from_numpy(cov_matrices.astype(np.float32)).to(device)

            # Compute the eigenvalues in batch
            eigenvalues_torch, _ = torch.linalg.eigh(cov_matrices_torch)

            # Convert to NumPy and store in the result array
            batch_eigenvalues = eigenvalues_torch.cpu().numpy()
            all_eigenvalues[start_idx:end_idx] = batch_eigenvalues

        # Combine all indices for later use
        self._last_indices = np.vstack(all_indices) if len(all_indices) > 1 else all_indices[0]

        # Apply smoothing if requested
        if smooth:
            smoothed_eigenvalues = np.zeros_like(all_eigenvalues)

            # Process smoothing in batches as well
            for start_idx in range(0, num_points, batch_size):
                end_idx = min(start_idx + batch_size, num_points)
                batch_indices = self._last_indices[start_idx:end_idx]

                # Get neighbor eigenvalues for this batch
                neighbor_eigenvalues = all_eigenvalues[batch_indices]

                # Average over the neighbor axis
                batch_avg_eigenvalues = np.mean(neighbor_eigenvalues, axis=1)
                smoothed_eigenvalues[start_idx:end_idx] = batch_avg_eigenvalues

            self._last_computed_eigenvalues = smoothed_eigenvalues
            return smoothed_eigenvalues

        self._last_computed_eigenvalues = all_eigenvalues
        return all_eigenvalues

    def eigenvalues_to_colors(self, eigenvalues=None):
        """
        Convert eigenvalues to RGB colors based on eigenvalue differences.

        If no eigenvalues are provided, uses the last computed eigenvalues if available.

        Args:
            eigenvalues (np.ndarray, optional): Eigenvalues array with shape (n_points, 3).
                                               If None, uses last computed eigenvalues.

        Returns:
            np.ndarray: RGB color values with shape (n_points, 3), normalized to [0, 1] range.

        Raises:
            ValueError: If no eigenvalues are provided and none have been computed previously.
        """
        if eigenvalues is None:
            if self._last_computed_eigenvalues is None:
                raise ValueError("No eigenvalues provided and none have been computed previously.")
            eigenvalues = self._last_computed_eigenvalues

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10

        # Normalize the eigenvalues row-wise
        eigenvalue_sums = np.sum(eigenvalues, axis=1, keepdims=True)
        # Replace zero sums with epsilon to avoid division by zero
        eigenvalue_sums = np.maximum(eigenvalue_sums, epsilon)
        normalized_eigenvalues = eigenvalues / eigenvalue_sums

        # Calculate the differences
        differences = np.zeros_like(normalized_eigenvalues)
        differences[:, 0] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 1]  # First minus Second
        differences[:, 1] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 2]  # First minus Third
        differences[:, 2] = normalized_eigenvalues[:, 2] - normalized_eigenvalues[:, 1]  # Third minus Second

        # Normalize the differences to [0, 1]
        min_vals = np.min(differences, axis=0)
        max_vals = np.max(differences, axis=0)

        # Avoid division by zero when min and max are equal
        divisors = np.maximum(max_vals - min_vals, epsilon)
        scaled_differences = (differences - min_vals) / divisors

        # Ensure all values are in [0, 1] range
        return np.clip(scaled_differences, 0, 1)

    def compute_geometric_features(self, eigenvalues=None):
        """
        Compute geometric features based on eigenvalues.

        If no eigenvalues are provided, uses the last computed eigenvalues if available.

        These features describe different aspects of the local geometry:
        - Planarity: Indicates how planar (flat) the local surface is
        - Linearity: Indicates how linear (curve-like) the local structure is
        - Sphericity: Indicates how spherical or corner-like the local area is
        - Anisotropy: Indicates the directional dependence of the local structure

        Args:
            eigenvalues (np.ndarray, optional): Eigenvalues array with shape (n_points, 3).
                                               If None, uses last computed eigenvalues.

        Returns:
            dict: Dictionary containing the computed geometric features, each with shape (n_points,).

        Raises:
            ValueError: If no eigenvalues are provided and none have been computed previously.
        """
        if eigenvalues is None:
            if self._last_computed_eigenvalues is None:
                raise ValueError("No eigenvalues provided and none have been computed previously.")
            eigenvalues = self._last_computed_eigenvalues

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10

        # Extract individual eigenvalues (sorted in ascending order: λ₁ ≤ λ₂ ≤ λ₃)
        lambda1 = eigenvalues[:, 0]  # Smallest eigenvalue
        lambda2 = eigenvalues[:, 1]  # Middle eigenvalue
        lambda3 = eigenvalues[:, 2]  # Largest eigenvalue

        # Calculate geometric features
        # Planarity: (λ₂ - λ₁)/λ₃ - how planar (flat) the surface is
        planarity = (lambda2 - lambda1) / np.maximum(lambda3, epsilon)

        # Linearity: (λ₃ - λ₂)/λ₃ - how linear (curve-like) the structure is
        linearity = (lambda3 - lambda2) / np.maximum(lambda3, epsilon)

        # Sphericity: λ₁/λ₃ - how spherical the local area is
        sphericity = lambda1 / np.maximum(lambda3, epsilon)

        # Anisotropy: (λ₃ - λ₁)/λ₃ - how directional the structure is
        anisotropy = (lambda3 - lambda1) / np.maximum(lambda3, epsilon)

        # Eigenvalue sum (total variance)
        total_variance = lambda1 + lambda2 + lambda3

        # Handle any NaN or infinity values
        features = {
            'planarity': np.nan_to_num(planarity),
            'linearity': np.nan_to_num(linearity),
            'sphericity': np.nan_to_num(sphericity),
            'anisotropy': np.nan_to_num(anisotropy),
            'total_variance': np.nan_to_num(total_variance)
        }

        return features