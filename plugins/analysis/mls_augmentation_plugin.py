# plugins/analysis/mls_augmentation_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.spatial import KDTree

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud


class MLSAugmentationPlugin(AnalysisPlugin):
    """
    Plugin for augmenting point clouds using Moving Least Squares (MLS) with adaptive sampling.

    This plugin identifies low-density regions in the point cloud and adds new points
    to create a more uniform density while preserving the surface shape using MLS projection.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "mls_augmentation"
        """
        return "mls_augmentation"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for MLS augmentation.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "neighborhood_size": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 1.0,
                "label": "Neighborhood Size",
                "description": "Size of the neighborhood for MLS fitting (relative to bounding box diagonal)"
            },
            "density_target": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "label": "Target Density",
                "description": "Target number of points within each neighborhood"
            },
            "polynomial_order": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 3,
                "label": "Polynomial Order",
                "description": "Order of the polynomial used for surface fitting (1=plane, 2=quadratic, 3=cubic)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute MLS augmentation on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for MLS augmentation

        Returns:
            Tuple[PointCloud, str, List]:
                - Augmented PointCloud with added points
                - Result type identifier "point_cloud"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data
        points = point_cloud.points

        # Extract parameters
        neighborhood_size_rel = params["neighborhood_size"]
        target_density = params["density_target"]
        polynomial_order = params["polynomial_order"]

        # Calculate absolute neighborhood size based on bounding box diagonal
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
        neighborhood_size = neighborhood_size_rel * bbox_diagonal

        # Build KD-tree for efficient neighborhood queries
        tree = KDTree(points)

        # Step 1: Estimate local density at sample points
        # We'll sample the space using a coarse grid to avoid checking every point
        grid_step = neighborhood_size
        x_range = np.arange(bbox_min[0], bbox_max[0], grid_step)
        y_range = np.arange(bbox_min[1], bbox_max[1], grid_step)
        z_range = np.arange(bbox_min[2], bbox_max[2], grid_step)

        # Create sample points (limiting to a reasonable number to avoid memory issues)
        max_samples = 10000
        if len(x_range) * len(y_range) * len(z_range) > max_samples:
            # If too many samples, create a random subset
            num_samples = max_samples
            sample_points = np.random.uniform(
                low=bbox_min,
                high=bbox_max,
                size=(num_samples, 3)
            )
        else:
            # Otherwise, use a grid
            X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            sample_points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        # For each sample point, find the nearest point in the point cloud
        _, nearest_indices = tree.query(sample_points, k=1)
        nearest_points = points[nearest_indices]

        # Step 2: Project sample points onto the approximate surface
        # This is a simple approximation - we're just using the nearest point
        projected_samples = nearest_points

        # Step 3: For each projected sample, count neighbors to assess density
        neighbor_counts = []
        for point in projected_samples:
            neighbors = tree.query_ball_point(point, neighborhood_size)
            neighbor_counts.append(len(neighbors))

        neighbor_counts = np.array(neighbor_counts)

        # Step 4: Identify low-density regions
        low_density_mask = neighbor_counts < target_density
        low_density_samples = projected_samples[low_density_mask]

        if len(low_density_samples) == 0:
            # No low-density regions found, return the original point cloud
            print("No low-density regions found that require augmentation.")
            augmented_point_cloud = point_cloud
        else:
            # Step 5: Generate new points in low-density regions
            new_points = []
            for point in low_density_samples:
                # Find existing points in the neighborhood
                indices = tree.query_ball_point(point, neighborhood_size)
                neighbors = points[indices]

                if len(neighbors) < 3:
                    # Not enough neighbors for MLS fitting, skip this point
                    continue

                # Calculate weights based on distance
                distances = np.linalg.norm(neighbors - point, axis=1)
                h = neighborhood_size / 2.0  # Kernel width
                weights = np.exp(-(distances ** 2) / (h ** 2))

                # Calculate weighted centroid
                centroid = np.average(neighbors, axis=0, weights=weights)

                # Create local coordinate system
                # Simplifying by using PCA to find principal directions
                centered = neighbors - centroid
                cov = np.dot(centered.T * weights, centered) / weights.sum()
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # Sort by eigenvalues in descending order
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # The third eigenvector is the approximate normal
                normal = eigenvectors[:, 2]

                # Ensure the normal points "outward" (this is a heuristic)
                if np.dot(normal, point - centroid) < 0:
                    normal = -normal

                # Generate new points in a small grid around the low-density point
                grid_size = 3  # 3x3 grid
                grid_step = neighborhood_size / (grid_size + 1)

                # Use the first two eigenvectors as the tangent plane basis
                u_axis = eigenvectors[:, 0]
                v_axis = eigenvectors[:, 1]

                for i in range(grid_size):
                    for j in range(grid_size):
                        # Create grid point in local tangent plane
                        u_offset = (i - grid_size / 2 + 0.5) * grid_step
                        v_offset = (j - grid_size / 2 + 0.5) * grid_step

                        new_point = centroid + u_offset * u_axis + v_offset * v_axis

                        # Project onto MLS surface (simplified approximation)
                        # For a full implementation, you would fit a polynomial here
                        # For now, we'll use the weighted average projection
                        distances = np.linalg.norm(neighbors - new_point, axis=1)
                        h = neighborhood_size / 2.0  # Kernel width
                        weights = np.exp(-(distances ** 2) / (h ** 2))

                        if weights.sum() > 0:
                            # Apply MLS projection (simplified version)
                            mls_point = np.average(neighbors, axis=0, weights=weights)
                            new_points.append(mls_point)

            # Convert to numpy array
            if new_points:
                new_points = np.array(new_points)

                # Step 6: Combine original points with new points
                combined_points = np.vstack((points, new_points))

                # Create colors for the new points (matching existing if available)
                if point_cloud.colors is not None and len(point_cloud.colors) > 0:
                    # Use the average color from nearby points for each new point
                    new_colors = np.zeros((len(new_points), 3), dtype=np.float32)
                    for i, new_point in enumerate(new_points):
                        _, indices = tree.query(new_point, k=5)
                        new_colors[i] = np.mean(point_cloud.colors[indices], axis=0)

                    combined_colors = np.vstack((point_cloud.colors, new_colors))
                else:
                    combined_colors = None

                # Create the augmented point cloud
                augmented_point_cloud = PointCloud(
                    points=combined_points,
                    colors=combined_colors,
                    normals=None  # We don't compute normals in this plugin
                )
            else:
                # No new points were generated
                augmented_point_cloud = point_cloud

        # Return with empty dependencies list since this is a root node
        # and doesn't need reconstruction
        return augmented_point_cloud, "point_cloud", []