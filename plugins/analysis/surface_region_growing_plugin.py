# plugins/analysis/surface_region_growing_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.spatial import KDTree

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class SurfaceRegionGrowingPlugin(AnalysisPlugin):
    """
    Surface Region Growing plugin for finding and extending surfaces from a seed region.

    This plugin uses eigenvalue analysis to identify surface properties and edge points,
    then grows the region by progressively adding nearby points that belong to the same
    surface based on local plane fitting and distance thresholds.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "surface_region_growing"
        """
        return "surface_region_growing"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for Surface Region Growing.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "eigenvalue_threshold": {
                "type": "float",
                "default": 0.15,
                "min": 0.01,
                "max": 0.5,
                "label": "Eigenvalue Threshold",
                "description": "Threshold for the ratio of smallest to middle eigenvalue to identify edge points"
            },
            "distance_threshold": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 1.0,
                "label": "Distance Threshold",
                "description": "Maximum distance from local surface for points to be added to the region"
            },
            "k_neighbors": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "label": "Number of Neighbors",
                "description": "Number of neighbors to consider for eigenvalue computation and local surface fitting"
            },
            "max_iterations": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 1000,
                "label": "Maximum Iterations",
                "description": "Maximum number of iterations for region growing"
            },
            "batch_size": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "label": "Batch Size",
                "description": "Number of edge points to process in each batch for efficiency"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the Surface Region Growing algorithm on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for Surface Region Growing

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the region mask
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Extract parameters
        eigenvalue_threshold = params["eigenvalue_threshold"]
        distance_threshold = params["distance_threshold"]
        k_neighbors = params["k_neighbors"]
        max_iterations = params["max_iterations"]
        batch_size = params["batch_size"]

        # Get all points from the point cloud
        points = point_cloud.points
        num_points = len(points)

        # Create a KD-tree for efficient nearest neighbor search
        tree = KDTree(points)

        # Compute eigenvalues for all points
        print(f"Computing eigenvalues for {num_points} points with k={k_neighbors}...")
        eigenvalues = point_cloud.get_eigenvalues(k=k_neighbors, smooth=True)

        # Identify edge points based on the eigenvalue ratio (smallest to middle eigenvalue)
        # Eigenvalues are sorted in ascending order: λ₁ ≤ λ₂ ≤ λ₃
        eigenvalue_ratio = eigenvalues[:, 0] / eigenvalues[:, 1]
        edge_mask = eigenvalue_ratio < eigenvalue_threshold

        # Initialize the region mask with edge points
        region_mask = np.zeros(num_points, dtype=bool)

        # Get indices of edge points
        edge_indices = np.where(edge_mask)[0]
        print(f"Found {len(edge_indices)} edge points using eigenvalue threshold {eigenvalue_threshold}")

        # If no edge points found, use all points as initial seeds
        if len(edge_indices) == 0:
            print("No edge points found. Using all points as initial seeds.")
            edge_indices = np.arange(num_points)

        # Add edge points to region
        region_mask[edge_indices] = True

        # Initialize the queue of edge points to process
        edge_queue = list(edge_indices)

        # Track points that have been processed
        processed = np.zeros(num_points, dtype=bool)

        print(f"Beginning region growing with max {max_iterations} iterations and batch size {batch_size}")

        # Grow the region
        iteration = 0
        points_added = 0

        while edge_queue and iteration < max_iterations:
            iteration += 1

            # Process a batch of edge points
            current_batch_size = min(batch_size, len(edge_queue))
            batch_indices = edge_queue[:current_batch_size]
            edge_queue = edge_queue[current_batch_size:]

            # Skip points that have already been processed
            new_batch_indices = []
            for idx in batch_indices:
                if not processed[idx]:
                    new_batch_indices.append(idx)
            batch_indices = new_batch_indices

            # Mark batch points as processed
            for idx in batch_indices:
                processed[idx] = True

            # For each point in the batch, find its neighbors and check if they should be added to the region
            for current_idx in batch_indices:
                # Find k nearest neighbors of the current point
                distances, indices = tree.query(points[current_idx], k=min(k_neighbors + 1, num_points))

                # Skip the point itself (first neighbor is the point itself)
                distances = distances[1:]
                indices = indices[1:]

                # Find neighboring points that are part of the current region
                region_neighbors = []
                for idx in indices:
                    if region_mask[idx]:
                        region_neighbors.append(idx)

                # If not enough region neighbors, skip this point
                if len(region_neighbors) < 3:
                    continue

                # Fit a local plane to region neighbors
                region_neighbor_points = np.array([points[idx] for idx in region_neighbors])
                center = np.mean(region_neighbor_points, axis=0)
                centered_points = region_neighbor_points - center

                # Compute the covariance matrix
                covariance = np.zeros((3, 3))
                for point in centered_points:
                    covariance += np.outer(point, point)
                covariance /= len(centered_points)

                # Compute eigenvalues and eigenvectors of the covariance matrix
                eigenvalues_local, eigenvectors_local = np.linalg.eigh(covariance)

                # The normal of the plane is the eigenvector corresponding to the smallest eigenvalue
                smallest_eigenvalue_idx = np.argmin(eigenvalues_local)
                normal = eigenvectors_local[:, smallest_eigenvalue_idx]

                # For nearby points that are not in the region, check if they are close to the plane
                for idx in indices:
                    if region_mask[idx] or processed[idx]:
                        continue

                    # Compute distance to the fitted plane
                    point = points[idx]
                    vector = point - center
                    distance = np.abs(np.dot(vector, normal))

                    # If the point is close to the plane, add it to the region
                    if distance < distance_threshold:
                        region_mask[idx] = True
                        edge_queue.append(idx)
                        points_added += 1

            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: Total points in region: {np.sum(region_mask)}, Points added: {points_added}")
                points_added = 0

        print(f"Region growing completed after {iteration} iterations")
        print(f"Final region contains {np.sum(region_mask)} points out of {num_points} total points")

        # Create a Masks object with the result
        mask = Masks(region_mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return mask, "masks", dependencies