# tasks/apply_dist_to_ground.py
"""
Task for applying distance-to-ground data to a point cloud.

This module provides functionality to apply distance-to-ground values
to a point cloud, enabling reconstruction of point clouds with vertical
distance information preserved.
"""

from core.point_cloud import PointCloud
from core.dist_to_ground import DistToGround
import numpy as np


class ApplyDistToGround:
    """
    Task for applying distance-to-ground values to a point cloud.

    This task takes a PointCloud instance and a DistToGround instance and
    applies the distance values to the point cloud as an attribute. It also
    automatically generates a color visualization based on the distance values,
    creating a green-to-red heatmap where green represents the minimum distance
    (potentially underground) and red represents the maximum distance.
    """

    def __init__(self, point_cloud: PointCloud, dist_to_ground: DistToGround):
        """
        Initialize the ApplyDistToGround task.

        Args:
            point_cloud (PointCloud): The point cloud to apply distance values to
            dist_to_ground (DistToGround): The distance-to-ground values to apply
        """
        self.point_cloud = point_cloud
        self.dist_to_ground = dist_to_ground.distances

    def execute(self) -> PointCloud:
        """
        Execute the task, applying distance-to-ground values to the point cloud.

        This method:
        1. Creates a copy of the point cloud to avoid modifying the original
        2. Adds the distance-to-ground values as an attribute
        3. Generates a light green to light red heatmap based on the distance values,
           where light green represents the lowest points and light red represents
           the highest points.

        Returns:
            PointCloud: A new point cloud with applied distance values and heatmap colors
        """
        # Create a copy of the point cloud
        result_point_cloud = self.point_cloud.get_subset(np.ones(self.point_cloud.size, dtype=bool))

        # Check if we have the right number of distance values
        if len(self.dist_to_ground) != result_point_cloud.size:
            raise ValueError(
                f"Distance count ({len(self.dist_to_ground)}) "
                f"does not match point count ({result_point_cloud.size})"
            )

        # Add distance-to-ground as an attribute
        result_point_cloud.add_attribute("dist_to_ground", self.dist_to_ground)

        # Generate a light green to light red heatmap based on distance values
        # Light green (0.5, 1.0, 0.5) for lowest points
        # Light red (1.0, 0.5, 0.5) for highest points

        # Initialize colors array with default neutral color
        colors = np.ones((len(self.dist_to_ground), 3), dtype=np.float32) * 0.5

        # Handle NaN or infinite values
        valid_distances = np.isfinite(self.dist_to_ground)
        if not np.any(valid_distances):
            # If no valid distances, use a neutral gray color
            print("Warning: No valid distance values found")
        else:
            # Get min and max distances (from valid values only)
            min_dist = np.min(self.dist_to_ground[valid_distances])
            max_dist = np.max(self.dist_to_ground[self.dist_to_ground[valid_distances] < 9999])

            if max_dist == min_dist:
                # If all distances are the same, use a neutral yellow-green
                print("All distance values are identical, using neutral color")
                colors[valid_distances, 0] = 0.7  # Red channel
                colors[valid_distances, 1] = 0.7  # Green channel
            else:
                # Normalize distances to 0-1 range for color mapping
                dist_range = max_dist - min_dist
                normalized_distances = (self.dist_to_ground - min_dist) / dist_range

                # Set light green (0.5, 1.0, 0.5) to light red (1.0, 0.5, 0.5) gradient
                # Red channel: 0.5 (low values) to 1.0 (high values)
                colors[valid_distances, 0] = 1 - normalized_distances[valid_distances]

                # Green channel: 1.0 (low values) to 0.5 (high values)
                colors[valid_distances, 1] = normalized_distances[valid_distances]

                # Blue channel: constant at 0.5 for a softer overall look
                colors[valid_distances, 2] = 0.5

        # Apply colors to the point cloud
        result_point_cloud.colors = colors

        print(f"Applied distance-to-ground values to point cloud")
        print(f"Distance range: [{np.nanmin(self.dist_to_ground):.3f}, {np.nanmax(self.dist_to_ground):.3f}]")
        print(f"Color mapping: Light green for minimum distance, light red for maximum distance")

        return result_point_cloud
