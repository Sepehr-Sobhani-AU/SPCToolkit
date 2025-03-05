# tasks/apply_values.py
import numpy as np
from core.point_cloud import PointCloud
from core.values import Values


class ApplyValues:
    """
    Task to apply point attribute values to a point cloud for visualization.

    This task creates a new point cloud with colors based on the attribute values.
    It's used by the NodeReconstructionManager to visualize attribute-based results.
    """

    def __init__(self, point_cloud: PointCloud, values: Values):
        """
        Initialize the ApplyValues task.

        Args:
            point_cloud (PointCloud): The original point cloud
            values (Values): Values object containing attribute values for each point
        """
        self.point_cloud = point_cloud
        self.values = values.values  # Extract values array from Values object

    def execute(self) -> PointCloud:
        """
        Apply the attribute values to create a colored point cloud.

        The method creates a new point cloud with the same geometry as the original,
        but with colors derived from the attribute values. Low values are colored blue,
        transitioning to red for high values.

        Returns:
            PointCloud: A new point cloud with colors based on attribute values
        """
        # Create a copy of the original point cloud
        colored_point_cloud = PointCloud(
            points=self.point_cloud.points.copy(),
            normals=self.point_cloud.normals.copy() if self.point_cloud.normals is not None else None
        )

        # Normalize attribute values to [0, 1] range for coloring
        min_val = np.min(self.values)
        max_val = np.max(self.values)

        # Handle the case where all values are the same
        if max_val == min_val:
            normalized_values = np.zeros_like(self.values)
        else:
            normalized_values = (self.values - min_val) / (max_val - min_val)

        # Create RGB colors: blue (low values) to red (high values)
        colors = np.zeros((len(normalized_values), 3), dtype=np.float32)
        colors[:, 0] = normalized_values  # Red channel increases with value
        colors[:, 2] = 1.0 - normalized_values  # Blue channel decreases with value

        # Set the colors in the point cloud
        colored_point_cloud.colors = colors

        return colored_point_cloud