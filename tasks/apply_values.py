# tasks/apply_values.py
"""
Task for applying Values data to a point cloud.
"""

from core.point_cloud import PointCloud
from core.values import Values
import numpy as np


class ApplyValues:
    """
    Task for applying Values data to a point cloud.

    This task handles both 1D and multi-dimensional value arrays.
    """

    def __init__(self, point_cloud: PointCloud, values: Values):
        """
        Initialize the ApplyValues task.

        Args:
            point_cloud (PointCloud): The point cloud to apply values to
            values (Values): The values to apply
        """
        self.point_cloud = point_cloud
        self.values = values.values

    def execute(self):
        """
        Execute the task, applying values to the point cloud.

        For multi-dimensional values, it handles them in one of two ways:
        1. If the first dimension matches the number of points, it adds the entire array as a
           single attribute
        2. If there's a dimension mismatch, it splits the values into separate attributes

        Returns:
            PointCloud: The point cloud with values applied
        """
        # Create shallow copy - shares array references (memory efficient)
        # This is safe because we only add attributes
        result_point_cloud = PointCloud(
            points=self.point_cloud.points,
            colors=self.point_cloud.colors,
            normals=self.point_cloud.normals,
            intensity=getattr(self.point_cloud, 'intensity', np.array([])),
            distToGround=getattr(self.point_cloud, 'distToGround', np.array([])),
            params=self.point_cloud.name,
        )
        # Copy attributes dictionary so we don't modify the original
        result_point_cloud.attributes = self.point_cloud.attributes.copy()

        # Check if values is a multi-dimensional array
        if len(self.values.shape) > 1:
            # For 2D arrays where first dimension matches point count
            if self.values.shape[0] == len(self.point_cloud.points):
                try:
                    # Try to add the multi-dimensional array directly
                    result_point_cloud.add_attribute("values", self.values)
                    print(f"Added multi-dimensional values with shape {self.values.shape} as attribute")
                except ValueError:
                    # If that fails, add each column as a separate attribute
                    for i in range(self.values.shape[1]):
                        attr_name = f"eigenvalue_{i}"
                        result_point_cloud.add_attribute(attr_name, self.values[:, i])
                    print(f"Added {self.values.shape[1]} separate eigenvalue attributes")
            else:
                # If dimensions don't match, we have a problem
                print(
                    f"Warning: Values shape {self.values.shape} doesn't match point count {len(self.point_cloud.points)}")

                # If the values are flattened eigenvalues, try to reshape them
                if len(self.values) == len(self.point_cloud.points) * 3:
                    reshaped_values = self.values.reshape(len(self.point_cloud.points), 3)
                    for i in range(3):
                        attr_name = f"eigenvalue_{i}"
                        result_point_cloud.add_attribute(attr_name, reshaped_values[:, i])
                    print(f"Reshaped flattened eigenvalues into 3 separate attributes")
                else:
                    # Just add the first value per point if everything else fails
                    result_point_cloud.add_attribute("values", self.values.flatten()[:len(self.point_cloud.points)])
                    print(f"Warning: Only using first {len(self.point_cloud.points)} values")
        else:
            # Regular 1D array case
            result_point_cloud.add_attribute("values", self.values)
            print(f"Added values as attribute to point cloud")

        return result_point_cloud