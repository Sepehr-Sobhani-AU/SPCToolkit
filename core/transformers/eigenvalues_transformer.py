# tasks/apply_eigenvalues.py
"""
Task for applying eigenvalue-based visualization to a point cloud.
"""

from core.entities.point_cloud import PointCloud
from core.entities.eigenvalues import Eigenvalues
from core.services.eigenvalue_utils import EigenvalueUtils
import numpy as np


class EigenvaluesTransformer:
    """
    Task for applying eigenvalues to a point cloud for visualization.

    This task maps eigenvalues to colors to create an intuitive visualization
    of geometric properties in the point cloud.
    """

    def __init__(self, point_cloud: PointCloud, eigenvalues: Eigenvalues):
        """
        Initialize the EigenvaluesTransformer task.

        Args:
            point_cloud (PointCloud): The point cloud to apply eigenvalues to
            eigenvalues (Eigenvalues): The eigenvalues to apply
        """
        self.point_cloud = point_cloud
        self.eigenvalues = eigenvalues

    def execute(self):
        """
        Execute the task, applying eigenvalues to the point cloud.

        This method:
        1. Maps eigenvalues to colors for visualization
        2. Stores eigenvalues in point_cloud.attributes for downstream use

        Returns:
            PointCloud: The point cloud with eigenvalue-based colors and attributes applied
        """
        # Create shallow copy - shares array references (memory efficient)
        # This is safe because we replace colors entirely and only add attributes
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

        # Check if we have the right number of eigenvalues_array
        if len(self.eigenvalues.eigenvalues) != result_point_cloud.size:
            raise ValueError(
                f"Eigenvalue count ({len(self.eigenvalues.eigenvalues)}) "
                f"does not match point count ({result_point_cloud.size})"
            )

        # Get the eigenvalue array
        eigenvalues_array = self.eigenvalues.eigenvalues

        # Apply eigenvalues_array to colors using the eigenvalue difference method
        eigenvalues_utils = EigenvalueUtils()
        colors = eigenvalues_utils.eigenvalues_to_colors(eigenvalues_array)

        # Apply colors to the point cloud
        result_point_cloud.colors = colors.astype(np.float32)

        # Store eigenvalues in attributes for downstream use (e.g., training data preparation)
        result_point_cloud.add_attribute('eigenvalues', eigenvalues_array)

        print(f"Applied eigenvalue-based colors to point cloud and stored eigenvalues in attributes")

        return result_point_cloud
