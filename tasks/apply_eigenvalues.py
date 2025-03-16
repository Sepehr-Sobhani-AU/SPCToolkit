# tasks/apply_eigenvalues.py
"""
Task for applying eigenvalue-based visualization to a point cloud.
"""

from core.point_cloud import PointCloud
from core.eigenvalues import Eigenvalues
from services.eigenvalue_utils import EigenvalueUtils
import numpy as np


class ApplyEigenvalues:
    """
    Task for applying eigenvalues to a point cloud for visualization.

    This task maps eigenvalues to colors to create an intuitive visualization
    of geometric properties in the point cloud.
    """

    def __init__(self, point_cloud: PointCloud, eigenvalues: Eigenvalues):
        """
        Initialize the ApplyEigenvalues task.

        Args:
            point_cloud (PointCloud): The point cloud to apply eigenvalues to
            eigenvalues (Eigenvalues): The eigenvalues to apply
        """
        self.point_cloud = point_cloud
        self.eigenvalues = eigenvalues

    def execute(self):
        """
        Execute the task, applying eigenvalues_array as colors to the point cloud.

        Returns:
            PointCloud: The point cloud with eigenvalue-based colors applied
        """
        # Create a copy of the point cloud to avoid modifying the original
        result_point_cloud = self.point_cloud.get_subset(np.ones(self.point_cloud.size, dtype=bool))

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

        print(f"Applied eigenvalue-based colors to point cloud")

        return result_point_cloud
