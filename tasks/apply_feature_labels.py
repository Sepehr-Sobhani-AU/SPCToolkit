# tasks/apply_feature_labels.py
"""
Task for applying feature labels to a point cloud.
"""

from core.point_cloud import PointCloud
from core.feature_labels import FeatureLabels
import numpy as np


class ApplyFeatureLabels:
    """
    Task for applying feature labels to a point cloud.

    This task takes a PointCloud instance and a FeatureLabels instance,
    and creates a new point cloud with feature class information applied
    as attributes and visualization colors.
    """

    def __init__(self, point_cloud: PointCloud, feature_labels: FeatureLabels):
        """
        Initialize the task.

        Args:
            point_cloud (PointCloud): The point cloud to apply labels to
            feature_labels (FeatureLabels): The feature labels to apply
        """
        self.point_cloud = point_cloud
        self.feature_labels = feature_labels

    def execute(self) -> PointCloud:
        """
        Execute the task to apply feature labels to the point cloud.

        Returns:
            PointCloud: A new point cloud with feature labels applied
        """
        # Create a copy of the point cloud
        result_point_cloud = self.point_cloud.get_subset(np.ones(self.point_cloud.size, dtype=bool))

        # Add feature class labels as attributes
        result_point_cloud.add_attribute("feature_labels", self.feature_labels.labels)

        # Get color values based on feature classes
        colors = self.feature_labels.get_point_colors()

        # Apply colors to the point cloud
        result_point_cloud.colors = colors

        print(f"Applied feature labels to point cloud")
        print(f"Feature classes: {self.feature_labels.get_unique_classes()}")

        return result_point_cloud
