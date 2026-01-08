# tasks/apply_class_reference.py
"""
Task for filtering a point cloud to show only points of a specific class.
"""

from core.point_cloud import PointCloud
from core.class_reference import ClassReference
import numpy as np


class ApplyClassReference:
    """
    Task for filtering a point cloud by semantic class.

    This task takes a PointCloud instance (which should have feature_class_ids
    from a parent FeatureClasses node) and a ClassReference, and creates a
    filtered point cloud showing only points of that class.
    """

    def __init__(self, point_cloud: PointCloud, class_reference: ClassReference):
        """
        Initialize the task.

        Args:
            point_cloud (PointCloud): The point cloud to filter (must have feature_class_ids attribute)
            class_reference (ClassReference): The class reference for filtering
        """
        self.point_cloud = point_cloud
        self.class_reference = class_reference

    def execute(self) -> PointCloud:
        """
        Execute the task to filter the point cloud by class.

        Returns:
            PointCloud: A new point cloud containing only points of the specified class

        Raises:
            ValueError: If point cloud doesn't have feature_class_ids attribute
        """
        # Check if point cloud has feature_class_ids attribute
        if not hasattr(self.point_cloud, 'get_attribute'):
            raise ValueError(
                "PointCloud must have feature_class_ids attribute. "
                "Ensure the parent is a FeatureClasses node."
            )

        feature_class_ids = self.point_cloud.get_attribute("feature_class_ids")
        if feature_class_ids is None:
            raise ValueError(
                "PointCloud must have feature_class_ids attribute. "
                "Ensure the parent is a FeatureClasses node."
            )

        # Create mask for points of this class
        class_mask = (feature_class_ids == self.class_reference.class_id)

        # Get subset of points
        filtered_point_cloud = self.point_cloud.get_subset(class_mask)

        # Set all points to the class color
        if filtered_point_cloud.size > 0:
            filtered_point_cloud.colors = np.tile(
                self.class_reference.color,
                (filtered_point_cloud.size, 1)
            ).astype(np.float32)

        return filtered_point_cloud
