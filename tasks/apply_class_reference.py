# tasks/apply_class_reference.py
"""
Task for filtering a point cloud to show only points of a specific class/name.
"""

from core.point_cloud import PointCloud
from core.class_reference import ClassReference
import numpy as np


class ApplyClassReference:
    """
    Task for filtering a point cloud by semantic class.

    This task takes a PointCloud instance (which should have cluster_labels
    from a parent Clusters node with names) and a ClassReference, and creates a
    filtered point cloud showing only points of that class.
    """

    def __init__(self, point_cloud: PointCloud, class_reference: ClassReference):
        """
        Initialize the task.

        Args:
            point_cloud (PointCloud): The point cloud to filter (must have cluster_labels attribute)
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
            ValueError: If point cloud doesn't have cluster_labels attribute
        """
        # Check if point cloud has required attribute
        if not hasattr(self.point_cloud, 'get_attribute'):
            raise ValueError(
                "PointCloud must have cluster_labels attribute. "
                "Ensure the parent is a Clusters node with names."
            )

        # Try new attribute name first, then fall back to old name for compatibility
        cluster_labels = self.point_cloud.get_attribute("cluster_labels")
        if cluster_labels is None:
            # Backward compatibility: try old attribute name
            cluster_labels = self.point_cloud.get_attribute("feature_class_ids")

        if cluster_labels is None:
            raise ValueError(
                "PointCloud must have cluster_labels attribute. "
                "Ensure the parent is a Clusters node with names."
            )

        # Create mask for points of this class
        # Use cluster_ids list if available (for Clusters with cluster_names)
        # Otherwise fall back to single class_id (for old FeatureClasses compatibility)
        if hasattr(self.class_reference, 'cluster_ids') and self.class_reference.cluster_ids:
            # Filter by all cluster_ids that belong to this class
            class_mask = np.isin(cluster_labels, self.class_reference.cluster_ids)
        else:
            # Fall back to single class_id
            class_mask = (cluster_labels == self.class_reference.class_id)

        # Get subset of points
        filtered_point_cloud = self.point_cloud.get_subset(class_mask)

        # Set all points to the class color
        if filtered_point_cloud.size > 0:
            filtered_point_cloud.colors = np.tile(
                self.class_reference.color,
                (filtered_point_cloud.size, 1)
            ).astype(np.float32)

        return filtered_point_cloud
