# tasks/apply_clusters.py
"""
Task for applying cluster labels and colors to a point cloud.

This unified task handles both simple clusters (with per-point colors)
and named clusters (with semantic names and class colors).
"""

from core.point_cloud import PointCloud
from core.clusters import Clusters


class ApplyClusters:
    """
    Task for applying cluster labels to a point cloud.

    This task takes a PointCloud instance and a Clusters instance,
    and creates a new point cloud with cluster information applied
    as attributes and visualization colors.

    If the Clusters object has cluster_names, colors are determined
    by the semantic names. Otherwise, per-point colors are used.
    """

    def __init__(self, point_cloud: PointCloud, clusters: Clusters):
        """
        Initialize the task.

        Args:
            point_cloud (PointCloud): The point cloud to apply clusters to
            clusters (Clusters): The clusters to apply
        """
        self.point_cloud = point_cloud
        self.clusters = clusters

    def execute(self) -> PointCloud:
        """
        Execute the task to apply clusters to the point cloud.

        Returns:
            PointCloud: A new point cloud with cluster labels and colors applied
        """
        # Determine colors: use named colors if available, otherwise per-point colors
        if self.clusters.has_names():
            colors = self.clusters.get_named_colors()
        else:
            colors = self.clusters.colors

        # Create a NEW point cloud instead of modifying the original
        # This ensures the original point cloud's colors are not changed
        new_point_cloud = PointCloud(
            points=self.point_cloud.points,
            colors=colors,
            normals=self.point_cloud.normals
        )

        # Add cluster labels as attribute
        new_point_cloud.cluster_labels = self.clusters.labels

        # Add cluster_ids attribute for downstream tasks (like ApplyClassReference)
        # This replaces the old "feature_class_ids" attribute from FeatureClasses
        new_point_cloud.add_attribute("cluster_ids", self.clusters.labels)

        return new_point_cloud
