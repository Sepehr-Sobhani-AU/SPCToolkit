# tasks/apply_clusters.py
"""
Task for applying cluster labels and colors to a point cloud.

This unified task handles both simple clusters (with per-point colors)
and named clusters (with semantic names and class colors).
"""

from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters


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

        # Copy existing attributes from source PointCloud
        # This preserves any metadata from upstream operations
        new_point_cloud.attributes = self.point_cloud.attributes.copy()

        # Add cluster labels as attribute for downstream tasks (like ApplyClassReference)
        # This replaces the old direct "cluster_labels" attribute and uses the attributes dict
        # which is properly masked during get_subset() operations
        new_point_cloud.add_attribute("cluster_labels", self.clusters.labels)

        # Store cluster metadata for downstream operations (e.g., Cluster by Class on filtered branches)
        # These are dict attributes (not per-point arrays) - store directly in attributes dict
        # since add_attribute() is for per-point arrays only. Dict attributes are preserved
        # unchanged through get_subset() operations.
        if self.clusters.has_names():
            new_point_cloud.attributes["_cluster_names"] = self.clusters.cluster_names
            new_point_cloud.attributes["_cluster_colors"] = self.clusters.cluster_colors

        return new_point_cloud
