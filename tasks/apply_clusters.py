# TODO: Docstrings
# TODO: Validations

from core.point_cloud import PointCloud
from core.clusters import Clusters


class ApplyClusters:

    def __init__(self, point_cloud: PointCloud, clusters: Clusters):
        self.point_cloud = point_cloud
        self.labels = clusters.labels
        self.colors = clusters.colors

    def execute(self):
        # Create a NEW point cloud instead of modifying the original
        # This ensures the original point cloud's colors are not changed
        new_point_cloud = PointCloud(
            points=self.point_cloud.points,
            colors=self.colors,  # Use cluster colors
            normals=self.point_cloud.normals
        )
        new_point_cloud.cluster_labels = self.labels
        return new_point_cloud


