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
        self.point_cloud.cluster_labels = self.labels
        self.point_cloud.colors = self.colors
        return self.point_cloud


