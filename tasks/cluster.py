from core.data_node import DataNode
from core.point_cloud import PointCloud


class Cluster:
    def __init__(self, params: dict):
        self.dependencies = []
        self.params = params

    def execute(self, data_node: DataNode):
        if data_node.data_type.upper() != "POINT_CLOUD":
            raise ValueError("DataNode must contain a point cloud for clustering.")
        else:
            point_cloud: PointCloud = data_node.data
            # Perform clustering on the point cloud data
            cluster_labels = point_cloud.dbscan(eps=self.params["eps"], min_points=self.params["min_samples"])
            self.dependencies.append(data_node.uid)

        return cluster_labels, "cluster_labels", self.dependencies
