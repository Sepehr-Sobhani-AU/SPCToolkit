from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters


class Dbscan:
    """
    Clusters data in the given DataNode

    Args:
        params (dict): The parameters for the clustering task.

    Return:
        cluster_labels (np.ndarray): The cluster labels for the data points.
    """
    def __init__(self, params: dict):
        self.dependencies = []
        self.params = params

    def execute(self, data_node: DataNode):
        point_cloud: PointCloud = data_node.data
        # Perform clustering on the point cloud data
        cluster_labels = point_cloud.dbscan(eps=self.params["eps"], min_points=self.params["min_samples"])
        # Create a new Clusters object
        cluster = Clusters(cluster_labels)
        # Set random colors for the cluster_labels
        cluster.set_random_color()

        self.dependencies.append(data_node.uid)

        return cluster, "cluster_labels", self.dependencies
