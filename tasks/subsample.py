"""
Subsample the point cloud data in the given DataNode.
"""

from core.data_node import DataNode
from core.point_cloud import PointCloud


class Subsample:
    """
    Subsample the point cloud data in the given DataNode.
    """
    def __init__(self, params: dict):
        self.params = params
        self.dependencies = []

    def execute(self, data_node: DataNode):
        """
        Subsample the point cloud data in the given DataNode.

        Args:
            data_node (DataNode): The DataNode containing the point cloud data to subsample.

        Returns:
            Tuple[str, PointCloud]: A tuple containing the data type and the subsampled PointCloud instance

        Raises:
            ValueError: If the DataNode does not contain point cloud data.
        """

        # Check upper case of data_node.data_type contains "POINT_CLOUD"
        if data_node.data_type.upper() != "POINT_CLOUD":
            raise ValueError("DataNode must contain a point cloud for subsampling.")
        else:
            # set point_cloud as a PointCloud instance
            point_cloud: PointCloud = data_node.data  # Type hinting for static analysis

            subsampled_point_cloud = point_cloud.subsample(rate=self.params["rate"], inplace=False)
            return subsampled_point_cloud, "point_cloud", self.dependencies
