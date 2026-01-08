"""
Subsample the point cloud data in the given DataNode.
"""

from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class Subsampling:
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

        # set point_cloud as a PointCloud instance
        point_cloud: PointCloud = data_node.data  # Type hinting for static analysis

        subsample_mask = point_cloud.subsample(rate=self.params["rate"], boolean=True)

        self.dependencies.append(data_node.uid)

        mask = Masks(subsample_mask)

        return mask, "masks", self.dependencies
