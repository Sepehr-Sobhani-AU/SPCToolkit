# TODO: Double check the filtering logic
"""
Filter the data in the given DataNode.
"""

from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class SelectClusters:
    """
    Filter data in the given DataNode.
    """
    def __init__(self, params: dict):
        self.params = params
        self.dependencies = []

    def execute(self, data_node: DataNode):

        filter_condition = self.params["condition"]

        # set point_cloud as a PointCloud instance
        point_cloud: PointCloud = data_node.data  # Type hinting for static analysis

        # Dynamically construct the condition and apply it using eval
        filter_mask = exec(f"point_cloud.{filter_condition}")

        self.dependencies.append(data_node.uid)

        mask = Masks(filter_mask)

        return mask, "masks", self.dependencies
