# TODO: Docstring
"""
Filter the data in the given DataNode.
"""
import numpy as np

from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.masks import Masks


class Subtracting:
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
        filter_mask = exec(f"{filter_condition}")

        self.dependencies.append(data_node.uid)

        mask = Masks(filter_mask)

        return mask, "masks", self.dependencies
