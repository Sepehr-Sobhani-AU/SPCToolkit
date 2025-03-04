# TODO: Double check the filtering logic
"""
Filter the data in the given DataNode.
"""
import numpy as np

from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class Filtering:
    """
    Filter data in the given DataNode.
    """

    def __init__(self, params: dict):
        self.params = params
        self.dependencies = []

    def execute(self, data_node: DataNode):
        filter_condition = self.params["condition"]
        point_cloud: PointCloud = data_node.data  # Type hinting for static analysis

        # Use a dictionary to capture the result of the exec statement
        local_vars = {"point_cloud": point_cloud}

        # Execute the condition in the provided namespace
        exec(f"filter_mask = {filter_condition}", globals(), local_vars)

        # Extract the filter_mask from the namespace
        filter_mask = local_vars.get("filter_mask")

        self.dependencies.append(data_node.uid)
        mask = Masks(filter_mask)

        return mask, "masks", self.dependencies
