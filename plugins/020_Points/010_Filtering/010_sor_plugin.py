# plugins/analysis/sor_plugin.py
from typing import Dict, Any, List, Tuple

from core.masks import Masks
from plugins.interfaces import Plugin
from core.data_node import DataNode
from core.point_cloud import PointCloud


class SORPlugin(Plugin):
    """
    Statistical Outlier Removal (SOR) plugin for noise filtering.

    This plugin implements the SOR algorithm to filter out noise points from point clouds.
    It identifies outliers by analyzing the distribution of distances between each point
    and its neighbors, removing points whose average distance is outside a defined standard
    deviation threshold.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "sor"
        """
        return "sor"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for Statistical Outlier Removal.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "nb_neighbors": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "label": "Number of Neighbors",
                "description": "Number of nearest neighbors to use for mean distance calculation"
            },
            "std_ratio": {
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 10.0,
                "label": "Standard Deviation Ratio",
                "description": "Points with a distance larger than (mean + std_ratio * std_dev) are considered outliers"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the SOR algorithm on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for SOR (nb_neighbors and std_ratio)

        Returns:
            Tuple[PointCloud, str, List]:
                - Filtered PointCloud with outliers removed
                - Result type identifier "point_cloud"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Extract parameters
        nb_neighbors = params["nb_neighbors"]
        std_ratio = params["std_ratio"]

        # Apply SOR filter to get the inlier mask
        sor_mask = point_cloud.SOR(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # Create a Masks object with the result
        mask = Masks(sor_mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return mask, "masks", dependencies