# plugins/analysis/dist_to_ground_slice_plugin.py
"""
Plugin for slicing point clouds based on their distance to ground.

This module provides functionality to create a mask that keeps only points
whose distance to ground falls within specified minimum and maximum bounds,
enabling the extraction of horizontal slices of point clouds.
"""

from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.masks import Masks


class GroundSlicePlugin(AnalysisPlugin):
    """
    Plugin for slicing a point cloud based on distance to ground.

    This plugin creates a mask that keeps only points whose distance
    to ground falls within specified minimum and maximum bounds.
    It requires that distance-to-ground values have already been
    calculated for the point cloud.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The unique name "ground_slice"
        """
        return "ground_slice"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for ground slicing.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "min_distance": {
                "type": "float",
                "default": 0.0,
                "min": -100.0,
                "max": 100.0,
                "label": "Minimum Distance",
                "description": "Minimum distance to ground (can be negative for underground)"
            },
            "max_distance": {
                "type": "float",
                "default": 10.0,
                "min": -100.0,
                "max": 100.0,
                "label": "Maximum Distance",
                "description": "Maximum distance to ground"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Create a mask for points within the specified distance to ground range.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters with min and max distance values

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the slice mask
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        try:
            # Get parameters
            min_distance = params["min_distance"]
            max_distance = params["max_distance"]

            # Validate parameters
            if min_distance > max_distance:
                raise ValueError("Minimum distance cannot be greater than maximum distance")

            # Get the point cloud
            point_cloud = data_node.data

            # Check if the point cloud has distance to ground attribute
            if not hasattr(point_cloud, 'attributes') or 'dist_to_ground' not in point_cloud.attributes:
                raise ValueError("Point cloud does not have distance to ground data. "
                                 "Please calculate distance to ground first.")

            # Get the distance values from the point cloud attributes
            distances = point_cloud.attributes['dist_to_ground']

            # Create the slice mask
            slice_mask = np.logical_and(
                distances >= min_distance,
                distances <= max_distance
            )

            # Create Masks object
            mask = Masks(slice_mask)

            # Print information about the slice
            total_points = len(slice_mask)
            kept_points = np.sum(slice_mask)
            print(f"Ground slice: keeping {kept_points} out of {total_points} points "
                  f"({kept_points / total_points * 100:.1f}%) "
                  f"with distances between {min_distance} and {max_distance}")

            # Return the mask with dependencies
            dependencies = [data_node.uid]
            return mask, "masks", dependencies

        except Exception as e:
            error_msg = f"Error in ground slice: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)
