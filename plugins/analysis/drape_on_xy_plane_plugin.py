# plugins/analysis/drape_on_xy_plane_plugin.py
"""
Drape on XY Plane analysis plugin.

This plugin projects all points in a point cloud to a constant Z height,
effectively "draping" them onto an XY plane. The default behavior is to
drape to the minimum Z value of the point cloud, with an optional offset.
"""

import numpy as np
from typing import Dict, Any, Tuple, List

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud


class DrapeOnXYPlanePlugin(AnalysisPlugin):
    """
    Analysis plugin for draping points onto an XY plane.

    This plugin takes a point cloud (reconstructed if necessary) and projects
    all points to a constant Z coordinate. The Z coordinate is calculated as
    the minimum Z value of the input point cloud plus an optional offset.

    The draping operation:
    1. Finds the minimum Z coordinate in the point cloud
    2. Adds the user-specified offset
    3. Sets all points' Z coordinates to this value
    4. Preserves colors and other attributes (except normals, which are discarded)
    """

    def get_name(self) -> str:
        """
        Return the unique name of this analysis plugin.

        Returns:
            str: The plugin identifier "drape_on_xy_plane"
        """
        return "drape_on_xy_plane"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for the draping operation.

        Returns:
            Dict[str, Any]: Parameter schema with z_offset parameter
        """
        return {
            "z_offset": {
                "type": "float",
                "default": 0.0,
                "label": "Z Offset from Minimum",
                "description": "Height above the lowest point (0 = drape to minimum Z)",
                "min": -1000.0,
                "max": 1000.0
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the draping operation on the point cloud.

        This method receives a DataNode containing a PointCloud (already reconstructed
        by DataManager if it was derived data), calculates the target Z coordinate,
        and creates a new PointCloud with all points at that height.

        Args:
            data_node (DataNode): The data node containing the PointCloud to drape
            params (Dict[str, Any]): Parameters including 'z_offset'

        Returns:
            Tuple[Any, str, List]: A tuple containing:
                - The draped PointCloud
                - The result type ("point_cloud")
                - List of dependencies (the input node's UID)

        Raises:
            ValueError: If the input data is not a PointCloud
            ValueError: If the PointCloud has no points
        """
        # Validate input
        if not isinstance(data_node.data, PointCloud):
            raise ValueError(
                f"Expected PointCloud but got {type(data_node.data).__name__}. "
                "Data should be reconstructed before draping."
            )

        point_cloud = data_node.data

        # Check if point cloud has points
        if point_cloud.size == 0:
            raise ValueError("Cannot drape an empty point cloud.")

        print(f"Draping point cloud with {point_cloud.size} points...")

        # Get points and make a copy to avoid modifying the original
        points = point_cloud.points.copy()

        # Calculate minimum Z coordinate
        min_z = np.min(points[:, 2])
        print(f"Minimum Z coordinate: {min_z:.4f}")

        # Get the offset parameter
        z_offset = params.get('z_offset', 0.0)

        # Calculate target Z coordinate
        target_z = min_z + z_offset
        print(f"Target Z coordinate: {target_z:.4f} (offset: {z_offset:.4f})")

        # Drape all points to the target Z coordinate
        points[:, 2] = target_z

        # Create new PointCloud with draped points
        # Preserve colors if they exist, but discard normals (no longer valid after projection)
        draped_pc = PointCloud(
            points=points,
            colors=point_cloud.colors.copy() if point_cloud.colors is not None else None,
            normals=None  # Normals are no longer valid after Z-projection
        )

        # Copy other attributes if they exist
        # Note: distToGround is no longer valid after draping, so we don't copy it
        if hasattr(point_cloud, 'intensity') and len(point_cloud.intensity) > 0:
            draped_pc.intensity = point_cloud.intensity.copy()

        # Copy custom attributes from the attributes dictionary
        if hasattr(point_cloud, 'attributes') and point_cloud.attributes:
            for attr_name, attr_value in point_cloud.attributes.items():
                if isinstance(attr_value, np.ndarray):
                    draped_pc.attributes[attr_name] = attr_value.copy()

        print(f"Draping completed successfully.")

        # Return the draped point cloud, its type, and dependencies
        return draped_pc, "point_cloud", [data_node.uid]