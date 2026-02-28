"""
Task for applying normal-based visualization to a point cloud.

Maps surface normal directions to RGB colors: the absolute XYZ components
of each normal vector become the Red, Green, Blue channels respectively.
"""

from core.entities.point_cloud import PointCloud
from core.entities.normals import Normals
import numpy as np


class NormalsTransformer:
    """
    Task for applying normals to a point cloud for visualization.

    Color mapping (standard normal colorization):
        R = |nx|  (surfaces facing X axis → red)
        G = |ny|  (surfaces facing Y axis → green)
        B = |nz|  (surfaces facing Z axis → blue)
    """

    def __init__(self, point_cloud: PointCloud, normals: Normals):
        self.point_cloud = point_cloud
        self.normals = normals

    def execute(self):
        """
        Apply normals to the point cloud as colors and attributes.

        Returns:
            PointCloud: The point cloud with normal-based colors and normals stored.
        """
        result_point_cloud = PointCloud(
            points=self.point_cloud.points,
            colors=self.point_cloud.colors,
            normals=self.normals.normals,
            intensity=getattr(self.point_cloud, 'intensity', np.array([])),
            distToGround=getattr(self.point_cloud, 'distToGround', np.array([])),
            params=self.point_cloud.name,
        )
        result_point_cloud.attributes = self.point_cloud.attributes.copy()

        if len(self.normals.normals) != result_point_cloud.size:
            raise ValueError(
                f"Normal count ({len(self.normals.normals)}) "
                f"does not match point count ({result_point_cloud.size})"
            )

        # Map absolute normal components to RGB
        colors = np.abs(self.normals.normals)  # (N, 3) in [0, 1]
        result_point_cloud.colors = colors.astype(np.float32)

        # Store normals in attributes for downstream use
        result_point_cloud.add_attribute('normals', self.normals.normals)

        print(f"Applied normal-based colors to point cloud and stored normals in attributes")

        return result_point_cloud
