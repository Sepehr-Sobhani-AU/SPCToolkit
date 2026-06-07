# tasks/apply_colors.py
import numpy as np

from core.entities.point_cloud import PointCloud
from core.entities.colors import Colors


class ColorsTransformer:
    """
    A task class for applying colors to a point cloud.

    This class takes a PointCloud instance and a Colors instance and returns a
    new PointCloud with those colors applied.
    """

    def __init__(self, point_cloud: PointCloud, colors: Colors):
        """
        Initialize the apply colors task.

        Args:
            point_cloud (PointCloud): The point cloud to recolor
            colors (Colors): The colors to apply to the point cloud
        """
        self.point_cloud = point_cloud
        self.colors = colors.colors  # Access the numpy array from the Colors object

    def execute(self) -> PointCloud:
        """
        Execute the task to apply colors to the point cloud.

        Returns a shallow copy that shares the points/normals arrays with the
        input but carries the new colors. This is important: the input is often
        the parent branch's *cached* PointCloud, so mutating it in place would
        also recolor the parent branch in the viewer.

        Returns:
            PointCloud: A new point cloud with the applied colors
        """
        if len(self.colors) != len(self.point_cloud.points):
            print(
                f"Warning: Number of colors ({len(self.colors)}) doesn't match "
                f"number of points ({len(self.point_cloud.points)})")
            return self.point_cloud

        # Shallow copy - shares point/normal arrays (memory efficient), only the
        # colors are replaced so the parent's cached cloud stays untouched.
        result_point_cloud = PointCloud(
            points=self.point_cloud.points,
            colors=self.colors,
            normals=self.point_cloud.normals,
            intensity=getattr(self.point_cloud, 'intensity', np.array([])),
            distToGround=getattr(self.point_cloud, 'distToGround', np.array([])),
            params=self.point_cloud.name,
        )
        # Copy attributes dictionary so we don't modify the original
        result_point_cloud.attributes = self.point_cloud.attributes.copy()

        return result_point_cloud