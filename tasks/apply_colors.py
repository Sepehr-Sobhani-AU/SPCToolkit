# tasks/apply_colors.py
from core.point_cloud import PointCloud
from core.colors import Colors


class ApplyColors:
    """
    A task class for applying colors to a point cloud.

    This class takes a PointCloud instance and a Colors instance and applies
    the colors to the point cloud directly.
    """

    def __init__(self, point_cloud: PointCloud, colors: Colors):
        """
        Initialize the apply colors task.

        Args:
            point_cloud (PointCloud): The point cloud to modify
            colors (Colors): The colors to apply to the point cloud
        """
        self.point_cloud = point_cloud
        self.colors = colors.colors  # Access the numpy array from the Colors object

    def execute(self) -> PointCloud:
        """
        Execute the task to apply colors to the point cloud.

        Returns:
            PointCloud: The point cloud with applied colors
        """
        # Apply the colors directly to the point cloud
        if len(self.colors) == len(self.point_cloud.points):
            self.point_cloud.colors = self.colors
        else:
            print(
                f"Warning: Number of colors ({len(self.colors)}) doesn't match number of points ({len(self.point_cloud.points)})")

        return self.point_cloud