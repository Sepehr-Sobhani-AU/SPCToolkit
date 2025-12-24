# tasks/apply_container.py
"""
Pass-through task for container nodes.

Container nodes are organizational only and don't modify the point cloud.
"""

from core.point_cloud import PointCloud


class ApplyContainer:
    """
    Pass-through task for container nodes.

    Container nodes are used for organizing the tree structure and don't
    contain actual data or modify the point cloud. This task simply returns
    the point cloud unchanged.
    """

    def __init__(self, point_cloud: PointCloud, container_data):
        """
        Initialize the task.

        Args:
            point_cloud (PointCloud): The point cloud to pass through
            container_data: Container data (typically None)
        """
        self.point_cloud = point_cloud
        # Container data is ignored

    def execute(self) -> PointCloud:
        """
        Execute the task (pass-through).

        Returns:
            PointCloud: The same point cloud unchanged
        """
        return self.point_cloud
