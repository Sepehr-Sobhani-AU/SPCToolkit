# TODO: Add merge point clouds functionality
"""
Module for managing multiple point cloud data within the application.

This module defines the `PointClouds` class, which serves as a container and manager for individual
`PointCloud` instances. It provides functionality for adding, removing, and retrieving point clouds
by unique identifiers. The class also emits a signal when the collection of point clouds is updated,
allowing other components to respond to any changes in the data set.

Classes:
    PointClouds: Manages a collection of `PointCloud` instances, enabling storage, retrieval, and deletion
    by UUIDs, and acting as a registry of point cloud data within the application.

Usage:
    point_clouds = PointClouds()
    point_cloud_uuid = point_clouds.add_point_cloud(point_cloud_instance)
    retrieved_cloud = point_clouds.get_point_cloud(point_cloud_uuid)
    point_clouds.remove_point_cloud(point_cloud_uuid)
"""


# Standard library imports
from typing import Optional
import uuid

# Third party imports
from PyQt5.QtCore import QObject, pyqtSignal

# Local application imports
from core.point_cloud import PointCloud


class PointClouds(QObject):
    """
    Manages a collection of `PointCloud` instances, each identified by a unique UUID.

    Provides methods for adding, removing, and retrieving point cloud data within the application.
    Internally, each `PointCloud` instance is stored with its metadata and referenced by a UUID,
    allowing efficient and consistent access to point cloud objects as needed.

    Attributes:
        _point_clouds (dict): Internal dictionary mapping UUIDs to their corresponding `PointCloud` instances.

    Methods:
        add_point_cloud(point_cloud): Adds a `PointCloud` instance to the collection and returns its UUID.
        remove_point_cloud(point_cloud_uuid): Removes a `PointCloud` from the collection by its UUID.
        get_point_cloud(point_cloud_uuid): Retrieves the `PointCloud` instance associated with a UUID.
    """

    # Signal to notify about changes in the point clouds dictionary
    point_clouds_updated = pyqtSignal(dict)

    def __init__(self):
        """Initializes the class with an empty dictionary for storing point clouds."""
        super().__init__()
        self._point_clouds = {}  # Dictionary to store point cloud instances by unique identifier
        self._previous_keys = set()  # Store the previous keys for detecting changes

    def add_point_cloud(self, point_cloud: PointCloud) -> str:
        """
        Adds a new `PointCloud` instance to the manager and assigns a UUID to it.

        Args:
            point_cloud (PointCloud): The `PointCloud` instance to add.

        Returns:
            UUID (str): A unique identifier assigned to the newly added point cloud.
        """

        # Generate a UUID and assign it to the item using the custom role
        point_cloud_uuid = str(uuid.uuid4())
        point_cloud.uuid = point_cloud_uuid

        # Store the point cloud data in the internal dictionary
        self._point_clouds[point_cloud_uuid] = point_cloud

        # Emit the signal to notify about the changes
        self._notify_changes()

        return point_cloud_uuid

    def remove_point_cloud(self, point_cloud_uuid: str) -> bool:
        """
        Removes a `PointCloud` instance from the manager using its UUID.

        Args:
            point_cloud_uuid (uuid.UUID): The unique identifier of the `PointCloud` to remove.

        Returns:
            bool: True if the point cloud was successfully removed, False if not found.
        """

        if point_cloud_uuid in self._point_clouds:
            # Remove the point cloud from the internal dictionary
            del self._point_clouds[point_cloud_uuid]

            # Emit the signal to notify about the changes
            self._notify_changes()

            return True
        else:
            return False

    def get_point_cloud(self, point_cloud_uuid: str) -> Optional[PointCloud]:
        """
        Retrieves a `PointCloud` instance by its UUID.

        Args:
            point_cloud_uuid (str): The unique identifier for the `PointCloud` to retrieve.

        Returns:
            Optional[PointCloud]: The `PointCloud` instance if found, otherwise None.
        """

        # Retrieve the point cloud from the internal dictionary
        point_cloud = self._point_clouds.get(point_cloud_uuid)

        if point_cloud:
            return point_cloud
        else:
            return None

    def _notify_changes(self):
        """
        Compares current and previous keys, and emits a signal when a change is detected.
        """
        current_keys = set(self._point_clouds.keys())
        if current_keys != self._previous_keys:

            # Emit the signal with the updated point clouds dictionary
            self.point_clouds_updated.emit(self._point_clouds)

            # Update previous keys to reflect the new state of the dictionary for the next comparison
            self._previous_keys = current_keys
