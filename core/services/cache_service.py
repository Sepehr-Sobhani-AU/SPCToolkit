"""
Cache service for managing reconstructed PointCloud caches.

Stores and retrieves cached PointCloud instances on DataNode objects.
Pure Core service — no GUI or framework imports.
"""
import time
import uuid
from typing import Optional


class CacheService:
    """Centralized cache management for reconstructed PointClouds."""

    def __init__(self, data_nodes):
        """
        Args:
            data_nodes: DataNodes collection instance.
        """
        self.data_nodes = data_nodes

    def get(self, uid) -> Optional[object]:
        """
        Get cached reconstruction for a node.

        Args:
            uid: UUID of the node (str or uuid.UUID).

        Returns:
            Cached PointCloud or None if not cached.
        """
        node = self._get_node(uid)
        if node is None:
            return None

        if node.is_cached and node.cached_point_cloud is not None:
            return node.cached_point_cloud

        return None

    def set(self, uid, point_cloud) -> None:
        """
        Cache a reconstruction result for a node.

        Args:
            uid: UUID of the node (str or uuid.UUID).
            point_cloud: PointCloud instance to cache.
        """
        node = self._get_node(uid)
        if node is None:
            return

        node.cached_point_cloud = point_cloud
        node.is_cached = True
        node.cache_timestamp = time.time()

    def invalidate(self, uid) -> None:
        """
        Remove cached data for a node.

        Args:
            uid: UUID of the node (str or uuid.UUID).
        """
        node = self._get_node(uid)
        if node is None:
            return

        node.cached_point_cloud = None
        node.is_cached = False
        node.cache_timestamp = None

    def invalidate_descendants(self, uid) -> None:
        """
        Invalidate caches for all descendants of a node.

        Clears cached_point_cloud and cache_timestamp but keeps
        is_cached=True so user preference is preserved.

        Args:
            uid: UUID of the parent node (str or uuid.UUID).
        """
        if isinstance(uid, str):
            uid_obj = uuid.UUID(uid)
        else:
            uid_obj = uid

        for node_uid, node in self.data_nodes.data_nodes.items():
            if self._is_descendant_of(node, uid_obj) and node.is_cached:
                node.cached_point_cloud = None
                node.cache_timestamp = None

    def is_cached(self, uid) -> bool:
        """
        Check if a node has a valid cache.

        Args:
            uid: UUID of the node (str or uuid.UUID).

        Returns:
            True if the node has cached data in memory.
        """
        node = self._get_node(uid)
        if node is None:
            return False

        return node.is_cached and node.cached_point_cloud is not None

    def get_memory_usage(self, uid) -> str:
        """
        Calculate approximate memory usage of cached point cloud.

        Args:
            uid: UUID of the node (str or uuid.UUID).

        Returns:
            Memory usage in human-readable format (e.g., "12.34 MB").
        """
        node = self._get_node(uid)
        if node is None:
            return "0 MB"

        if hasattr(node, 'memory_size') and node.memory_size:
            return node.memory_size

        if node.is_cached and node.cached_point_cloud is not None:
            return self._calculate_point_cloud_memory(node.cached_point_cloud)

        return "0 MB"

    def _get_node(self, uid):
        """Resolve uid to DataNode."""
        if isinstance(uid, str):
            uid = uuid.UUID(uid)
        return self.data_nodes.get_node(uid)

    def _is_descendant_of(self, node, ancestor_uid) -> bool:
        """Check if node is a descendant of ancestor_uid."""
        current = node
        while current.parent_uid is not None:
            if current.parent_uid == ancestor_uid:
                return True
            current = self.data_nodes.get_node(current.parent_uid)
            if current is None:
                break
        return False

    @staticmethod
    def _calculate_point_cloud_memory(point_cloud) -> str:
        """Calculate approximate memory usage of a PointCloud."""
        if point_cloud is None:
            return "0 MB"

        bytes_used = 0

        if hasattr(point_cloud, 'points') and point_cloud.points is not None:
            bytes_used += point_cloud.points.nbytes

        if hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
            bytes_used += point_cloud.colors.nbytes

        if hasattr(point_cloud, 'normals') and point_cloud.normals is not None:
            bytes_used += point_cloud.normals.nbytes

        if hasattr(point_cloud, 'attributes') and point_cloud.attributes is not None:
            for key, value in point_cloud.attributes.items():
                if hasattr(value, 'nbytes'):
                    bytes_used += value.nbytes

        mb_used = bytes_used / (1024 * 1024)
        return f"{mb_used:.2f} MB"
