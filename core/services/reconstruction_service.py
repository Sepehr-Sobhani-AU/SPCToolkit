"""
Reconstruction service for rebuilding PointCloud from derived nodes.

Walks up the DataNode hierarchy, applies transformers sequentially,
and leverages cached results when available.
"""
import uuid
from typing import Dict, Type, Optional

from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode

from core.transformers.values_transformer import ValuesTransformer
from core.transformers.clusters_transformer import ClustersTransformer
from core.transformers.masks_transformer import MasksTransformer
from core.transformers.eigenvalues_transformer import EigenvaluesTransformer
from core.transformers.colors_transformer import ColorsTransformer
from core.transformers.dist_to_ground_transformer import DistToGroundTransformer
from core.transformers.class_reference_transformer import ClassReferenceTransformer


class ReconstructionService:
    """Rebuilds PointCloud from derived nodes by applying transformations."""

    def __init__(self, data_nodes):
        """
        Args:
            data_nodes: DataNodes collection instance.
        """
        self.data_nodes = data_nodes
        self.transformer_registry = {
            "masks": MasksTransformer,
            "cluster_labels": ClustersTransformer,
            "values": ValuesTransformer,
            "eigenvalues": EigenvaluesTransformer,
            "colors": ColorsTransformer,
            "dist_to_ground": DistToGroundTransformer,
            "class_reference": ClassReferenceTransformer,
        }

    def reconstruct(self, uid) -> PointCloud:
        """
        Reconstruct a branch by applying transformations from nearest
        cached ancestor to target.

        Args:
            uid: UUID of the branch to reconstruct (str or uuid.UUID).

        Returns:
            PointCloud: The reconstructed point cloud.

        Raises:
            ValueError: If the data node is not found or type is unknown.
        """
        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        target_node = self.data_nodes.get_node(uid)
        if target_node is None:
            raise ValueError(f"DataNode with UID {uid} not found")

        # If target is cached, return immediately
        if target_node.is_cached and target_node.cached_point_cloud is not None:
            return target_node.cached_point_cloud

        # Build path from target back to root (or first cached ancestor)
        path = [target_node]
        current_node = target_node

        while current_node.parent_uid is not None:
            parent_node = self.data_nodes.get_node(current_node.parent_uid)
            if parent_node is None:
                break

            path.append(parent_node)

            if parent_node.is_cached and parent_node.cached_point_cloud is not None:
                break
            elif parent_node.data_type == "point_cloud" and parent_node.parent_uid is None:
                break

            current_node = parent_node

        # Reverse path to go from start point (cache/root) to target
        path.reverse()

        # Start from the first node (cached ancestor or root)
        start_node = path[0]
        if start_node.is_cached and start_node.cached_point_cloud is not None:
            point_cloud = start_node.cached_point_cloud
        else:
            point_cloud = start_node.data

        # Apply transformations from start to target
        for node in path[1:]:
            if node.data_type == "point_cloud":
                point_cloud = node.data
            elif node.data_type == "container":
                continue
            else:
                point_cloud = self._apply_transformer(point_cloud, node)

        return point_cloud

    def _apply_transformer(self, point_cloud: PointCloud, data_node: DataNode) -> PointCloud:
        """
        Apply the appropriate transformer to reconstruct a single node.

        Args:
            point_cloud: Input PointCloud to transform.
            data_node: DataNode containing the derived data.

        Returns:
            PointCloud: Transformed point cloud.

        Raises:
            ValueError: If the data type has no registered transformer.
        """
        data_type = data_node.data_type
        if data_type not in self.transformer_registry:
            raise ValueError(f"Data type '{data_type}' not found in transformer registry.")

        transformer_class = self.transformer_registry[data_type]
        transformer = transformer_class(point_cloud, data_node.data)
        return transformer.execute()

    def get_reconstruction_path(self, uid) -> list:
        """
        Get ordered UIDs from root PointCloud to target.

        Args:
            uid: UUID of the target node.

        Returns:
            List of UIDs from root to target.
        """
        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        path = []
        current_node = self.data_nodes.get_node(uid)

        while current_node is not None:
            path.append(current_node.uid)
            if current_node.parent_uid is None:
                break
            current_node = self.data_nodes.get_node(current_node.parent_uid)

        path.reverse()
        return path
