"""
Application Controller — main orchestrator for all operations.

GUI components call methods here, never Core directly.
Bridges GUI ↔ Core with proper layer separation.
"""
import logging
import uuid
from typing import Optional, Callable, Dict, Any

import numpy as np

from core.entities.data_node import DataNode
from core.entities.data_nodes import DataNodes
from core.entities.point_cloud import PointCloud
from core.services.reconstruction_service import ReconstructionService
from core.services.cache_service import CacheService
from plugins.plugin_manager import PluginManager

logger = logging.getLogger(__name__)


class ApplicationController:
    """
    Main entry point for all operations.
    GUI components call methods here via global_variables singleton.
    """

    def __init__(self,
                 data_nodes: DataNodes,
                 reconstruction_service: ReconstructionService,
                 cache_service: CacheService,
                 plugin_manager: PluginManager):
        self.data_nodes = data_nodes
        self.reconstruction_service = reconstruction_service
        self.cache_service = cache_service
        self.plugin_manager = plugin_manager

        # Will be set after construction (circular dependency with Application layer)
        self.analysis_executor = None
        self.rendering_coordinator = None

        # Selection state
        self.selected_branches = []

        # Single-level undo for cluster operations: {str(node_uid): previous Clusters}
        self._cluster_undo = {}

    @classmethod
    def create(cls, plugin_manager: PluginManager, file_manager=None):
        """
        Factory method that creates ApplicationController with all services.

        Creates DataNodes, ReconstructionService, CacheService, AnalysisService,
        AnalysisExecutor, and RenderingCoordinator.

        Args:
            plugin_manager: The PluginManager instance.
            file_manager: The FileManager instance (unused, reserved for future).

        Returns:
            Fully wired ApplicationController instance.
        """
        from core.services.analysis_service import AnalysisService
        from application.analysis_executor import AnalysisExecutor
        from application.rendering_coordinator import RenderingCoordinator

        data_nodes = DataNodes()
        reconstruction_service = ReconstructionService(data_nodes)
        cache_service = CacheService(data_nodes)
        analysis_service = AnalysisService()

        controller = cls(data_nodes, reconstruction_service, cache_service, plugin_manager)

        controller.analysis_executor = AnalysisExecutor(
            reconstruction_service, cache_service, analysis_service
        )
        controller.rendering_coordinator = RenderingCoordinator(
            data_nodes, reconstruction_service, cache_service
        )

        # When the reconstruction cache is invalidated for any uid, drop the
        # rendering coordinator's per-branch vertex cache for the same uid so
        # the next render reflects the updated point cloud.
        def _on_cache_invalidated(uids):
            rc = controller.rendering_coordinator
            if rc is None:
                return
            for uid in uids:
                rc.invalidate_branch(uid)
        cache_service.add_invalidate_listener(_on_cache_invalidated)

        return controller

    # === Data Operations ===

    def add_point_cloud(self, point_cloud: PointCloud, name: str) -> str:
        """
        Add a new root point cloud.

        Args:
            point_cloud: The PointCloud instance.
            name: Display name for the node.

        Returns:
            UID of the new node as string.
        """
        data_node = DataNode(
            params=name,
            data=point_cloud,
            data_type="point_cloud",
            parent_uid=None,
            depends_on=None,
            tags=[]
        )
        data_node.memory_size = self._calculate_point_cloud_memory(point_cloud)

        uid = self.data_nodes.add_node(data_node)
        return str(uid)

    def add_analysis_result(self, result, result_type: str, dependencies: list,
                            parent_node: DataNode, analysis_type: str,
                            params: dict) -> str:
        """
        Add an analysis result as a child node.

        Args:
            result: The analysis result data.
            result_type: Type identifier for the result.
            dependencies: List of dependency UIDs.
            parent_node: The parent DataNode.
            analysis_type: Name of the analysis performed.
            params: Parameters used for the analysis.

        Returns:
            UID of the new node as string.
        """
        data_node_result = DataNode(
            analysis_type,
            data=result,
            data_type=result_type,
            parent_uid=parent_node.uid,
            depends_on=dependencies,
            tags=[analysis_type, params]
        )

        if result_type == "point_cloud":
            data_node_result.memory_size = self._calculate_point_cloud_memory(result)
        else:
            data_node_result.memory_size = self._estimate_derived_memory(result)

        uid = self.data_nodes.add_node(data_node_result)
        return str(uid)

    def remove_node(self, uid: str) -> bool:
        """Remove a node and all descendants."""
        try:
            # DataNodes.remove_node, not remove_data — the latter has never
            # existed, so every call through here failed into the except below
            # and only logged it. The node survived; only its tree row went.
            self.data_nodes.remove_node(uuid.UUID(uid))
            self.cache_service.invalidate_descendants(uid)
            return True
        except Exception as e:
            logger.error(f"Failed to remove node {uid}: {e}")
            return False

    def get_node(self, uid: str) -> Optional[DataNode]:
        """Get a node by UID."""
        return self.data_nodes.get_node(uuid.UUID(uid))

    # === Selection ===

    def set_selected_branches(self, uids: list):
        """Update the currently selected branches."""
        self.selected_branches = uids

    # === Analysis Operations ===

    def run_analysis(self,
                     plugin_name: str,
                     params: dict,
                     on_progress: Callable[[int, str], None] = None,
                     on_complete: Callable[[str], None] = None,
                     on_error: Callable[[str], None] = None) -> None:
        """
        Run analysis on all selected branches.

        Args:
            plugin_name: Name of analysis plugin.
            params: Analysis parameters from dialog.
            on_progress: Callback for progress updates (percent, message).
            on_complete: Callback on success (new_uid).
            on_error: Callback on failure (error_message).
        """
        if self.analysis_executor is None:
            if on_error:
                on_error("AnalysisExecutor not initialized")
            return

        plugins = self.plugin_manager.get_analysis_plugins()
        if plugin_name not in plugins:
            if on_error:
                on_error(f"Plugin '{plugin_name}' not found")
            return

        plugin_class = plugins[plugin_name]

        for uid in self.selected_branches:
            data_node = self.data_nodes.get_node(uuid.UUID(uid))
            if data_node is None:
                if on_error:
                    on_error(f"DataNode '{uid}' not found")
                continue

            self.analysis_executor.execute(
                plugin_class=plugin_class,
                data_node=data_node,
                params=params,
                analysis_type=plugin_name,
                on_progress=on_progress,
                on_complete=on_complete,
                on_error=on_error
            )

    def get_analysis_confirmation(self, plugin_name: str, params: dict) -> Optional[str]:
        """
        Ask the plugin whether it wants to confirm a potentially-expensive run
        before dispatch. Returns the first warning message produced across the
        selected branches (to be shown as a Yes/No prompt on the main thread),
        or None if no confirmation is needed.

        This is a cheap, main-thread pre-flight check -- see
        ``Plugin.confirm_before_execute``.
        """
        plugins = self.plugin_manager.get_analysis_plugins()
        plugin_class = plugins.get(plugin_name)
        if plugin_class is None:
            return None

        try:
            plugin = plugin_class()
        except Exception as e:
            logger.debug(f"Could not instantiate '{plugin_name}' for confirmation: {e}")
            return None

        for uid in self.selected_branches:
            node = self.data_nodes.get_node(uuid.UUID(uid))
            if node is None:
                continue
            try:
                message = plugin.confirm_before_execute(node, params)
            except Exception as e:
                logger.debug(f"confirm_before_execute failed for '{plugin_name}': {e}")
                message = None
            if message:
                return message
        return None

    def is_analysis_running(self) -> bool:
        """Check if analysis is in progress."""
        if self.analysis_executor is None:
            return False
        return self.analysis_executor.is_running()

    # === Reconstruction & Cache ===

    def reconstruct(self, uid) -> PointCloud:
        """
        Reconstruct a node to PointCloud.

        Args:
            uid: UUID of the node (str or uuid.UUID).

        Returns:
            Reconstructed PointCloud.
        """
        return self.reconstruction_service.reconstruct(uid)

    def cache_node(self, uid: str) -> None:
        """Cache a node's reconstruction."""
        if self.cache_service.is_cached(uid):
            return

        point_cloud = self.reconstruction_service.reconstruct(uid)
        self.cache_service.set(uid, point_cloud)
        logger.debug(f"Cached node: {uid}")

    def uncache_node(self, uid: str) -> None:
        """Remove cached data for a node."""
        self.cache_service.invalidate(uid)
        logger.debug(f"Uncached node: {uid}")

    def is_cached(self, uid: str) -> bool:
        """Check if a node is cached."""
        return self.cache_service.is_cached(uid)

    def get_cache_memory_usage(self, uid: str) -> str:
        """Get memory usage of a cached node."""
        return self.cache_service.get_memory_usage(uid)

    # === Memory Helpers ===

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

    @staticmethod
    def _estimate_derived_memory(data) -> str:
        """Estimate memory usage of derived data types."""
        bytes_used = 0

        if hasattr(data, 'nbytes'):
            bytes_used = data.nbytes
        elif hasattr(data, '__dict__'):
            for attr_name, attr_value in data.__dict__.items():
                if hasattr(attr_value, 'nbytes'):
                    bytes_used += attr_value.nbytes

        mb_used = bytes_used / (1024 * 1024)
        return f"{mb_used:.2f} MB" if mb_used > 0 else "< 0.01 MB"

    def get_node_point_count(self, node) -> int:
        """Get point count from a node (cached or from data)."""
        if node.is_cached and node.cached_point_cloud:
            return node.cached_point_cloud.size
        elif node.data:
            if hasattr(node.data, 'size'):
                return node.data.size
            elif hasattr(node.data, 'labels'):
                return len(node.data.labels)
            elif hasattr(node.data, 'points'):
                return len(node.data.points)
        if node.data_type in ("class_reference", "container"):
            return 100000
        return 0

    # Primary per-point array attribute for each derived data class.
    _DATA_LENGTH_ATTRS = ("points", "mask", "labels", "values",
                          "eigenvalues", "normals", "colors", "distances")

    def get_node_reconstructed_count(self, node) -> Optional[int]:
        """
        Number of points the branch yields when reconstructed, computed cheaply
        without reconstructing the cloud.

        - masks: number of selected points (count of True in the mask).
        - class_reference: points whose label is in cluster_ids (from the
          nearest cluster_labels ancestor).
        - transform_matrix: the parent's count (the transform moves points but
          preserves their identity, order and count).
        - identity / point_cloud / other per-point data: the array length
          (one entry per reconstructed point).

        Returns None when it can't be determined (e.g. container nodes).
        """
        data = getattr(node, "data", None)
        if data is None:
            return None

        data_type = node.data_type
        if data_type == "masks":
            mask = getattr(data, "mask", None)
            return int(np.count_nonzero(mask)) if mask is not None else None
        if data_type == "class_reference":
            labels = self._nearest_cluster_labels(node)
            if labels is None:
                return None
            return int(np.count_nonzero(np.isin(labels, getattr(data, "cluster_ids", []))))
        if data_type == "transform_matrix":
            parent = self.data_nodes.get_node(node.parent_uid) if node.parent_uid else None
            return self.get_node_reconstructed_count(parent) if parent is not None else None

        for attr in self._DATA_LENGTH_ATTRS:
            arr = getattr(data, attr, None)
            if arr is not None and hasattr(arr, "__len__"):
                return len(arr)
        return None

    def _nearest_cluster_labels(self, node):
        """Labels of the nearest cluster_labels ancestor of ``node``, or None."""
        current = self.data_nodes.get_node(node.parent_uid) if node.parent_uid else None
        while current is not None:
            if current.data_type == "cluster_labels":
                return getattr(current.data, "labels", None)
            current = self.data_nodes.get_node(current.parent_uid) if current.parent_uid else None
        return None

    def update_all_branch_memory_labels(self) -> dict:
        """
        Calculate memory labels for all nodes.

        Returns:
            Dict mapping uid_str to memory_size string.
        """
        result = {}
        for uid, node in self.data_nodes.data_nodes.items():
            uid_str = str(uid)
            if hasattr(node, 'memory_size') and node.memory_size:
                result[uid_str] = node.memory_size
            elif node.data_type == "point_cloud":
                memory_size = self._calculate_point_cloud_memory(node.data)
                node.memory_size = memory_size
                result[uid_str] = memory_size
            else:
                memory_size = self._estimate_derived_memory(node.data)
                node.memory_size = memory_size
                result[uid_str] = memory_size
        return result

    def load_project(self, loaded_data_nodes: DataNodes):
        """
        Replace current data_nodes with loaded ones and update all service references.

        Called by load_project_plugin after deserializing a .pcdtk file.

        Args:
            loaded_data_nodes: The DataNodes instance from the loaded project.
        """
        from config.config import global_variables

        self.data_nodes = loaded_data_nodes
        global_variables.global_data_nodes = loaded_data_nodes

        # Update service references so reconstruction/cache use the new data
        self.reconstruction_service.data_nodes = loaded_data_nodes
        self.cache_service.data_nodes = loaded_data_nodes
        if self.rendering_coordinator is not None:
            self.rendering_coordinator.data_nodes = loaded_data_nodes
