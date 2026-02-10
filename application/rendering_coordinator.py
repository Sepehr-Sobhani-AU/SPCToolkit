"""
Rendering Coordinator — manages visibility state and prepares render data.

Handles LOD, reconstruction, caching, and vertex buffer assembly for the viewer.
"""
import logging
import time
import traceback
import uuid
from typing import Dict, List, Optional

import numpy as np

from core.entities.data_nodes import DataNodes
from core.services.reconstruction_service import ReconstructionService
from core.services.cache_service import CacheService
from application.lod_manager import LODManager

logger = logging.getLogger(__name__)


class RenderingCoordinator:
    """
    Manages visibility state and prepares render data for the viewer.

    Handles reconstruction, caching, LOD subsampling, and vertex buffer
    assembly. No direct GUI imports — returns data for the GUI to display.
    """

    def __init__(self,
                 data_nodes: DataNodes,
                 reconstruction_service: ReconstructionService,
                 cache_service: CacheService):
        self.data_nodes = data_nodes
        self._reconstruction_service = reconstruction_service
        self._cache_service = cache_service

        # LOD state
        self._current_sample_rate: float = 1.0
        self._point_budget: int = 50_000_000
        self._total_visible_points: int = 0

        # Per-branch index ranges in the combined vertex array: uid -> (start, end)
        self.branch_offsets: Dict[str, tuple] = {}

    @property
    def current_sample_rate(self) -> float:
        return self._current_sample_rate

    @property
    def total_visible_points(self) -> int:
        return self._total_visible_points

    def prepare_vertices(self, visibility_status: dict,
                         sample_rate: float = 1.0,
                         camera_distance: float = 1.0,
                         zoom_factor: float = 1.0,
                         max_extent: float = 1.0) -> Optional[np.ndarray]:
        """
        Prepare combined vertex array for all visible nodes.

        Handles reconstruction, caching, LOD subsampling, and vertex
        buffer assembly.

        Args:
            visibility_status: Dict mapping uid strings to visibility bools.
            sample_rate: LOD sample rate (0.01 to 1.0), 1.0 = full resolution.
            camera_distance: Current camera distance (for LOD).
            zoom_factor: Current zoom factor (for LOD).
            max_extent: Max extent of visible point cloud (for LOD).

        Returns:
            Vertex array (N, 6) of [x,y,z,r,g,b] float32, or None if empty.
            Also returns a dict of per-node metadata for GUI updates.
        """
        from infrastructure.memory_manager import MemoryManager

        self.branch_offsets = {}
        uids_to_show = [uid for uid, vis in visibility_status.items() if vis]
        logger.debug(f"Visible branches: {len(uids_to_show)}")

        if not uids_to_show:
            self._total_visible_points = 0
            return None

        # Estimate total points and cache status
        total_points = 0
        all_cached = True
        for uid in uids_to_show:
            try:
                node = self.data_nodes.get_node(uuid.UUID(uid))
                if node:
                    if not (node.is_cached and node.cached_point_cloud):
                        all_cached = False
                    total_points += self._get_node_point_count(node)
            except Exception:
                pass

        self._total_visible_points = total_points
        logger.debug(f"Total points: {total_points:,}, all_cached: {all_cached}")

        # Compute dynamic point budget from available VRAM
        self._point_budget = LODManager.compute_dynamic_point_budget()
        logger.debug(f"Dynamic point budget: {self._point_budget:,}")

        # Auto-compute LOD: enforce point budget regardless of requested sample_rate
        if total_points > self._point_budget:
            max_safe_rate = self._point_budget / total_points
            if sample_rate > max_safe_rate:
                old_rate = sample_rate
                sample_rate = LODManager.compute_sample_rate(
                    total_points,
                    camera_distance,
                    zoom_factor,
                    max_extent or 1.0,
                    self._point_budget
                )
                sample_rate = min(sample_rate, max_safe_rate)
                logger.info(
                    f"AUTO-LOD: Capped {old_rate:.1%} -> {sample_rate:.1%} "
                    f"({total_points:,} points, {self._point_budget:,} budget)"
                )

        self._current_sample_rate = sample_rate

        # Debug memory estimate
        points_to_check = int(total_points * sample_rate) if sample_rate < 1.0 else total_points
        estimates = MemoryManager.estimate_render_memory(points_to_check, cached=all_cached)
        logger.debug(
            f"Memory estimate for {points_to_check:,} points: "
            f"RAM={estimates['ram_mb']}MB, VRAM={estimates['vram_mb']}MB"
        )

        # Pre-allocate vertex array
        if sample_rate < 1.0:
            base_size = int(total_points * sample_rate)
            buffer_size = max(int(base_size * 0.2), 100000)
            alloc_size = min(total_points, base_size + buffer_size)
        else:
            alloc_size = total_points
        vertices = np.empty((alloc_size, 6), dtype=np.float32)
        offset = 0

        # Per-node metadata for GUI updates
        node_metadata = {}

        # Process each visible branch
        for uid_idx, uid in enumerate(uids_to_show):
            logger.debug(f"Processing branch {uid_idx + 1}/{len(uids_to_show)}: {uid[:8]}...")
            try:
                node = self.data_nodes.get_node(uuid.UUID(uid))
                if node is None:
                    logger.warning(f"Node not found: {uid}")
                    continue

                # Reconstruct (uses cache if available)
                point_cloud = self._reconstruction_service.reconstruct(uid)
                n = point_cloud.size
                logger.debug(f"Reconstructed: {n:,} points")

                # Calculate memory usage
                memory_usage = self._calculate_point_cloud_memory(point_cloud)
                node.memory_size = memory_usage

                # Auto-cache if not already cached
                was_newly_cached = False
                if not node.is_cached:
                    self._cache_service.set(uid, point_cloud)
                    was_newly_cached = True
                    logger.debug(f"[AUTO-CACHE] {node.params} ({memory_usage})")

                node_metadata[uid] = {
                    'memory_usage': memory_usage,
                    'newly_cached': was_newly_cached,
                    'is_cached': node.is_cached,
                }

                # Apply per-node subsampling if LOD is active
                if sample_rate < 1.0:
                    indices = LODManager.subsample_indices(n, sample_rate)
                    if indices is not None:
                        n_to_add = len(indices)
                        pts = point_cloud.points[indices]
                        clrs = point_cloud.colors[indices] if point_cloud.colors is not None else None
                        logger.debug(f"LOD subsampled: {n:,} -> {n_to_add:,}")
                    else:
                        n_to_add = n
                        pts = point_cloud.points
                        clrs = point_cloud.colors
                else:
                    n_to_add = n
                    pts = point_cloud.points
                    clrs = point_cloud.colors

                # Resize buffer if needed
                if offset + n_to_add > len(vertices):
                    new_size = int((offset + n_to_add) * 1.2) + 100000
                    logger.warning(f"Resizing vertex buffer: {len(vertices):,} -> {new_size:,}")
                    new_vertices = np.empty((new_size, 6), dtype=np.float32)
                    new_vertices[:offset] = vertices[:offset]
                    vertices = new_vertices
                    del new_vertices

                vertices[offset:offset + n_to_add, :3] = pts
                if clrs is not None:
                    vertices[offset:offset + n_to_add, 3:] = clrs
                else:
                    vertices[offset:offset + n_to_add, 3:] = 1.0  # White
                offset += n_to_add
                self.branch_offsets[uid] = (offset - n_to_add, offset)

            except Exception as e:
                logger.error(f"Error processing branch {uid}: {e}")
                logger.error(traceback.format_exc())
                continue

        logger.info(f"Rendering {offset:,} points (LOD: {sample_rate:.1%})")
        return vertices[:offset]

    def get_node_metadata(self) -> dict:
        """Get per-node metadata from the last prepare_vertices() call."""
        return getattr(self, '_last_node_metadata', {})

    def _get_node_point_count(self, node) -> int:
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
