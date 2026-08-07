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

# Renderable vector-feature node types. "cad_object" is the pre-rename string
# kept for backwards compatibility with .pcdtk projects saved before 2026-05-17.
_VECTOR_FEATURE_TYPES = ("vector_feature", "cad_object")


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

        # Per-branch cached Nx6 vertex slices (post-LOD). Reused across
        # visibility toggles so a toggle becomes O(toggled branch) instead
        # of O(all visible points). Cleared on sample-rate change.
        self._branch_vertex_cache: Dict[str, np.ndarray] = {}
        self._branch_cache_sample_rate: Optional[float] = None

        # Which point of the source cloud each rendered point came from, per
        # branch — the LOD subsample indices, or None when the branch is drawn
        # whole. Kept in step with _branch_vertex_cache. Without it nothing
        # downstream can tell rendered row k from cloud row k, and per-point
        # lookups against the source data (cluster labels, and so what may be
        # picked) silently read the wrong point. See PCDViewerWidget.cloud_index.
        self._branch_sample_indices: Dict[str, Optional[np.ndarray]] = {}

    def invalidate_branch(self, uid) -> None:
        """Drop cached vertex slice for a branch (call when its data changes)."""
        key = str(uid)
        self._branch_vertex_cache.pop(key, None)
        self._branch_sample_indices.pop(key, None)

    def invalidate_all(self) -> None:
        """Drop all cached vertex slices."""
        self._branch_vertex_cache.clear()
        self._branch_sample_indices.clear()
        self._branch_cache_sample_rate = None

    @property
    def branch_sample_indices(self) -> Dict[str, Optional[np.ndarray]]:
        """Rendered-row -> source-row map per branch from the last prepare."""
        return self._branch_sample_indices

    @property
    def current_sample_rate(self) -> float:
        return self._current_sample_rate

    @property
    def total_visible_points(self) -> int:
        return self._total_visible_points

    def prepare_branches(self, visibility_status: dict,
                         sample_rate: float = 1.0,
                         camera_distance: float = 1.0,
                         zoom_factor: float = 1.0,
                         max_extent: float = 1.0):
        """Prepare per-branch vertex slices for all visible nodes.

        Handles reconstruction, caching, and LOD subsampling. Does NOT
        concatenate — that work belongs to the viewer (and only happens
        lazily there when picking/zoom needs a global index).

        Args:
            visibility_status: Dict mapping uid strings to visibility bools.
            sample_rate: LOD sample rate (0.01 to 1.0), 1.0 = full resolution.
            camera_distance: Current camera distance (for LOD).
            zoom_factor: Current zoom factor (for LOD).
            max_extent: Max extent of visible point cloud (for LOD).

        Returns:
            (slices_by_uid, visible_order) where:
              - slices_by_uid: ``{uid: Nx6 float32}``, only point branches
                that have data (vector features are skipped — they render
                via prepare_mesh_lines()).
              - visible_order: ordered list of uids matching the user's
                visibility intent.
        """
        from infrastructure.memory_manager import MemoryManager

        uids_to_show = [uid for uid, vis in visibility_status.items() if vis]
        logger.debug(f"Visible branches: {len(uids_to_show)}")

        if not uids_to_show:
            self._total_visible_points = 0
            self._last_node_metadata = {}
            return {}, []

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

        # LOD change invalidates per-branch vertex cache (subsample bound to rate).
        if self._branch_cache_sample_rate != sample_rate:
            self._branch_vertex_cache.clear()
            self._branch_sample_indices.clear()
            self._branch_cache_sample_rate = sample_rate

        # Debug memory estimate
        points_to_check = int(total_points * sample_rate) if sample_rate < 1.0 else total_points
        estimates = MemoryManager.estimate_render_memory(points_to_check, cached=all_cached)
        logger.debug(
            f"Memory estimate for {points_to_check:,} points: "
            f"RAM={estimates['ram_mb']}MB, VRAM={estimates['vram_mb']}MB"
        )

        # Per-node metadata for GUI updates
        node_metadata = {}

        # Build (or reuse) per-branch Nx6 slices. The viewer receives the
        # dict directly — no concatenation here, so toggling visibility
        # never pays for an O(total visible points) memcpy.
        slices_by_uid: Dict[str, np.ndarray] = {}
        visible_order: List[str] = []
        total_rendered = 0

        for uid_idx, uid in enumerate(uids_to_show):
            logger.debug(f"Processing branch {uid_idx + 1}/{len(uids_to_show)}: {uid[:8]}...")
            try:
                node = self.data_nodes.get_node(uuid.UUID(uid))
                if node is None:
                    logger.warning(f"Node not found: {uid}")
                    continue

                # Vector features render as line geometry, not points
                if node.data_type in _VECTOR_FEATURE_TYPES:
                    continue

                cached_slice = self._branch_vertex_cache.get(uid)
                if cached_slice is not None:
                    node_metadata[uid] = {
                        'memory_usage': getattr(node, 'memory_size', 0),
                        'newly_cached': False,
                        'is_cached': node.is_cached,
                    }
                    slices_by_uid[uid] = cached_slice
                    visible_order.append(uid)
                    total_rendered += len(cached_slice)
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
                sample_indices = None
                if sample_rate < 1.0:
                    indices = LODManager.subsample_indices(n, sample_rate)
                    if indices is not None:
                        pts = point_cloud.points[indices]
                        clrs = point_cloud.colors[indices] if point_cloud.colors is not None else None
                        sample_indices = np.asarray(indices)
                        logger.debug(f"LOD subsampled: {n:,} -> {len(indices):,}")
                    else:
                        pts = point_cloud.points
                        clrs = point_cloud.colors
                else:
                    pts = point_cloud.points
                    clrs = point_cloud.colors
                self._branch_sample_indices[uid] = sample_indices

                slice_n = len(pts)
                branch_slice = np.empty((slice_n, 6), dtype=np.float32)
                branch_slice[:, :3] = pts
                if clrs is not None:
                    branch_slice[:, 3:] = clrs
                else:
                    branch_slice[:, 3:] = 1.0  # White

                self._branch_vertex_cache[uid] = branch_slice
                slices_by_uid[uid] = branch_slice
                visible_order.append(uid)
                total_rendered += slice_n

            except Exception as e:
                logger.error(f"Error processing branch {uid}: {e}")
                logger.error(traceback.format_exc())
                continue

        # Drop cache entries for branches no longer visible (bound memory).
        visible_set = set(uids_to_show)
        for stale_uid in [k for k in self._branch_vertex_cache if k not in visible_set]:
            self._branch_vertex_cache.pop(stale_uid, None)
            self._branch_sample_indices.pop(stale_uid, None)

        self._last_node_metadata = node_metadata

        logger.info(f"Rendering {total_rendered:,} points across "
                    f"{len(visible_order)} branches (LOD: {sample_rate:.1%})")
        return slices_by_uid, visible_order

    def get_node_metadata(self) -> dict:
        """Get per-node metadata from the last prepare_branches() call."""
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
        if node.data_type in _VECTOR_FEATURE_TYPES:
            return 0
        return 0

    def prepare_mesh_lines(self, visibility_status: dict):
        """
        Collect wireframe/polyline data from visible VectorFeature nodes.

        Scans visible nodes for ``data_type in ("vector_feature", "cad_object")``,
        reads their geometry, applies each object's transform_matrix, and
        assembles the result into arrays suitable for ``PCDViewerWidget.set_lines()``.
        (The "cad_object" string is accepted for back-compat with pre-2026-05-17
        projects.)

        Returns:
            Tuple of (vertices, edges, colors) or (None, None, None) when
            there are no visible CAD objects.

            - vertices: (V, 3) float32 — all transformed vertex positions.
            - edges: (E, 2) uint32 — index pairs into *vertices*.
            - colors: (V, 3) float32 — per-vertex RGB colours.
        """
        all_verts = []
        all_edges = []
        all_colors = []
        vertex_offset = 0

        for uid, vis in visibility_status.items():
            if not vis:
                continue
            try:
                node = self.data_nodes.get_node(uuid.UUID(uid))
            except (ValueError, AttributeError):
                continue
            if node is None or node.data_type not in _VECTOR_FEATURE_TYPES:
                continue

            feature = node.data
            geom = feature.geometry
            T = feature.transform_matrix

            if feature.geometry_type == "mesh":
                verts = np.asarray(geom["vertices"], dtype=np.float64)
                edges = np.asarray(geom["edges"], dtype=np.uint32)
            elif feature.geometry_type == "polyline":
                verts = np.asarray(geom["vertices"], dtype=np.float64)
                n = len(verts)
                pairs = [[i, i + 1] for i in range(n - 1)]
                if geom.get("closed", False) and n > 2:
                    pairs.append([n - 1, 0])
                if not pairs:
                    continue
                edges = np.array(pairs, dtype=np.uint32)
            else:
                continue

            # Apply transform: world_verts = (verts @ R_S) + t
            ones = np.ones((len(verts), 1), dtype=np.float64)
            homo = np.hstack([verts, ones])            # (V, 4)
            transformed = (T @ homo.T).T[:, :3]        # (V, 3)
            transformed = transformed.astype(np.float32)

            # Offset edge indices for the combined array
            all_verts.append(transformed)
            all_edges.append(edges + vertex_offset)
            all_colors.append(
                np.tile(feature.color, (len(transformed), 1))
            )
            vertex_offset += len(transformed)

        if not all_verts:
            return None, None, None

        vertices = np.concatenate(all_verts, axis=0)
        edges = np.concatenate(all_edges, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        return vertices, edges, colors

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
