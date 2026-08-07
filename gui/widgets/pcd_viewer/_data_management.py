import logging
import traceback
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class DataManagementMixin:
    """Point cloud data loading and management for PCDViewerWidget.

    Holds vertex data as a dict of per-branch Nx6 (xyz + rgb) float32 slices,
    each with its own VBO created lazily on first paint. A visibility toggle
    only updates the visible-uid list — no concat, no upload, no GC.

    A combined Nx6 view (``self.points``) and a ``_branch_offsets`` map are
    materialised lazily, on first access, for code that still uses global
    indices (picking, polygon select, zoom-to-fit).
    """

    def _init_data(self):
        """Initialize point cloud data attributes."""
        # Per-branch vertex storage (uid -> Nx6 float32). The viewer owns
        # the references; the rendering coordinator hands these in via
        # set_branches() and identity is preserved when nothing changed.
        self._branch_vertices: Dict[str, np.ndarray] = {}

        # Per-branch GPU buffers. Created lazily in render_point_cloud()
        # because VBO construction needs an active OpenGL context.
        self._branch_vbos: Dict[str, "object"] = {}

        # VBOs scheduled for deletion. Filled by set_branches() when a
        # branch slice is replaced/removed, drained in render_point_cloud()
        # where the GL context is current.
        self._pending_vbo_deletions: List["object"] = []

        # Ordered list of visible branch uids. Also defines the index order
        # of the lazily-built combined array.
        self._visible_branches: List[str] = []

        # uid -> source-row index per rendered row, when LOD drew only a subset
        # of a branch (None = drawn whole). Set by set_branches from the
        # rendering coordinator; read by cloud_index().
        self._branch_sample_indices: Dict[str, Optional[np.ndarray]] = {}

        # Lazy derived state, invalidated by set_branches().
        self._combined_points_cache: Optional[np.ndarray] = None
        self._branch_offsets_cache: Optional[Dict[str, Tuple[int, int]]] = None
        self._kdtree: Optional[cKDTree] = None

        # Line geometry (e.g. mesh wireframes, CAD polylines). Independent of point data.
        self.line_vertices = None  # Nx3 float32
        self.line_indices = None   # (2*M,) uint32 — flattened edge endpoint indices
        self.line_colors = None    # Nx3 float32 per-vertex colors, or None for uniform gray

        # Initialize list to store indices of picked points
        self.picked_points_indices = []

    # ------------------------------------------------------------------
    # Lazy combined view
    # ------------------------------------------------------------------

    @property
    def points(self) -> Optional[np.ndarray]:
        """Lazy combined Nx6 (xyz + rgb) view of all visible branches.

        Concatenated on first access after set_branches(); cached until the
        next branch change. Returns None when nothing is visible.
        """
        if not self._visible_branches:
            return None
        if self._combined_points_cache is None:
            self._build_combined()
        return self._combined_points_cache

    @property
    def _branch_offsets(self) -> Dict[str, Tuple[int, int]]:
        """uid -> (start, end) ranges in the combined array. Lazy."""
        if not self._visible_branches:
            return {}
        if self._branch_offsets_cache is None:
            self._build_combined()
        return self._branch_offsets_cache or {}

    def _build_combined(self) -> None:
        """Build the combined Nx6 array and the matching offsets map."""
        slices: List[np.ndarray] = []
        offsets: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for uid in self._visible_branches:
            slc = self._branch_vertices.get(uid)
            if slc is None or len(slc) == 0:
                continue
            slices.append(slc)
            offsets[uid] = (offset, offset + len(slc))
            offset += len(slc)

        if not slices:
            self._combined_points_cache = None
            self._branch_offsets_cache = {}
            return

        self._combined_points_cache = np.concatenate(slices, axis=0)
        self._branch_offsets_cache = offsets

    # ------------------------------------------------------------------
    # New per-branch API
    # ------------------------------------------------------------------

    def set_branches(self,
                     slices_by_uid: Dict[str, np.ndarray],
                     visible_order: List[str],
                     sample_indices_by_uid: Dict[str, np.ndarray] = None) -> None:
        """Replace per-branch vertex storage.

        Args:
            slices_by_uid: ``uid -> Nx6 float32`` slice. Identity is honoured;
                if the slice for a uid is the *same* numpy object as before,
                its VBO is kept and no GPU work is needed.
            visible_order: ordered list of visible uids; defines draw order
                and the index order of the lazy combined array.
            sample_indices_by_uid: ``uid -> (N,) source rows`` when LOD drew a
                subset of that branch, ``None``/absent when it was drawn whole.
                Anything looking a rendered point up in the branch's source data
                needs this — see ``cloud_index``.

        Cost:
            O(toggled branches) on the toggle hot path — no concat, no
            VBO upload, no GC. Replaced/removed VBOs are queued for
            deletion in the next paint frame.
        """
        new_uids = set(slices_by_uid.keys())
        old_uids = set(self._branch_vertices.keys())

        # Drop branches no longer present.
        for uid in old_uids - new_uids:
            v = self._branch_vbos.pop(uid, None)
            if v is not None:
                self._pending_vbo_deletions.append(v)
            self._branch_vertices.pop(uid, None)

        # Update or add slices. Identity check keeps the VBO when the
        # producer hands us back the same cached slice.
        for uid, slc in slices_by_uid.items():
            old = self._branch_vertices.get(uid)
            if old is not slc:
                v = self._branch_vbos.pop(uid, None)
                if v is not None:
                    self._pending_vbo_deletions.append(v)
            self._branch_vertices[uid] = slc

        self._visible_branches = list(visible_order)
        self._branch_sample_indices = dict(sample_indices_by_uid or {})

        # Invalidate lazy derived state. self.points / _branch_offsets /
        # _kdtree will be rebuilt on first access.
        self._combined_points_cache = None
        self._branch_offsets_cache = None
        self._kdtree = None

        # Picked-point indices reference the OLD combined order. Clamp by
        # eventual size when rendered; preserved here for back-compat.

        self.update()

    # ------------------------------------------------------------------
    # Legacy single-branch API (still used by training-data preview)
    # ------------------------------------------------------------------

    def set_points(self, points: np.ndarray, colors: np.ndarray = None):
        """Single-branch convenience: replace all state with one slice.

        Pass ``points=None`` to clear the display.
        """
        logger.debug("PCDViewerWidget.set_points() called")

        if points is None:
            logger.debug("  Clearing display")
            self._clear_display()
            return

        logger.debug(f"  Points: {points.shape}, {points.nbytes / 1024 / 1024:.1f} MB")
        assert points.shape[1] == 3, "Points array must have shape Nx3"
        assert points.dtype == np.float32, f"Points array must be float32, not {points.dtype}"

        if colors is not None:
            assert colors.shape[0] == points.shape[0], "Points and colors must have same length"
            assert colors.shape[1] == 3, "Colors array must have shape Nx3"
            assert colors.dtype == np.float32, "Colors array must be float32"
        else:
            colors = np.ones_like(points, dtype=np.float32)

        n = len(points)
        slice_n6 = np.empty((n, 6), dtype=np.float32)
        slice_n6[:, :3] = points
        slice_n6[:, 3:] = colors

        self.set_branches({"_single": slice_n6}, ["_single"])

    def cloud_index(self, uid: str, local_index: int) -> int:
        """Translate a rendered row of branch *uid* into a row of its source data.

        The two are only the same when the branch was drawn whole. Under LOD the
        viewer holds ``points[indices]``, so rendered row 5 may be cloud row 50 —
        and anything reading per-point source data by rendered row (cluster
        labels, and therefore what may be selected) would consult an unrelated
        point. Callers that own an index into ``self.points`` subtract the
        branch's start offset first.
        """
        indices = self._branch_sample_indices.get(uid)
        if indices is None:
            return local_index
        if 0 <= local_index < len(indices):
            return int(indices[local_index])
        return -1

    def cloud_indices(self, uid: str, local_indices: np.ndarray) -> np.ndarray:
        """Array form of ``cloud_index``."""
        indices = self._branch_sample_indices.get(uid)
        local_indices = np.asarray(local_indices, dtype=np.int64)
        if indices is None:
            return local_indices
        valid = (local_indices >= 0) & (local_indices < len(indices))
        out = np.full(local_indices.shape, -1, dtype=np.int64)
        out[valid] = np.asarray(indices)[local_indices[valid]]
        return out

    # ------------------------------------------------------------------
    # KDTree (lazy, derived from combined points)
    # ------------------------------------------------------------------

    def _ensure_kdtree(self):
        """Build the KDTree on demand. Used lazily by point picking."""
        if self._kdtree is None:
            pts = self.points
            if pts is not None:
                self._kdtree = cKDTree(pts[:, :3])

    # ------------------------------------------------------------------
    # Clear / release
    # ------------------------------------------------------------------

    def _clear_display(self):
        """Clear all point and line geometry, then trigger repaint."""
        self._release_point_data()
        self._release_line_data()
        self.update()

    def _release_point_data(self):
        """Drop all per-branch VBOs and vertex slices.

        Notes:
            - VBO deletion is queued via _pending_vbo_deletions so it
              runs in render_point_cloud() with the GL context current.
              closeEvent() flushes the queue directly.
            - We deliberately do NOT run gc.collect() / CuPy pool flush
              here — that used to fire on every visibility change and
              made cached toggles feel slow.
        """
        for v in self._branch_vbos.values():
            self._pending_vbo_deletions.append(v)
        self._branch_vbos.clear()
        self._branch_vertices.clear()
        self._visible_branches = []
        self._combined_points_cache = None
        self._branch_offsets_cache = None
        self._kdtree = None

    def _release_line_data(self):
        """Drop any stored line geometry."""
        self.line_vertices = None
        self.line_indices = None
        self.line_colors = None

    def set_lines(self, vertices: np.ndarray, edges: np.ndarray,
                  colors: np.ndarray = None):
        """
        Set wireframe line geometry to be rendered in the widget.

        Line data is independent of point data — both can be displayed
        simultaneously. When no point data is present, camera bounds are
        recomputed from the line vertices so zoom_to_extent() works.

        Args:
            vertices: Nx3 float32 array of vertex positions, or None to clear.
            edges: Mx2 integer array of vertex-index pairs defining line segments.
            colors: Nx3 float32 per-vertex RGB colors in [0, 1], or None for
                uniform gray (0.85, 0.85, 0.85).
        """
        if vertices is None or edges is None or len(edges) == 0:
            self._release_line_data()
            self.update()
            return

        assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices must be Nx3"
        assert vertices.dtype == np.float32, f"vertices must be float32, not {vertices.dtype}"
        assert edges.ndim == 2 and edges.shape[1] == 2, "edges must be Mx2"

        self._release_line_data()

        self.line_vertices = vertices
        self.line_indices = edges.reshape(-1).astype(np.uint32, copy=False)
        if colors is not None:
            assert colors.shape == vertices.shape, "colors must match vertices shape Nx3"
            self.line_colors = np.asarray(colors, dtype=np.float32)

        # Compute camera bounds from lines only when no point data exists.
        if not self._visible_branches:
            min_bounds = np.min(vertices, axis=0)
            max_bounds = np.max(vertices, axis=0)
            self.center = (min_bounds + max_bounds) / 2.0
            self.size = max_bounds - min_bounds
            self.max_extent = float(np.max(self.size)) or 1.0

        self.update()
