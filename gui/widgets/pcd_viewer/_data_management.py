import logging
import threading
import traceback
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from core.services.point_grid import PointGrid

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

        # uid -> (emphasised, faded) SOURCE-row indices: points to draw bigger,
        # and points to draw see-through. Held in source rows, not rendered rows,
        # so a change of LOD cannot silently point them at other points — the
        # translation happens at paint time. See set_point_emphasis.
        self._branch_emphasis: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._emphasis_rows_cache: Dict[str, Optional[Tuple[np.ndarray, ...]]] = {}

        # Lazy derived state, invalidated by set_branches().
        self._combined_points_cache: Optional[np.ndarray] = None
        self._branch_offsets_cache: Optional[Dict[str, Tuple[int, int]]] = None
        # (fingerprint, positions) for the picked-point highlight draw. Keyed
        # rather than invalidated — see _picked_positions().
        self._picked_positions_cache: Optional[Tuple[Tuple, np.ndarray]] = None

        # uid -> (drawn slice, PointGrid over it), so a click measures one cell
        # instead of the whole cloud. Built in the background on first pick and
        # dropped alongside the VBO when the drawn rows are replaced — see
        # _pick_grid_for(). Not built eagerly: a session that never clicks never
        # pays for one.
        #
        # The slice is stored *with* the grid rather than checked separately,
        # so that reading the pair and confirming it is current is a single
        # dict lookup. See _build_pick_grid for the race that closes.
        self._pick_grids: Dict[str, Tuple[np.ndarray, PointGrid]] = {}
        self._pick_grid_building: Set[str] = set()
        # uids whose build raised. Cleared when the branch's rows are replaced,
        # so a failure costs one attempt per render rather than one per click.
        self._pick_grid_failed: Set[str] = set()

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
        """Build the combined Nx6 array and the matching offsets map.

        With one visible branch — the normal case — this aliases that branch's
        slice rather than copying it. ``np.concatenate`` always allocates and
        copies, even from a one-item list, so the old code held the same points
        twice and paid for it on every branch change: measured at 0.38 s and a
        doubled footprint for a single 20M-point branch, which is ~3 s and
        4.1 GB at 170M.

        Aliasing is only safe while nothing writes through ``self.points``,
        because a write would land in the branch's live render data and leave
        the VBO showing something else. Nothing does — ``points`` is a read-only
        property, every use of it in the viewer is a read, and the only writes to
        a branch array happen in ``rendering_coordinator`` on a freshly created
        array before anyone else sees it. The array is marked non-writeable in
        both paths so that stays true: a future write fails loudly here instead
        of silently corrupting the display, and it fails the same way whether one
        branch is visible or several.
        """
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

        if len(slices) == 1:
            combined = slices[0]
        else:
            combined = np.concatenate(slices, axis=0)

        combined.flags.writeable = False
        self._combined_points_cache = combined
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
            self._pick_grids.pop(uid, None)
            self._pick_grid_failed.discard(uid)

        # Update or add slices. Identity check keeps the VBO when the
        # producer hands us back the same cached slice.
        for uid, slc in slices_by_uid.items():
            old = self._branch_vertices.get(uid)
            if old is not slc:
                v = self._branch_vbos.pop(uid, None)
                if v is not None:
                    self._pending_vbo_deletions.append(v)
                # The pick grid numbers the rows of the OLD slice, so it goes
                # exactly where the VBO does. Same condition, so the two can
                # never drift apart. A previous build failure is forgotten
                # here too, so new rows always get a fresh attempt.
                self._pick_grids.pop(uid, None)
                self._pick_grid_failed.discard(uid)
            self._branch_vertices[uid] = slc

        self._visible_branches = list(visible_order)
        self._branch_sample_indices = dict(sample_indices_by_uid or {})

        # Invalidate lazy derived state. self.points / _branch_offsets will be
        # rebuilt on first access.
        self._combined_points_cache = None
        self._branch_offsets_cache = None
        # The emphasis itself survives — it is held in source rows, which a
        # re-render does not change. Only the rendered rows it maps onto do.
        self._emphasis_rows_cache.clear()

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
    # Pick grid
    # ------------------------------------------------------------------

    def _pick_grid_for(self, uid: str) -> Optional[PointGrid]:
        """The pick grid for branch *uid*, or None while it is not ready.

        Starts the build on first ask and returns None until it finishes, so the
        window never blocks: numbering every point takes about 11 s at 170M on
        the CPU and 1 s on the graphics card. Callers fall back to the plain
        scan meanwhile, which is what they did before the grid existed.
        """
        slc = self._branch_vertices.get(uid)
        if slc is None or len(slc) == 0:
            return None

        # One lookup yields both the grid and the rows it was built over, so a
        # grid can never be paired with a slice it does not describe.
        cached = self._pick_grids.get(uid)
        if cached is not None:
            built_over, grid = cached
            if built_over is slc:
                return grid

        # A branch whose build already failed is not retried until its rows are
        # replaced. It would otherwise restart on every click, and each attempt
        # redoes the full O(N) numbering — so a cloud whose build runs out of
        # memory would stack up a fresh multi-second thread every time the user
        # clicks. Recorded by uid, not by array, so it pins nothing.
        if uid in self._pick_grid_failed:
            return None

        if uid not in self._pick_grid_building:
            self._pick_grid_building.add(uid)
            threading.Thread(
                target=self._build_pick_grid, args=(uid, slc),
                name=f"pick-grid-{uid[:8]}", daemon=True,
            ).start()
        return None

    def _build_pick_grid(self, uid: str, slc: np.ndarray) -> None:
        """Number *slc*'s points by cell, off the GUI thread.

        Only reads *slc*, which is non-writeable anyway (see _build_combined),
        and publishes by storing the grid together with the array it describes.

        That pairing is what makes this safe without a lock. Checking
        ``self._branch_vertices[uid] is slc`` and *then* assigning would be two
        statements on this thread while set_branches() swaps the slice on the
        GUI thread; interleaved between them, a grid built over rows that are no
        longer drawn would be stored against the rows that are, and a click
        would index past the end of a shorter buffer. Storing the pair moves the
        check to the reader, where it is a single dict lookup.
        """
        try:
            grid = PointGrid.build(slc)
        except Exception:
            logger.error(f"Failed to build the pick grid for {uid[:8]}:\n"
                         f"{traceback.format_exc()}")
            grid = None

        if grid is None:
            # Remember the failure so clicks fall back to the scan instead of
            # restarting the build every time. set_branches() clears it when the
            # branch's rows are replaced, so a re-render gets a fresh attempt.
            self._pick_grid_failed.add(uid)
        elif self._branch_vertices.get(uid) is slc:
            # Publish only while the branch still draws these rows. Without this
            # a build that lands after its branch was hidden re-adds an entry
            # that set_branches() has already dropped and nothing can reach
            # again — leaking the grid and pinning the whole render slice.
            self._pick_grids[uid] = (slc, grid)
            logger.debug(f"Pick grid ready for {uid[:8]}: {len(slc):,} points")

        self._pick_grid_building.discard(uid)

    def rendered_rows(self, uid: str, cloud_indices: np.ndarray) -> np.ndarray:
        """The rendered rows of branch *uid* showing the given source rows.

        The inverse of ``cloud_indices``, and lossy in the same direction LOD is:
        a source row the viewer did not draw has no rendered row and simply does
        not come back.
        """
        cloud_indices = np.asarray(cloud_indices, dtype=np.int64)
        sample = self._branch_sample_indices.get(uid)
        if sample is None:
            slc = self._branch_vertices.get(uid)
            limit = 0 if slc is None else len(slc)
            return cloud_indices[(cloud_indices >= 0) & (cloud_indices < limit)]
        return np.flatnonzero(np.isin(np.asarray(sample), cloud_indices))

    # ------------------------------------------------------------------
    # Emphasis: drawing some points bigger and others see-through
    # ------------------------------------------------------------------

    def set_point_emphasis(self, uid: str, emphasised=None, faded=None) -> None:
        """Single out some of branch *uid*'s points from the rest.

        *emphasised* points are drawn ``emphasis_size_factor`` times the normal
        point size; *faded* points are drawn at ``faded_opacity`` and write no
        depth, so anything behind them still shows through instead of being
        hidden by clutter. Everything else in the branch draws normally.

        Both are SOURCE row indices — indices into the branch's own data, the
        same space cluster labels live in — not rendered rows, which change
        whenever LOD does.

        Pass neither to clear the branch's emphasis. Whoever sets it owns
        clearing it: the viewer holds it until told otherwise, exactly like
        visibility.
        """
        if emphasised is None and faded is None:
            self._branch_emphasis.pop(uid, None)
        else:
            empty = np.empty(0, dtype=np.int64)
            self._branch_emphasis[uid] = (
                empty if emphasised is None else np.asarray(emphasised, dtype=np.int64),
                empty if faded is None else np.asarray(faded, dtype=np.int64),
            )
        self._emphasis_rows_cache.pop(uid, None)
        self.update()

    def clear_point_emphasis(self) -> None:
        """Drop every branch's emphasis — back to one size, fully opaque."""
        if not self._branch_emphasis:
            return
        self._branch_emphasis.clear()
        self._emphasis_rows_cache.clear()
        self.update()

    def _emphasis_draw_groups(self, uid: str, n_rows: int):
        """``(emphasised, opaque)`` rendered-row index arrays for *uid*, or None
        when the branch has no emphasis and draws in one pass.

        The FADED rows are deliberately not among them. Drawing is done as a
        faded pass over the whole branch followed by these two opaque subsets on
        top (see ``_draw_emphasised``), because the faded set is normally almost
        the entire cloud — an index array for it would be tens of megabytes
        pushed from client memory every frame, where the whole branch draws
        straight out of its VBO for nothing.

        Cached per branch: the translation from source rows costs O(n) and the
        answer only changes when the emphasis or the rendered slice does, both
        of which drop the entry.
        """
        if uid not in self._branch_emphasis:
            return None
        cached = self._emphasis_rows_cache.get(uid, False)
        if cached is not False:
            return cached

        emphasised, faded = self._branch_emphasis[uid]
        emph_rows = self.rendered_rows(uid, emphasised)
        # A row named as both is emphasised: being offered for picking outranks
        # being clutter.
        faded_rows = np.setdiff1d(self.rendered_rows(uid, faded), emph_rows)
        opaque_rows = np.setdiff1d(np.arange(n_rows, dtype=np.int64),
                                   np.union1d(emph_rows, faded_rows))

        groups = (emph_rows.astype(np.uint32), opaque_rows.astype(np.uint32))
        self._emphasis_rows_cache[uid] = groups
        return groups

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
        self._pick_grids.clear()
        self._pick_grid_failed.clear()
        self._visible_branches = []
        self._combined_points_cache = None
        self._branch_offsets_cache = None

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
