import logging
import numpy as np
from PyQt5.QtCore import Qt

from core.services.screen_selection import select_in_polygon

logger = logging.getLogger(__name__)


class PolygonSelectionMixin:
    """Polygon selection and deselection mode for PCDViewerWidget."""

    def _init_polygon_state(self):
        """Initialize polygon selection state."""
        self._polygon_mode = False        # Whether polygon selection mode is active
        self._polygon_deselect_mode = False  # Whether polygon is for deselection (vs selection)
        self._polygon_vertices = []       # List of (x, y) tuples in Qt widget coordinates

        # Stored polygons + matrices for full-resolution re-testing by plugins.
        # Each entry is a (polygon, mv, proj, viewport) tuple.
        self._selection_polygons = []

    def enter_polygon_mode(self):
        """Activate polygon selection mode. User clicks to add vertices."""
        if self.points is None:
            return
        if self._zoom_window_mode:
            self.exit_zoom_window_mode()
        self._polygon_mode = True
        self._polygon_vertices = []
        self.setCursor(Qt.CrossCursor)
        self.update()

    def enter_polygon_deselect_mode(self):
        """Activate polygon deselect mode. User draws a polygon to remove points from selection."""
        if self.points is None:
            return
        if self._zoom_window_mode:
            self.exit_zoom_window_mode()
        self._polygon_mode = True
        self._polygon_deselect_mode = True
        self._polygon_vertices = []
        self.setCursor(Qt.CrossCursor)
        self.update()

    def exit_polygon_mode(self):
        """Deactivate polygon selection mode and restore normal cursor."""
        self._polygon_mode = False
        self._polygon_deselect_mode = False
        self._polygon_vertices = []
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def _close_polygon_and_select(self):
        """Close the polygon and select all 3D points whose screen projections fall inside it."""
        if len(self._polygon_vertices) < 3:
            self.exit_polygon_mode()
            return

        # Nothing to select against. enter_polygon_mode() refuses to start
        # without points, but a lasso already in progress survives the branches
        # being hidden under it, and there is no cloud left to test by the time
        # it closes.
        if self.points is None:
            self.exit_polygon_mode()
            return

        polygon = np.array(self._polygon_vertices, dtype=np.float64)  # (M, 2)

        # OpenGL's glGetDoublev returns column-major matrices. In numpy (row-major)
        # these appear transposed: mv_np = ModelView^T, proj_np = Projection^T.
        # Use row-vector multiplication: clip_row = point_row @ MV^T @ P^T
        mv = np.array(self.model_view_matrix, dtype=np.float64)   # 4x4 (M^T)
        proj = np.array(self.projection_matrix, dtype=np.float64)  # 4x4 (P^T)

        # Append polygon + matrices for full-resolution re-testing by plugins
        self._selection_polygons.append((
            polygon.copy(), mv.copy(), proj.copy(), tuple(self.viewport)
        ))

        # Which points land inside the lasso. Blocked and float32 — see
        # core.services.screen_selection for why the whole-cloud float64 version
        # this replaced needed 19 GB of scratch at 170M points.
        inside = select_in_polygon(self.points, polygon, mv, proj, self.viewport)

        # Get indices of selected points and apply selection filters
        new_indices = np.flatnonzero(inside)
        new_indices = self._filter_selection(new_indices)

        # Avoid duplicates. Done with numpy rather than a set + per-index
        # append: a lasso over a dense area yields millions of indices, and at
        # that size the Python loop costs more than the projection and the
        # polygon test put together. ``np.where`` above already returns unique,
        # ascending indices, so masking is enough and keeps their order.
        if new_indices.size > 0:
            if self.picked_points_indices:
                existing = np.fromiter(
                    self.picked_points_indices, dtype=np.int64,
                    count=len(self.picked_points_indices)
                )
                new_indices = new_indices[~np.isin(new_indices, existing)]
            self.picked_points_indices.extend(new_indices.tolist())

        self.exit_polygon_mode()

    def clear_selection(self):
        """Drop all picked points and stored selection polygons, then repaint.

        Called after a plugin consumes the selection so highlighted points are
        de-highlighted once the operation completes. This is the one place that
        knows what "clearing the selection" means — everywhere else calls it
        rather than clearing the two lists by hand, which is how 17 of the 20
        previous call sites came to leave the stored polygons behind.
        """
        self.picked_points_indices.clear()
        self._selection_polygons.clear()
        self.update()

    def get_selection_mask_for(self, points_3d, allowed=None):
        """Boolean mask over *full-resolution* ``points_3d`` marking the user's
        current selection.

        The viewer's combined ``self.points`` buffer is LOD-subsampled and may
        concatenate several visible branches, so ``picked_points_indices`` are
        indices into that rendered buffer — they do NOT map to a branch's
        full-resolution point order. Plugins must therefore re-derive the
        selection against the cloud they actually operate on:

        - When polygon selections are stored, re-test the full-resolution points
          against the saved polygons + camera matrices (exact, resolution
          independent).
        - Otherwise fall back to coordinate-matching the picked points' world
          coordinates (exact float32 row equality, since the rendered slice is
          just a subsample of the same source coordinates).

        **The polygon re-test does not go through the viewer's selection
        filters.** The viewer applies them when the lasso closes
        (``_filter_selection``: branch membership, cluster locks, noise) — but
        they work in rendered-index space, and re-testing works in cloud-index
        space, so the widened result comes back ungated. A lasso drawn over a
        few permitted points otherwise returns every point it encloses, locked
        clusters and noise included.

        A caller that knows which rows are admissible passes them as *allowed*;
        the mask is intersected with them. ``selection_gate`` has
        ``selectable_cloud_indices(node, len(points))`` for exactly this, and it
        is the same gate ``picked_cloud_indices`` takes — anything that cares
        what it is operating on should pass it.

        Args:
            points_3d: (N, >=3) array of full-resolution world coordinates.
            allowed: optional row indices of *points_3d* that may be selected.

        Returns:
            (N,) boolean mask aligned to ``points_3d``.
        """
        # float32 throughout. The cloud is stored float32 and the comparison
        # below is done in float32 anyway, so the float64 copy this used to make
        # bought nothing and cost 4 GB on a 170M-point cloud.
        pts = np.asarray(points_3d, dtype=np.float32)
        n = len(pts)

        poly_mask = self.retest_polygon_selection(pts[:, :3])
        if poly_mask is not None:
            return self._gate_mask(poly_mask, allowed)

        if not self.picked_points_indices or self.points is None:
            return np.zeros(n, dtype=bool)

        sel = np.asarray(self.picked_points_indices, dtype=np.int64)
        sel = sel[(sel >= 0) & (sel < len(self.points))]
        if sel.size == 0:
            return np.zeros(n, dtype=bool)

        full32 = np.ascontiguousarray(pts[:, :3])
        pick32 = np.ascontiguousarray(
            np.unique(self.points[sel, :3].astype(np.float32), axis=0)
        )
        void_dtype = np.dtype((np.void, full32.dtype.itemsize * 3))
        fv = full32.reshape(-1).view(void_dtype)
        pv = pick32.reshape(-1).view(void_dtype)
        # The coordinate-match branch starts from points the viewer already
        # filtered, so it needs no gate of its own — but honour one if given, so
        # both branches of this method mean the same thing.
        return self._gate_mask(np.isin(fv, pv), allowed)

    @staticmethod
    def _gate_mask(mask, allowed):
        """*mask* restricted to the rows in *allowed*, or unchanged if None."""
        if allowed is None:
            return mask
        gate = np.zeros(len(mask), dtype=bool)
        allowed = np.asarray(allowed, dtype=np.intp)
        allowed = allowed[(allowed >= 0) & (allowed < len(mask))]
        gate[allowed] = True
        return mask & gate

    def retest_polygon_selection(self, points_3d):
        """
        Re-test arbitrary 3D points against all stored polygon selections.

        Uses the stored polygons and camera matrices from polygon selections,
        so it works correctly even if the camera has moved.  Returns the union
        of all polygon tests (a point inside ANY stored polygon is True).

        Args:
            points_3d: (N, 3) float array of 3D world coordinates.

        Returns:
            (N,) boolean mask (True = inside any polygon), or None if no polygons stored.
        """
        if not self._selection_polygons:
            return None

        pts = np.asarray(points_3d, dtype=np.float32)
        combined_mask = np.zeros(pts.shape[0], dtype=bool)

        for polygon, mv, proj, viewport in self._selection_polygons:
            combined_mask |= select_in_polygon(pts, polygon, mv, proj, viewport)

        return combined_mask

    def _close_polygon_and_deselect(self):
        """Close the polygon and remove all points inside it from the current selection."""
        if len(self._polygon_vertices) < 3:
            self.exit_polygon_mode()
            return

        if not self.picked_points_indices:
            self.exit_polygon_mode()
            return

        if self.points is None:
            self.exit_polygon_mode()
            return

        polygon = np.array(self._polygon_vertices, dtype=np.float64)  # (M, 2)

        mv = np.array(self.model_view_matrix, dtype=np.float64)
        proj = np.array(self.projection_matrix, dtype=np.float64)

        # Only test the currently selected points, not the whole cloud — a
        # deselect lasso can only ever remove points that are already picked.
        selected_indices = np.fromiter(
            self.picked_points_indices, dtype=np.int64,
            count=len(self.picked_points_indices)
        )

        # Drop picks that no longer address a drawn point. The list deliberately
        # outlives set_branches(), so after LOD drops a level — or a branch is
        # hidden — it still holds indices past the end of the shorter buffer,
        # and indexing with them raises. Every path that indexes with this list
        # has to clamp: deselect_point_at, deselect_cluster_at,
        # _picked_positions and this one.
        in_range = (selected_indices >= 0) & (selected_indices < len(self.points))
        selected_indices = selected_indices[in_range]
        if selected_indices.size == 0:
            self.clear_selection()
            self.exit_polygon_mode()
            return

        inside = select_in_polygon(self.points[selected_indices, :3], polygon,
                                   mv, proj, self.viewport)

        # Remove points that fall inside the polygon from the selection.
        # ``selected_indices`` is picked_points_indices in order, so masking it
        # keeps the survivors in order without a per-index membership test.
        self.picked_points_indices[:] = selected_indices[~inside].tolist()
        # Invalidate stored polygons so plugins fall back to coordinate matching
        self._selection_polygons.clear()

        self.exit_polygon_mode()

