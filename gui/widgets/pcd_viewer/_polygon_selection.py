import logging
import numpy as np
from PyQt5.QtCore import Qt

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

        # Project all 3D points to screen coordinates
        pts_3d = self.points[:, :3].astype(np.float64)
        screen_x, screen_y, valid_mask = self._project_points_to_screen(
            pts_3d, mv, proj, self.viewport
        )

        # Point-in-polygon test (ray casting)
        inside = self._points_in_polygon(screen_x, screen_y, polygon, valid_mask)

        # Get indices of selected points and apply selection filters
        new_indices = np.where(inside)[0]
        new_indices = self._filter_selection(new_indices)

        # Avoid duplicates
        if new_indices.size > 0:
            existing = set(self.picked_points_indices)
            for idx in new_indices:
                idx_int = int(idx)
                if idx_int not in existing:
                    self.picked_points_indices.append(idx_int)
                    existing.add(idx_int)

        self.exit_polygon_mode()

    def clear_selection(self):
        """Drop all picked points and stored selection polygons, then repaint.

        Called after a plugin consumes the selection so highlighted points are
        de-highlighted once the operation completes.
        """
        self.picked_points_indices.clear()
        self._selection_polygons.clear()
        self.update()

    def get_selection_mask_for(self, points_3d):
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

        Args:
            points_3d: (N, >=3) array of full-resolution world coordinates.

        Returns:
            (N,) boolean mask aligned to ``points_3d``.
        """
        pts = np.asarray(points_3d, dtype=np.float64)
        n = len(pts)

        poly_mask = self.retest_polygon_selection(pts[:, :3])
        if poly_mask is not None:
            return poly_mask

        if not self.picked_points_indices or self.points is None:
            return np.zeros(n, dtype=bool)

        sel = np.asarray(self.picked_points_indices, dtype=np.int64)
        sel = sel[(sel >= 0) & (sel < len(self.points))]
        if sel.size == 0:
            return np.zeros(n, dtype=bool)

        full32 = np.ascontiguousarray(pts[:, :3].astype(np.float32))
        pick32 = np.ascontiguousarray(
            np.unique(self.points[sel, :3].astype(np.float32), axis=0)
        )
        void_dtype = np.dtype((np.void, full32.dtype.itemsize * 3))
        fv = full32.reshape(-1).view(void_dtype)
        pv = pick32.reshape(-1).view(void_dtype)
        return np.isin(fv, pv)

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

        pts = np.asarray(points_3d, dtype=np.float64)
        combined_mask = np.zeros(pts.shape[0], dtype=bool)

        for polygon, mv, proj, viewport in self._selection_polygons:
            screen_x, screen_y, valid_mask = self._project_points_to_screen(
                pts, mv, proj, viewport
            )
            combined_mask |= self._points_in_polygon(screen_x, screen_y, polygon, valid_mask)

        return combined_mask

    def _close_polygon_and_deselect(self):
        """Close the polygon and remove all points inside it from the current selection."""
        if len(self._polygon_vertices) < 3:
            self.exit_polygon_mode()
            return

        if not self.picked_points_indices:
            self.exit_polygon_mode()
            return

        polygon = np.array(self._polygon_vertices, dtype=np.float64)  # (M, 2)

        mv = np.array(self.model_view_matrix, dtype=np.float64)
        proj = np.array(self.projection_matrix, dtype=np.float64)

        # Only project the currently selected points (not all points)
        selected_indices = np.array(self.picked_points_indices, dtype=np.int64)
        pts_3d = self.points[selected_indices, :3].astype(np.float64)
        screen_x, screen_y, valid_mask = self._project_points_to_screen(
            pts_3d, mv, proj, self.viewport
        )

        inside = self._points_in_polygon(screen_x, screen_y, polygon, valid_mask)

        # Remove points that fall inside the polygon from the selection
        indices_to_remove = set(int(selected_indices[i]) for i in np.where(inside)[0])
        self.picked_points_indices[:] = [i for i in self.picked_points_indices if i not in indices_to_remove]
        # Invalidate stored polygons so plugins fall back to coordinate matching
        self._selection_polygons.clear()

        self.exit_polygon_mode()

    @staticmethod
    def _points_in_polygon(screen_x, screen_y, polygon, valid_mask):
        """
        Vectorized ray-casting point-in-polygon test.

        Args:
            screen_x: (N,) array of screen X coordinates
            screen_y: (N,) array of screen Y coordinates
            polygon: (M, 2) array of polygon vertices in screen coords
            valid_mask: (N,) boolean mask for points in front of camera

        Returns:
            (N,) boolean array — True for points inside the polygon
        """
        n = len(screen_x)
        m = len(polygon)
        inside = np.zeros(n, dtype=bool)

        # Ray casting: count crossings for each point
        crossings = np.zeros(n, dtype=np.int32)
        for i in range(m):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % m]

            # For each edge, test all valid points simultaneously
            # A ray from (px, py) going right crosses edge if:
            #   1. The edge straddles the ray's y-level
            #   2. The intersection x is to the right of px
            cond1 = (y1 <= screen_y) & (y2 > screen_y)
            cond2 = (y2 <= screen_y) & (y1 > screen_y)
            straddle = cond1 | cond2

            # Compute x-intersection of the edge with the horizontal ray
            # x_intersect = x1 + (screen_y - y1) / (y2 - y1) * (x2 - x1)
            test = straddle & valid_mask
            if not np.any(test):
                continue

            dy = y2 - y1
            if dy == 0:
                continue

            t = (screen_y[test] - y1) / dy
            x_intersect = x1 + t * (x2 - x1)

            crosses = x_intersect > screen_x[test]
            crossings[test] += crosses.astype(np.int32)

        # Odd number of crossings = inside
        inside = (crossings % 2 == 1) & valid_mask
        return inside
