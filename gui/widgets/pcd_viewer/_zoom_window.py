import logging
import numpy as np
from PyQt5.QtCore import Qt

from core.services.screen_selection import select_in_rect

logger = logging.getLogger(__name__)


class ZoomWindowMixin:
    """Zoom window (rectangle drag to zoom) mode for PCDViewerWidget."""

    def _init_zoom_window_state(self):
        """Initialize zoom window state."""
        self._zoom_window_mode = False
        self._zoom_window_start = None   # (x, y) on mouse press
        self._zoom_window_current = None  # (x, y) during drag

    def _zoom_window_geometry(self):
        """Collect the visible geometry a zoom rectangle can be measured against.

        Points and line vertices render independently and either may be the only
        thing on screen, so both are candidates.

        Returns:
            numpy.ndarray: (N, 3) float32 world coordinates, or None if nothing
            is visible. When points are the only source this is a *view* on the
            render buffer rather than a copy — the old float64 cast duplicated
            the whole cloud, 4 GB of it at 170M points, for no gain.
        """
        sources = []
        if self.points is not None and len(self.points) > 0:
            sources.append(self.points[:, :3])
        if self.line_vertices is not None and len(self.line_vertices) > 0:
            sources.append(np.asarray(self.line_vertices, dtype=np.float32))

        if not sources:
            return None
        if len(sources) == 1:
            return sources[0]
        return np.concatenate(sources, axis=0)

    def enter_zoom_window_mode(self):
        """Activate zoom window mode. User drags a rectangle to zoom into that region."""
        if self._zoom_window_geometry() is None:
            return
        if self._polygon_mode:
            self.exit_polygon_mode()
        self._zoom_window_mode = True
        self._zoom_window_start = None
        self._zoom_window_current = None
        self.setCursor(Qt.CrossCursor)
        self.update()

    def exit_zoom_window_mode(self):
        """Deactivate zoom window mode and restore normal cursor."""
        self._zoom_window_mode = False
        self._zoom_window_start = None
        self._zoom_window_current = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def _execute_zoom_window(self):
        """Project all visible geometry to screen, find what is in the rectangle, zoom to its 3D bbox."""
        if self._zoom_window_start is None or self._zoom_window_current is None:
            self.exit_zoom_window_mode()
            return

        pts_3d = self._zoom_window_geometry()
        if pts_3d is None:
            self.exit_zoom_window_mode()
            return

        x1, y1 = self._zoom_window_start
        x2, y2 = self._zoom_window_current

        # Minimum rectangle size check to avoid accidental clicks
        if abs(x2 - x1) < self._MIN_ZOOM_WINDOW_SIZE_PX or abs(y2 - y1) < self._MIN_ZOOM_WINDOW_SIZE_PX:
            self.exit_zoom_window_mode()
            return

        # Normalize rectangle corners
        rect_left = min(x1, x2)
        rect_right = max(x1, x2)
        rect_top = min(y1, y2)
        rect_bottom = max(y1, y2)

        # Which points land inside the rectangle. A rectangle is a four-sided
        # polygon, so this goes through the same blocked float32 code as the
        # lasso rather than projecting the whole cloud at once in float64 —
        # which is what made zooming into a 170M-point cloud swap.
        mv = np.array(self.model_view_matrix, dtype=np.float64)
        proj = np.array(self.projection_matrix, dtype=np.float64)
        inside = select_in_rect(
            pts_3d, (rect_left, rect_top, rect_right, rect_bottom),
            mv, proj, self.viewport,
        )

        if not np.any(inside):
            self.exit_zoom_window_mode()
            return

        # Compute 3D bounding box of filtered points
        selected_pts = pts_3d[inside]
        min_bounds = np.min(selected_pts, axis=0)
        max_bounds = np.max(selected_pts, axis=0)

        new_center = (min_bounds + max_bounds) / 2.0
        new_size = max_bounds - min_bounds
        new_max_extent = max(np.max(new_size), 1e-6)

        # Compute the new camera distance for this region
        half_fov_rad = np.radians(self.fov / 2)
        new_camera_distance = new_max_extent / (2 * np.tan(half_fov_rad)) * self._CAMERA_DISTANCE_PADDING

        # Guard: if the new distance is >= old, the rectangle doesn't zoom in
        old_effective_distance = self.camera_distance * self.zoom_factor
        if new_camera_distance >= old_effective_distance:
            self.exit_zoom_window_mode()
            return

        # Set camera parameters (preserve rotation)
        self.center = new_center
        self.size = new_size
        self.max_extent = float(new_max_extent)
        self.camera_distance = new_camera_distance
        self.zoom_factor = 1.0

        self.pan_x = -self.center[0]
        self.pan_y = -self.center[1]
        self.pan_z = -self.center[2]
        # rot_x, rot_y, rot_z are preserved (no change)

        # Allow zooming back out well past the original view.
        # FOV is capped at base value in the renderer, so no distortion risk —
        # only camera distance increases when zoom_factor > 1.
        max_zoom_out = old_effective_distance / self.camera_distance
        self._zoom_max_factor = max(1.0, max_zoom_out * 3.0)

        self.exit_zoom_window_mode()
