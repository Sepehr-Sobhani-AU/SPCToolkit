import logging
import numpy as np
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)


class ZoomWindowMixin:
    """Zoom window (rectangle drag to zoom) mode for PCDViewerWidget."""

    def _init_zoom_window_state(self):
        """Initialize zoom window state."""
        self._zoom_window_mode = False
        self._zoom_window_start = None   # (x, y) on mouse press
        self._zoom_window_current = None  # (x, y) during drag

    def enter_zoom_window_mode(self):
        """Activate zoom window mode. User drags a rectangle to zoom into that region."""
        if self.points is None:
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
        """Project all visible points to screen, find those in rectangle, zoom to their 3D bbox."""
        if self._zoom_window_start is None or self._zoom_window_current is None:
            self.exit_zoom_window_mode()
            return

        if self.points is None or len(self.points) == 0:
            self.exit_zoom_window_mode()
            return

        x1, y1 = self._zoom_window_start
        x2, y2 = self._zoom_window_current

        # Minimum rectangle size check to avoid accidental clicks
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.exit_zoom_window_mode()
            return

        # Normalize rectangle corners
        rect_left = min(x1, x2)
        rect_right = max(x1, x2)
        rect_top = min(y1, y2)
        rect_bottom = max(y1, y2)

        # Project all 3D points to screen coords (same approach as polygon selection)
        mv = np.array(self.model_view_matrix, dtype=np.float64)
        proj = np.array(self.projection_matrix, dtype=np.float64)

        pts_3d = self.points[:, :3].astype(np.float64)
        n = pts_3d.shape[0]
        ones = np.ones((n, 1), dtype=np.float64)
        pts_homo = np.hstack([pts_3d, ones])

        clip = pts_homo @ mv @ proj
        w = clip[:, 3]
        valid_mask = w > 0

        ndc_x = np.zeros(n, dtype=np.float64)
        ndc_y = np.zeros(n, dtype=np.float64)
        ndc_x[valid_mask] = clip[valid_mask, 0] / w[valid_mask]
        ndc_y[valid_mask] = clip[valid_mask, 1] / w[valid_mask]

        vp = self.viewport
        vp_x, vp_y, vp_w, vp_h = vp[0], vp[1], vp[2], vp[3]
        screen_x = (ndc_x + 1.0) * 0.5 * vp_w + vp_x
        screen_y = (1.0 - ndc_y) * 0.5 * vp_h + vp_y

        # Filter points inside the rectangle and in front of camera
        inside = (valid_mask
                  & (screen_x >= rect_left) & (screen_x <= rect_right)
                  & (screen_y >= rect_top) & (screen_y <= rect_bottom))

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

        # Save old effective distance so the user can zoom back out to it
        old_effective_distance = self.camera_distance * self.zoom_factor

        # Set camera parameters (preserve rotation)
        self.center = new_center
        self.size = new_size
        self.max_extent = float(new_max_extent)

        half_fov_rad = np.radians(self.fov / 2)
        self.camera_distance = new_max_extent / (2 * np.tan(half_fov_rad)) * 1.2
        self.zoom_factor = 1.0

        self.pan_x = -self.center[0]
        self.pan_y = -self.center[1]
        self.pan_z = -self.center[2]
        # rot_x, rot_y, rot_z are preserved (no change)

        # Raise zoom_max_factor so mouse wheel can zoom back out to the old view.
        # FOV is capped at base value in the renderer, so no distortion risk —
        # only camera distance increases when zoom_factor > 1.
        max_zoom_out = old_effective_distance / self.camera_distance if self.camera_distance > 0 else 1.0
        self._zoom_max_factor = max(1.0, max_zoom_out * 1.5)

        self.exit_zoom_window_mode()
