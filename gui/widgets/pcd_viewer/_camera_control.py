import logging
import traceback
import numpy as np
from PyQt5.QtCore import QTimer
from OpenGL.GL import glReadPixels, GL_DEPTH_COMPONENT, GL_FLOAT
from OpenGL.GLU import gluUnProject

from config.config import global_variables
from application.lod_manager import LODManager

logger = logging.getLogger(__name__)


class CameraControlMixin:
    """Camera manipulation (rotation center, reset, zoom-to-extent, LOD) for PCDViewerWidget."""

    def _init_camera(self):
        """Initialize camera and viewport state."""
        # Initialize variables for zoom to extent
        self.center = np.array([0.0, 0.0, 0.0])
        self.size = np.array([1.0, 1.0, 1.0])
        self.max_extent = None
        self.fov = 60.0  # Field of view in degrees
        self.near_plane = 0.1
        self.far_plane = 1000.0

        # Visibility of the point cloud
        self.visible = True

        # Initialize default values
        self.camera_distance = self._default_camera_distance
        self.zoom_factor = self._default_zoom_factor

        # Timer for hiding the axis symbol after panning
        self.axis_timer = None

        # Initialize matrices used in the pick_point method
        self.model_view_matrix = np.identity(4)
        self.projection_matrix = np.identity(4)
        self.viewport = np.array([0, 0, self.width(), self.height()], dtype=np.int32)

    def update_rotation_center(self, mouse_pos):
        """
        Update the centre of rotation based on the given mouse position.

        This method is used to update the centre of rotation of the point cloud view based on a double-click event
        at a specific mouse position. It reads the depth value at the mouse position, unprojects the screen coordinates
        to world coordinates, and sets the centre of rotation to the closest point in the point cloud if it lies within
        a specified threshold.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Use stored matrices
        modelview = self.model_view_matrix
        projection = self.projection_matrix
        viewport = self.viewport

        # Get the window coordinates
        win_x = mouse_pos.x()
        win_y = viewport[3] - mouse_pos.y()  # Invert Y coordinate

        # Read the depth value at the mouse position
        z_buffer = glReadPixels(int(win_x), int(win_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        win_z = z_buffer[0][0]

        # Handle cases where the depth value is 1.0 (background)
        if win_z == 1.0:
            # No depth information; do not update center
            return

        # Unproject the window coordinates to get the world coordinates
        world_coords = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
        click_point = np.array(world_coords[:3])

        # Set a threshold to determine if a point is close enough
        threshold = self.max_extent * self.picking_point_threshold_factor

        # Find the closest point using KDTree for O(log n) performance
        if self._kdtree is not None:
            min_distance, min_distance_index = self._kdtree.query(click_point, k=1)
        else:
            # Fallback to brute-force if KDTree not available
            distances = np.linalg.norm(self.points[:, :3] - click_point, axis=1)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]

        if min_distance < threshold:
            # Save old center before updating
            old_center = self.center.copy()
            new_center = self.points[min_distance_index, :3].copy()

            # Adjust pan to compensate for center change (prevents view shift)
            # paintGL applies: P_view = R*P + (pan + center - R*center) + (0,0,-cam_dist)
            # For stable view when center changes: pan_new = pan_old + (I - R) * delta
            R = np.array(self.model_view_matrix[:3, :3]).T  # transpose: OpenGL stores column-major
            delta = old_center - new_center
            rotated_delta = R @ delta
            self.pan_x += delta[0] - rotated_delta[0]
            self.pan_y += delta[1] - rotated_delta[1]
            self.pan_z += delta[2] - rotated_delta[2]

            # Update the center to the new point
            self.center = new_center

            # Show axis briefly at new center + repaint
            self.show_axis = True
            self.update()
            if self.axis_timer is not None:
                self.axis_timer.stop()
            self.axis_timer = QTimer(self)
            self.axis_timer.setSingleShot(True)
            self.axis_timer.timeout.connect(self.hide_axis_after_zoom)
            self.axis_timer.start(500)

    def reset_view(self):
        """
        Reset the camera to its default position and orientation.

        This method restores the default settings for the camera, including zoom, rotation, and panning,
        allowing the user to return to the initial view of the point cloud.
        Specifically, the following attributes are reset:

        - `zoom_factor`: Set to `default_zoom_factor` to restore the original zoom level.
        - `rot_x`, `rot_y`, `rot_z`: Rotation angles around the X, Y, and Z axes are reset to default values, default_rot_x, etc.
        - `pan_x`, `pan_y`, `pan_z`: Panning offsets along the X, Y, and Z axes are reset to default values, default_pan_x.
        - `camera_distance`: Set to `default_camera_distance` to restore the original distance from the point cloud.

        After resetting these parameters, the view is updated to reflect the changes.
        """

        self._zoom_max_factor = 1.0
        self.zoom_factor = self.default_zoom_factor
        self.rot_x = self.default_rot_x
        self.rot_y = self.default_rot_y
        self.rot_z = self.default_rot_z
        self.pan_x = self.default_pan_x
        self.pan_y = self.default_pan_y
        self.pan_z = self.default_pan_z
        self.camera_distance = self.default_camera_distance

        self.center[0] = -self.default_pan_x
        self.center[1] = -self.default_pan_y
        self.center[2] = -self.default_pan_z

        self.update()

    def zoom_to_extent(self, preserve_rotation=False):
        """
        Zoom the camera to fit all visible points in the viewport.

        This method calculates the bounding box of the currently displayed point cloud
        and adjusts the camera parameters to frame the entire dataset optimally.
        It recenters the view on the data and calculates an appropriate camera distance.

        Args:
            preserve_rotation: If True, keeps current rotation angles (camera angle).
                             If False, resets rotation to default values.

        The following parameters are adjusted:
        - `center`: Set to the center of the point cloud bounding box
        - `camera_distance`: Calculated based on bounding box size and FOV
        - `zoom_factor`: Reset to 1.0
        - `rot_x`, `rot_y`, `rot_z`: Reset to default (or preserved if preserve_rotation=True)
        - `pan_x`, `pan_y`, `pan_z`: Adjusted to center the view
        - Also updates default values for reset_view functionality

        After adjusting parameters, the view is updated to reflect the changes.
        """
        logger.debug("zoom_to_extent() called")

        # Check if points are available
        if self.points is None or len(self.points) == 0:
            logger.debug("  No points to zoom to")
            return

        logger.debug(f"  Processing {len(self.points)} points")

        try:
            # Calculate bounding box of visible points
            logger.debug("  Calculating bounding box...")
            min_bounds = np.min(self.points[:, :3], axis=0)
            max_bounds = np.max(self.points[:, :3], axis=0)

            # Calculate center, size, and maximum extent
            self.center = (min_bounds + max_bounds) / 2.0
            self.size = max_bounds - min_bounds
            self.max_extent = np.max(self.size)

            logger.debug(f"  Center: {self.center}")
            logger.debug(f"  Size: {self.size}")
            logger.debug(f"  Max extent: {self.max_extent}")

            # Avoid division by zero for degenerate cases
            if self.max_extent == 0:
                self.max_extent = 1.0

            # Calculate optimal camera distance based on FOV and bounding box
            half_fov_rad = np.radians(self.fov / 2)
            self.default_camera_distance = self.max_extent / (2 * np.tan(half_fov_rad)) * 1.2  # 20% padding
            self.camera_distance = self.default_camera_distance

            logger.debug(f"  Camera distance: {self.camera_distance}")

            # Update panning offsets to align the center with the view
            self.pan_x = -self.center[0]
            self.pan_y = -self.center[1]
            self.pan_z = -self.center[2]

            # Save these values as defaults for reset_view
            self.default_pan_x = self.pan_x
            self.default_pan_y = self.pan_y
            self.default_pan_z = self.pan_z

            # Reset rotation to default values (unless preserving rotation)
            if not preserve_rotation:
                self.rot_x = self.default_rot_x
                self.rot_y = self.default_rot_y
                self.rot_z = self.default_rot_z

            # Reset zoom factor and max (zoom window may have raised it)
            self._zoom_max_factor = 1.0
            self.zoom_factor = 1.0
            self.default_zoom_factor = 1.0

            # Show axis briefly for orientation
            self.show_axis = True

            # Note: Do NOT reset _current_sample_rate here - it was set by DataManager
            # during render to respect the point budget. Resetting would cause
            # _on_zoom_changed to trigger an unnecessary (and potentially OOM) re-render.

            # Update the view
            logger.debug("  Updating view...")
            self.update()

            # Set a timer to hide the axis after a short time
            if hasattr(self, 'axis_timer') and self.axis_timer is not None:
                self.axis_timer.stop()

            self.axis_timer = QTimer(self)
            self.axis_timer.setSingleShot(True)
            self.axis_timer.timeout.connect(self.hide_axis_after_zoom)
            self.axis_timer.start(500)  # 500ms delay

            logger.debug("  zoom_to_extent() completed")

        except Exception as e:
            logger.error(f"  ERROR in zoom_to_extent(): {e}")
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            raise

    def show_point_cloud(self, visible=True):
        """Show or hide the point cloud by setting visibility and updating the view."""
        if self.visible != visible:
            self.visible = visible
            self.update()

    def _on_zoom_changed(self):
        """Called after zoom changes to potentially update LOD.

        Note: LOD only DECREASES detail when zooming out, never increases.
        This prevents OOM when zooming in on large datasets.
        """
        if not self._lod_enabled:
            return

        controller = global_variables.global_application_controller
        if controller is None or controller.rendering_coordinator is None:
            return

        rc = controller.rendering_coordinator
        if rc.total_visible_points <= 0:
            return

        total_points = rc.total_visible_points
        point_budget = rc._point_budget

        # Calculate max safe rate (never exceed budget)
        max_safe_rate = min(1.0, point_budget / total_points) if total_points > 0 else 1.0

        new_rate = LODManager.compute_sample_rate(
            total_points,
            self.camera_distance,
            self.zoom_factor,
            self.max_extent or 1.0,
            point_budget
        )

        # Cap at max safe rate
        new_rate = min(new_rate, max_safe_rate)

        # IMPORTANT: Only re-render if DECREASING detail (zooming out)
        # Never increase detail dynamically - it risks OOM on large datasets
        # Users who want more detail should adjust point_budget or hide branches
        if new_rate < self._current_sample_rate - 0.05:
            logger.debug(f"LOD: {self._current_sample_rate:.1%} -> {new_rate:.1%} (zoom out)")
            self._current_sample_rate = new_rate
            main_window = global_variables.global_main_window
            if main_window and hasattr(main_window, 'render_visible_with_lod'):
                main_window.render_visible_with_lod(new_rate)
