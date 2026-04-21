import logging
import numpy as np
from OpenGL.GL import glReadPixels, GL_DEPTH_COMPONENT, GL_FLOAT
from OpenGL.GLU import gluProject, gluUnProject

logger = logging.getLogger(__name__)


class PointPickingMixin:
    """Point selection and deselection logic for PCDViewerWidget."""

    @staticmethod
    def _project_points_to_screen(pts_3d, mv, proj, viewport):
        """Project an array of 3D points to 2D screen coordinates (Qt: top-left origin, Y-down).

        Args:
            pts_3d: (N, 3) float64 array of world coordinates.
            mv: 4x4 model-view matrix (column-major transposed).
            proj: 4x4 projection matrix (same convention).
            viewport: (vp_x, vp_y, vp_w, vp_h).

        Returns:
            tuple: (screen_x, screen_y, valid_mask) arrays of shape (N,).
        """
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

        vp_x, vp_y, vp_w, vp_h = viewport[0], viewport[1], viewport[2], viewport[3]
        screen_x = (ndc_x + 1.0) * 0.5 * vp_w + vp_x
        screen_y = (1.0 - ndc_y) * 0.5 * vp_h + vp_y

        return screen_x, screen_y, valid_mask

    def _unproject_mouse_to_nearest_point(self, mouse_pos):
        """Unproject a mouse position through the depth buffer and find the nearest point.

        Returns:
            tuple: (point_index, point_3d_coords) or (None, None) if no point found.
        """
        self.makeCurrent()

        modelview = self.model_view_matrix
        projection = self.projection_matrix
        viewport = self.viewport

        win_x = mouse_pos.x()
        win_y = viewport[3] - mouse_pos.y()

        z_buffer = glReadPixels(int(win_x), int(win_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
        win_z = z_buffer[0][0]

        if win_z == 1.0:
            return None, None

        world_coords = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)
        pick_point = np.array(world_coords[:3])

        threshold = self.max_extent * self.picking_point_threshold_factor

        if self._kdtree is not None:
            min_distance, min_index = self._kdtree.query(pick_point, k=1)
        else:
            distances = np.linalg.norm(self.points[:, :3] - pick_point, axis=1)
            min_index = np.argmin(distances)
            min_distance = distances[min_index]

        if min_distance < threshold:
            return min_index, self.points[min_index, :3].copy()
        return None, None

    def pick_point(self, mouse_pos, select=True):
        """
        Handle point picking or deselecting points in the point cloud.

        This method is used to pick or deselect points in the point cloud based on a mouse click position. It uses
        OpenGL to project the clicked point onto the screen space and determines whether a point in the point cloud
        is close enough to be picked or deselected. If `select` is True, the method attempts to pick a point;
        otherwise, it attempts to deselect a point.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
            select (bool, optional): A flag indicating whether to select (True) or deselect (False) the point. Defaults
                to True.
        """

        if select:
            self.select_point_at(mouse_pos)
        else:
            self.deselect_point_at(mouse_pos)

    def project_to_screen(self, point_3d):
        """
        Project a 3D point onto the screen space.

        This method takes a 3D point in the world coordinate system and projects it onto the 2D screen space using the
        current model-view matrix, projection matrix, and viewport settings. The resulting screen coordinates can be
        used for tasks such as selecting or highlighting points in the point cloud.

        Args:
            point_3d (numpy.ndarray): A 3-element array representing the (x, y, z) coordinates of the point to be
                projected.

        Returns:
            numpy.ndarray: A 3-element array representing the (x, y, z) coordinates of the projected point in screen
            space.
        """

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Use stored matrices
        modelview = self.model_view_matrix
        projection = self.projection_matrix
        viewport = self.viewport

        # Use gluProject to project the point
        screen_pos = gluProject(
            point_3d[0], point_3d[1], point_3d[2],
            modelview, projection, viewport
        )
        return np.array(screen_pos)

    def select_point_at(self, mouse_pos):
        """
        Select a point in the point cloud at the given mouse position.

        This method is used to select a point in the point cloud based on the mouse click position in widget coordinates.
        It reads the depth buffer to get the depth value at the mouse position and then unprojects the screen coordinates
        to world coordinates. The closest point to the unprojected coordinates is selected if it lies within a specified
        threshold.

        The distance threshold used for selecting points is defined by the `picking_point_threshold_factor` attribute.
        You can adjust this attribute to control how close a point must be to be considered selectable.

        If a point is successfully selected, its index is added to the `picked_points_indices` attribute, which stores the
        indices of all currently selected points.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """

        idx, _ = self._unproject_mouse_to_nearest_point(mouse_pos)
        if idx is None:
            return

        if not self._is_point_selectable(idx):
            return
        if idx not in self.picked_points_indices:
            self.picked_points_indices.append(idx)

    def deselect_point_at(self, mouse_pos):
        """
        Deselect a point in the point cloud at the given mouse position.

        This method is used to deselect a previously selected point in the point cloud based on the mouse click position
        in widget coordinates. It projects the picked points to screen space and determines the closest point to the
        mouse click position. If the closest point lies within a specified pixel threshold, it is deselected.

        The pixel threshold used for deselecting points is defined by the `pixel_threshold` attribute. You can adjust
        this attribute to control how close a point must be to be considered for deselection.

        If a point is successfully deselected, its index is removed from the `picked_points_indices` attribute, which
        stores the indices of all currently selected points.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """

        # Ensure OpenGL context is current
        self.makeCurrent()

        # Use stored matrices
        viewport = self.viewport

        # Convert mouse position to screen space coordinates
        click_x = mouse_pos.x()
        click_y = viewport[3] - mouse_pos.y()  # Invert Y coordinate (adjust if necessary)

        # Project picked points to screen space
        screen_positions = []
        for index in self.picked_points_indices:
            point_3d = self.points[index, :3]
            screen_pos = self.project_to_screen(point_3d)
            screen_positions.append((index, screen_pos))

        # Find the closest picked point to the mouse click
        min_distance = float('inf')
        closest_index = None
        for index, screen_pos in screen_positions:
            dx = screen_pos[0] - click_x
            dy = screen_pos[1] - click_y
            distance = np.hypot(dx, dy)
            if distance < min_distance:
                min_distance = distance
                closest_index = index

        # Define a threshold in pixels (e.g., radius of the sphere in screen space)
        # TODO: Not sure if the pixel threshold is appropriate for all cases
        pixel_threshold = self.pixel_threshold

        if min_distance <= pixel_threshold:
            # Remove the point from picked points
            self.picked_points_indices.remove(closest_index)
            # Invalidate stored polygons so plugins fall back to coordinate matching
            self._selection_polygons.clear()

    def deselect_cluster_at(self, mouse_pos):
        """
        Deselect all selected points that share the same color as the clicked point.

        This removes an entire cluster from the selection by matching the clicked point's
        RGB color against the colors of all currently selected points. Points with matching
        colors are removed from picked_points_indices.

        Args:
            mouse_pos (QPoint): The position of the mouse click in widget coordinates.
        """
        if not self.picked_points_indices:
            return

        clicked_index, _ = self._unproject_mouse_to_nearest_point(mouse_pos)
        if clicked_index is None:
            return

        # Get the color of the clicked point
        target_color = self.points[clicked_index, 3:6]

        # Vectorized removal of all selected points matching this color
        selected = np.array(self.picked_points_indices, dtype=np.int64)
        colors = self.points[selected, 3:6]
        matches = np.all(colors == target_color, axis=1)
        self.picked_points_indices[:] = selected[~matches].tolist()
        # Invalidate stored polygons so plugins fall back to coordinate matching
        self._selection_polygons.clear()

        self.update()
