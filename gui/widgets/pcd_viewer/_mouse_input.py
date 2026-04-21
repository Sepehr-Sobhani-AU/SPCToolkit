import logging
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)


class MouseInputMixin:
    """Mouse event handling for PCDViewerWidget."""

    def _init_mouse_state(self):
        """Initialize mouse interaction state."""
        self.last_mouse_pos = None

        self.rot_x = 0.0  # Rotation around X-axis
        self.rot_y = 0.0  # Rotation around Y-axis
        self.rot_z = 0.0  # Rotation around Z-axis

        self.pan_x = 0.0  # Panning along X-axis
        self.pan_y = 0.0  # Panning along Y-axis
        self.pan_z = 0.0  # Panning along Z-axis

        # Interaction flags
        self.is_rotating = False
        self.is_rotating_z = False  # New flag for Z-axis rotation
        self.is_panning = False
        self.is_panning_z = False  # New flag for Z-axis panning
        self.show_axis = False  # Flag to show/hide the axis symbol

    def mousePressEvent(self, event):
        """
        Handle mouse press events for interaction with the point cloud.

        This method processes mouse press events to initiate interactions such as rotation, panning, and point
        selection. Depending on the mouse button and modifier keys pressed, the method determines the type of
        interaction (e.g., rotating, panning, or selecting points).

        Args:
            event (QMouseEvent): The mouse event containing details such as the button pressed and the mouse position.
        """

        # Zoom window mode: left-click starts rectangle, right-click cancels
        if self._zoom_window_mode:
            if event.button() == Qt.LeftButton:
                self._zoom_window_start = (event.x(), event.y())
                self._zoom_window_current = (event.x(), event.y())
                self.update()
            elif event.button() == Qt.RightButton:
                self.exit_zoom_window_mode()
            return

        # Polygon mode: left-click adds vertex, right-click closes polygon
        if self._polygon_mode:
            if event.button() == Qt.LeftButton:
                self._polygon_vertices.append((event.x(), event.y()))
                self.update()
                return
            elif event.button() == Qt.RightButton:
                if self._polygon_deselect_mode:
                    self._close_polygon_and_deselect()
                else:
                    self._close_polygon_and_select()
                return

        self.last_mouse_pos = event.pos()
        modifiers = event.modifiers()

        if (modifiers & Qt.ShiftModifier) and (modifiers & Qt.ControlModifier):
            if event.button() == Qt.RightButton:
                # Ctrl + Shift + Right Click: Deselect cluster by color
                self.deselect_cluster_at(event.pos())
        elif modifiers & Qt.ShiftModifier:
            if event.button() == Qt.LeftButton:
                # Shift + Left Click: Select a point
                self.pick_point(event.pos(), select=True)
            elif event.button() == Qt.RightButton:
                # Shift + Right Click: Deselect a point
                self.pick_point(event.pos(), select=False)
        elif modifiers & Qt.ControlModifier:
            if event.button() == Qt.LeftButton:
                # Ctrl + Left Click: Rotate around Z-axis
                self.setCursor(Qt.OpenHandCursor)
                self.is_rotating_z = True
            elif event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
                # Ctrl + Right/Middle Click: Pan along Z-axis
                self.setCursor(Qt.ClosedHandCursor)
                self.is_panning_z = True

            self.show_axis = True  # Show axis when panning
        else:
            if event.button() == Qt.LeftButton:
                # Rotate around X and Y axes
                self.setCursor(Qt.OpenHandCursor)
                self.is_rotating = True
            elif event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
                # Pan along X and Y axes
                self.setCursor(Qt.ClosedHandCursor)
                self.is_panning = True

            self.show_axis = True  # Show axis when panning

    def mouseDoubleClickEvent(self, event):
        """
        Handle mouse double-click events for updating the centre of rotation.

        This method processes double-click events to update the centre of rotation of the point cloud view. When the
        user double-clicks on a point, the centre of rotation is moved to that point, making it the new focal point
        for subsequent rotations.

        Args:
            event (QMouseEvent): The mouse event containing details such as the button pressed and the mouse position.
        """

        # Zoom window mode: ignore double-clicks
        if self._zoom_window_mode:
            return

        # Polygon mode: double-click closes polygon
        if self._polygon_mode:
            if event.button() == Qt.LeftButton:
                if self._polygon_deselect_mode:
                    self._close_polygon_and_deselect()
                else:
                    self._close_polygon_and_select()
            return

        if event.button() == Qt.LeftButton:
            # Update the rotation center on double left-click
            self.update_rotation_center(event.pos())

    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events for ending interactions.

        This method processes mouse release events to end interactions such as rotation or panning. When the user
        releases the mouse button, any ongoing rotation or panning is stopped, and the cursor is reset to its default
        state. The axis symbol is also hidden after the interaction ends.

        Args:
            event (QMouseEvent): The mouse event containing details such as the button released and the mouse position.
        """
        # Zoom window mode: left-release executes zoom
        if self._zoom_window_mode:
            if event.button() == Qt.LeftButton and self._zoom_window_start is not None:
                self._zoom_window_current = (event.x(), event.y())
                self._execute_zoom_window()
            return

        # Suppress release events in polygon mode
        if self._polygon_mode:
            return

        if event.button() == Qt.LeftButton:
            self.is_rotating = False
            self.is_rotating_z = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.is_panning_z = False
            self.setCursor(Qt.ArrowCursor)

        self.show_axis = False  # Hide axis when panning stops
        self.update()

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events for updating the interaction state.

        This method processes mouse movement events to update the state of ongoing interactions such as rotation and
        panning. Depending on the interaction mode (e.g., rotating or panning), it adjusts the rotation angles or
        panning offsets based on the mouse movement distance.

        Args:
            event (QMouseEvent): The mouse event containing details such as the current mouse position.
        """

        # Zoom window mode: update current corner during drag
        if self._zoom_window_mode:
            if self._zoom_window_start is not None:
                self._zoom_window_current = (event.x(), event.y())
                self.update()
            return

        # Suppress drag in polygon mode
        if self._polygon_mode:
            return

        if self.last_mouse_pos is None:
            return

        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()

        if self.is_rotating:
            # Rotate around X and Y axes
            # / 10 is a scaling factor to adjust the rotation sensitivity
            self.rot_x += dy * self.rotate_sensitivity / self._ROTATION_SENSITIVITY_DIVISOR
            self.rot_y += dx * self.rotate_sensitivity / self._ROTATION_SENSITIVITY_DIVISOR
        elif self.is_rotating_z:
            # Rotate around Z-axis
            self.rot_z += dx * self.rotate_sensitivity / self._ROTATION_SENSITIVITY_DIVISOR
        elif self.is_panning:
            # Pan along X and Y axes
            self.pan_x += dx * self.pan_sensitivity / self._PAN_SENSITIVITY_DIVISOR * self.camera_distance
            self.pan_y -= dy * self.pan_sensitivity / self._PAN_SENSITIVITY_DIVISOR * self.camera_distance
        elif self.is_panning_z:
            # Pan along Z-axis
            self.pan_z += dy * self.pan_sensitivity / self._PAN_SENSITIVITY_DIVISOR * self.camera_distance

        self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        """
        Handle mouse wheel events for zooming the view.

        This method processes mouse wheel events to adjust the zoom factor of the camera. The zoom sensitivity is
        controlled by the `zoom_sensitivity` attribute, and the resulting zoom factor is clamped between the
        `zoom_min_factor` and `zoom_max_factor` attributes to prevent excessive zooming in or out.

        Args:
            event (QWheelEvent): The wheel event containing details such as the direction and magnitude of the scroll.
        """

        delta = event.angleDelta().y()
        zoom_step = delta * self.zoom_sensitivity / self._ZOOM_WHEEL_DIVISOR
        self.zoom_factor *= (1 + zoom_step)
        self.zoom_factor = max(self.zoom_min_factor, min(self.zoom_factor, self.zoom_max_factor))  # Limit zoom factor

        self._show_axis_briefly()

        # Check if LOD needs to be updated
        self._on_zoom_changed()

        self.update()

    def hide_axis_after_zoom(self):
        """Hide the axis symbol after zooming is completed."""
        self.show_axis = False
        self.update()
