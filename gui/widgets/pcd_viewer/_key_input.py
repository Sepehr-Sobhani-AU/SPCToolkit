import logging
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from config.config import global_variables

logger = logging.getLogger(__name__)


class KeyInputMixin:
    """Keyboard event handling for PCDViewerWidget."""

    def keyPressEvent(self, event):
        """
        Handle key press events for interaction with the point cloud.

        This method processes key press events to allow specific actions, such as deselecting all picked points. If
        the Escape key is pressed, a confirmation dialog is displayed, and if confirmed, all selected points are
        deselected.

        Args:
            event (QKeyEvent): The key event containing details such as the key pressed.
        """

        if event.key() == Qt.Key_Z and not (event.modifiers() & Qt.ControlModifier):
            # Z: Toggle zoom window mode
            if self._zoom_window_mode:
                self.exit_zoom_window_mode()
            else:
                self.enter_zoom_window_mode()
            return
        elif event.key() == Qt.Key_P:
            if event.modifiers() & Qt.ShiftModifier:
                # Shift+P: Enter polygon deselect mode
                if self._polygon_mode:
                    self.exit_polygon_mode()
                else:
                    self.enter_polygon_deselect_mode()
            else:
                # P: Toggle polygon selection mode
                if self._polygon_mode:
                    self.exit_polygon_mode()
                else:
                    self.enter_polygon_mode()
        elif event.key() == Qt.Key_Escape:
            if self._zoom_window_mode:
                self.exit_zoom_window_mode()
            elif self._polygon_mode:
                # Cancel polygon mode without selecting
                self.exit_polygon_mode()
            else:
                # Create a confirmation dialog box
                reply = QMessageBox.question(self, 'Confirmation', 'Deselect all selected points?',
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    # Clear all selected points if confirmed
                    self.picked_points_indices.clear()
                    self._selection_polygons.clear()
                    self.update()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_R:
            self.reset_view()
        elif event.key() == Qt.Key_F:
            self.zoom_to_extent()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            new_size = min(self._point_size * self._POINT_SIZE_FACTOR, self._POINT_SIZE_MAX)
            self.point_size = new_size
        elif event.key() == Qt.Key_Minus:
            new_size = max(self._point_size / self._POINT_SIZE_FACTOR, self._POINT_SIZE_MIN)
            self.point_size = new_size
        elif event.key() == Qt.Key_C:
            main_window = global_variables.global_main_window
            if main_window:
                main_window.execute_action_plugin("cut_cluster")
        elif event.key() == Qt.Key_M:
            main_window = global_variables.global_main_window
            if main_window:
                main_window.execute_action_plugin("merge_clusters")
        elif event.key() == Qt.Key_Delete:
            main_window = global_variables.global_main_window
            if main_window:
                main_window.execute_action_plugin("remove_clusters")

        # Ensure the parent class handles other key events
        super().keyPressEvent(event)
