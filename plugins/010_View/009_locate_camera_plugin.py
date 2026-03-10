"""
Plugin for displaying the current camera position and state.

Workflow:
1. User clicks View > Locate Camera
2. Plugin reads the viewer's model-view matrix and camera parameters
3. A dialog shows eye position, look-at center, rotation, pan, zoom, FOV, etc.
"""

from typing import Dict, Any

import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox
from PyQt5.QtGui import QFont

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class LocateCameraPlugin(ActionPlugin):
    """Action plugin that shows the current camera state in a copyable dialog."""

    def get_name(self) -> str:
        return "locate_camera"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer = global_variables.global_pcd_viewer_widget
        if viewer is None:
            return

        mv = np.array(viewer.model_view_matrix, dtype=np.float64)

        # OpenGL stores column-major; numpy sees the transpose.
        # The upper-left 3x3 of the true model-view is R, last column is t.
        # inv(MV) gives the camera frame: last column = eye position.
        try:
            mv_inv = np.linalg.inv(mv.T)  # transpose to get true MV, then invert
            eye = mv_inv[:3, 3]
        except np.linalg.LinAlgError:
            eye = np.array([0.0, 0.0, 0.0])

        effective_distance = viewer.camera_distance * viewer.zoom_factor
        num_points = len(viewer.points) if viewer.points is not None else 0

        lines = [
            "Camera State",
            "=" * 40,
            "",
            f"Eye position (world):",
            f"  X: {eye[0]:>14.4f}",
            f"  Y: {eye[1]:>14.4f}",
            f"  Z: {eye[2]:>14.4f}",
            "",
            f"Look-at center:",
            f"  X: {viewer.center[0]:>14.4f}",
            f"  Y: {viewer.center[1]:>14.4f}",
            f"  Z: {viewer.center[2]:>14.4f}",
            "",
            f"Rotation (degrees):",
            f"  rot_x: {viewer.rot_x:>10.2f}",
            f"  rot_y: {viewer.rot_y:>10.2f}",
            f"  rot_z: {viewer.rot_z:>10.2f}",
            "",
            f"Pan offsets:",
            f"  pan_x: {viewer.pan_x:>14.6f}",
            f"  pan_y: {viewer.pan_y:>14.6f}",
            f"  pan_z: {viewer.pan_z:>14.6f}",
            "",
            f"Camera distance:     {viewer.camera_distance:>12.4f}",
            f"Zoom factor:         {viewer.zoom_factor:>12.6f}",
            f"Effective distance:  {effective_distance:>12.4f}",
            "",
            f"FOV (degrees):       {viewer.fov:>12.1f}",
            f"Point size:          {viewer.point_size:>12.1f}",
            f"Visible points:      {num_points:>12,}",
        ]

        text = "\n".join(lines)

        dialog = QDialog(main_window)
        dialog.setWindowTitle("Locate Camera")
        dialog.setMinimumSize(420, 460)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Monospace", 10))
        text_edit.setPlainText(text)
        layout.addWidget(text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec_()
