"""
Plugin for camera flythrough animation along a smooth Catmull-Rom spline path.

Workflow:
1. User clicks View > Flythrough — a non-modal dialog opens (viewer stays interactive)
2. User navigates to desired views and clicks "Add Waypoint" at each position
3. User sets duration per segment and FPS, then clicks Play
4. Camera animates smoothly through all waypoints along a curved path
"""

from typing import Dict, Any, List

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import QTimer

from plugins.interfaces import ActionPlugin
from config.config import global_variables


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _catmull_rom(p0, p1, p2, p3, t: float):
    """
    Uniform Catmull-Rom spline interpolation between p1 and p2.
    p0 and p3 are the neighbouring control points that shape the curve.
    Works for floats and numpy arrays.
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _unwrap_waypoint_angles(waypoints: list) -> list:
    """
    Return a copy of waypoints with rot_x/y/z unwrapped so Catmull-Rom can
    interpolate through them smoothly without 360° jumps.
    Each angle is adjusted to be within ±180° of the previous waypoint's angle.
    """
    import copy
    result = copy.deepcopy(waypoints)
    for key in ("rot_x", "rot_y", "rot_z"):
        for i in range(1, len(result)):
            diff = (result[i][key] - result[i - 1][key] + 180.0) % 360.0 - 180.0
            result[i][key] = result[i - 1][key] + diff
    return result


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class FlythroughDialog(QDialog):
    """Non-modal dialog for managing camera waypoints and running the flythrough."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flythrough")
        self.setMinimumSize(460, 460)

        self._waypoints: List[dict] = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animation_tick)

        self._frame_index = 0
        self._total_frames = 0
        self._frames_per_segment = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Waypoints
        wp_group = QGroupBox("Waypoints")
        wp_layout = QVBoxLayout(wp_group)

        self._wp_list = QListWidget()
        self._wp_list.setSelectionMode(QListWidget.SingleSelection)
        wp_layout.addWidget(self._wp_list)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add Current View")
        self._btn_remove = QPushButton("Remove")
        self._btn_up = QPushButton("Up")
        self._btn_down = QPushButton("Down")
        for b in (self._btn_add, self._btn_remove, self._btn_up, self._btn_down):
            btn_row.addWidget(b)
        wp_layout.addLayout(btn_row)
        layout.addWidget(wp_group)

        # Settings
        settings_group = QGroupBox("Animation Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Duration per segment (s):"))
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0.5, 30.0)
        self._duration_spin.setValue(3.0)
        self._duration_spin.setSingleStep(0.5)
        settings_layout.addWidget(self._duration_spin)

        settings_layout.addStretch()

        settings_layout.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(10, 60)
        self._fps_spin.setValue(30)
        settings_layout.addWidget(self._fps_spin)

        layout.addWidget(settings_group)

        # Status
        self._status_label = QLabel("Add at least 2 waypoints, then press Play.")
        self._status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self._status_label)

        # Playback
        pb_row = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        pb_row.addWidget(self._btn_play)
        pb_row.addWidget(self._btn_stop)
        layout.addLayout(pb_row)

        # Wire up
        self._btn_add.clicked.connect(self._add_waypoint)
        self._btn_remove.clicked.connect(self._remove_waypoint)
        self._btn_up.clicked.connect(self._move_up)
        self._btn_down.clicked.connect(self._move_down)
        self._btn_play.clicked.connect(self._start_play)
        self._btn_stop.clicked.connect(self._stop_animation)

    # ------------------------------------------------------------------
    # Waypoint management
    # ------------------------------------------------------------------

    def _capture_camera(self) -> dict:
        v = global_variables.global_pcd_viewer_widget
        return {
            "rot_x": v.rot_x,
            "rot_y": v.rot_y,
            "rot_z": v.rot_z,
            "pan_x": v.pan_x,
            "pan_y": v.pan_y,
            "pan_z": v.pan_z,
            "camera_distance": v.camera_distance,
            "zoom_factor": v.zoom_factor,
            "center": v.center.copy(),
        }

    @staticmethod
    def _make_label(idx: int, wp: dict) -> QListWidgetItem:
        text = (
            f"Waypoint {idx}  —  "
            f"rot({wp['rot_x']:.1f}, {wp['rot_y']:.1f}, {wp['rot_z']:.1f})  "
            f"dist={wp['camera_distance']:.1f}"
        )
        return QListWidgetItem(text)

    def _refresh_labels(self):
        self._wp_list.clear()
        for i, wp in enumerate(self._waypoints):
            self._wp_list.addItem(self._make_label(i + 1, wp))

    def _add_waypoint(self):
        wp = self._capture_camera()
        self._waypoints.append(wp)
        self._wp_list.addItem(self._make_label(len(self._waypoints), wp))
        self._status_label.setText(
            f"{len(self._waypoints)} waypoint(s) added."
            if len(self._waypoints) < 2
            else f"{len(self._waypoints)} waypoints — ready to Play."
        )

    def _remove_waypoint(self):
        row = self._wp_list.currentRow()
        if row < 0:
            return
        self._wp_list.takeItem(row)
        self._waypoints.pop(row)
        self._refresh_labels()

    def _move_up(self):
        row = self._wp_list.currentRow()
        if row <= 0:
            return
        self._waypoints[row - 1], self._waypoints[row] = (
            self._waypoints[row], self._waypoints[row - 1]
        )
        self._refresh_labels()
        self._wp_list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self._wp_list.currentRow()
        if row < 0 or row >= len(self._waypoints) - 1:
            return
        self._waypoints[row], self._waypoints[row + 1] = (
            self._waypoints[row + 1], self._waypoints[row]
        )
        self._refresh_labels()
        self._wp_list.setCurrentRow(row + 1)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _start_play(self):
        if len(self._waypoints) < 2:
            QMessageBox.warning(self, "Flythrough", "Add at least 2 waypoints.")
            return

        fps = self._fps_spin.value()
        duration = self._duration_spin.value()
        segments = len(self._waypoints) - 1
        self._frames_per_segment = max(2, int(fps * duration))
        self._total_frames = self._frames_per_segment * segments
        self._frame_index = 0
        self._anim_waypoints = _unwrap_waypoint_angles(self._waypoints)

        self._btn_play.setEnabled(False)
        self._btn_stop.setEnabled(True)

        interval_ms = max(1, int(1000 / fps))
        self._timer.start(interval_ms)

    def _stop_animation(self):
        self._timer.stop()
        self._btn_play.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_label.setText("Stopped.")

    def _animation_tick(self):
        if self._frame_index >= self._total_frames:
            self._stop_animation()
            self._status_label.setText("Playback complete.")
            return

        n = len(self._waypoints)
        last = n - 1

        seg = self._frame_index // self._frames_per_segment
        local = self._frame_index % self._frames_per_segment
        t = local / max(self._frames_per_segment - 1, 1)

        # Catmull-Rom neighbours (clamp to valid range)
        i0 = max(seg - 1, 0)
        i1 = seg
        i2 = seg + 1
        i3 = min(seg + 2, last)

        w0, w1, w2, w3 = (
            self._anim_waypoints[i0],
            self._anim_waypoints[i1],
            self._anim_waypoints[i2],
            self._anim_waypoints[i3],
        )

        v = global_variables.global_pcd_viewer_widget

        # Positions — Catmull-Rom
        v.pan_x = _catmull_rom(w0["pan_x"], w1["pan_x"], w2["pan_x"], w3["pan_x"], t)
        v.pan_y = _catmull_rom(w0["pan_y"], w1["pan_y"], w2["pan_y"], w3["pan_y"], t)
        v.pan_z = _catmull_rom(w0["pan_z"], w1["pan_z"], w2["pan_z"], w3["pan_z"], t)
        v.camera_distance = _catmull_rom(
            w0["camera_distance"], w1["camera_distance"],
            w2["camera_distance"], w3["camera_distance"], t
        )
        v.zoom_factor = _catmull_rom(
            w0["zoom_factor"], w1["zoom_factor"],
            w2["zoom_factor"], w3["zoom_factor"], t
        )
        v.center = _catmull_rom(w0["center"], w1["center"], w2["center"], w3["center"], t)

        # Rotations — Catmull-Rom on unwrapped angles
        v.rot_x = _catmull_rom(w0["rot_x"], w1["rot_x"], w2["rot_x"], w3["rot_x"], t)
        v.rot_y = _catmull_rom(w0["rot_y"], w1["rot_y"], w2["rot_y"], w3["rot_y"], t)
        v.rot_z = _catmull_rom(w0["rot_z"], w1["rot_z"], w2["rot_z"], w3["rot_z"], t)

        v.update()

        self._status_label.setText(
            f"Frame {self._frame_index + 1}/{self._total_frames}  "
            f"(segment {seg + 1}/{last},  t={t:.2f})"
        )
        self._frame_index += 1

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

class FlythroughPlugin(ActionPlugin):
    """Opens the Flythrough dialog for smooth camera path animation."""

    def get_name(self) -> str:
        return "flythrough"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        if global_variables.global_pcd_viewer_widget is None:
            return
        dialog = FlythroughDialog(main_window)
        dialog.show()
