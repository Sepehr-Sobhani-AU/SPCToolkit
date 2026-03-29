"""
Plugin for camera flythrough animation along a smooth Catmull-Rom spline path.

Workflow:
1. User clicks View > Flythrough — a non-modal dialog opens (viewer stays interactive)
2. Select an existing flythrough from the combobox, or click New to start fresh
3. Navigate to desired views and click "Add Waypoint"; set per-waypoint segment duration
4. Double-click a waypoint to jump the camera there instantly
5. Click Save to store the flythrough (it appears in the combobox)
6. Click Play to animate; File > Save Project persists everything to disk
"""

import copy
from typing import Dict, Any, List

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QComboBox, QInputDialog, QMessageBox, QSizePolicy,
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
    Return a deep copy of waypoints with rot_x/y/z unwrapped so Catmull-Rom
    interpolates through them without 360° jumps.
    """
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
    """Non-modal dialog for managing flythroughs and running camera animation."""

    _DEFAULT_DURATION = 3.0
    _DEFAULT_FPS = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flythrough")
        self.setMinimumSize(500, 560)

        self._waypoints: List[dict] = []
        self._current_ft_index: int = -1  # -1 = unsaved new flythrough

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animation_tick)
        self._frame_index = 0
        self._total_frames = 0
        self._seg_starts: List[int] = []
        self._seg_frames: List[int] = []
        self._anim_waypoints: List[dict] = []

        self._building_ui = True
        self._build_ui()
        self._building_ui = False
        self._populate_combobox()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- Flythrough selector ----
        ft_row = QHBoxLayout()
        ft_row.addWidget(QLabel("Flythrough:"))
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        ft_row.addWidget(self._combo)
        self._btn_new = QPushButton("New")
        self._btn_new.clicked.connect(self._new_flythrough)
        ft_row.addWidget(self._btn_new)
        layout.addLayout(ft_row)

        # ---- Waypoints ----
        wp_group = QGroupBox("Waypoints  (double-click to jump)")
        wp_layout = QVBoxLayout(wp_group)

        self._wp_list = QListWidget()
        self._wp_list.setSelectionMode(QListWidget.SingleSelection)
        self._wp_list.currentRowChanged.connect(self._on_waypoint_selected)
        self._wp_list.itemDoubleClicked.connect(self._jump_to_waypoint)
        wp_layout.addWidget(self._wp_list)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add Current View")
        self._btn_remove = QPushButton("Remove")
        self._btn_up = QPushButton("Up")
        self._btn_down = QPushButton("Down")
        self._btn_rename = QPushButton("Rename")
        for b in (self._btn_add, self._btn_remove, self._btn_up,
                  self._btn_down, self._btn_rename):
            btn_row.addWidget(b)
        wp_layout.addLayout(btn_row)

        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Segment duration (s):"))
        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.5, 60.0)
        self._dur_spin.setValue(self._DEFAULT_DURATION)
        self._dur_spin.setSingleStep(0.5)
        self._dur_spin.setEnabled(False)
        self._dur_spin.valueChanged.connect(self._on_duration_changed)
        dur_row.addWidget(self._dur_spin)
        dur_row.addStretch()
        wp_layout.addLayout(dur_row)

        layout.addWidget(wp_group)

        # ---- Settings ----
        settings_group = QGroupBox("Animation Settings")
        settings_layout = QHBoxLayout(settings_group)
        settings_layout.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(10, 60)
        self._fps_spin.setValue(self._DEFAULT_FPS)
        settings_layout.addWidget(self._fps_spin)
        settings_layout.addStretch()
        layout.addWidget(settings_group)

        # ---- Status ----
        self._status_label = QLabel("Select a flythrough or click New.")
        self._status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # ---- Playback ----
        pb_row = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_stop = QPushButton("Stop")
        self._btn_save = QPushButton("Save")
        self._btn_stop.setEnabled(False)
        for b in (self._btn_play, self._btn_stop, self._btn_save):
            pb_row.addWidget(b)
        layout.addLayout(pb_row)

        # ---- Wire up ----
        self._btn_add.clicked.connect(self._add_waypoint)
        self._btn_remove.clicked.connect(self._remove_waypoint)
        self._btn_up.clicked.connect(self._move_up)
        self._btn_down.clicked.connect(self._move_down)
        self._btn_rename.clicked.connect(self._rename_waypoint)
        self._btn_play.clicked.connect(self._start_play)
        self._btn_stop.clicked.connect(self._stop_animation)
        self._btn_save.clicked.connect(self._save_flythrough)

    # ------------------------------------------------------------------
    # Flythrough management (combobox)
    # ------------------------------------------------------------------

    def _populate_combobox(self):
        self._building_ui = True
        self._combo.clear()
        fm = global_variables.global_file_manager
        if fm:
            for ft in fm.flythroughs:
                self._combo.addItem(ft["name"])
        self._building_ui = False

        if self._combo.count() > 0:
            self._combo.setCurrentIndex(0)
            self._load_flythrough(0)
        else:
            self._new_flythrough()

    def _on_combo_changed(self, index: int):
        if self._building_ui or index < 0:
            return
        self._load_flythrough(index)

    def _load_flythrough(self, index: int):
        fm = global_variables.global_file_manager
        if fm is None or index < 0 or index >= len(fm.flythroughs):
            return
        ft = fm.flythroughs[index]
        self._current_ft_index = index
        self._fps_spin.setValue(ft.get("fps", self._DEFAULT_FPS))
        self._waypoints = copy.deepcopy(ft["waypoints"])
        self._refresh_list()
        self._status_label.setText(
            f"Loaded '{ft['name']}' — {len(self._waypoints)} waypoint(s)."
        )

    def _new_flythrough(self):
        self._current_ft_index = -1
        self._waypoints = []
        self._building_ui = True
        self._combo.setCurrentIndex(-1)
        self._building_ui = False
        self._refresh_list()
        self._dur_spin.setEnabled(False)
        self._status_label.setText("New flythrough — add waypoints, then click Save.")

    def _save_flythrough(self):
        fm = global_variables.global_file_manager
        if fm is None:
            return
        if len(self._waypoints) < 2:
            QMessageBox.warning(self, "Flythrough", "Add at least 2 waypoints before saving.")
            return

        if self._current_ft_index == -1:
            # New flythrough — ask for a name
            name, ok = QInputDialog.getText(
                self, "Save Flythrough", "Name:", text="Flythrough 1"
            )
            if not ok or not name.strip():
                return
            name = name.strip()
            ft = {
                "name": name,
                "fps": self._fps_spin.value(),
                "waypoints": copy.deepcopy(self._waypoints),
            }
            fm.flythroughs.append(ft)
            self._current_ft_index = len(fm.flythroughs) - 1
            self._building_ui = True
            self._combo.addItem(name)
            self._combo.setCurrentIndex(self._current_ft_index)
            self._building_ui = False
        else:
            # Overwrite existing
            fm.flythroughs[self._current_ft_index]["fps"] = self._fps_spin.value()
            fm.flythroughs[self._current_ft_index]["waypoints"] = copy.deepcopy(self._waypoints)

        self._status_label.setText(
            f"Saved — use File > Save Project to persist to disk."
        )

    # ------------------------------------------------------------------
    # Waypoint management
    # ------------------------------------------------------------------

    def _capture_camera(self) -> dict:
        v = global_variables.global_pcd_viewer_widget
        n = len(self._waypoints) + 1
        return {
            "name": f"Waypoint {n}",
            "duration": self._DEFAULT_DURATION,
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
    def _make_label(wp: dict) -> QListWidgetItem:
        text = (
            f"{wp['name']}  ({wp.get('duration', 3.0):.1f}s)  —  "
            f"rot({wp['rot_x']:.1f}, {wp['rot_y']:.1f}, {wp['rot_z']:.1f})"
        )
        return QListWidgetItem(text)

    def _refresh_list(self):
        self._wp_list.clear()
        for wp in self._waypoints:
            self._wp_list.addItem(self._make_label(wp))

    def _on_waypoint_selected(self, row: int):
        if row < 0 or row >= len(self._waypoints):
            self._dur_spin.setEnabled(False)
            return
        self._dur_spin.setEnabled(True)
        # Temporarily block valueChanged so we don't write back while loading
        self._dur_spin.blockSignals(True)
        self._dur_spin.setValue(self._waypoints[row].get("duration", self._DEFAULT_DURATION))
        self._dur_spin.blockSignals(False)

    def _on_duration_changed(self, value: float):
        row = self._wp_list.currentRow()
        if row < 0 or row >= len(self._waypoints):
            return
        self._waypoints[row]["duration"] = value
        # Refresh label in place
        self._wp_list.item(row).setText(self._make_label(self._waypoints[row]).text())

    def _jump_to_waypoint(self, item: QListWidgetItem):
        row = self._wp_list.row(item)
        if row < 0 or row >= len(self._waypoints):
            return
        wp = self._waypoints[row]
        v = global_variables.global_pcd_viewer_widget
        v.rot_x = wp["rot_x"]
        v.rot_y = wp["rot_y"]
        v.rot_z = wp["rot_z"]
        v.pan_x = wp["pan_x"]
        v.pan_y = wp["pan_y"]
        v.pan_z = wp["pan_z"]
        v.camera_distance = wp["camera_distance"]
        v.zoom_factor = wp["zoom_factor"]
        v.center = wp["center"].copy()
        v.update()

    def _add_waypoint(self):
        wp = self._capture_camera()
        self._waypoints.append(wp)
        self._wp_list.addItem(self._make_label(wp))
        self._wp_list.setCurrentRow(len(self._waypoints) - 1)
        n = len(self._waypoints)
        self._status_label.setText(
            f"{n} waypoint(s)." if n < 2 else f"{n} waypoints — ready to Play or Save."
        )

    def _remove_waypoint(self):
        row = self._wp_list.currentRow()
        if row < 0:
            return
        self._wp_list.takeItem(row)
        self._waypoints.pop(row)

    def _move_up(self):
        row = self._wp_list.currentRow()
        if row <= 0:
            return
        self._waypoints[row - 1], self._waypoints[row] = (
            self._waypoints[row], self._waypoints[row - 1]
        )
        self._refresh_list()
        self._wp_list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self._wp_list.currentRow()
        if row < 0 or row >= len(self._waypoints) - 1:
            return
        self._waypoints[row], self._waypoints[row + 1] = (
            self._waypoints[row + 1], self._waypoints[row]
        )
        self._refresh_list()
        self._wp_list.setCurrentRow(row + 1)

    def _rename_waypoint(self):
        row = self._wp_list.currentRow()
        if row < 0:
            return
        current = self._waypoints[row]["name"]
        new_name, ok = QInputDialog.getText(
            self, "Rename Waypoint", "Name:", text=current
        )
        if ok and new_name.strip():
            self._waypoints[row]["name"] = new_name.strip()
            self._wp_list.item(row).setText(self._make_label(self._waypoints[row]).text())

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _start_play(self):
        if len(self._waypoints) < 2:
            QMessageBox.warning(self, "Flythrough", "Add at least 2 waypoints.")
            return

        fps = self._fps_spin.value()
        self._anim_waypoints = _unwrap_waypoint_angles(self._waypoints)

        # Per-segment frame counts (N-1 segments for N waypoints)
        self._seg_frames = [
            max(2, int(fps * self._waypoints[i].get("duration", self._DEFAULT_DURATION)))
            for i in range(len(self._waypoints) - 1)
        ]
        # Cumulative start frame for each segment
        self._seg_starts = []
        cumulative = 0
        for f in self._seg_frames:
            self._seg_starts.append(cumulative)
            cumulative += f
        self._total_frames = cumulative
        self._frame_index = 0

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

        # Find current segment
        seg = len(self._seg_starts) - 1
        for i in range(len(self._seg_starts) - 1, -1, -1):
            if self._frame_index >= self._seg_starts[i]:
                seg = i
                break

        local = self._frame_index - self._seg_starts[seg]
        t = local / max(self._seg_frames[seg] - 1, 1)

        last = len(self._anim_waypoints) - 1
        i0 = max(seg - 1, 0)
        i1 = seg
        i2 = seg + 1
        i3 = min(seg + 2, last)

        w0 = self._anim_waypoints[i0]
        w1 = self._anim_waypoints[i1]
        w2 = self._anim_waypoints[i2]
        w3 = self._anim_waypoints[i3]

        v = global_variables.global_pcd_viewer_widget

        v.pan_x = _catmull_rom(w0["pan_x"], w1["pan_x"], w2["pan_x"], w3["pan_x"], t)
        v.pan_y = _catmull_rom(w0["pan_y"], w1["pan_y"], w2["pan_y"], w3["pan_y"], t)
        v.pan_z = _catmull_rom(w0["pan_z"], w1["pan_z"], w2["pan_z"], w3["pan_z"], t)
        v.camera_distance = _catmull_rom(
            w0["camera_distance"], w1["camera_distance"],
            w2["camera_distance"], w3["camera_distance"], t,
        )
        v.zoom_factor = _catmull_rom(
            w0["zoom_factor"], w1["zoom_factor"],
            w2["zoom_factor"], w3["zoom_factor"], t,
        )
        v.center = _catmull_rom(w0["center"], w1["center"], w2["center"], w3["center"], t)
        v.rot_x = _catmull_rom(w0["rot_x"], w1["rot_x"], w2["rot_x"], w3["rot_x"], t)
        v.rot_y = _catmull_rom(w0["rot_y"], w1["rot_y"], w2["rot_y"], w3["rot_y"], t)
        v.rot_z = _catmull_rom(w0["rot_z"], w1["rot_z"], w2["rot_z"], w3["rot_z"], t)

        v.update()

        segments = len(self._waypoints) - 1
        self._status_label.setText(
            f"Frame {self._frame_index + 1}/{self._total_frames}  "
            f"(segment {seg + 1}/{segments},  t={t:.2f})"
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
