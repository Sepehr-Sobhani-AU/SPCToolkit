"""
Export E57 Plugin

Exports the selected branches as an E57 file.
Multiple selected branches are merged before export.
Preserves RGB colors and intensity if available.
"""

from typing import Dict, Any
import os
import logging
import uuid as _uuid
import numpy as np
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QDialog, QVBoxLayout,
                               QLabel, QLineEdit, QDialogButtonBox,
                               QGroupBox, QGridLayout)

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud

logger = logging.getLogger(__name__)


class ExportE57Plugin(ActionPlugin):

    def get_name(self) -> str:
        return "export_e57"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        try:
            import pye57
        except ImportError:
            QMessageBox.critical(
                main_window, "Missing Library",
                "The 'pye57' library is required to export E57 files.\n"
                "Install it with: pip install pye57"
            )
            return

        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes

        if not controller.selected_branches:
            QMessageBox.warning(
                main_window, "No Selection",
                "Please select one or more branches to export."
            )
            return

        # Reconstruct each selected branch
        point_clouds = []
        for uid in controller.selected_branches:
            try:
                pc = controller.reconstruct(uid)
                if pc is not None:
                    point_clouds.append(pc)
            except Exception as e:
                QMessageBox.critical(
                    main_window, "Reconstruction Error",
                    f"Failed to reconstruct branch {uid}:\n{str(e)}"
                )
                return

        if not point_clouds:
            QMessageBox.warning(
                main_window, "No Data",
                "Selected branches produced no point cloud data."
            )
            return

        # Merge if multiple branches
        if len(point_clouds) == 1:
            merged = point_clouds[0]
        else:
            merged = PointCloud.merge(point_clouds, name="Export")

        # Detect coordinate translation from root PointCloud
        detected_translation = _find_root_translation(
            data_nodes, controller.selected_branches[0]
        )
        detected_shift = -detected_translation

        dialog = _ShiftDialog(main_window, detected_shift)
        if dialog.exec_() != QDialog.Accepted:
            return
        shift = dialog.get_shift()

        # File save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            main_window,
            "Export E57",
            "",
            "E57 Files (*.e57)"
        )

        if not file_path:
            return

        if not file_path.lower().endswith('.e57'):
            file_path += '.e57'

        try:
            _write_e57(merged, file_path, shift)
            logger.info(f"Exported {merged.size} points to {file_path}")
        except Exception as e:
            QMessageBox.critical(
                main_window, "Export Error",
                f"Failed to export:\n{str(e)}"
            )


def _find_root_translation(data_nodes, uid_str: str) -> np.ndarray:
    """Walk up to the root PointCloud and return its translation."""
    node = data_nodes.get_node(_uuid.UUID(uid_str))
    visited = set()
    while node is not None and node.uid not in visited:
        visited.add(node.uid)
        if node.data_type == "point_cloud" and node.parent_uid is None:
            return getattr(node.data, 'translation', np.zeros(3))
        if node.parent_uid is None:
            break
        node = data_nodes.get_node(node.parent_uid)
    return np.zeros(3)


class _ShiftDialog(QDialog):
    """Dialog showing the detected coordinate shift, editable by the user."""

    def __init__(self, parent, shift: np.ndarray):
        super().__init__(parent)
        self.setWindowTitle("Coordinate Shift")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)

        info = QLabel(
            "On import, the point cloud was shifted to the origin.\n"
            "The detected shift to restore original coordinates is shown below.\n"
            "Adjust if needed, or set all to 0 for no shift."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        group = QGroupBox("Shift (added to exported points)")
        grid = QGridLayout(group)

        self._edits = {}
        for row, (axis, val) in enumerate(zip(("X", "Y", "Z"), shift)):
            grid.addWidget(QLabel(f"{axis}:"), row, 0)
            edit = QLineEdit(f"{val:.6f}")
            self._edits[axis] = edit
            grid.addWidget(edit, row, 1)

        layout.addWidget(group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_shift(self) -> np.ndarray:
        values = []
        for axis in ("X", "Y", "Z"):
            try:
                values.append(float(self._edits[axis].text()))
            except ValueError:
                values.append(0.0)
        return np.array(values, dtype=np.float64)


def _apply_shift_gpu(points_f32, shift):
    """
    Apply coordinate shift and convert to float64.
    Uses GPU (CuPy) when available, falls back to CPU.
    """
    from infrastructure.hardware_detector import HardwareDetector

    apply_shift = shift is not None and np.any(shift != 0)

    if HardwareDetector.can_use_cupy():
        try:
            import cupy as cp
            pts_gpu = cp.asarray(points_f32).astype(cp.float64)
            if apply_shift:
                pts_gpu += cp.asarray(shift, dtype=cp.float64)
            result = cp.asnumpy(pts_gpu)
            del pts_gpu
            logger.info(f"GPU-accelerated coordinate shift ({len(result)} pts)")
            return result
        except Exception as e:
            logger.warning(f"GPU shift failed, falling back to CPU: {e}")

    points = points_f32.astype(np.float64)
    if apply_shift:
        points += shift.astype(np.float64)
    return points


def _write_e57(pc: PointCloud, file_path: str, shift: np.ndarray = None) -> None:
    """
    Write a PointCloud to an E57 file as a single scan.

    Uses float64 for coordinates to preserve precision on large values.
    """
    import pye57

    n = pc.size

    # Apply coordinate shift in float64 (GPU-accelerated)
    points = _apply_shift_gpu(pc.points, shift)

    # Build data dictionary for pye57
    data = {
        "cartesianX": points[:, 0],
        "cartesianY": points[:, 1],
        "cartesianZ": points[:, 2],
    }

    # Add colors if available (E57 expects 0-255 integer range)
    if pc.colors is not None and len(pc.colors) > 0:
        colors = pc.colors
        if colors.max() <= 1.0:
            data["colorRed"] = (colors[:, 0] * 255).astype(np.uint8)
            data["colorGreen"] = (colors[:, 1] * 255).astype(np.uint8)
            data["colorBlue"] = (colors[:, 2] * 255).astype(np.uint8)
        else:
            data["colorRed"] = colors[:, 0].astype(np.uint8)
            data["colorGreen"] = colors[:, 1].astype(np.uint8)
            data["colorBlue"] = colors[:, 2].astype(np.uint8)

    # Add intensity if available
    intensity = pc.attributes.get('intensity')
    if isinstance(intensity, np.ndarray) and len(intensity) == n:
        data["intensity"] = intensity.astype(np.float32)

    # Remove existing file (pye57 doesn't overwrite)
    if os.path.exists(file_path):
        os.remove(file_path)

    e57 = pye57.E57(file_path, mode="w")
    e57.write_scan_raw(data)
    e57.close()
