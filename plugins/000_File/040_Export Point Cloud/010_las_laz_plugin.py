"""
Export LAS/LAZ Plugin

Exports the selected branches as a LAS or LAZ file.
Multiple selected branches are merged before export.
Preserves per-point attributes (intensity, classification, etc.).
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


class ExportLASPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "export_las"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        try:
            import laspy
        except ImportError:
            QMessageBox.critical(
                main_window, "Missing Library",
                "The 'laspy' library is required to export LAS/LAZ files.\n"
                "Install it with: pip install laspy[lazrs]"
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
        file_path, selected_filter = QFileDialog.getSaveFileName(
            main_window,
            "Export LAS/LAZ",
            "",
            "LAS Files (*.las);;LAZ Files (*.laz)"
        )

        if not file_path:
            return

        # Ensure correct extension
        if "laz" in selected_filter.lower():
            if not file_path.lower().endswith('.laz'):
                file_path += '.laz'
        else:
            if not file_path.lower().endswith(('.las', '.laz')):
                file_path += '.las'

        try:
            _write_las(merged, file_path, shift)
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


def _write_las(pc: PointCloud, file_path: str, shift: np.ndarray = None) -> None:
    """
    Write a PointCloud to a LAS/LAZ file.

    Uses float64 for coordinates to preserve precision on large values.
    Automatically uses LAZ compression if the file extension is .laz.
    """
    import laspy

    n = pc.size

    # Apply coordinate shift in float64 (GPU-accelerated)
    points = _apply_shift_gpu(pc.points, shift)

    # Determine if we need extra attributes
    has_classification = (pc.attributes.get('classification') is not None
                          and isinstance(pc.attributes['classification'], np.ndarray))
    has_intensity = (pc.attributes.get('intensity') is not None
                     and isinstance(pc.attributes['intensity'], np.ndarray))
    has_gps_time = (pc.attributes.get('gps_time') is not None
                    and isinstance(pc.attributes['gps_time'], np.ndarray))
    has_return_number = (pc.attributes.get('return_number') is not None
                         and isinstance(pc.attributes['return_number'], np.ndarray))

    # Use point format 2 (XYZ + RGB) or 3 (XYZ + RGB + GPS time)
    point_format_id = 3 if has_gps_time else 2
    header = laspy.LasHeader(point_format=point_format_id, version="1.2")

    # Set scale and offset for coordinate precision
    offset = points.min(axis=0)
    header.offsets = offset
    header.scales = np.array([0.001, 0.001, 0.001])  # mm precision

    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Colors (LAS stores as 16-bit unsigned integers)
    if pc.colors is not None and len(pc.colors) > 0:
        colors = pc.colors
        if colors.max() <= 1.0:
            las.red = (colors[:, 0] * 65535).astype(np.uint16)
            las.green = (colors[:, 1] * 65535).astype(np.uint16)
            las.blue = (colors[:, 2] * 65535).astype(np.uint16)
        else:
            las.red = colors[:, 0].astype(np.uint16)
            las.green = colors[:, 1].astype(np.uint16)
            las.blue = colors[:, 2].astype(np.uint16)

    # Standard LAS attributes
    if has_intensity:
        intensity = pc.attributes['intensity']
        las.intensity = np.clip(intensity, 0, 65535).astype(np.uint16)

    if has_classification:
        classification = pc.attributes['classification']
        las.classification = np.clip(classification, 0, 255).astype(np.uint8)

    if has_return_number:
        ret_num = pc.attributes['return_number']
        las.return_number = np.clip(ret_num, 0, 15).astype(np.uint8)

    if has_gps_time:
        las.gps_time = pc.attributes['gps_time'].astype(np.float64)

    las.write(file_path)
