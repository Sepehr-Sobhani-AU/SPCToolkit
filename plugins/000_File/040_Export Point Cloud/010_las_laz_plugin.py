"""
Export LAS/LAZ Plugin

Exports the selected branches as a LAS or LAZ file.
Multiple selected branches are merged before export.
Preserves per-point attributes (intensity, classification, etc.).
"""

from typing import Dict, Any
import os
import logging
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from services.coordinate_service import apply_shift, find_root_translation
from plugins.dialogs.shift_dialog import ShiftDialog

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
        detected_translation = find_root_translation(
            data_nodes, controller.selected_branches[0]
        )
        detected_shift = -detected_translation

        dialog = ShiftDialog(main_window, detected_shift)
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


def _write_las(pc: PointCloud, file_path: str, shift: np.ndarray = None) -> None:
    """
    Write a PointCloud to a LAS/LAZ file.

    Uses float64 for coordinates to preserve precision on large values.
    Automatically uses LAZ compression if the file extension is .laz.
    """
    import laspy

    n = pc.size

    # Apply coordinate shift in float64 (GPU-accelerated)
    points = apply_shift(pc.points, shift)

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
