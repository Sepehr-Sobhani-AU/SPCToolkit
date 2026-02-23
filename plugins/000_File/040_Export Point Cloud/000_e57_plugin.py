"""
Export E57 Plugin

Exports the selected branches as an E57 file.
Multiple selected branches are merged before export.
Preserves RGB colors and intensity if available.
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
        total_branches = len(controller.selected_branches)
        point_clouds = []
        for i, uid in enumerate(controller.selected_branches):
            percent = int(((i + 1) / total_branches) * 70)
            main_window.show_progress(
                f"Reconstructing branch {i + 1}/{total_branches}...", percent
            )
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
        detected_shift = detected_translation

        dialog = ShiftDialog(main_window, detected_shift)
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

        main_window.disable_menus()
        main_window.disable_tree()
        try:
            main_window.show_progress("Writing E57 file...", 80)
            _write_e57(merged, file_path, shift)
            main_window.show_progress("Export complete", 100)
            logger.info(f"Exported {merged.size} points to {file_path}")
        except Exception as e:
            QMessageBox.critical(
                main_window, "Export Error",
                f"Failed to export:\n{str(e)}"
            )
        finally:
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()


def _write_e57(pc: PointCloud, file_path: str, shift: np.ndarray = None) -> None:
    """
    Write a PointCloud to an E57 file as a single scan.

    Uses float64 for coordinates to preserve precision on large values.
    """
    import pye57

    n = pc.size

    # Apply coordinate shift in float64 (GPU-accelerated)
    points = apply_shift(pc.points, shift)

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
