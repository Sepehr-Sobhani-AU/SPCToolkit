"""
Import LAS/LAZ Point Cloud Plugin

Imports LAS and LAZ files using the laspy library.
Supports multi-file selection. Extracts XYZ, RGB colors, intensity,
classification, and other standard LAS attributes.
"""

from typing import Dict, Any
import os
import logging
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QSettings, Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode
from services.coordinate_service import translate_and_convert

logger = logging.getLogger(__name__)


class ImportLASPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "import_las"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        try:
            import laspy
        except ImportError:
            QMessageBox.critical(
                main_window, "Missing Library",
                "The 'laspy' library is required to import LAS/LAZ files.\n"
                "Install it with: pip install laspy[lazrs]"
            )
            return

        settings = QSettings("SPCToolkit", "ImportLAS")
        last_dir = settings.value("last_directory", "")

        file_paths, _ = QFileDialog.getOpenFileNames(
            main_window,
            "Import LAS/LAZ Files",
            last_dir,
            "LAS/LAZ Files (*.las *.laz);;All Files (*)"
        )

        if not file_paths:
            return

        settings.setValue("last_directory", os.path.dirname(file_paths[0]))

        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        success_count = 0
        failed_files = []

        for file_path in file_paths:
            try:
                las = laspy.read(file_path)
                points_xyz = np.column_stack([
                    np.array(las.x, dtype=np.float64),
                    np.array(las.y, dtype=np.float64),
                    np.array(las.z, dtype=np.float64),
                ])

                # Extract colors if available (LAS stores as 16-bit)
                colors = None
                if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                    r = np.array(las.red, dtype=np.float32)
                    g = np.array(las.green, dtype=np.float32)
                    b = np.array(las.blue, dtype=np.float32)
                    max_val = max(r.max(), g.max(), b.max(), 1.0)
                    if max_val > 255:
                        # 16-bit color values
                        colors = np.column_stack([r, g, b]) / 65535.0
                    else:
                        colors = np.column_stack([r, g, b]) / max(max_val, 1.0)

                # Fallback: use intensity as grayscale color
                if colors is None and hasattr(las, 'intensity'):
                    intensity = np.array(las.intensity, dtype=np.float32)
                    i_min, i_max = intensity.min(), intensity.max()
                    intensity_norm = (intensity - i_min) / (i_max - i_min + 1e-6)
                    colors = np.column_stack([intensity_norm] * 3)

                # Translate to origin for float32 precision (GPU-accelerated)
                min_bound = points_xyz.min(axis=0)
                points_translated, colors = translate_and_convert(
                    points_xyz, min_bound, colors
                )

                # Create PointCloud
                filename = os.path.basename(file_path)
                point_cloud = PointCloud(points_translated, colors=colors)
                point_cloud.name = filename
                point_cloud.translation = min_bound

                # Add standard LAS attributes
                if hasattr(las, 'intensity'):
                    point_cloud.add_attribute('intensity',
                                              np.array(las.intensity, dtype=np.float32))
                if hasattr(las, 'classification'):
                    point_cloud.add_attribute('classification',
                                              np.array(las.classification, dtype=np.int32))
                if hasattr(las, 'return_number'):
                    point_cloud.add_attribute('return_number',
                                              np.array(las.return_number, dtype=np.int32))
                if hasattr(las, 'number_of_returns'):
                    point_cloud.add_attribute('number_of_returns',
                                              np.array(las.number_of_returns, dtype=np.int32))
                if hasattr(las, 'gps_time'):
                    point_cloud.add_attribute('gps_time',
                                              np.array(las.gps_time, dtype=np.float64))

                # Add to data manager
                uid = self._add_point_cloud(point_cloud, controller, data_nodes, tree_widget)
                success_count += 1
                logger.info(f"Imported {filename}: {len(points_translated)} points, UID={uid}")

            except Exception as e:
                logger.error(f"Failed to import {file_path}: {e}")
                failed_files.append((os.path.basename(file_path), str(e)))

        # Summary
        if failed_files:
            msg = f"Imported {success_count} of {len(file_paths)} files.\n\nFailed:\n"
            for name, err in failed_files[:5]:
                msg += f"- {name}: {err}\n"
            QMessageBox.warning(main_window, "Import Complete", msg)
        elif success_count > 0:
            QMessageBox.information(
                main_window, "Import Complete",
                f"Successfully imported {success_count} file(s)."
            )

    def _add_point_cloud(self, point_cloud, controller, data_nodes, tree_widget):
        """Add a PointCloud as a root DataNode."""
        data_node = DataNode(
            params=point_cloud.name,
            data=point_cloud,
            data_type="point_cloud",
            parent_uid=None,
            depends_on=None,
            tags=[]
        )
        data_node.memory_size = controller._calculate_point_cloud_memory(point_cloud)
        uid = data_nodes.add_node(data_node)
        tree_widget.add_branch(str(uid), "", point_cloud.name, is_root=True)
        tree_widget.update_cache_tooltip(str(uid), data_node.memory_size)
        return uid
