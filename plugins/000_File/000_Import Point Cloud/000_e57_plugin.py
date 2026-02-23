"""
Import E57 Point Cloud Plugin

Imports E57 files using the pye57 library.
Supports multi-scan E57 files — each scan is imported as a separate branch.
Extracts XYZ, RGB colors, and intensity.
"""

from typing import Dict, Any
import os
import logging
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QSettings

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode
from services.coordinate_service import translate_and_convert

logger = logging.getLogger(__name__)


class ImportE57Plugin(ActionPlugin):

    def get_name(self) -> str:
        return "import_e57"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        try:
            import pye57
        except ImportError:
            QMessageBox.critical(
                main_window, "Missing Library",
                "The 'pye57' library is required to import E57 files.\n"
                "Install it with: pip install pye57"
            )
            return

        settings = QSettings("SPCToolkit", "ImportE57")
        last_dir = settings.value("last_directory", "")

        file_paths, _ = QFileDialog.getOpenFileNames(
            main_window,
            "Import E57 Files",
            last_dir,
            "E57 Files (*.e57);;All Files (*)"
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
                e57_file = pye57.E57(file_path)
                scan_count = e57_file.scan_count
                filename = os.path.basename(file_path)

                for scan_idx in range(scan_count):
                    try:
                        header = e57_file.get_header(scan_idx)
                        data = e57_file.read_scan(scan_idx, ignore_missing_fields=True)

                        # Extract XYZ coordinates
                        x = np.asarray(data["cartesianX"], dtype=np.float64)
                        y = np.asarray(data["cartesianY"], dtype=np.float64)
                        z = np.asarray(data["cartesianZ"], dtype=np.float64)
                        points_xyz = np.column_stack([x, y, z])

                        # Remove invalid points (NaN or Inf)
                        valid = np.isfinite(points_xyz).all(axis=1)
                        if not valid.all():
                            points_xyz = points_xyz[valid]
                            logger.info(f"Removed {(~valid).sum()} invalid points from scan {scan_idx}")

                        if len(points_xyz) == 0:
                            logger.warning(f"Scan {scan_idx} has no valid points, skipping")
                            continue

                        # Extract colors if available
                        colors = None
                        if "colorRed" in data and "colorGreen" in data and "colorBlue" in data:
                            r = np.asarray(data["colorRed"], dtype=np.float32)
                            g = np.asarray(data["colorGreen"], dtype=np.float32)
                            b = np.asarray(data["colorBlue"], dtype=np.float32)
                            if valid is not None and not valid.all():
                                r, g, b = r[valid], g[valid], b[valid]
                            max_val = max(r.max(), g.max(), b.max(), 1.0)
                            if max_val > 255:
                                colors = np.column_stack([r, g, b]) / 65535.0
                            elif max_val > 1.0:
                                colors = np.column_stack([r, g, b]) / 255.0
                            else:
                                colors = np.column_stack([r, g, b])

                        # Fallback: intensity as grayscale
                        intensity_arr = None
                        if "intensity" in data:
                            intensity_arr = np.asarray(data["intensity"], dtype=np.float32)
                            if valid is not None and not valid.all():
                                intensity_arr = intensity_arr[valid]
                            if colors is None:
                                i_min, i_max = intensity_arr.min(), intensity_arr.max()
                                intensity_norm = (intensity_arr - i_min) / (i_max - i_min + 1e-6)
                                colors = np.column_stack([intensity_norm] * 3)

                        # Translate to origin for float32 precision (GPU-accelerated)
                        min_bound = points_xyz.min(axis=0)
                        points_translated, colors = translate_and_convert(
                            points_xyz, min_bound, colors
                        )

                        # Create PointCloud
                        scan_name = f"{filename}" if scan_count == 1 else f"{filename} [scan {scan_idx}]"
                        point_cloud = PointCloud(points_translated, colors=colors)
                        point_cloud.name = scan_name
                        point_cloud.translation = min_bound

                        if intensity_arr is not None:
                            point_cloud.add_attribute('intensity', intensity_arr)

                        # Add to data manager
                        uid = self._add_point_cloud(
                            point_cloud, controller, data_nodes, tree_widget
                        )
                        success_count += 1
                        logger.info(
                            f"Imported {scan_name}: {len(points_translated)} points, UID={uid}"
                        )

                    except Exception as e:
                        logger.error(f"Failed scan {scan_idx} in {filename}: {e}")
                        failed_files.append((f"{filename} scan {scan_idx}", str(e)))

            except Exception as e:
                logger.error(f"Failed to open {file_path}: {e}")
                failed_files.append((os.path.basename(file_path), str(e)))

        # Summary
        if failed_files:
            msg = f"Imported {success_count} scan(s).\n\nFailed:\n"
            for name, err in failed_files[:5]:
                msg += f"- {name}: {err}\n"
            QMessageBox.warning(main_window, "Import Complete", msg)
        elif success_count > 0:
            QMessageBox.information(
                main_window, "Import Complete",
                f"Successfully imported {success_count} scan(s)."
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
