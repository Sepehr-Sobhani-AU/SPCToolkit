"""
Import NPY/NPZ Point Cloud Plugin

Imports NumPy .npy and .npz files as point clouds.
Supports multi-file selection.

.npy files: Interprets columns based on shape (N, C):
  C=3: XYZ, C=4: XYZ+intensity, C=6: XYZ+RGB,
  C=7: XYZ+RGB+intensity, C=9: XYZ+RGB+normals,
  C=10: XYZ+RGB+normals+intensity

.npz files: Looks for named arrays (points/xyz/pos, colors/rgb,
  normals/normal, intensity). Falls back to single-array interpretation.
"""

from typing import Dict, Any, Optional, Tuple
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

# Key aliases for .npz files
_POINTS_KEYS = ("points", "xyz", "pos", "features")
_COLORS_KEYS = ("colors", "rgb")
_NORMALS_KEYS = ("normals", "normal")
_INTENSITY_KEYS = ("intensity",)
_LABELS_KEYS = ("labels", "label", "classification")
_ALL_KNOWN_KEYS = _POINTS_KEYS + _COLORS_KEYS + _NORMALS_KEYS + _INTENSITY_KEYS + _LABELS_KEYS


class ImportNPYPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "import_npy_npz"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        settings = QSettings("SPCToolkit", "ImportNPY")
        last_dir = settings.value("last_directory", "")

        file_paths, _ = QFileDialog.getOpenFileNames(
            main_window,
            "Import NPY/NPZ Files",
            last_dir,
            "NumPy Files (*.npy *.npz);;All Files (*)"
        )

        if not file_paths:
            return

        settings.setValue("last_directory", os.path.dirname(file_paths[0]))

        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        total_files = len(file_paths)
        success_count = 0
        failed_files = []

        main_window.disable_menus()
        main_window.disable_tree()
        try:
            for file_idx, file_path in enumerate(file_paths):
                filename = os.path.basename(file_path)
                percent = int(((file_idx + 1) / total_files) * 100)
                main_window.show_progress(
                    f"Importing {filename} ({file_idx + 1}/{total_files})...", percent
                )

                try:
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == ".npz":
                        points_xyz, colors, normals, attributes = self._load_npz(file_path)
                    else:
                        points_xyz, colors, normals, attributes = self._load_npy(file_path)

                    # Translate to origin for float32 precision (GPU-accelerated)
                    min_bound = points_xyz.min(axis=0)
                    points_translated, colors = translate_and_convert(
                        points_xyz, min_bound, colors
                    )

                    # Create PointCloud
                    point_cloud = PointCloud(
                        points_translated, colors=colors, normals=normals
                    )
                    point_cloud.name = filename
                    point_cloud.translation = min_bound

                    # Add extra attributes
                    for attr_name, attr_values in attributes.items():
                        point_cloud.add_attribute(attr_name, attr_values)

                    # Add to data manager
                    uid = self._add_point_cloud(
                        point_cloud, controller, data_nodes, tree_widget
                    )
                    success_count += 1
                    logger.info(
                        f"Imported {filename}: {len(points_translated)} points, UID={uid}"
                    )

                except Exception as e:
                    logger.error(f"Failed to import {file_path}: {e}")
                    failed_files.append((filename, str(e)))
        finally:
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()

        # Summary
        if failed_files:
            msg = f"Imported {success_count} of {total_files} files.\n\nFailed:\n"
            for name, err in failed_files[:5]:
                msg += f"- {name}: {err}\n"
            QMessageBox.warning(main_window, "Import Complete", msg)
        elif success_count > 0:
            QMessageBox.information(
                main_window, "Import Complete",
                f"Successfully imported {success_count} file(s)."
            )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_npy(self, file_path: str):
        """Load a .npy file and interpret columns by shape."""
        data = np.load(file_path, allow_pickle=False)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(
                f"Expected 2D array with >= 3 columns, got shape {data.shape}"
            )
        return self._parse_columns(data)

    def _load_npz(self, file_path: str):
        """Load a .npz file, looking for named arrays or falling back to single array."""
        archive = np.load(file_path, allow_pickle=False)
        keys = list(archive.keys())

        # Try to find points by known key names
        points_key = self._find_key(archive, _POINTS_KEYS)
        points_arr = archive[points_key] if points_key else None

        if points_arr is not None:
            if points_arr.ndim != 2 or points_arr.shape[1] < 3:
                raise ValueError(
                    f"Points array must be (N, >=3), got shape {points_arr.shape}"
                )

            points_xyz = points_arr[:, :3].astype(np.float64)
            n_points = len(points_xyz)
            normals = None
            attributes = {}

            # "features" arrays may have extra columns: XYZ + normals + eigenvalues
            if points_arr.shape[1] > 3:
                extra = points_arr[:, 3:]
                if extra.shape[1] >= 3:
                    normals = extra[:, :3].astype(np.float32)
                if extra.shape[1] > 3:
                    attributes["eigenvalues"] = extra[:, 3:].astype(np.float32)

            colors = self._find_array(archive, _COLORS_KEYS)
            if colors is not None:
                colors = self._normalize_colors(colors.astype(np.float32))

            if normals is None:
                found_normals = self._find_array(archive, _NORMALS_KEYS)
                if found_normals is not None:
                    normals = found_normals.astype(np.float32)

            intensity_arr = self._find_array(archive, _INTENSITY_KEYS)
            if intensity_arr is not None:
                attributes["intensity"] = intensity_arr.astype(np.float32)

            labels_arr = self._find_array(archive, _LABELS_KEYS)
            if labels_arr is not None:
                attributes["labels"] = labels_arr

            # Collect remaining arrays with matching row count
            for key in keys:
                if key in _ALL_KNOWN_KEYS or key in attributes:
                    continue
                arr = archive[key]
                if hasattr(arr, "__len__") and len(arr) == n_points:
                    attributes[key] = arr

            return points_xyz, colors, normals, attributes

        # Fallback: single array in the archive
        if len(keys) == 1:
            data = archive[keys[0]]
            if data.ndim == 2 and data.shape[1] >= 3:
                return self._parse_columns(data)

        raise ValueError(
            f"NPZ file has no recognized point cloud keys. Found: {keys}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_columns(self, data: np.ndarray):
        """Interpret a 2D array's columns as XYZ + optional fields."""
        C = data.shape[1]
        points_xyz = data[:, :3].astype(np.float64)
        colors = None
        normals = None
        attributes = {}

        if C == 4:
            # XYZ + intensity
            attributes["intensity"] = data[:, 3].astype(np.float32)
        elif C == 6:
            # XYZ + RGB
            colors = self._normalize_colors(data[:, 3:6].astype(np.float32))
        elif C == 7:
            # XYZ + RGB + intensity
            colors = self._normalize_colors(data[:, 3:6].astype(np.float32))
            attributes["intensity"] = data[:, 6].astype(np.float32)
        elif C == 9:
            # XYZ + RGB + normals
            colors = self._normalize_colors(data[:, 3:6].astype(np.float32))
            normals = data[:, 6:9].astype(np.float32)
        elif C >= 10:
            # XYZ + RGB + normals + intensity
            colors = self._normalize_colors(data[:, 3:6].astype(np.float32))
            normals = data[:, 6:9].astype(np.float32)
            attributes["intensity"] = data[:, 9].astype(np.float32)

        return points_xyz, colors, normals, attributes

    @staticmethod
    def _normalize_colors(colors: np.ndarray) -> np.ndarray:
        """Normalize colors to [0, 1]. Detects 0-255 vs 0-1 range."""
        max_val = colors.max()
        if max_val > 1.0:
            colors = colors / 255.0
        return colors

    @staticmethod
    def _find_key(archive, key_names: Tuple[str, ...]) -> Optional[str]:
        """Return the first matching key name from an npz archive."""
        for key in key_names:
            if key in archive:
                return key
        return None

    @staticmethod
    def _find_array(archive, key_names: Tuple[str, ...]) -> Optional[np.ndarray]:
        """Return the first matching array from an npz archive."""
        for key in key_names:
            if key in archive:
                return archive[key]
        return None

    @staticmethod
    def _add_point_cloud(point_cloud, controller, data_nodes, tree_widget):
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
