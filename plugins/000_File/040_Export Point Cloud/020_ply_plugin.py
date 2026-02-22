"""
Export Point Cloud Plugin

Exports the selected branches as a PLY file including all per-point
attributes (cluster labels, classification, eigenvalues, etc.).
Multiple selected branches are merged before export.
"""

from typing import Dict, Any, List, Tuple

import numpy as np
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QDialog, QVBoxLayout,
                               QLabel, QLineEdit, QDialogButtonBox,
                               QGroupBox, QGridLayout)

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud


class ExportPointCloudPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "export_ply"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes

        if not controller.selected_branches:
            QMessageBox.warning(
                main_window,
                "No Selection",
                "Please select one or more branches to export."
            )
            return

        # Reconstruct each selected branch into a PointCloud
        point_clouds = []
        for uid in controller.selected_branches:
            try:
                pc = controller.reconstruct(uid)
                if pc is not None:
                    point_clouds.append(pc)
            except Exception as e:
                QMessageBox.critical(
                    main_window,
                    "Reconstruction Error",
                    f"Failed to reconstruct branch {uid}:\n{str(e)}"
                )
                return

        if not point_clouds:
            QMessageBox.warning(
                main_window,
                "No Data",
                "Selected branches produced no point cloud data."
            )
            return

        # Merge if multiple branches selected
        if len(point_clouds) == 1:
            merged = point_clouds[0]
        else:
            merged = PointCloud.merge(point_clouds, name="Export")

        # Recover the original coordinate translation from the root PointCloud.
        # On import, points are shifted to the origin for float32 precision;
        # the offset is stored as root_pc.translation = -min_bound.
        detected_translation = _find_root_translation(
            data_nodes, controller.selected_branches[0]
        )

        # Show shift dialog so user can verify / adjust
        # The shift to restore = -translation (i.e. add min_bound back)
        detected_shift = -detected_translation
        dialog = _ShiftDialog(main_window, detected_shift)
        if dialog.exec_() != QDialog.Accepted:
            return
        shift = dialog.get_shift()

        # Open file save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            main_window,
            "Export Point Cloud",
            "",
            "PLY Files (*.ply)"
        )

        if not file_path:
            return

        if not file_path.lower().endswith('.ply'):
            file_path += '.ply'

        try:
            _write_ply(merged, file_path, shift)
        except Exception as e:
            QMessageBox.critical(
                main_window,
                "Export Error",
                f"Failed to export point cloud:\n{str(e)}"
            )


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


def _find_root_translation(data_nodes, uid_str: str) -> np.ndarray:
    """
    Walk up from a node to the root PointCloud and return its translation.

    On import, points are shifted to the origin (points -= min_bound) and
    the offset is stored as translation = -min_bound on the root PointCloud.
    To restore original coordinates: original = points - translation.
    """
    import uuid as _uuid
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


def _collect_scalar_attributes(pc: PointCloud) -> List[Tuple[str, np.ndarray, str]]:
    """
    Collect per-point attributes suitable for PLY export.

    Returns list of (name, array, ply_type) tuples. Multi-column arrays
    are split into separate scalar properties (e.g. eigenvalues_0, eigenvalues_1).
    Dict attributes (like _cluster_names) are skipped.
    """
    props = []

    for attr_name, attr_value in pc.attributes.items():
        # Skip non-array attributes (dicts like _cluster_names, _cluster_colors)
        if not isinstance(attr_value, np.ndarray):
            continue

        # Skip if length doesn't match point count
        if attr_value.shape[0] != pc.size:
            continue

        if attr_value.ndim == 1:
            ply_type = _numpy_to_ply_type(attr_value.dtype)
            props.append((attr_name, attr_value, ply_type))
        elif attr_value.ndim == 2:
            ply_type = _numpy_to_ply_type(attr_value.dtype)
            for col in range(attr_value.shape[1]):
                props.append(
                    (f"{attr_name}_{col}", attr_value[:, col], ply_type)
                )

    # If cluster_names mapping exists and no classification attribute was
    # already collected, synthesize a classification field from the mapping
    existing_names = {name for name, _, _ in props}
    cluster_names = pc.attributes.get("_cluster_names")
    cluster_labels = pc.attributes.get("cluster_labels")
    if (isinstance(cluster_names, dict) and isinstance(cluster_labels, np.ndarray)
            and "classification" not in existing_names):
        unique_names = sorted(set(cluster_names.values()))
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        classification = np.full(pc.size, -1, dtype=np.int32)
        for label_id, name in cluster_names.items():
            mask = cluster_labels == label_id
            classification[mask] = name_to_id[name]
        props.append(("classification", classification, "int"))

    return props


def _numpy_to_ply_type(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.floating):
        return "float"
    if np.issubdtype(dtype, np.signedinteger):
        return "int"
    if np.issubdtype(dtype, np.unsignedinteger):
        return "uint"
    return "float"


def _write_ply(pc: PointCloud, file_path: str,
               shift: np.ndarray = None) -> None:
    """
    Write a PointCloud to a binary little-endian PLY file with all attributes.

    Includes: XYZ, RGB (as uchar), normals (if present), and all per-point
    scalar attributes from the attributes dict. Uses numpy structured arrays
    for fast vectorized writing.

    Args:
        shift: Additive offset applied to points (exported = points + shift).
               Uses float64 to avoid precision loss on large coordinates.
    """
    n = pc.size

    has_colors = pc.colors is not None and len(pc.colors) > 0
    has_normals = pc.normals is not None and len(pc.normals) > 0
    scalar_attrs = _collect_scalar_attributes(pc)

    # Use float64 for XYZ when a non-zero shift is applied (e.g. UTM)
    # to avoid losing precision on large values
    apply_shift = shift is not None and np.any(shift != 0)
    xyz_dtype = '<f8' if apply_shift else '<f4'
    xyz_ply_type = "double" if apply_shift else "float"

    # Build numpy structured dtype and PLY header simultaneously
    dtype_fields = [('x', xyz_dtype), ('y', xyz_dtype), ('z', xyz_dtype)]
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
    ]

    # Add classification name mapping as comments
    cluster_names = pc.attributes.get("_cluster_names") if hasattr(pc, 'attributes') else None
    if isinstance(cluster_names, dict) and cluster_names:
        unique_names = sorted(set(cluster_names.values()))
        for idx, name in enumerate(unique_names):
            header_lines.append(f"comment classification {idx} {name}")

    header_lines.append(f"element vertex {n}")
    header_lines.append(f"property {xyz_ply_type} x")
    header_lines.append(f"property {xyz_ply_type} y")
    header_lines.append(f"property {xyz_ply_type} z")

    if has_colors:
        dtype_fields.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        header_lines.append("property uchar red")
        header_lines.append("property uchar green")
        header_lines.append("property uchar blue")

    if has_normals:
        dtype_fields.extend([('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')])
        header_lines.append("property float nx")
        header_lines.append("property float ny")
        header_lines.append("property float nz")

    _PLY_TO_NUMPY_DTYPE = {"float": "<f4", "int": "<i4", "uint": "<u4"}
    for prop_name, _, ply_type in scalar_attrs:
        dtype_fields.append((prop_name, _PLY_TO_NUMPY_DTYPE[ply_type]))
        header_lines.append(f"property {ply_type} {prop_name}")

    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    # Build structured array and fill columns
    vertex_dtype = np.dtype(dtype_fields)
    vertices = np.empty(n, dtype=vertex_dtype)

    # Apply coordinate shift (exported = points + shift) in float64
    if apply_shift:
        points = pc.points.astype(np.float64) + shift.astype(np.float64)
    else:
        points = pc.points
    vertices['x'] = points[:, 0]
    vertices['y'] = points[:, 1]
    vertices['z'] = points[:, 2]

    if has_colors:
        colors = pc.colors
        if colors.max() <= 1.0:
            colors_u8 = (colors * 255).astype(np.uint8)
        else:
            colors_u8 = colors.astype(np.uint8)
        vertices['red'] = colors_u8[:, 0]
        vertices['green'] = colors_u8[:, 1]
        vertices['blue'] = colors_u8[:, 2]

    if has_normals:
        normals = pc.normals.astype(np.float32)
        vertices['nx'] = normals[:, 0]
        vertices['ny'] = normals[:, 1]
        vertices['nz'] = normals[:, 2]

    for prop_name, arr, ply_type in scalar_attrs:
        np_dtype = _PLY_TO_NUMPY_DTYPE[ply_type]
        vertices[prop_name] = arr.astype(np_dtype)

    # Write header + binary data
    with open(file_path, 'wb') as f:
        f.write(header.encode('ascii'))
        vertices.tofile(f)
