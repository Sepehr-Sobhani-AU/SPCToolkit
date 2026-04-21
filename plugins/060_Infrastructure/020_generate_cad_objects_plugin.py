"""
Generate CAD Objects Plugin

Creates CAD object branches for classified clusters. For each class (e.g. Pole,
Tree, Kerb Edge) a container is created, and under it one CADObject node per
cluster instance, fitted to the cluster's oriented bounding box.

Geometry types:
- "mesh"     — bounding-box wireframe fitted to the cluster OBB.
- "polyline" — ordered 3D polyline fitted along the cluster's principal axis.

The user chooses the geometry type per class in the dialog.
"""

import uuid
import logging
import traceback
from typing import Dict, Any, List, Tuple

import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QListWidget, QListWidgetItem, QComboBox,
)
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.cad_object import CADObject
from core.entities.data_node import DataNode

logger = logging.getLogger(__name__)

# ── Unit geometries ──────────────────────────────────────────────────────

# Unit cube: base at Z=0, top at Z=1, centred on XY at (0.5, 0.5).
_BOX_VERTICES = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
], dtype=np.float32)

_BOX_FACES = [
    [0, 1, 2, 3], [4, 5, 6, 7],  # bottom, top
    [0, 1, 5, 4], [2, 3, 7, 6],  # front, back
    [1, 2, 6, 5], [0, 3, 7, 4],  # right, left
]

_BOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
    [4, 5], [5, 6], [6, 7], [7, 4],  # top
    [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
], dtype=np.int32)


# ── Default colour palette for classes ───────────────────────────────────

_CLASS_COLORS = [
    [0.00, 1.00, 1.00],  # cyan
    [1.00, 0.65, 0.00],  # orange
    [0.00, 1.00, 0.00],  # green
    [1.00, 0.00, 1.00],  # magenta
    [1.00, 1.00, 0.00],  # yellow
    [0.40, 0.80, 1.00],  # light blue
    [1.00, 0.40, 0.40],  # salmon
    [0.60, 1.00, 0.60],  # light green
]


# ── Dialog ───────────────────────────────────────────────────────────────

class CADClassDialog(QDialog):
    """Let the user choose geometry type per class."""

    def __init__(self, parent, class_info: List[Tuple[str, int]]):
        """
        Args:
            parent: Parent widget.
            class_info: List of (class_name, cluster_count) tuples.
        """
        super().__init__(parent)
        self.setWindowTitle("Generate CAD Objects")
        self.setModal(True)
        self.setMinimumWidth(450)
        self.class_info = class_info
        self._combos: Dict[str, QComboBox] = {}
        self._checks: Dict[str, QListWidgetItem] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Select classes and geometry type:"))

        self.list_widget = QListWidget()
        self.list_widget.setMinimumHeight(250)

        for class_name, count in self.class_info:
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
            self._checks[class_name] = item

            # Custom widget: label + combo in a row
            row = QHBoxLayout()
            row.setContentsMargins(4, 2, 4, 2)
            row.addWidget(QLabel(f"{class_name}  ({count} clusters)"), stretch=1)

            combo = QComboBox()
            combo.addItems(["mesh", "polyline"])
            combo.setFixedWidth(100)
            row.addWidget(combo)
            self._combos[class_name] = combo

            container = QLabel()  # dummy to host the layout
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(4, 2, 4, 2)
            container_layout.addWidget(QLabel(f"{class_name}  ({count} clusters)"), stretch=1)
            container_layout.addWidget(combo)

            item.setSizeHint(container.sizeHint())
            self.list_widget.setItemWidget(item, container)

        layout.addWidget(self.list_widget)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_layout.addWidget(ok)
        btn_layout.addWidget(cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def get_selections(self) -> List[Tuple[str, str]]:
        """Return list of (class_name, geometry_type) for checked classes."""
        result = []
        for class_name, item in self._checks.items():
            if item.checkState() == Qt.Checked:
                geom = self._combos[class_name].currentText()
                result.append((class_name, geom))
        return result


# ── OBB / geometry helpers ───────────────────────────────────────────────

def _compute_obb(points: np.ndarray):
    """
    Compute the oriented bounding box for a set of points.

    Returns:
        (center, R, extent, height_axis_index) where R is 3x3 rotation,
        extent is (3,) with dimensions along each axis of R, and
        height_axis_index identifies which axis is most vertical.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    obb = pcd.get_oriented_bounding_box()

    # Identify the height axis (most aligned with world Z)
    projections = np.abs(obb.R.T @ np.array([0, 0, 1]))
    height_idx = int(np.argmax(projections))

    return np.asarray(obb.center), np.asarray(obb.R), np.asarray(obb.extent), height_idx


def _build_box_transform(center, R, extent, height_idx):
    """
    Build a 4x4 transform that maps the unit box onto the OBB.

    The unit box spans [0,1]^3 with base at Z=0.  We reorder axes so
    the OBB height axis maps to the box Z axis, then scale/rotate/translate.
    """
    # Reorder so height is last (maps to box Z)
    order = [i for i in range(3) if i != height_idx] + [height_idx]
    dims = extent[order]                 # (width, length, height)
    axes = R[:, order]                   # columns = reordered OBB axes

    # Scale: stretch unit box to cluster dimensions
    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = dims

    # Offset: unit box is [0,1] so shift by -0.5 to centre on origin
    C = np.eye(4)
    C[0, 3] = -0.5
    C[1, 3] = -0.5
    C[2, 3] = -0.5

    # Rotation: OBB axes → world
    Rot = np.eye(4)
    Rot[:3, :3] = axes

    # Translation: OBB centre
    T = np.eye(4)
    T[:3, 3] = center

    return T @ Rot @ S @ C, dims


def _build_polyline_geometry(points: np.ndarray):
    """
    Fit an ordered polyline through a point cluster along its principal axis.

    Returns (vertices_Nx3, closed=False) in the cluster's world coordinates.
    The transform_matrix for polylines is identity (vertices are absolute).
    """
    if len(points) < 2:
        return points.copy(), False

    # PCA: project onto first principal component, sort by projection
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Use SVD for principal direction (robust even for 2 points)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    principal = Vt[0]  # first principal direction

    projections = centered @ principal
    order = np.argsort(projections)
    sorted_pts = points[order]

    # Subsample to reasonable vertex count (max ~200 for rendering)
    n = len(sorted_pts)
    if n > 200:
        indices = np.linspace(0, n - 1, 200, dtype=int)
        sorted_pts = sorted_pts[indices]

    return sorted_pts.astype(np.float32), False


# ── Plugin ───────────────────────────────────────────────────────────────

class GenerateCADObjectsPlugin(ActionPlugin):
    """
    Generate CAD object branches for classified clusters.

    Creates one container per class, with individual CADObject children
    for each cluster instance.
    """

    def get_name(self) -> str:
        return "generate_cad_objects"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # ── Validate selection ───────────────────────────────────────
        selected = controller.selected_branches
        if not selected:
            QMessageBox.warning(main_window, "No Selection",
                                "Select a branch with classified clusters.")
            return
        if len(selected) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch.")
            return

        selected_uid = selected[0]

        # ── Reconstruct and extract metadata ─────────────────────────
        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error", str(e))
            return

        cluster_labels = point_cloud.get_attribute("cluster_labels")
        cluster_names = point_cloud.get_attribute("_cluster_names")

        if cluster_labels is None or cluster_names is None or len(cluster_names) == 0:
            QMessageBox.warning(main_window, "No Classifications",
                                "Selected branch has no classified clusters.\n"
                                "Run clustering + classification first.")
            return

        # Group cluster IDs by class name
        class_to_cids: Dict[str, List[int]] = {}
        for cid, cname in cluster_names.items():
            class_to_cids.setdefault(cname, []).append(cid)

        class_info = sorted(
            [(name, len(cids)) for name, cids in class_to_cids.items()],
            key=lambda x: x[0],
        )

        # ── Show dialog ──────────────────────────────────────────────
        dialog = CADClassDialog(main_window, class_info)
        if dialog.exec_() != QDialog.Accepted:
            return

        selections = dialog.get_selections()
        if not selections:
            QMessageBox.warning(main_window, "Nothing Selected",
                                "No classes were selected.")
            return

        # ── Generate CAD objects ──────────────────────────────────────
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.tree_overlay.show_processing("Generating CAD objects...")

        parent_uuid = uuid.UUID(selected_uid)
        points = point_cloud.points
        created_count = 0

        try:
            tree_widget.blockSignals(True)

            for class_idx, (class_name, geom_type) in enumerate(selections):
                cids = class_to_cids[class_name]
                color = np.array(
                    _CLASS_COLORS[class_idx % len(_CLASS_COLORS)],
                    dtype=np.float32,
                )

                # Create class container
                container_node = DataNode(
                    params=f"{class_name} CAD",
                    data=None,
                    data_type="container",
                    parent_uid=parent_uuid,
                    depends_on=[parent_uuid],
                    tags=["cad", class_name],
                )
                container_uid = data_nodes.add_node(container_node)
                tree_widget.add_branch(
                    str(container_uid), selected_uid, f"{class_name} CAD"
                )
                item = tree_widget.branches_dict.get(str(container_uid))
                if item:
                    item.setCheckState(0, Qt.Unchecked)
                    tree_widget.visibility_status[str(container_uid)] = False

                # Process each cluster in this class
                for seq, cid in enumerate(cids):
                    mask = cluster_labels == cid
                    cluster_pts = points[mask]
                    if len(cluster_pts) < 3:
                        continue

                    try:
                        cad_obj = _make_cad_object(
                            cluster_pts, class_name, geom_type,
                            color, parent_uuid,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Skipping {class_name}_{seq}: {exc}"
                        )
                        continue

                    node_name = f"{class_name}_{seq:03d}"
                    cad_node = DataNode(
                        params=node_name,
                        data=cad_obj,
                        data_type="cad_object",
                        parent_uid=container_uid,
                        depends_on=[parent_uuid],
                        tags=["cad", class_name],
                    )
                    cad_uid = data_nodes.add_node(cad_node)
                    tree_widget.add_branch(
                        str(cad_uid), str(container_uid), node_name
                    )
                    cad_item = tree_widget.branches_dict.get(str(cad_uid))
                    if cad_item:
                        cad_item.setCheckState(0, Qt.Unchecked)
                        tree_widget.visibility_status[str(cad_uid)] = False

                    created_count += 1

                print(f"  {class_name}: {len(cids)} clusters -> "
                      f"{geom_type} CAD objects")

        except Exception:
            logger.error(traceback.format_exc())
            QMessageBox.critical(main_window, "Error",
                                 f"CAD generation failed:\n{traceback.format_exc()}")
        finally:
            tree_widget.blockSignals(False)
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

        QMessageBox.information(
            main_window, "CAD Objects Created",
            f"Created {created_count} CAD objects across "
            f"{len(selections)} classes.\n\n"
            f"All branches are unchecked by default.\n"
            f"Check individual branches to view wireframes.",
        )


# ── Factory ──────────────────────────────────────────────────────────────

def _make_cad_object(cluster_pts, class_name, geom_type, color, parent_uid):
    """Build a CADObject for one cluster."""

    if geom_type == "mesh":
        center, R, extent, h_idx = _compute_obb(cluster_pts)
        transform, dims = _build_box_transform(center, R, extent, h_idx)

        geometry = {
            "vertices": _BOX_VERTICES.copy(),
            "faces": [list(f) for f in _BOX_FACES],
            "edges": _BOX_EDGES.copy(),
        }
        return CADObject(
            symbol_type=class_name,
            geometry_type="mesh",
            geometry=geometry,
            transform_matrix=transform,
            dimensions=dims.astype(np.float32),
            cluster_reference=parent_uid,
            color=color,
        )

    elif geom_type == "polyline":
        verts, closed = _build_polyline_geometry(cluster_pts)

        # For polylines the geometry is already in world coords,
        # so transform is identity and dimensions come from AABB.
        aabb_min = np.min(verts, axis=0)
        aabb_max = np.max(verts, axis=0)
        dims = aabb_max - aabb_min

        geometry = {
            "vertices": verts,
            "closed": closed,
        }
        return CADObject(
            symbol_type=class_name,
            geometry_type="polyline",
            geometry=geometry,
            transform_matrix=np.eye(4),
            dimensions=dims.astype(np.float32),
            cluster_reference=parent_uid,
            color=color,
        )

    else:
        raise ValueError(f"Unknown geometry_type: {geom_type}")
