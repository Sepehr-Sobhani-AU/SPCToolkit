"""
Cluster Boundary Plugin

Creates a closed 3D polyline for the outer edge of each selected cluster.
Uses an alpha-shape (Delaunay + edge-length filter) to find boundary points,
then resamples at a user-defined spacing, snapping each vertex to the nearest
actual edge point so the polyline preserves real XYZ coordinates.
"""

import uuid
import logging
import traceback
from typing import Dict, Any, List

import numpy as np
from scipy.spatial import KDTree
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.cad_object import CADObject
from core.entities.data_node import DataNode

logger = logging.getLogger(__name__)

_DEFAULT_COLOR = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # yellow


def _trace_outer_contour(grid):
    """
    Moore-neighbour contour tracing on a boolean grid.

    Returns an ordered list of (row, col) cells forming the outer
    boundary of the occupied region.  The grid must have at least one
    empty-cell border on all sides.
    """
    ny, nx = grid.shape

    # Find leftmost occupied cell (break ties by bottom-most row)
    start = None
    for c in range(nx):
        for r in range(ny):
            if grid[r, c]:
                start = (r, c)
                break
        if start is not None:
            break
    if start is None:
        return []

    # 8 directions in CW order (grid coords: row+ = Y+)
    #   0=E  1=SE  2=S  3=SW  4=W  5=NW  6=N  7=NE
    dr = [0, -1, -1, -1, 0, 1, 1, 1]
    dc = [1, 1, 0, -1, -1, -1, 0, 1]

    contour = [start]
    r, c = start
    # Cell to the left (W) is guaranteed empty → backtrack = W = 4
    initial_bt = 4
    backtrack = initial_bt

    for _ in range(ny * nx * 4):
        search = (backtrack + 1) % 8         # first CW from backtrack
        found = False
        for i in range(8):
            d = (search + i) % 8
            nr, nc = r + dr[d], c + dc[d]
            if 0 <= nr < ny and 0 <= nc < nx and grid[nr, nc]:
                r, c = nr, nc
                backtrack = (d + 4) % 8      # opposite direction
                found = True
                break
        if not found:
            break
        # Terminate when we re-enter start from same direction
        if (r, c) == start and backtrack == initial_bt:
            break
        contour.append((r, c))

    # Remove duplicate consecutive cells (can happen at sharp corners)
    cleaned = [contour[0]]
    for cell in contour[1:]:
        if cell != cleaned[-1]:
            cleaned.append(cell)
    return cleaned


def _compute_boundary_polyline(points: np.ndarray,
                               vertex_spacing: float = 0.10) -> np.ndarray:
    """
    Compute the outer boundary of a point cluster on the XY plane using
    grid-based contour tracing, then snap each vertex to the nearest
    actual point (preserving real Z).

    1. Build a 2D occupancy grid with ``cell_size = vertex_spacing``.
    2. Trace the outer contour with Moore-neighbour tracing.
    3. For each contour cell, pick the nearest real point.

    Args:
        points: (N, 3) float32 array of XYZ coordinates.
        vertex_spacing: Grid cell size / approximate vertex spacing (metres).

    Returns:
        (M, 3) float32 array of ordered boundary vertices (closed loop).
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points for a boundary.")

    xy = points[:, :2].astype(np.float64)

    spread = np.max(xy, axis=0) - np.min(xy, axis=0)
    if min(spread) < 1e-6:
        raise ValueError("Points are collinear — no 2D boundary.")

    # ── Build occupancy grid (1-cell padding on every side) ─────────
    cell_size = float(vertex_spacing)
    xy_min = np.min(xy, axis=0)
    grid_origin = xy_min - cell_size          # one cell of padding
    nx = int(np.ceil(spread[0] / cell_size)) + 3
    ny = int(np.ceil(spread[1] / cell_size)) + 3

    cell_idx = ((xy - grid_origin) / cell_size).astype(int)
    grid = np.zeros((ny, nx), dtype=bool)
    for cx, cy in cell_idx:
        if 0 <= cy < ny and 0 <= cx < nx:
            grid[cy, cx] = True

    # ── Trace outer contour ─────────────────────────────────────────
    contour = _trace_outer_contour(grid)
    if len(contour) < 3:
        raise ValueError("Boundary contour too short.")

    # ── Snap each contour cell to the nearest actual point ──────────
    tree = KDTree(xy)
    result = np.empty((len(contour), 3), dtype=np.float32)
    for i, (r, c) in enumerate(contour):
        center = grid_origin + np.array([c + 0.5, r + 0.5]) * cell_size
        _, idx = tree.query(center)
        result[i] = points[idx]

    # Remove consecutive duplicates (adjacent cells can snap to same point)
    keep = np.ones(len(result), dtype=bool)
    for i in range(1, len(result)):
        if np.array_equal(result[i], result[i - 1]):
            keep[i] = False
    result = result[keep]

    return result


class ClusterBoundaryPlugin(ActionPlugin):
    """Create a closed 3D polyline boundary for each cluster."""

    def get_name(self) -> str:
        return "cluster_boundary"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "vertex_spacing": {
                "type": "float",
                "default": 0.10,
                "min": 0.01,
                "max": 10.0,
                "label": "Vertex Spacing (m)",
                "description": "Distance between boundary vertices in metres",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget
        viewer_widget = global_variables.global_pcd_viewer_widget

        # Validate selection
        selected = controller.selected_branches
        if not selected:
            QMessageBox.warning(main_window, "No Selection",
                                "Select a branch to extract boundaries from.")
            return
        if len(selected) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch.")
            return

        selected_uid = selected[0]

        # Reconstruct
        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error", str(e))
            return

        points = point_cloud.points
        if points is None or len(points) < 3:
            QMessageBox.warning(main_window, "Insufficient Points",
                                "Selected branch has too few points.")
            return

        cluster_labels = point_cloud.get_attribute("cluster_labels")
        cluster_names = point_cloud.get_attribute("_cluster_names")

        if cluster_labels is None:
            QMessageBox.warning(main_window, "No Cluster Labels",
                                "Selected branch has no cluster labels.\n"
                                "Run clustering first.")
            return

        # Determine which clusters to process from picked points
        picked_indices = viewer_widget.picked_points_indices
        if not picked_indices:
            QMessageBox.warning(
                main_window, "No Clusters Selected",
                "No clusters are selected.\n\n"
                "Shift+Click on points in the clusters you want\n"
                "to extract boundaries for.")
            return

        # Find cluster IDs from picked point indices
        target_cids = set()
        for idx in picked_indices:
            if idx < len(cluster_labels):
                cid = cluster_labels[idx]
                if cid >= 0:  # exclude noise
                    target_cids.add(cid)

        if not target_cids:
            QMessageBox.warning(main_window, "No Valid Clusters",
                                "Selected points do not belong to any "
                                "valid clusters (all noise points).")
            return

        groups = []
        for cid in sorted(target_cids):
            mask = cluster_labels == cid
            name = (cluster_names or {}).get(cid, f"cluster_{cid}")
            groups.append((name, points[mask]))

        vertex_spacing = params.get("vertex_spacing", 0.10)

        # Generate boundaries
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.tree_overlay.show_processing("Computing boundaries...")

        parent_uuid = uuid.UUID(selected_uid)
        created = 0

        try:
            tree_widget.blockSignals(True)

            # Create container
            container_node = DataNode(
                params="Boundaries",
                data=None,
                data_type="container",
                parent_uid=parent_uuid,
                depends_on=[parent_uuid],
                tags=["cad", "boundary"],
            )
            container_uid = data_nodes.add_node(container_node)
            tree_widget.add_branch(
                str(container_uid), selected_uid, "Boundaries"
            )
            item = tree_widget.branches_dict.get(str(container_uid))
            if item:
                item.setCheckState(0, Qt.Unchecked)
                tree_widget.visibility_status[str(container_uid)] = False

            for name, cluster_pts in groups:
                try:
                    hull_verts = _compute_boundary_polyline(cluster_pts, vertex_spacing)
                except (ValueError, Exception) as exc:
                    logger.warning(f"Skipping {name}: {exc}")
                    continue

                aabb_min = np.min(hull_verts, axis=0)
                aabb_max = np.max(hull_verts, axis=0)
                dims = (aabb_max - aabb_min).astype(np.float32)

                cad_obj = CADObject(
                    symbol_type=name,
                    geometry_type="polyline",
                    geometry={
                        "vertices": hull_verts,
                        "closed": True,
                    },
                    transform_matrix=np.eye(4),
                    dimensions=dims,
                    cluster_reference=parent_uuid,
                    color=_DEFAULT_COLOR,
                )

                node_name = f"{name}_boundary"
                cad_node = DataNode(
                    params=node_name,
                    data=cad_obj,
                    data_type="cad_object",
                    parent_uid=container_uid,
                    depends_on=[parent_uuid],
                    tags=["cad", "boundary"],
                )
                cad_uid = data_nodes.add_node(cad_node)
                tree_widget.add_branch(
                    str(cad_uid), str(container_uid), node_name
                )
                cad_item = tree_widget.branches_dict.get(str(cad_uid))
                if cad_item:
                    cad_item.setCheckState(0, Qt.Unchecked)
                    tree_widget.visibility_status[str(cad_uid)] = False

                created += 1

        except Exception:
            logger.error(traceback.format_exc())
            QMessageBox.critical(main_window, "Error",
                                 f"Boundary extraction failed:\n"
                                 f"{traceback.format_exc()}")
        finally:
            tree_widget.blockSignals(False)
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

        # Clear picked points after processing
        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()

        QMessageBox.information(
            main_window, "Boundaries Created",
            f"Created {created} boundary polyline(s) from "
            f"{len(target_cids)} selected cluster(s).\n\n"
            f"Check individual branches to view them.",
        )
