"""
Mesh Drape Plugin

Drapes a 2.5D rectangular mesh over a point cloud surface:

1. Reconstruct the selected branch to a PointCloud (works for any branch type:
   point_cloud / mask / cluster_labels / …).
2. Determine the target subset:
   - cluster_labels branch  -> require Shift-picked point(s); use the
     corresponding cluster id(s).
   - other branches with picked points -> drape only those points.
   - otherwise -> drape all points.
3. Require precomputed normals. Build an orthonormal (u, v, n) frame with
   `n = normalize(mean(normals[subset]))`.
4. Project points into the uv-plane, bin into square cells of `cell_size`,
   and store the per-cell median of the out-of-plane coordinate w.
5. Shared-corner vertex heights = mean of adjacent populated cell medians;
   emit one quad per populated cell whose 4 corners are all valid.
6. Optional Gaussian smoothing on the height grid (empty-cell aware).
7. Output a CADObject (geometry_type='mesh') as a child node, rendered as a
   rectangular wireframe by the existing mesh-lines path.
"""

import logging
import threading
import time
import traceback
import uuid
from typing import Any, Dict

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMessageBox
from scipy.ndimage import gaussian_filter as _np_gaussian_filter
from scipy.spatial import cKDTree

try:
    import cupy as _cp
    from cupyx.scipy.ndimage import gaussian_filter as _cp_gaussian_filter
    _HAS_CUPY = True
except Exception:  # pragma: no cover - CuPy is optional
    _cp = None
    _cp_gaussian_filter = None
    _HAS_CUPY = False

from config.config import global_variables
from core.entities.cad_object import CADObject
from core.entities.data_node import DataNode
from plugins.interfaces import ActionPlugin

logger = logging.getLogger(__name__)

_DRAPE_COLOR = np.array([0.2, 1.0, 0.4], dtype=np.float32)  # green


class MeshDrapePlugin(ActionPlugin):

    def get_name(self) -> str:
        return "mesh_drape"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "cell_size": {
                "type": "float",
                "default": 0.1,
                "min": 1e-4,
                "max": 100.0,
                "label": "Cell Size",
                "description": "Square cell edge length in world units. "
                               "Smaller cells give a finer drape.",
            },
            "smoothing_sigma": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 5.0,
                "label": "Smoothing Sigma",
                "description": "Gaussian sigma (in cells) applied to the height "
                               "grid. 0 disables smoothing.",
            },
            "min_points_per_cell": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 50,
                "label": "Min Points / Cell",
                "description": "Cells with fewer points are treated as empty.",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        logger.info("MeshDrapePlugin.execute() called")
        logger.debug(f"Params: {params}")

        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget
        data_nodes = global_variables.global_data_nodes

        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a branch first.")
            return
        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch.")
            return

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)
        if node is None:
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Selected branch not found.")
            return

        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            logger.error(f"Failed to reconstruct {selected_uid}: {e}")
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{e}")
            return

        all_points = np.asarray(point_cloud.points, dtype=np.float32)
        if all_points is None or len(all_points) == 0:
            QMessageBox.warning(main_window, "Empty Branch",
                                "Reconstructed point cloud has no points.")
            return

        all_normals = point_cloud.normals
        if all_normals is None:
            QMessageBox.warning(
                main_window, "Missing Normals",
                "This branch has no normals. Run normal estimation on it first.",
            )
            return
        all_normals = np.asarray(all_normals, dtype=np.float32)
        if all_normals.shape != all_points.shape:
            QMessageBox.warning(
                main_window, "Normals Mismatch",
                f"Normals shape {all_normals.shape} does not match points "
                f"shape {all_points.shape}.",
            )
            return

        # --- Resolve subset ---
        picked_indices = list(viewer_widget.picked_points_indices)

        if node.data_type == "cluster_labels":
            if not picked_indices:
                QMessageBox.warning(
                    main_window, "No Cluster Picked",
                    "Shift+click one or more points on the target cluster(s), "
                    "then run this plugin.",
                )
                return
            cluster_labels = point_cloud.get_attribute("cluster_labels")
            if cluster_labels is None:
                QMessageBox.warning(main_window, "No Cluster Labels",
                                    "Branch has no cluster_labels attribute.")
                return
            cluster_labels = cluster_labels.astype(np.int32)

            # Map viewer-picked world XYZs to nearest points in the reconstructed
            # cloud to read their cluster ids.
            viewer_pts = viewer_widget.points
            picked_xyz = []
            for idx in picked_indices:
                if 0 <= idx < len(viewer_pts):
                    picked_xyz.append(viewer_pts[idx, :3])
            if not picked_xyz:
                QMessageBox.warning(main_window, "Invalid Pick",
                                    "Picked point indices are out of range.")
                return
            picked_xyz = np.asarray(picked_xyz, dtype=np.float32)
            kd = cKDTree(all_points)
            _, nearest = kd.query(picked_xyz)
            picked_cluster_ids = {int(cluster_labels[i]) for i in nearest}
            picked_cluster_ids.discard(-1)
            if not picked_cluster_ids:
                QMessageBox.warning(main_window, "Noise Pick",
                                    "Picked points are all noise (label -1). "
                                    "Pick a clustered point.")
                return
            subset_mask = np.isin(cluster_labels, np.array(sorted(picked_cluster_ids),
                                                           dtype=np.int32))
            scope_tag = f"clusters_{'_'.join(str(c) for c in sorted(picked_cluster_ids))}"
        elif picked_indices:
            subset_mask = np.zeros(len(all_points), dtype=bool)
            valid = [i for i in picked_indices if 0 <= i < len(all_points)]
            subset_mask[valid] = True
            scope_tag = "selection"
        else:
            subset_mask = np.ones(len(all_points), dtype=bool)
            scope_tag = "all"

        points = all_points[subset_mask]
        normals = all_normals[subset_mask]
        n_pts = len(points)
        if n_pts < 4:
            QMessageBox.warning(main_window, "Too Few Points",
                                f"Need at least 4 points, got {n_pts}.")
            return

        # Clear picks so the next run starts clean.
        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()

        cell_size = float(params["cell_size"])
        smoothing_sigma = float(params["smoothing_sigma"])
        min_points_per_cell = int(params["min_points_per_cell"])

        main_window.disable_menus()
        main_window.disable_tree()
        global_variables.global_progress = (None, "Draping mesh...")
        main_window.show_progress("Draping mesh...")
        main_window.show_cancel_button()

        cancel_event = global_variables.global_cancel_event
        cancel_event.clear()
        state = {"result": None, "error": None, "done": False}

        def _run():
            try:
                state["result"] = _drape(
                    points, normals, cell_size, smoothing_sigma,
                    min_points_per_cell, cancel_event,
                )
            except Exception as e:
                logger.exception(f"Mesh drape failed: {e}")
                state["error"] = f"{e}\n\n{traceback.format_exc()}"
            finally:
                state["done"] = True

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        poll_timer = QTimer(main_window)

        def _poll():
            percent, msg = global_variables.global_progress
            if msg:
                main_window.show_progress(msg, percent)
            if not state["done"]:
                return
            poll_timer.stop()
            _finalize()

        def _finalize():
            global_variables.global_progress = (None, "")
            main_window.hide_cancel_button()

            if state["error"]:
                main_window.clear_progress()
                main_window.enable_menus()
                main_window.enable_tree()
                QMessageBox.critical(main_window, "Mesh Drape Failed",
                                     state["error"])
                return

            if state["result"] is None:
                main_window.clear_progress()
                main_window.enable_menus()
                main_window.enable_tree()
                QMessageBox.information(main_window, "Cancelled",
                                        "Mesh drape cancelled.")
                return

            vertices, faces, edges, aabb_dims = state["result"]
            if len(faces) == 0:
                main_window.clear_progress()
                main_window.enable_menus()
                main_window.enable_tree()
                QMessageBox.warning(main_window, "Empty Mesh",
                                    "No quads were produced. Try a larger cell "
                                    "size or check the input subset.")
                return

            parent_uuid = uuid.UUID(selected_uid)
            node_name = f"drape_{scope_tag}_cs{cell_size:g}"

            cad_obj = CADObject(
                symbol_type=node_name,
                geometry_type="mesh",
                geometry={
                    "vertices": vertices,
                    "faces": faces,
                    "edges": edges,
                },
                transform_matrix=np.eye(4),
                dimensions=aabb_dims,
                cluster_reference=parent_uuid,
                color=_DRAPE_COLOR,
            )

            try:
                tree_widget.blockSignals(True)
                cad_node = DataNode(
                    params=node_name,
                    data=cad_obj,
                    data_type="cad_object",
                    parent_uid=parent_uuid,
                    depends_on=[parent_uuid],
                    tags=["cad", "mesh_drape"],
                )
                cad_uid = data_nodes.add_node(cad_node)
                tree_widget.add_branch(str(cad_uid), selected_uid, node_name)
                cad_item = tree_widget.branches_dict.get(str(cad_uid))
                if cad_item:
                    cad_item.setCheckState(0, Qt.Checked)
                    tree_widget.visibility_status[str(cad_uid)] = True
            finally:
                tree_widget.blockSignals(False)

            main_window.render_visible_data(zoom_extent=False)
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()

            logger.info(
                f"Mesh drape branch added: uid={cad_uid}, "
                f"quads={len(faces)}, vertices={len(vertices)}"
            )

        poll_timer.timeout.connect(_poll)
        poll_timer.start(100)


# ---------------------------------------------------------------------------
# Drape algorithm
# ---------------------------------------------------------------------------

def _drape(
    points: np.ndarray,
    normals: np.ndarray,
    cell_size: float,
    smoothing_sigma: float,
    min_points_per_cell: int,
    cancel_event: threading.Event,
):
    """Return (vertices, faces, edges, aabb_dims) or None if cancelled.

    Runs on GPU via CuPy when available, otherwise NumPy. Arrays stay on the
    device from projection through edge dedup; only the final mesh
    (vertices/faces/edges/dims) is copied back to host for CADObject.
    """
    t0 = time.time()

    if cell_size <= 0:
        raise ValueError("cell_size must be positive.")

    xp = _cp if _HAS_CUPY else np
    gaussian = _cp_gaussian_filter if _HAS_CUPY else _np_gaussian_filter
    backend = "GPU (CuPy)" if _HAS_CUPY else "CPU (NumPy)"

    points_x = xp.asarray(points, dtype=xp.float32)
    normals_x = xp.asarray(normals, dtype=xp.float32)

    # --- Mean normal and orthonormal (u, v, n) frame ---
    global_variables.global_progress = (None, "Building frame...")
    if cancel_event.is_set():
        return None

    n_mean = normals_x.mean(axis=0).astype(xp.float64)
    nrm = float(xp.linalg.norm(n_mean))
    if nrm < 1e-6:
        raise RuntimeError(
            "Mean normal is ~0 (normals cancel out). Re-run normal estimation "
            "with a consistent orientation, or pick a more uniform subset."
        )
    n_axis = n_mean / nrm

    # Reference axis not parallel to n_axis; build u, v.
    nz = float(n_axis[2])
    ref = xp.array([0.0, 0.0, 1.0], dtype=xp.float64) if abs(nz) < 0.9 \
        else xp.array([1.0, 0.0, 0.0], dtype=xp.float64)
    u_axis = xp.cross(n_axis, ref)
    u_axis = u_axis / xp.linalg.norm(u_axis)
    v_axis = xp.cross(n_axis, u_axis)
    basis = xp.stack([u_axis, v_axis, n_axis], axis=1)  # (3, 3)

    # --- Project points ---
    global_variables.global_progress = (None, "Projecting points...")
    if cancel_event.is_set():
        return None

    pts64 = points_x.astype(xp.float64)
    centroid = pts64.mean(axis=0)
    uvw = (pts64 - centroid) @ basis  # (N, 3)

    # --- Bin into cells ---
    global_variables.global_progress = (None, "Binning cells...")
    if cancel_event.is_set():
        return None

    uv_min = uvw[:, :2].min(axis=0)
    uv_max = uvw[:, :2].max(axis=0)
    extent = uv_max - uv_min
    nu = int(xp.ceil(extent[0] / cell_size)) + 1
    nv = int(xp.ceil(extent[1] / cell_size)) + 1
    if nu < 1 or nv < 1:
        raise RuntimeError("Degenerate grid extent.")
    if nu * nv > 50_000_000:
        raise RuntimeError(
            f"Grid too large ({nu} x {nv} = {nu * nv:,} cells). "
            f"Increase cell_size."
        )

    iu = xp.clip(((uvw[:, 0] - uv_min[0]) / cell_size).astype(xp.int64), 0, nu - 1)
    iv = xp.clip(((uvw[:, 1] - uv_min[1]) / cell_size).astype(xp.int64), 0, nv - 1)
    flat = iu * nv + iv
    w = uvw[:, 2]

    # --- Per-cell median of w (vectorised via sort) ---
    global_variables.global_progress = (None, "Computing medians...")
    if cancel_event.is_set():
        return None

    order = xp.lexsort(xp.stack([w, flat]))
    flat_sorted = flat[order]
    w_sorted = w[order]
    unique_cells, starts, counts = xp.unique(
        flat_sorted, return_index=True, return_counts=True
    )

    keep = counts >= min_points_per_cell
    unique_cells = unique_cells[keep]
    starts = starts[keep]
    counts = counts[keep]

    # Average the two middle elements of each segment (works for even/odd).
    lo = starts + (counts - 1) // 2
    hi = starts + counts // 2
    medians = 0.5 * (w_sorted[lo] + w_sorted[hi])

    height = xp.zeros((nu, nv), dtype=xp.float64)
    populated = xp.zeros((nu, nv), dtype=bool)
    ci = unique_cells // nv
    cj = unique_cells % nv
    height[ci, cj] = medians
    populated[ci, cj] = True

    if not bool(populated.any()):
        raise RuntimeError(
            f"No cells met min_points_per_cell={min_points_per_cell}. "
            f"Try lowering the threshold or increasing cell_size."
        )

    # --- Optional smoothing (empty-cell aware) ---
    if smoothing_sigma > 0:
        global_variables.global_progress = (None, "Smoothing...")
        if cancel_event.is_set():
            return None
        mask_f = populated.astype(xp.float64)
        num = gaussian(height * mask_f, sigma=smoothing_sigma, mode="constant")
        den = gaussian(mask_f, sigma=smoothing_sigma, mode="constant")
        smoothed = xp.where(den > 1e-9, num / xp.maximum(den, 1e-9), height)
        height = xp.where(populated, smoothed, height)

    # --- Corner heights = mean of up-to-4 adjacent populated cells ---
    global_variables.global_progress = (None, "Building corners...")
    if cancel_event.is_set():
        return None

    cu, cv = nu + 1, nv + 1
    corner_sum = xp.zeros((cu, cv), dtype=xp.float64)
    corner_cnt = xp.zeros((cu, cv), dtype=xp.int32)
    h_pop = xp.where(populated, height, 0.0)
    p_int = populated.astype(xp.int32)

    # Each cell (i, j) contributes to corners (i, j), (i+1, j), (i, j+1), (i+1, j+1).
    corner_sum[0:nu,     0:nv    ] += h_pop
    corner_sum[1:nu + 1, 0:nv    ] += h_pop
    corner_sum[0:nu,     1:nv + 1] += h_pop
    corner_sum[1:nu + 1, 1:nv + 1] += h_pop
    corner_cnt[0:nu,     0:nv    ] += p_int
    corner_cnt[1:nu + 1, 0:nv    ] += p_int
    corner_cnt[0:nu,     1:nv + 1] += p_int
    corner_cnt[1:nu + 1, 1:nv + 1] += p_int

    corner_valid = corner_cnt > 0
    corner_height = xp.where(corner_valid, corner_sum / xp.maximum(corner_cnt, 1), 0.0)

    quad_mask = (
        populated
        & corner_valid[0:nu,     0:nv    ]
        & corner_valid[1:nu + 1, 0:nv    ]
        & corner_valid[0:nu,     1:nv + 1]
        & corner_valid[1:nu + 1, 1:nv + 1]
    )
    n_quads = int(quad_mask.sum())
    if n_quads == 0:
        return (np.zeros((0, 3), dtype=np.float32),
                [],
                np.zeros((0, 2), dtype=np.int32),
                np.zeros(3, dtype=np.float32))

    # --- Emit vertices only for corners that are part of some quad ---
    global_variables.global_progress = (None, "Building mesh...")
    if cancel_event.is_set():
        return None

    used_corner = xp.zeros((cu, cv), dtype=bool)
    qi, qj = xp.nonzero(quad_mask)
    used_corner[qi,     qj    ] = True
    used_corner[qi + 1, qj    ] = True
    used_corner[qi,     qj + 1] = True
    used_corner[qi + 1, qj + 1] = True

    vertex_index = -xp.ones((cu, cv), dtype=xp.int64)
    uc_i, uc_j = xp.nonzero(used_corner)
    vertex_index[uc_i, uc_j] = xp.arange(uc_i.shape[0], dtype=xp.int64)

    corner_u = uv_min[0] + uc_i.astype(xp.float64) * cell_size
    corner_v = uv_min[1] + uc_j.astype(xp.float64) * cell_size
    corner_w = corner_height[uc_i, uc_j]

    uvw_corners = xp.stack([corner_u, corner_v, corner_w], axis=1)
    vertices_world = centroid + uvw_corners @ basis.T
    vertices_dev = vertices_world.astype(xp.float32)

    # --- Build quad faces ---
    v00 = vertex_index[qi,     qj    ]
    v10 = vertex_index[qi + 1, qj    ]
    v11 = vertex_index[qi + 1, qj + 1]
    v01 = vertex_index[qi,     qj + 1]
    faces_dev = xp.stack([v00, v10, v11, v01], axis=1).astype(xp.int64)

    # --- Build edges (dedup via single-int key min*V + max) ---
    e1 = xp.stack([v00, v10], axis=1)
    e2 = xp.stack([v10, v11], axis=1)
    e3 = xp.stack([v11, v01], axis=1)
    e4 = xp.stack([v01, v00], axis=1)
    all_edges = xp.concatenate([e1, e2, e3, e4], axis=0)
    e_min = xp.minimum(all_edges[:, 0], all_edges[:, 1])
    e_max = xp.maximum(all_edges[:, 0], all_edges[:, 1])
    n_verts = int(vertices_dev.shape[0])
    key = e_min.astype(xp.int64) * xp.int64(n_verts) + e_max.astype(xp.int64)
    unique_keys = xp.unique(key)
    edges_dev = xp.stack([unique_keys // xp.int64(n_verts),
                          unique_keys %  xp.int64(n_verts)], axis=1).astype(xp.int32)

    # --- AABB dimensions ---
    aabb_min = vertices_dev.min(axis=0)
    aabb_max = vertices_dev.max(axis=0)
    aabb_dims_dev = (aabb_max - aabb_min).astype(xp.float32)

    # --- Copy back to host (CADObject expects numpy) ---
    if _HAS_CUPY:
        vertices = xp.asnumpy(vertices_dev)
        faces_np = xp.asnumpy(faces_dev)
        edges = xp.asnumpy(edges_dev)
        aabb_dims = xp.asnumpy(aabb_dims_dev)
    else:
        vertices = vertices_dev
        faces_np = faces_dev
        edges = edges_dev
        aabb_dims = aabb_dims_dev
    faces_list = faces_np.tolist()

    dt = time.time() - t0
    logger.info(
        f"Mesh drape [{backend}]: {len(points):,} pts -> {nu}x{nv} grid, "
        f"{n_quads:,} quads, {len(vertices):,} verts, "
        f"{len(edges):,} edges in {dt:.2f}s"
    )
    return vertices, faces_list, edges, aabb_dims
