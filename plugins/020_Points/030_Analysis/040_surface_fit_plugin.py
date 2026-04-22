"""
Surface Fit Plugin

Fits a bivariate polynomial surface to a single cluster picked in the viewer
and emits the result as a `cad_object` mesh node so the existing
`prepare_mesh_lines` → `set_lines` path renders it as a wireframe overlay.

Workflow:
1. User selects a `cluster_labels` branch.
2. User Shift+clicks a point on the target cluster.
3. Plugin isolates that cluster's points, runs PCA on GPU for a local (u, v, w)
   frame, solves a least-squares polynomial fit `w = Σ c_ij · u^i v^j`
   (i + j ≤ degree) on GPU via torch.linalg.lstsq.
4. Delaunay-triangulates the uv scatter, drops triangles whose longest edge
   exceeds `alpha` (alpha-shape), samples a regular uv grid clipped to the
   resulting hull, evaluates the polynomial, and transforms back to world.
5. Produces a CADObject (geometry_type="mesh") child node under the selected
   cluster_labels branch; rendered as a cyan wireframe.
"""

import logging
import threading
import time
import traceback
import uuid
from typing import Any, Dict

import numpy as np
import torch
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMessageBox
from scipy.spatial import Delaunay, cKDTree

from config.config import global_variables
from core.entities.cad_object import CADObject
from core.entities.data_node import DataNode
from plugins.interfaces import ActionPlugin

logger = logging.getLogger(__name__)

_SURFACE_COLOR = np.array([0.0, 1.0, 1.0], dtype=np.float32)  # cyan


class SurfaceFitPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "surface_fit"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "polynomial_degree": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 4,
                "label": "Polynomial Degree",
                "description": "Total degree of the bivariate polynomial "
                               "z(u, v) = Σ c_ij u^i v^j with i + j ≤ degree.",
            },
            "grid_resolution": {
                "type": "int",
                "default": 40,
                "min": 5,
                "max": 200,
                "label": "Grid Resolution",
                "description": "Samples per uv axis used for wireframe output.",
            },
            "distance_threshold": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "label": "Inlier Distance Threshold",
                "description": "Max |w - w_fit| for a cluster point to count "
                               "as an inlier (logged only).",
            },
            "alpha": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "label": "Alpha (0 = auto)",
                "description": "Alpha-shape edge-length filter for the uv hull. "
                               "0 uses 4× median nearest-neighbour uv spacing.",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        logger.info("SurfaceFitPlugin.execute() called")
        logger.debug(f"Params: {params}")

        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget
        data_nodes = global_variables.global_data_nodes

        # --- Validate: exactly one branch selected ---
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a cluster_labels branch.")
            return
        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE cluster_labels branch.")
            return

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)
        if node is None or node.data_type != "cluster_labels":
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Please select a cluster_labels branch.")
            return

        picked_indices = viewer_widget.picked_points_indices
        if not picked_indices:
            QMessageBox.warning(main_window, "No Point Selected",
                                "Shift+click a point on the target cluster, "
                                "then run this plugin.")
            return

        picked_idx = picked_indices[0]
        if picked_idx >= len(viewer_widget.points):
            QMessageBox.warning(main_window, "Invalid Pick",
                                "Picked point index is out of range.")
            return
        picked_xyz = viewer_widget.points[picked_idx, :3].astype(np.float32)

        try:
            clusters_pc = controller.reconstruct(selected_uid)
        except Exception as e:
            logger.error(f"Failed to reconstruct {selected_uid}: {e}")
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{e}")
            return

        cluster_labels = clusters_pc.get_attribute("cluster_labels")
        if cluster_labels is None:
            QMessageBox.warning(main_window, "No Cluster Labels",
                                "Branch has no cluster_labels attribute.")
            return
        cluster_labels = cluster_labels.astype(np.int32)

        kd = cKDTree(clusters_pc.points)
        _, local_idx = kd.query(picked_xyz)
        seed_cluster_id = int(cluster_labels[local_idx])
        if seed_cluster_id == -1:
            QMessageBox.warning(main_window, "Noise Point Selected",
                                "Picked point is noise (label -1). "
                                "Pick a clustered point.")
            return

        cluster_mask = cluster_labels == seed_cluster_id
        cluster_points = clusters_pc.points[cluster_mask].astype(np.float32)
        n_cluster = len(cluster_points)
        degree = int(params["polynomial_degree"])
        min_points = (degree + 1) * (degree + 2) // 2
        if n_cluster < max(min_points, 6):
            QMessageBox.warning(main_window, "Cluster Too Small",
                                f"Cluster has {n_cluster} points; need at least "
                                f"{max(min_points, 6)} for degree {degree}.")
            return

        logger.info(
            f"Fitting surface to cluster id={seed_cluster_id}, "
            f"{n_cluster:,} points, degree={degree}"
        )

        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()

        grid_resolution = int(params["grid_resolution"])
        distance_threshold = float(params["distance_threshold"])
        alpha_param = float(params["alpha"])

        main_window.disable_menus()
        main_window.disable_tree()
        global_variables.global_progress = (None, "Fitting surface...")
        main_window.show_progress("Fitting surface...")
        main_window.show_cancel_button()

        cancel_event = global_variables.global_cancel_event
        cancel_event.clear()
        state = {"result": None, "error": None, "done": False}

        def _run():
            try:
                state["result"] = _fit_and_triangulate(
                    cluster_points,
                    degree,
                    grid_resolution,
                    distance_threshold,
                    alpha_param,
                    cancel_event,
                )
            except Exception as e:
                logger.exception(f"Surface fit failed: {e}")
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
                QMessageBox.critical(main_window, "Surface Fit Failed",
                                     state["error"])
                return

            if state["result"] is None:
                main_window.clear_progress()
                main_window.enable_menus()
                main_window.enable_tree()
                QMessageBox.information(main_window, "Cancelled",
                                        "Surface fit cancelled.")
                return

            vertices, faces, edges, aabb_dims = state["result"]

            parent_uuid = uuid.UUID(selected_uid)
            node_name = f"cluster_{seed_cluster_id}_surface_d{degree}"

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
                color=_SURFACE_COLOR,
            )

            try:
                tree_widget.blockSignals(True)
                cad_node = DataNode(
                    params=node_name,
                    data=cad_obj,
                    data_type="cad_object",
                    parent_uid=parent_uuid,
                    depends_on=[parent_uuid],
                    tags=["cad", "surface_fit"],
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

            logger.info(f"Surface fit branch added: uid={cad_uid}")

        poll_timer.timeout.connect(_poll)
        poll_timer.start(100)


# ---------------------------------------------------------------------------
# Fit + triangulation
# ---------------------------------------------------------------------------

def _fit_and_triangulate(
    points: np.ndarray,
    degree: int,
    grid_resolution: int,
    distance_threshold: float,
    alpha_param: float,
    cancel_event: threading.Event,
):
    """Fit polynomial surface, return (vertices, faces, edges, aabb_dims).

    Returns ``None`` if ``cancel_event`` is set at any stage boundary.
    """
    t0 = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available — surface fit requires a GPU. "
            "Check your PyTorch/CUDA installation."
        )
    device = torch.device("cuda")

    global_variables.global_progress = (None, "PCA...")
    if cancel_event.is_set():
        return None

    pts_t = torch.as_tensor(points, dtype=torch.float64, device=device)
    centroid = pts_t.mean(dim=0)
    centered = pts_t - centroid

    # PCA: eigh on 3x3 covariance.
    cov = (centered.T @ centered) / max(len(points) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # eigh returns ascending eigenvalues — the normal is the first column.
    # Reorder so columns are (u_axis, v_axis, w_axis) with w_axis = smallest eig.
    axes = torch.stack([eigvecs[:, 2], eigvecs[:, 1], eigvecs[:, 0]], dim=1)
    uvw = centered @ axes
    u = uvw[:, 0]
    v = uvw[:, 1]
    w = uvw[:, 2]

    # Vandermonde for total degree ≤ `degree`.
    exponents = [(i, j) for i in range(degree + 1)
                 for j in range(degree + 1 - i)]
    A = torch.stack(
        [(u ** i) * (v ** j) for (i, j) in exponents], dim=1
    )

    global_variables.global_progress = (None, "Least squares fit...")
    if cancel_event.is_set():
        return None

    solution = torch.linalg.lstsq(A, w.unsqueeze(1))
    coeffs = solution.solution.squeeze(1)

    residual = (A @ coeffs) - w
    rms = float(torch.sqrt(torch.mean(residual ** 2)).item())
    inlier_frac = float(
        (residual.abs() < distance_threshold).float().mean().item()
    )
    logger.info(
        f"Polynomial fit RMS={rms:.6f}, "
        f"inlier_frac@{distance_threshold}={inlier_frac:.3f}"
    )

    # --- Delaunay + alpha shape in uv ---
    global_variables.global_progress = (None, "Triangulating...")
    if cancel_event.is_set():
        return None

    uv_np = uvw[:, :2].detach().cpu().numpy()
    try:
        tri = Delaunay(uv_np)
    except Exception as exc:
        raise RuntimeError(f"Delaunay triangulation failed: {exc}") from exc

    if alpha_param <= 0:
        kd2d = cKDTree(uv_np)
        nn_dists, _ = kd2d.query(uv_np, k=2)
        median_spacing = float(np.median(nn_dists[:, 1]))
        alpha = 4.0 * max(median_spacing, 1e-6)
    else:
        alpha = alpha_param
    logger.info(f"Alpha-shape edge cap = {alpha:.6f}")

    simplex_verts = uv_np[tri.simplices]  # (T, 3, 2)
    d01 = np.linalg.norm(simplex_verts[:, 0] - simplex_verts[:, 1], axis=1)
    d12 = np.linalg.norm(simplex_verts[:, 1] - simplex_verts[:, 2], axis=1)
    d20 = np.linalg.norm(simplex_verts[:, 2] - simplex_verts[:, 0], axis=1)
    keep_tri = (d01 <= alpha) & (d12 <= alpha) & (d20 <= alpha)
    if not np.any(keep_tri):
        raise RuntimeError(
            "All triangles exceed alpha; increase alpha or lower "
            "cluster sparsity."
        )
    hull_simplices = tri.simplices[keep_tri]

    # --- Regular uv grid clipped to the retained hull ---
    global_variables.global_progress = (None, "Evaluating grid...")
    if cancel_event.is_set():
        return None

    u_min, v_min = uv_np.min(axis=0)
    u_max, v_max = uv_np.max(axis=0)
    uu = np.linspace(u_min, u_max, grid_resolution)
    vv = np.linspace(v_min, v_max, grid_resolution)
    grid_u, grid_v = np.meshgrid(uu, vv, indexing="xy")
    grid_uv = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1)

    # Restrict Delaunay to kept triangles by feeding it only their vertices.
    # Using find_simplex on the full tri and filtering by membership is cheaper.
    containing = tri.find_simplex(grid_uv)
    kept_mask = np.zeros(len(tri.simplices), dtype=bool)
    kept_mask[np.where(keep_tri)[0]] = True
    inside = (containing >= 0) & kept_mask[np.clip(containing, 0, None)]

    # Evaluate polynomial at grid vertices (GPU).
    grid_u_t = torch.as_tensor(grid_uv[:, 0], dtype=torch.float64, device=device)
    grid_v_t = torch.as_tensor(grid_uv[:, 1], dtype=torch.float64, device=device)
    A_grid = torch.stack(
        [(grid_u_t ** i) * (grid_v_t ** j) for (i, j) in exponents], dim=1
    )
    grid_w = (A_grid @ coeffs).detach().cpu().numpy()

    # Transform (u, v, w_fit) → world.
    axes_np = axes.detach().cpu().numpy()
    centroid_np = centroid.detach().cpu().numpy()
    local = np.stack([grid_uv[:, 0], grid_uv[:, 1], grid_w], axis=1)
    world = local @ axes_np.T + centroid_np

    # --- Build quad mesh faces/edges over inside grid cells (vectorized) ---
    global_variables.global_progress = (None, "Building mesh...")
    if cancel_event.is_set():
        return None

    inside_grid = inside.reshape(grid_resolution, grid_resolution)
    quad_ok = (inside_grid[:-1, :-1] & inside_grid[:-1, 1:]
               & inside_grid[1:, :-1] & inside_grid[1:, 1:])
    if not np.any(quad_ok):
        raise RuntimeError(
            "No grid cells lie inside the cluster hull — try a finer grid "
            "or a larger alpha."
        )

    rr, cc = np.where(quad_ok)
    v00 = rr * grid_resolution + cc
    v10 = rr * grid_resolution + (cc + 1)
    v01 = (rr + 1) * grid_resolution + cc
    v11 = (rr + 1) * grid_resolution + (cc + 1)
    faces_arr = np.concatenate(
        [np.stack([v00, v10, v11], axis=1),
         np.stack([v00, v11, v01], axis=1)],
        axis=0,
    )

    # Unique undirected edges from triangle edges.
    tri_edges = np.concatenate([
        faces_arr[:, [0, 1]],
        faces_arr[:, [1, 2]],
        faces_arr[:, [2, 0]],
    ], axis=0)
    tri_edges = np.sort(tri_edges, axis=1)
    edges_arr = np.unique(tri_edges, axis=0)

    # Compact vertex set to only those referenced.
    used = np.zeros(len(world), dtype=bool)
    used[faces_arr.ravel()] = True
    remap = -np.ones(len(world), dtype=np.int64)
    remap[used] = np.arange(used.sum())
    vertices = world[used].astype(np.float32)
    faces = remap[faces_arr].astype(np.int32).tolist()
    edges = remap[edges_arr].astype(np.int32)

    aabb_dims = (vertices.max(axis=0) - vertices.min(axis=0)).astype(np.float32)
    logger.info(
        f"Surface mesh: {len(vertices)} verts, {len(faces)} tris, "
        f"{len(edges)} edges, built in {time.time() - t0:.2f}s"
    )
    return vertices, faces, edges, aabb_dims
