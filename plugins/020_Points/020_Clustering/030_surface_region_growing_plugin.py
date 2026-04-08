"""
Plugin for surface region growing on a pre-existing DBSCAN Clusters branch.

Workflow:
1. User prepares a Clusters branch via: normal estimation → normal filter → DBSCAN
2. User Shift+clicks a point on the desired seed cluster in the 3D viewer
3. User runs this plugin
4. Plugin identifies the seed cluster from the clicked point
5. Expands the seed across unassigned points using a voxel grid + per-boundary-voxel
   RANSAC plane fitting — no normal consistency gate (intentional)
6. Returns a new Clusters branch: label 0 = grown surface, -1 = non-surface
"""

import threading
import time
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any, Tuple, Optional, List
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.clusters import Clusters


class SurfaceRegionGrowingPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "surface_region_growing"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "voxel_size": {
                "type": "float",
                "default": 0.5,
                "min": 0.001,
                "max": 100.0,
                "label": "Voxel Size",
                "description": "Uniform voxel edge length in world units. "
                               "Should match approximate point spacing."
            },
            "neighbor_radius": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 5,
                "label": "Neighbor Radius (voxels)",
                "description": "Chebyshev voxel radius to search for candidate points "
                               "around each boundary voxel."
            },
            "distance_threshold": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "label": "Distance Threshold",
                "description": "Maximum perpendicular distance to the RANSAC plane "
                               "for a point to be accepted into the surface."
            },
            "ransac_iterations": {
                "type": "int",
                "default": 50,
                "min": 10,
                "max": 500,
                "label": "RANSAC Iterations",
                "description": "Number of RANSAC iterations per boundary voxel. "
                               "More iterations = more robust plane fit but slower."
            },
            "ransac_inlier_threshold": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "label": "RANSAC Inlier Threshold",
                "description": "Maximum distance to the plane for a point to count "
                               "as a RANSAC inlier during plane fitting."
            },
            "min_voxel_surface_points": {
                "type": "int",
                "default": 5,
                "min": 3,
                "max": 100,
                "label": "Min Surface Points per Voxel",
                "description": "Minimum number of surface points required in/around "
                               "a boundary voxel to attempt RANSAC fitting."
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget

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
                                "Please select a cluster_labels branch "
                                "(output of DBSCAN or similar).")
            return

        # --- Validate: a point must be picked ---
        picked_indices = viewer_widget.picked_points_indices
        if not picked_indices:
            QMessageBox.warning(main_window, "No Point Selected",
                                "Shift+click a point on the seed surface cluster "
                                "in the viewer, then run this plugin.")
            return

        # --- Determine seed cluster ID from picked point ---
        picked_idx = picked_indices[0]
        if picked_idx >= len(viewer_widget.points):
            QMessageBox.warning(main_window, "Invalid Pick",
                                "Picked point index is out of range. "
                                "Please Shift+click again.")
            return

        picked_xyz = viewer_widget.points[picked_idx, :3].astype(np.float32)

        try:
            clusters_pc = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct clusters branch:\n{e}")
            return

        cluster_labels_attr = clusters_pc.get_attribute("cluster_labels")
        if cluster_labels_attr is None:
            QMessageBox.warning(main_window, "No Cluster Labels",
                                "Reconstructed point cloud has no cluster_labels attribute.")
            return

        cluster_labels = cluster_labels_attr.astype(np.int32)

        kd = cKDTree(clusters_pc.points)
        _, local_idx = kd.query(picked_xyz)
        seed_cluster_id = int(cluster_labels[local_idx])

        if seed_cluster_id == -1:
            QMessageBox.warning(main_window, "Noise Point Selected",
                                "The clicked point belongs to noise (label -1). "
                                "Please Shift+click a non-noise cluster point.")
            return

        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()

        # --- Reconstruct parent PointCloud ---
        parent_uid = node.parent_uid
        if parent_uid is None:
            QMessageBox.warning(main_window, "No Parent",
                                "The clusters branch has no parent PointCloud node.")
            return

        parent_uid_str = str(parent_uid)
        parent_node = controller.get_node(parent_uid_str)
        if parent_node is None:
            QMessageBox.warning(main_window, "Parent Not Found",
                                "Could not find the parent PointCloud node.")
            return

        try:
            full_pc = controller.reconstruct(parent_uid_str)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct parent point cloud:\n{e}")
            return

        full_points = full_pc.points.astype(np.float32)
        n_points = len(full_points)

        # cluster_labels is aligned to the clusters_pc (same parent), same N
        if len(cluster_labels) != n_points:
            QMessageBox.critical(main_window, "Shape Mismatch",
                                 f"Cluster labels ({len(cluster_labels)}) do not match "
                                 f"parent point count ({n_points}).")
            return

        voxel_size = float(params["voxel_size"])
        neighbor_radius = int(params["neighbor_radius"])
        distance_threshold = float(params["distance_threshold"])
        ransac_iterations = int(params["ransac_iterations"])
        ransac_inlier_threshold = float(params["ransac_inlier_threshold"])
        min_voxel_surface_points = int(params["min_voxel_surface_points"])

        # --- Disable UI ---
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.show_progress("Growing surface region...")

        state = {
            'result': None,
            'error': None,
            'done': False,
        }

        def _grow():
            try:
                labels_out = _region_grow(
                    full_points,
                    cluster_labels,
                    seed_cluster_id,
                    voxel_size,
                    neighbor_radius,
                    distance_threshold,
                    ransac_iterations,
                    ransac_inlier_threshold,
                    min_voxel_surface_points,
                )
                clusters = Clusters(labels_out)
                clusters.set_random_color()
                state['result'] = clusters
            except Exception as e:
                state['error'] = str(e)
            finally:
                state['done'] = True

        thread = threading.Thread(target=_grow, daemon=True)
        thread.start()

        while not state['done']:
            percent, msg = global_variables.global_progress
            if msg:
                main_window.show_progress(msg, percent)
            QtWidgets.QApplication.processEvents()
            time.sleep(0.1)

        global_variables.global_progress = (None, "")

        if state['error']:
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()
            QMessageBox.critical(main_window, "Region Growing Failed", state['error'])
            return

        # --- Add result branch ---
        main_window.show_progress("Adding result...", 95)
        tree_widget.blockSignals(True)

        result_uid = controller.add_analysis_result(
            state['result'],
            "cluster_labels",
            [parent_node.uid],
            parent_node,
            "surface_region_growing",
            params
        )

        tree_widget.add_branch(
            result_uid,
            parent_uid_str,
            "surface_region_growing",
            tooltip=f"surface_region_growing,{params}"
        )

        tree_widget.visibility_status[result_uid] = True

        tree_widget.blockSignals(False)

        main_window.render_visible_data(zoom_extent=False)
        main_window.clear_progress()
        main_window.enable_menus()
        main_window.enable_tree()


# ---------------------------------------------------------------------------
# Region growing implementation
# ---------------------------------------------------------------------------

def _region_grow(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    seed_cluster_id: int,
    voxel_size: float,
    neighbor_radius: int,
    distance_threshold: float,
    ransac_iterations: int,
    ransac_inlier_threshold: float,
    min_voxel_surface_points: int,
) -> np.ndarray:
    """
    Grow the seed cluster across unassigned points using voxel grid + RANSAC planes.

    Returns:
        labels_out (N,) int32 — 0 = surface, -1 = non-surface
    """
    n = len(points)

    surface_mask = (cluster_labels == seed_cluster_id)
    unassigned_mask = ~surface_mask

    # Build voxel grid for ALL points
    voxel_grid, voxel_keys_all = _build_voxel_grid(points, voxel_size)

    # Separate voxel sets
    surface_voxels: set = set()
    unassigned_voxels: set = set()
    for i in range(n):
        key = tuple(voxel_keys_all[i])
        if surface_mask[i]:
            surface_voxels.add(key)
        else:
            unassigned_voxels.add(key)

    iteration = 0
    while True:
        boundary_voxels = _get_boundary_voxels(surface_voxels, unassigned_voxels)
        if not boundary_voxels:
            break

        newly_added = 0

        for bv in boundary_voxels:
            # Gather surface points in/around this boundary voxel
            surface_pts_indices = _gather_surface_points_near_voxel(
                bv, voxel_grid, surface_mask, voxel_keys_all,
                min_voxel_surface_points
            )

            if len(surface_pts_indices) < 3:
                continue

            surface_pts = points[surface_pts_indices]
            plane = _fit_ransac_plane(surface_pts, ransac_iterations, ransac_inlier_threshold)
            if plane is None:
                continue

            plane_normal, plane_d = plane

            # Collect all unassigned candidate points in neighbouring voxels
            candidate_indices = _get_candidates_in_radius(
                bv, neighbor_radius, voxel_grid, unassigned_mask
            )

            if len(candidate_indices) == 0:
                continue

            candidate_pts = points[candidate_indices]
            perp_dist = np.abs(candidate_pts @ plane_normal + plane_d)
            accept_mask = perp_dist < distance_threshold
            new_indices = candidate_indices[accept_mask]

            if len(new_indices) == 0:
                continue

            surface_mask[new_indices] = True
            unassigned_mask[new_indices] = False

            # Update voxel sets with point-level correctness
            updated_voxels = set()
            for idx in new_indices:
                updated_voxels.add(tuple(voxel_keys_all[idx]))

            for key in updated_voxels:
                voxel_pts = voxel_grid[key]
                surface_count = int(np.sum(surface_mask[voxel_pts]))
                # Only promote to surface voxel if enough points are surface
                if surface_count >= min_voxel_surface_points:
                    surface_voxels.add(key)
                # Only remove from unassigned if ALL points in voxel are surface
                if surface_count == len(voxel_pts):
                    unassigned_voxels.discard(key)

            newly_added += len(new_indices)

        iteration += 1
        total_surface = int(np.sum(surface_mask))
        global_variables.global_progress = (
            None,
            f"Region growing — iteration {iteration}, "
            f"surface points: {total_surface:,}"
        )

        if newly_added == 0:
            break

    labels_out = np.full(n, -1, dtype=np.int32)
    labels_out[surface_mask] = 0
    return labels_out


def _build_voxel_grid(
    points: np.ndarray, voxel_size: float
) -> Tuple[Dict[tuple, List[int]], np.ndarray]:
    """
    Build a spatial hash grid.

    Returns:
        voxel_grid: dict mapping voxel key → list of point indices
        voxel_keys_all: (N, 3) int32 array of each point's voxel key
    """
    voxel_keys_all = np.floor(points / voxel_size).astype(np.int32)
    voxel_grid: Dict[tuple, List[int]] = {}
    for i, key in enumerate(map(tuple, voxel_keys_all)):
        if key not in voxel_grid:
            voxel_grid[key] = []
        voxel_grid[key].append(i)
    # Convert lists to numpy arrays for efficient fancy indexing
    for key in voxel_grid:
        voxel_grid[key] = np.array(voxel_grid[key], dtype=np.int64)
    return voxel_grid, voxel_keys_all


def _get_boundary_voxels(
    surface_voxels: set, unassigned_voxels: set
) -> set:
    """
    Return voxels that are 26-connectivity neighbours of surface_voxels
    and also present in unassigned_voxels.
    """
    boundary = set()
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    for vk in surface_voxels:
        for dx, dy, dz in offsets:
            nb = (vk[0] + dx, vk[1] + dy, vk[2] + dz)
            if nb in unassigned_voxels:
                boundary.add(nb)
    return boundary


def _get_voxels_in_radius(center: tuple, radius: int) -> List[tuple]:
    """Return all voxel keys within Chebyshev distance `radius` of center."""
    cx, cy, cz = center
    return [
        (cx + dx, cy + dy, cz + dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
    ]


def _gather_surface_points_near_voxel(
    bv: tuple,
    voxel_grid: Dict[tuple, List[int]],
    surface_mask: np.ndarray,
    voxel_keys_all: np.ndarray,
    min_count: int,
) -> np.ndarray:
    """
    Collect surface point indices in bv. If fewer than min_count, expand to
    26-connectivity neighbours until enough points are gathered.
    """
    _empty = np.array([], dtype=np.int64)

    # Start with the boundary voxel itself
    voxel_pts = voxel_grid.get(bv, _empty)
    if len(voxel_pts) > 0:
        indices = voxel_pts[surface_mask[voxel_pts]]
    else:
        indices = _empty

    if len(indices) >= min_count:
        return indices

    # Expand to 26-neighbours
    parts = [indices]
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    for dx, dy, dz in offsets:
        nb = (bv[0] + dx, bv[1] + dy, bv[2] + dz)
        nb_pts = voxel_grid.get(nb, _empty)
        if len(nb_pts) > 0:
            parts.append(nb_pts[surface_mask[nb_pts]])

    return np.concatenate(parts) if len(parts) > 1 else indices


def _get_candidates_in_radius(
    bv: tuple,
    radius: int,
    voxel_grid: Dict[tuple, List[int]],
    unassigned_mask: np.ndarray,
) -> np.ndarray:
    """Collect unassigned point indices in all voxels within Chebyshev radius of bv."""
    _empty = np.array([], dtype=np.int64)
    parts = []
    for vk in _get_voxels_in_radius(bv, radius):
        voxel_pts = voxel_grid.get(vk, _empty)
        if len(voxel_pts) > 0:
            unassigned = voxel_pts[unassigned_mask[voxel_pts]]
            if len(unassigned) > 0:
                parts.append(unassigned)
    if not parts:
        return _empty
    return np.concatenate(parts)


def _fit_ransac_plane(
    points: np.ndarray,
    n_iterations: int,
    inlier_threshold: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Fit a plane to points using RANSAC.

    Returns:
        (unit_normal, d) for plane equation: normal·x + d = 0
        None if no valid plane found.
    """
    n = len(points)
    if n < 3:
        return None

    best_normal = None
    best_d = None
    best_inliers = 0

    for _ in range(n_iterations):
        sample_idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = points[sample_idx]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue

        normal = normal / norm
        d = -np.dot(normal, p0)

        distances = np.abs(points @ normal + d)
        inlier_count = int(np.sum(distances < inlier_threshold))

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_normal = normal
            best_d = d

    if best_normal is None:
        return None

    return best_normal.astype(np.float32), float(best_d)