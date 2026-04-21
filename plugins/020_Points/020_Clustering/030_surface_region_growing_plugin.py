"""
Plugin for surface region growing on a pre-existing DBSCAN Clusters branch.

Workflow:
1. User prepares a Clusters branch by clustering the parent cloud (e.g. DBSCAN).
   The parent cloud must have per-point normals — either embedded at import
   (PLY nx/ny/nz) or produced by the normal_estimation plugin.
2. User Shift+clicks a point on the desired seed cluster in the 3D viewer
3. User runs this plugin
4. Plugin identifies the seed cluster from the clicked point and derives a
   reference normal from the picked point + its K seed-cluster neighbors
5. Expands the seed across unassigned points using a GPU-resident voxel grid +
   batched RANSAC plane fitting, gated by the reference normal
6. Returns a new Clusters branch: label 0 = grown surface, -1 = non-surface
"""

import threading
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from typing import Dict, Any, Tuple, List
from PyQt5.QtWidgets import QMessageBox, QApplication

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
            "max_normal_angle_deg": {
                "type": "float",
                "default": 20.0,
                "min": 0.0,
                "max": 90.0,
                "label": "Max Normal Deviation (deg)",
                "description": "Maximum angle between a RANSAC plane's normal and the "
                               "reference normal (mean of picked point and its K seed "
                               "neighbors). 90° disables the gate."
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

        # --- Reference normal: sign-aligned mean of picked point + K seed neighbors ---
        # Normals may be embedded at import (PLY nx/ny/nz → pc.normals) or produced
        # by the normal_estimation plugin (→ pc.get_attribute("normals") AND pc.normals).
        parent_normals = full_pc.get_attribute("normals")
        if parent_normals is None:
            parent_normals = getattr(full_pc, "normals", None)
        if parent_normals is None or len(parent_normals) == 0:
            QMessageBox.warning(main_window, "No Normals",
                                "Parent point cloud has no normals. Either import a "
                                "cloud with embedded normals (e.g. PLY nx/ny/nz), or "
                                "run the normal_estimation plugin first.")
            return

        anchor = np.asarray(parent_normals[local_idx], dtype=np.float32)
        anchor_n = np.linalg.norm(anchor)
        if anchor_n < 1e-10:
            QMessageBox.warning(main_window, "Invalid Normal",
                                "Picked point has a zero-length normal.")
            return
        anchor = anchor / anchor_n

        initial_k = min(len(clusters_pc.points), _REF_NORMAL_K * 5)
        _, nearest = kd.query(picked_xyz, k=initial_k)
        nearest = np.atleast_1d(nearest)
        seed_hits = nearest[cluster_labels[nearest] == seed_cluster_id][:_REF_NORMAL_K]
        if len(seed_hits) == 0:
            seed_hits = np.array([local_idx])

        nbhd = np.asarray(parent_normals[seed_hits], dtype=np.float32)
        nbhd_norms = np.linalg.norm(nbhd, axis=1)
        keep = nbhd_norms > 1e-10
        if not keep.any():
            ref_normal = anchor
        else:
            nbhd = nbhd[keep] / nbhd_norms[keep, None]
            # Sign-align to anchor; plane normals have sign ambiguity.
            flip = (nbhd @ anchor) < 0
            nbhd = np.where(flip[:, None], -nbhd, nbhd)
            ref_normal = nbhd.mean(axis=0)
            ref_n = np.linalg.norm(ref_normal)
            ref_normal = anchor if ref_n < 1e-10 else (ref_normal / ref_n)
        ref_normal = ref_normal.astype(np.float32)

        voxel_size = float(params["voxel_size"])
        neighbor_radius = int(params["neighbor_radius"])
        distance_threshold = float(params["distance_threshold"])
        ransac_iterations = int(params["ransac_iterations"])
        ransac_inlier_threshold = float(params["ransac_inlier_threshold"])
        min_voxel_surface_points = int(params["min_voxel_surface_points"])
        max_normal_angle_deg = float(params["max_normal_angle_deg"])

        # --- Disable UI ---
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.show_progress("Growing surface region...")
        main_window.show_cancel_button()

        state = {
            'result': None,
            'error': None,
            'stopped_early': False,
            'done': False,
        }

        def _grow():
            try:
                labels_out, stopped_early = _region_grow(
                    full_points,
                    cluster_labels,
                    seed_cluster_id,
                    voxel_size,
                    neighbor_radius,
                    distance_threshold,
                    ransac_iterations,
                    ransac_inlier_threshold,
                    min_voxel_surface_points,
                    ref_normal,
                    max_normal_angle_deg,
                )
                clusters = Clusters(labels_out)
                clusters.set_random_color()
                state['result'] = clusters
                state['stopped_early'] = stopped_early
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
            QApplication.processEvents()
            time.sleep(0.1)

        global_variables.global_progress = (None, "")
        main_window.hide_cancel_button()

        if state['error']:
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()
            QMessageBox.critical(main_window, "Region Growing Failed", state['error'])
            return

        # --- Add result branch (always, even if stopped early) ---
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

        if state['stopped_early']:
            QMessageBox.information(
                main_window, "Cancelled",
                "Region growing stopped early. Partial result saved as a new branch."
            )


# ---------------------------------------------------------------------------
# Region growing implementation (GPU-resident voxel grid, batched RANSAC)
# ---------------------------------------------------------------------------

_SURFACE_POINTS_CAP = 512
_BOUNDARY_CHUNK = 2048
_REF_NORMAL_K = 30

# 21-bit-per-axis packing: supports voxel grids with spans up to ~2M per axis.
_BIT_SHIFT_X = 42
_BIT_SHIFT_Y = 21
_BIT_AXIS_MAX = (1 << 21) - 1


def _pack_keys_np(keys_shifted: np.ndarray) -> np.ndarray:
    return (
        (keys_shifted[..., 0].astype(np.int64) << _BIT_SHIFT_X)
        | (keys_shifted[..., 1].astype(np.int64) << _BIT_SHIFT_Y)
        | keys_shifted[..., 2].astype(np.int64)
    )


def _pack_keys_torch(keys_shifted: torch.Tensor) -> torch.Tensor:
    return (
        (keys_shifted[..., 0] << _BIT_SHIFT_X)
        | (keys_shifted[..., 1] << _BIT_SHIFT_Y)
        | keys_shifted[..., 2]
    )


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
    ref_normal: np.ndarray,
    max_normal_angle_deg: float,
):
    """
    Grow the seed cluster across unassigned points using a GPU-resident voxel
    grid + batched RANSAC planes, with an orientation gate against `ref_normal`.

    Returns:
        (labels_out, stopped_early)
        labels_out: (N,) int32 — 0 = surface, -1 = non-surface
        stopped_early: True if the user cancelled via global_cancel_event.
                       Partial progress is preserved in labels_out.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Surface region growing requires a GPU.")
    device = torch.device('cuda')
    n = len(points)

    # --- Voxel grid build (vectorised NumPy) ---
    voxel_keys_all = np.floor(points / voxel_size).astype(np.int64)
    vk_min = voxel_keys_all.min(axis=0)
    vk_shifted_all = voxel_keys_all - vk_min  # non-negative

    # Safety bound: leave room for neighbor-radius offsets staying in [0, 2^21).
    span_limit = _BIT_AXIS_MAX - neighbor_radius
    if int(vk_shifted_all.max()) > span_limit:
        raise RuntimeError(
            f"Voxel grid span exceeds 21-bit packing limit "
            f"(max axis index {int(vk_shifted_all.max())} > {span_limit}). "
            f"Increase voxel_size or reduce point-cloud extent."
        )

    packed_all = _pack_keys_np(vk_shifted_all)
    order = np.argsort(packed_all, kind='stable').astype(np.int64)
    sorted_packed = packed_all[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = sorted_packed[1:] != sorted_packed[:-1]
    unique_packed = sorted_packed[is_new]
    V = int(len(unique_packed))

    voxel_point_starts = np.concatenate(
        (np.where(is_new)[0], np.array([n], dtype=np.int64))
    ).astype(np.int64)
    voxel_point_indices = order  # length-N
    voxel_total_counts = np.diff(voxel_point_starts).astype(np.int64)
    point_voxel_idx = np.searchsorted(unique_packed, packed_all).astype(np.int64)

    # Shifted unique keys (for neighbor offset computation)
    unique_voxel_keys_shifted = vk_shifted_all[voxel_point_indices[voxel_point_starts[:-1]]]

    surface_mask_np = (cluster_labels == seed_cluster_id)
    voxel_surface_counts = np.bincount(
        point_voxel_idx[surface_mask_np], minlength=V
    ).astype(np.int64)
    is_surface_voxel = voxel_surface_counts > 0
    is_unassigned_voxel = voxel_surface_counts < voxel_total_counts

    # --- Upload to GPU (one-time) ---
    points_t = torch.from_numpy(points).to(device)
    voxel_point_starts_t = torch.from_numpy(voxel_point_starts).to(device)
    voxel_point_indices_t = torch.from_numpy(voxel_point_indices).to(device)
    unique_packed_t = torch.from_numpy(unique_packed).to(device)
    unique_voxel_keys_t = torch.from_numpy(unique_voxel_keys_shifted).to(device)
    point_voxel_idx_t = torch.from_numpy(point_voxel_idx).to(device)
    surface_mask_t = torch.from_numpy(surface_mask_np).to(device)
    voxel_surface_counts_t = torch.from_numpy(voxel_surface_counts).to(device)
    voxel_total_counts_t = torch.from_numpy(voxel_total_counts).to(device)
    is_surface_voxel_t = torch.from_numpy(is_surface_voxel).to(device)
    is_unassigned_voxel_t = torch.from_numpy(is_unassigned_voxel).to(device)
    ref_normal_t = torch.from_numpy(np.asarray(ref_normal, dtype=np.float32)).to(device)
    cos_threshold = float(np.cos(np.deg2rad(max_normal_angle_deg)))

    # --- Neighbor offset tables on GPU ---
    neigh_offsets_26 = torch.tensor(
        [(dx, dy, dz)
         for dx in (-1, 0, 1)
         for dy in (-1, 0, 1)
         for dz in (-1, 0, 1)
         if not (dx == 0 and dy == 0 and dz == 0)],
        dtype=torch.int64, device=device,
    )
    neigh_offsets_27 = torch.tensor(
        [(dx, dy, dz)
         for dx in (-1, 0, 1)
         for dy in (-1, 0, 1)
         for dz in (-1, 0, 1)],
        dtype=torch.int64, device=device,
    )
    r = neighbor_radius
    radius_offsets = torch.tensor(
        [(dx, dy, dz)
         for dx in range(-r, r + 1)
         for dy in range(-r, r + 1)
         for dz in range(-r, r + 1)],
        dtype=torch.int64, device=device,
    )

    cancel_event = global_variables.global_cancel_event
    stopped_early = False
    iteration = 0

    while True:
        if cancel_event.is_set():
            stopped_early = True
            break

        # --- Find boundary voxels (vectorised on GPU) ---
        surface_vidx = torch.nonzero(is_surface_voxel_t, as_tuple=False).squeeze(1)
        if surface_vidx.numel() == 0:
            break

        neigh_keys = (
            unique_voxel_keys_t[surface_vidx].unsqueeze(1)
            + neigh_offsets_26.unsqueeze(0)
        )  # (S, 26, 3)
        in_range = ((neigh_keys >= 0) & (neigh_keys <= _BIT_AXIS_MAX)).all(dim=-1)
        keys_safe = neigh_keys.clamp(min=0, max=_BIT_AXIS_MAX)
        neigh_packed = _pack_keys_torch(keys_safe)
        flat = neigh_packed.reshape(-1)
        idx = torch.searchsorted(unique_packed_t, flat).clamp(max=V - 1)
        hits = (unique_packed_t[idx] == flat) & in_range.reshape(-1)
        valid_nb = hits & is_unassigned_voxel_t[idx]
        boundary_vidx_t = torch.unique(idx[valid_nb])

        if boundary_vidx_t.numel() == 0:
            break

        newly_added_total = 0
        B_total = int(boundary_vidx_t.numel())
        for chunk_start in range(0, B_total, _BOUNDARY_CHUNK):
            if cancel_event.is_set():
                stopped_early = True
                break
            chunk = boundary_vidx_t[chunk_start:chunk_start + _BOUNDARY_CHUNK]
            newly_added_total += _process_chunk_gpu(
                chunk, device, V,
                unique_voxel_keys_t, unique_packed_t,
                voxel_point_starts_t, voxel_point_indices_t, point_voxel_idx_t,
                points_t, surface_mask_t,
                voxel_surface_counts_t, voxel_total_counts_t,
                is_surface_voxel_t, is_unassigned_voxel_t,
                neigh_offsets_27, radius_offsets,
                ransac_iterations, ransac_inlier_threshold, distance_threshold,
                min_voxel_surface_points, ref_normal_t, cos_threshold,
            )

        if stopped_early:
            break

        iteration += 1
        total_surface = int(surface_mask_t.sum().item())
        global_variables.global_progress = (
            None,
            f"Region growing — iter {iteration}, surface: {total_surface:,}, "
            f"+{newly_added_total}"
        )

        if newly_added_total == 0:
            break

    surface_mask_final = surface_mask_t.cpu().numpy()
    labels_out = np.full(n, -1, dtype=np.int32)
    labels_out[surface_mask_final] = 0
    return labels_out, stopped_early


def _resolve_neighbors(
    chunk_vidx: torch.Tensor,
    unique_voxel_keys_t: torch.Tensor,
    unique_packed_t: torch.Tensor,
    offsets: torch.Tensor,
    V: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each voxel in `chunk_vidx`, resolve neighbor voxel indices for each offset.

    Returns:
        idx   : (B, K) int64 — voxel index (clamped to [0, V-1]) for each (chunk, offset)
        hits  : (B, K) bool  — True where the neighbor voxel actually exists
    """
    B = int(chunk_vidx.numel())
    K = int(offsets.shape[0])
    keys = unique_voxel_keys_t[chunk_vidx].unsqueeze(1) + offsets.unsqueeze(0)  # (B,K,3)
    in_range = ((keys >= 0) & (keys <= _BIT_AXIS_MAX)).all(dim=-1)  # (B,K)
    keys_safe = keys.clamp(min=0, max=_BIT_AXIS_MAX)
    packed = _pack_keys_torch(keys_safe)  # (B,K)
    idx = torch.searchsorted(unique_packed_t, packed.reshape(-1)).clamp(max=V - 1).view(B, K)
    hits = (unique_packed_t[idx] == packed) & in_range
    return idx, hits


def _gather_points_compacted(
    nb_vidx: torch.Tensor,
    nb_valid: torch.Tensor,
    voxel_point_starts_t: torch.Tensor,
    voxel_point_indices_t: torch.Tensor,
    points_t: torch.Tensor,
    surface_mask_t: torch.Tensor,
    take_surface: bool,
    cap: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather points from neighbor voxels, filter by surface/unassigned, compact to left.

    Returns:
        pts_out     : (B, max_kept, 3) float32 coords (zeros in padded slots)
        abs_idx_out : (B, max_kept) int64 absolute point indices (-1 in padded slots)
        counts      : (B,) int64 kept-count per row (clamped to cap if cap > 0)
    """
    B, K = nb_vidx.shape
    nb_starts = voxel_point_starts_t[nb_vidx]
    nb_ends = voxel_point_starts_t[nb_vidx + 1]
    nb_counts = torch.where(nb_valid, nb_ends - nb_starts, torch.zeros_like(nb_starts))

    nb_cum = torch.cat(
        [torch.zeros(B, 1, dtype=torch.int64, device=device),
         nb_counts.cumsum(dim=1)],
        dim=1,
    )  # (B, K+1)
    row_totals = nb_counts.sum(dim=1)  # (B,)
    max_row = int(row_totals.max().item())
    if max_row == 0:
        empty_pts = torch.zeros(B, 0, 3, dtype=torch.float32, device=device)
        empty_idx = torch.zeros(B, 0, dtype=torch.int64, device=device)
        return empty_pts, empty_idx, torch.zeros(B, dtype=torch.int64, device=device)

    s_grid = torch.arange(max_row, device=device).unsqueeze(0).expand(B, -1)
    k_per_slot = (torch.searchsorted(nb_cum, s_grid, right=True) - 1).clamp(min=0, max=K - 1)
    inner = s_grid - nb_cum.gather(1, k_per_slot)
    flat_pt_idx = (nb_starts.gather(1, k_per_slot) + inner).clamp(
        min=0, max=voxel_point_indices_t.numel() - 1
    )
    abs_pt_idx = voxel_point_indices_t[flat_pt_idx]
    valid_pad = s_grid < row_totals.unsqueeze(1)

    if take_surface:
        keep_mask = surface_mask_t[abs_pt_idx] & valid_pad
    else:
        keep_mask = (~surface_mask_t[abs_pt_idx]) & valid_pad

    # Stable sort: keep=True rows come first in each row.
    order = (~keep_mask).to(torch.int64).argsort(dim=1, stable=True)
    abs_idx_sorted = abs_pt_idx.gather(1, order)
    pts_sorted = points_t[abs_idx_sorted]  # (B, max_row, 3)

    counts = keep_mask.sum(dim=1)
    if cap > 0:
        counts = counts.clamp(max=cap)

    max_kept = int(counts.max().item())
    if max_kept == 0:
        empty_pts = torch.zeros(B, 0, 3, dtype=torch.float32, device=device)
        empty_idx = torch.zeros(B, 0, dtype=torch.int64, device=device)
        return empty_pts, empty_idx, counts

    abs_idx_out = abs_idx_sorted[:, :max_kept]
    pts_out = pts_sorted[:, :max_kept, :]
    valid_out = torch.arange(max_kept, device=device).unsqueeze(0) < counts.unsqueeze(1)
    abs_idx_out = torch.where(valid_out, abs_idx_out, torch.full_like(abs_idx_out, -1))
    pts_out = pts_out * valid_out.unsqueeze(-1).to(pts_out.dtype)
    return pts_out, abs_idx_out, counts


def _process_chunk_gpu(
    chunk: torch.Tensor,
    device: torch.device,
    V: int,
    unique_voxel_keys_t: torch.Tensor,
    unique_packed_t: torch.Tensor,
    voxel_point_starts_t: torch.Tensor,
    voxel_point_indices_t: torch.Tensor,
    point_voxel_idx_t: torch.Tensor,
    points_t: torch.Tensor,
    surface_mask_t: torch.Tensor,
    voxel_surface_counts_t: torch.Tensor,
    voxel_total_counts_t: torch.Tensor,
    is_surface_voxel_t: torch.Tensor,
    is_unassigned_voxel_t: torch.Tensor,
    neigh_offsets_27: torch.Tensor,
    radius_offsets: torch.Tensor,
    ransac_iterations: int,
    ransac_inlier_threshold: float,
    distance_threshold: float,
    min_voxel_surface_points: int,
    ref_normal_t: torch.Tensor,
    cos_threshold: float,
) -> int:
    B = int(chunk.numel())
    if B == 0:
        return 0

    # --- Surface points (self + 26 neighbors, surface-only, capped) ---
    surf_vidx, surf_valid = _resolve_neighbors(
        chunk, unique_voxel_keys_t, unique_packed_t, neigh_offsets_27, V
    )
    surf_padded_t, _, surf_cnt_t = _gather_points_compacted(
        surf_vidx, surf_valid,
        voxel_point_starts_t, voxel_point_indices_t, points_t, surface_mask_t,
        take_surface=True, cap=_SURFACE_POINTS_CAP, device=device,
    )

    # --- Candidate points ((2r+1)^3 neighbors, unassigned-only, uncapped) ---
    cand_vidx, cand_valid = _resolve_neighbors(
        chunk, unique_voxel_keys_t, unique_packed_t, radius_offsets, V
    )
    cand_padded_t, cand_idx_t, cand_cnt_t = _gather_points_compacted(
        cand_vidx, cand_valid,
        voxel_point_starts_t, voxel_point_indices_t, points_t, surface_mask_t,
        take_surface=False, cap=0, device=device,
    )

    if cand_cnt_t.numel() == 0 or int(cand_cnt_t.max().item()) == 0:
        return 0

    # --- Batched RANSAC (unchanged math, now fully on GPU) ---
    I = ransac_iterations
    max_s = int(surf_padded_t.shape[1])
    max_c = int(cand_padded_t.shape[1])
    if max_s < 3:
        return 0

    counts_float = surf_cnt_t.clamp(min=1).float().view(B, 1, 1)
    rand = torch.rand(B, I, 3, device=device)
    max_idx = (surf_cnt_t - 1).clamp(min=0).view(B, 1, 1)
    sample_idx = (rand * counts_float).long().clamp(max=max_idx)

    gather_idx = sample_idx.unsqueeze(-1).expand(B, I, 3, 3)
    surf_exp = surf_padded_t.unsqueeze(1).expand(B, I, max_s, 3)
    triples = torch.gather(surf_exp, 2, gather_idx)

    p0 = triples[:, :, 0]
    v1 = triples[:, :, 1] - p0
    v2 = triples[:, :, 2] - p0
    normals = torch.cross(v1, v2, dim=-1)
    norms = torch.norm(normals, dim=-1, keepdim=True)
    valid_iter = norms.squeeze(-1) > 1e-10
    normals_unit = normals / norms.clamp(min=1e-10)
    d = -(normals_unit * p0).sum(dim=-1)

    dists = torch.einsum('bsk,bik->bsi', surf_padded_t, normals_unit) + d.unsqueeze(1)
    dists = dists.abs()
    pts_valid = (
        torch.arange(max_s, device=device).unsqueeze(0) < surf_cnt_t.unsqueeze(1)
    ).unsqueeze(-1)
    inliers = ((dists < ransac_inlier_threshold) & pts_valid).sum(dim=1)
    inliers = torch.where(valid_iter, inliers, torch.full_like(inliers, -1))

    # Per-trial orientation gate (abs — plane normals have sign ambiguity).
    normal_dots = (normals_unit * ref_normal_t.view(1, 1, 3)).sum(dim=-1).abs()
    orientation_ok = normal_dots >= cos_threshold
    inliers = torch.where(orientation_ok, inliers, torch.full_like(inliers, -1))

    best_iter = inliers.argmax(dim=1)
    best_count = inliers.gather(1, best_iter.unsqueeze(1)).squeeze(1)
    best_normal = normals_unit.gather(
        1, best_iter.view(B, 1, 1).expand(B, 1, 3)
    ).squeeze(1)
    best_d = d.gather(1, best_iter.unsqueeze(1)).squeeze(1)

    valid_voxel = surf_cnt_t >= 3
    voxel_has_plane = (best_count > 0) & valid_voxel

    # --- Accept candidates ---
    cand_validity = cand_idx_t >= 0
    perp = (cand_padded_t * best_normal.unsqueeze(1)).sum(dim=-1) + best_d.unsqueeze(1)
    accept = (
        (perp.abs() < distance_threshold)
        & cand_validity
        & voxel_has_plane.unsqueeze(1)
    )

    accepted = cand_idx_t[accept]  # (K,) absolute indices
    if accepted.numel() == 0:
        return 0

    accepted_unique = torch.unique(accepted)
    newly_added_mask = ~surface_mask_t[accepted_unique]
    newly_added = accepted_unique[newly_added_mask]
    if newly_added.numel() == 0:
        return 0

    # --- Commit to GPU state ---
    surface_mask_t[newly_added] = True
    added_voxels = point_voxel_idx_t[newly_added]
    voxel_surface_counts_t.scatter_add_(
        0, added_voxels, torch.ones_like(newly_added)
    )
    changed_voxels = torch.unique(added_voxels)
    is_surface_voxel_t[changed_voxels] = is_surface_voxel_t[changed_voxels] | (
        voxel_surface_counts_t[changed_voxels] >= min_voxel_surface_points
    )
    is_unassigned_voxel_t[changed_voxels] = (
        voxel_surface_counts_t[changed_voxels] < voxel_total_counts_t[changed_voxels]
    )

    return int(newly_added.numel())
