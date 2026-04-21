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

import logging
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

logger = logging.getLogger(__name__)


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
                "label": "Max Normal Angle (deg)",
                "description": "Maximum angle (on |cos|) between (a) boundary-voxel and "
                               "candidate-voxel planes, and (b) optionally the reference "
                               "normal and each boundary plane. 90° disables the gate."
            },
            "use_ref_normal_gate": {
                "type": "bool",
                "default": True,
                "label": "Gate by Reference Normal",
                "description": "If enabled, boundary-voxel RANSAC planes must also "
                               "satisfy the angle threshold against the seed-cluster "
                               "reference normal."
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        logger.info("SurfaceRegionGrowingPlugin.execute() called")
        logger.debug(f"Params: {params}")

        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget

        # --- Validate: exactly one branch selected ---
        selected_branches = controller.selected_branches
        if not selected_branches:
            logger.warning("No branch selected")
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a cluster_labels branch.")
            return
        if len(selected_branches) > 1:
            logger.warning(f"Multiple branches selected: {len(selected_branches)}")
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE cluster_labels branch.")
            return

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)

        if node is None or node.data_type != "cluster_labels":
            logger.warning(
                f"Invalid branch: node={node}, "
                f"data_type={getattr(node, 'data_type', None)}"
            )
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Please select a cluster_labels branch "
                                "(output of DBSCAN or similar).")
            return

        # --- Validate: a point must be picked ---
        picked_indices = viewer_widget.picked_points_indices
        if not picked_indices:
            logger.warning("No picked point in viewer")
            QMessageBox.warning(main_window, "No Point Selected",
                                "Shift+click a point on the seed surface cluster "
                                "in the viewer, then run this plugin.")
            return

        # --- Determine seed cluster ID from picked point ---
        picked_idx = picked_indices[0]
        if picked_idx >= len(viewer_widget.points):
            logger.warning(
                f"Picked index {picked_idx} out of range "
                f"(viewer has {len(viewer_widget.points)} points)"
            )
            QMessageBox.warning(main_window, "Invalid Pick",
                                "Picked point index is out of range. "
                                "Please Shift+click again.")
            return

        picked_xyz = viewer_widget.points[picked_idx, :3].astype(np.float32)
        logger.debug(f"Picked point idx={picked_idx} xyz={picked_xyz.tolist()}")

        try:
            clusters_pc = controller.reconstruct(selected_uid)
        except Exception as e:
            logger.error(f"Failed to reconstruct clusters branch {selected_uid}: {e}")
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct clusters branch:\n{e}")
            return

        cluster_labels_attr = clusters_pc.get_attribute("cluster_labels")
        if cluster_labels_attr is None:
            logger.warning("Reconstructed clusters branch has no cluster_labels attribute")
            QMessageBox.warning(main_window, "No Cluster Labels",
                                "Reconstructed point cloud has no cluster_labels attribute.")
            return

        cluster_labels = cluster_labels_attr.astype(np.int32)

        kd = cKDTree(clusters_pc.points)
        _, local_idx = kd.query(picked_xyz)
        seed_cluster_id = int(cluster_labels[local_idx])
        seed_size = int((cluster_labels == seed_cluster_id).sum())
        logger.info(
            f"Seed cluster id={seed_cluster_id}, size={seed_size} points "
            f"(local_idx={int(local_idx)})"
        )

        if seed_cluster_id == -1:
            logger.warning("Picked point is noise (label -1)")
            QMessageBox.warning(main_window, "Noise Point Selected",
                                "The clicked point belongs to noise (label -1). "
                                "Please Shift+click a non-noise cluster point.")
            return

        viewer_widget.picked_points_indices.clear()
        viewer_widget.update()

        # --- Reconstruct parent PointCloud ---
        parent_uid = node.parent_uid
        if parent_uid is None:
            logger.warning("Clusters branch has no parent_uid")
            QMessageBox.warning(main_window, "No Parent",
                                "The clusters branch has no parent PointCloud node.")
            return

        parent_uid_str = str(parent_uid)
        parent_node = controller.get_node(parent_uid_str)
        if parent_node is None:
            logger.warning(f"Parent node not found for uid={parent_uid_str}")
            QMessageBox.warning(main_window, "Parent Not Found",
                                "Could not find the parent PointCloud node.")
            return

        try:
            full_pc = controller.reconstruct(parent_uid_str)
        except Exception as e:
            logger.error(f"Failed to reconstruct parent {parent_uid_str}: {e}")
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct parent point cloud:\n{e}")
            return

        full_points = full_pc.points.astype(np.float32)
        n_points = len(full_points)
        logger.info(f"Parent point cloud: {n_points:,} points")

        # cluster_labels is aligned to the clusters_pc (same parent), same N
        if len(cluster_labels) != n_points:
            logger.error(
                f"Shape mismatch: labels={len(cluster_labels)}, parent={n_points}"
            )
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
            logger.warning("Parent point cloud has no normals")
            QMessageBox.warning(main_window, "No Normals",
                                "Parent point cloud has no normals. Either import a "
                                "cloud with embedded normals (e.g. PLY nx/ny/nz), or "
                                "run the normal_estimation plugin first.")
            return

        anchor = np.asarray(parent_normals[local_idx], dtype=np.float32)
        anchor_n = np.linalg.norm(anchor)
        if anchor_n < 1e-10:
            logger.warning(f"Picked point normal has zero length (norm={anchor_n})")
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
        logger.info(
            f"Reference normal: {ref_normal.tolist()} "
            f"(from {len(seed_hits)} seed neighbors)"
        )

        voxel_size = float(params["voxel_size"])
        distance_threshold = float(params["distance_threshold"])
        ransac_iterations = int(params["ransac_iterations"])
        ransac_inlier_threshold = float(params["ransac_inlier_threshold"])
        min_voxel_surface_points = int(params["min_voxel_surface_points"])
        max_normal_angle_deg = float(params["max_normal_angle_deg"])
        use_ref_normal_gate = bool(params["use_ref_normal_gate"])

        logger.info(
            f"Starting region growing: "
            f"voxel_size={voxel_size}, "
            f"distance_threshold={distance_threshold}, "
            f"ransac_iterations={ransac_iterations}, "
            f"ransac_inlier_threshold={ransac_inlier_threshold}, "
            f"min_voxel_surface_points={min_voxel_surface_points}, "
            f"max_normal_angle_deg={max_normal_angle_deg}, "
            f"use_ref_normal_gate={use_ref_normal_gate}"
        )

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
            t_start = time.time()
            try:
                labels_out, stopped_early = _region_grow(
                    full_points,
                    cluster_labels,
                    seed_cluster_id,
                    voxel_size,
                    distance_threshold,
                    ransac_iterations,
                    ransac_inlier_threshold,
                    min_voxel_surface_points,
                    ref_normal,
                    max_normal_angle_deg,
                    use_ref_normal_gate,
                )
                clusters = Clusters(labels_out)
                clusters.set_random_color()
                state['result'] = clusters
                state['stopped_early'] = stopped_early
                elapsed = time.time() - t_start
                n_surface = int((labels_out == 0).sum())
                logger.info(
                    f"Region growing done in {elapsed:.2f}s: "
                    f"surface={n_surface:,}/{len(labels_out):,} "
                    f"({100.0 * n_surface / max(len(labels_out), 1):.2f}%), "
                    f"stopped_early={stopped_early}"
                )
            except Exception as e:
                logger.exception(f"Region growing thread failed: {e}")
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
            logger.error(f"Region growing failed: {state['error']}")
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

        logger.info(f"Result branch added: uid={result_uid}")

        if state['stopped_early']:
            logger.info("Region growing was cancelled by user")
            QMessageBox.information(
                main_window, "Cancelled",
                "Region growing stopped early. Partial result saved as a new branch."
            )


# ---------------------------------------------------------------------------
# Region growing implementation (GPU-resident voxel grid, batched RANSAC)
# ---------------------------------------------------------------------------

# Per candidate voxel, we retry RANSAC up to this many times when the
# candidate-plane/boundary-plane angle gate fails. Implemented as
# RETRIES × ransac_iterations total iterations with an angle-gate filter.
_CANDIDATE_RANSAC_TRIES = 3

# Max points per candidate voxel fed to RANSAC. The final surface-point
# filter still uses ALL points in the voxel — this cap only limits the
# point set RANSAC samples triples from and evaluates inliers against.
_CANDIDATE_RANSAC_POINTS_CAP = 256

_BOUNDARY_CHUNK = 1024
# After flattening to (boundary, candidate) pairs, process this many per sub-batch.
_CANDIDATE_CHUNK = 1024
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
    distance_threshold: float,
    ransac_iterations: int,
    ransac_inlier_threshold: float,
    min_voxel_surface_points: int,
    ref_normal: np.ndarray,
    max_normal_angle_deg: float,
    use_ref_normal_gate: bool,
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

    # Safety bound: leave room for ±1 neighbor offsets staying in [0, 2^21).
    span_limit = _BIT_AXIS_MAX - 1
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

    logger.info(
        f"Voxel grid: {V:,} voxels from {n:,} points, "
        f"initial surface voxels={int(is_surface_voxel.sum()):,}, "
        f"initial seed points={int(surface_mask_np.sum()):,}"
    )

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
    # Voxels that have already served as a boundary voxel — excluded from
    # future candidate selection so we don't re-fit the same neighborhood.
    is_processed_voxel_t = torch.zeros(V, dtype=torch.bool, device=device)
    ref_normal_t = torch.from_numpy(np.asarray(ref_normal, dtype=np.float32)).to(device)
    vk_min_t = torch.from_numpy(np.asarray(vk_min, dtype=np.float64)).to(device).to(torch.float32)
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

    cancel_event = global_variables.global_cancel_event
    stopped_early = False
    iteration = 0

    while True:
        if cancel_event.is_set():
            stopped_early = True
            logger.info(f"Cancel requested at iteration {iteration}")
            break

        # --- Find boundary voxels: unprocessed surface voxels with at least
        #     one unassigned neighbor in the 26-neighborhood. ---
        eligible = is_surface_voxel_t & ~is_processed_voxel_t
        surface_vidx = torch.nonzero(eligible, as_tuple=False).squeeze(1)
        if surface_vidx.numel() == 0:
            logger.info("No unprocessed surface voxels remain; terminating")
            break

        nb_idx_all, nb_hits_all = _resolve_neighbors(
            surface_vidx, unique_voxel_keys_t, unique_packed_t,
            neigh_offsets_26, V,
        )
        has_unassigned_nb = (nb_hits_all & is_unassigned_voxel_t[nb_idx_all]).any(dim=1)
        boundary_vidx_t = surface_vidx[has_unassigned_nb]

        if boundary_vidx_t.numel() == 0:
            logger.info(f"No boundary voxels at iteration {iteration}; terminating")
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
                is_processed_voxel_t,
                neigh_offsets_26,
                vk_min_t, voxel_size,
                ransac_iterations, ransac_inlier_threshold, distance_threshold,
                min_voxel_surface_points, ref_normal_t, cos_threshold,
                use_ref_normal_gate,
            )
            # Mark this chunk's boundary voxels as processed so they're not
            # revisited on future iterations.
            is_processed_voxel_t[chunk] = True

        if stopped_early:
            break

        iteration += 1
        total_surface = int(surface_mask_t.sum().item())
        logger.info(
            f"Iter {iteration}: boundary_voxels={B_total:,}, "
            f"newly_added={newly_added_total:,}, total_surface={total_surface:,}"
        )
        global_variables.global_progress = (
            None,
            f"Region growing — iter {iteration}, surface: {total_surface:,}, "
            f"+{newly_added_total}"
        )

        if newly_added_total == 0:
            logger.info(f"Converged at iteration {iteration} (no new points added)")
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


def _gather_all_points_in_voxels(
    vidx: torch.Tensor,
    voxel_point_starts_t: torch.Tensor,
    voxel_point_indices_t: torch.Tensor,
    points_t: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each voxel in `vidx` (shape (C,)), gather ALL its points (no filter).

    Returns:
        pts      : (C, max_p, 3) float32 — padded with zeros beyond per-row count
        abs_idx  : (C, max_p) int64     — absolute point indices; -1 in padded slots
        counts   : (C,) int64           — per-voxel point count
    """
    C = int(vidx.numel())
    starts = voxel_point_starts_t[vidx]          # (C,)
    ends = voxel_point_starts_t[vidx + 1]        # (C,)
    counts = ends - starts
    if C == 0 or int(counts.max().item()) == 0:
        empty_pts = torch.zeros(C, 0, 3, dtype=points_t.dtype, device=device)
        empty_idx = torch.zeros(C, 0, dtype=torch.int64, device=device)
        return empty_pts, empty_idx, counts.to(torch.int64)

    max_p = int(counts.max().item())
    col = torch.arange(max_p, device=device).unsqueeze(0)      # (1, max_p)
    valid = col < counts.unsqueeze(1)                           # (C, max_p)
    flat = (starts.unsqueeze(1) + col).clamp(
        min=0, max=voxel_point_indices_t.numel() - 1
    )
    abs_idx = voxel_point_indices_t[flat]
    abs_idx = torch.where(valid, abs_idx, torch.full_like(abs_idx, -1))
    pts = points_t[abs_idx.clamp(min=0)]
    pts = pts * valid.unsqueeze(-1).to(pts.dtype)
    return pts, abs_idx, counts.to(torch.int64)


def _batched_ransac(
    pts: torch.Tensor,         # (N, max_p, 3) candidate points per row (zeros in padding)
    counts: torch.Tensor,      # (N,) valid point count per row
    iterations: int,
    inlier_threshold: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched RANSAC over `iterations` trials per row.

    Returns per-trial tensors (the caller picks best):
        normals_unit : (N, I, 3)
        d            : (N, I)
        inliers      : (N, I) int64 — -1 for degenerate / out-of-range iterations
        iter_valid   : (N, I) bool
    """
    N = int(pts.shape[0])
    max_p = int(pts.shape[1])
    I = iterations

    counts_f = counts.clamp(min=1).to(torch.float32).view(N, 1, 1)
    rand = torch.rand(N, I, 3, device=device)
    max_idx = (counts - 1).clamp(min=0).view(N, 1, 1)
    sample_idx = (rand * counts_f).long().clamp(max=max_idx)

    gather_idx = sample_idx.unsqueeze(-1).expand(N, I, 3, 3)
    pts_exp = pts.unsqueeze(1).expand(N, I, max_p, 3)
    triples = torch.gather(pts_exp, 2, gather_idx)

    p0 = triples[:, :, 0]
    v1 = triples[:, :, 1] - p0
    v2 = triples[:, :, 2] - p0
    normals = torch.cross(v1, v2, dim=-1)
    mag = torch.norm(normals, dim=-1, keepdim=True)
    iter_valid = mag.squeeze(-1) > 1e-10
    normals_unit = normals / mag.clamp(min=1e-10)
    d = -(normals_unit * p0).sum(dim=-1)

    dists = torch.einsum('npk,nik->npi', pts, normals_unit) + d.unsqueeze(1)
    pts_valid = (
        torch.arange(max_p, device=device).unsqueeze(0) < counts.unsqueeze(1)
    ).unsqueeze(-1)
    inliers = ((dists.abs() < inlier_threshold) & pts_valid).sum(dim=1)
    inliers = torch.where(iter_valid, inliers, torch.full_like(inliers, -1))

    # Enough valid points to even form a plane?
    row_ok = counts >= 3
    inliers = torch.where(row_ok.unsqueeze(1), inliers, torch.full_like(inliers, -1))
    return normals_unit, d, inliers, iter_valid


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
    is_processed_voxel_t: torch.Tensor,
    neigh_offsets_26: torch.Tensor,
    vk_min_t: torch.Tensor,
    voxel_size: float,
    ransac_iterations: int,
    ransac_inlier_threshold: float,
    distance_threshold: float,
    min_voxel_surface_points: int,
    ref_normal_t: torch.Tensor,
    cos_threshold: float,
    use_ref_normal_gate: bool,
) -> int:
    B = int(chunk.numel())
    if B == 0:
        return 0
    I = ransac_iterations

    # -------------------------------------------------------------------
    # Step 1: Boundary-voxel RANSAC — surface points WITHIN each boundary
    # voxel (no neighbor expansion).
    # -------------------------------------------------------------------
    b_self_vidx = chunk.view(B, 1)
    b_self_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
    b_surf_pts, _, b_surf_cnt = _gather_points_compacted(
        b_self_vidx, b_self_valid,
        voxel_point_starts_t, voxel_point_indices_t, points_t, surface_mask_t,
        take_surface=True, cap=0, device=device,
    )
    if b_surf_pts.shape[1] < 3:
        return 0

    b_normals, b_d_all, b_inliers, _ = _batched_ransac(
        b_surf_pts, b_surf_cnt, I, ransac_inlier_threshold, device
    )

    # Skip voxels that don't meet the minimum surface-point count.
    enough_pts = b_surf_cnt >= min_voxel_surface_points
    b_inliers = torch.where(
        enough_pts.unsqueeze(1), b_inliers, torch.full_like(b_inliers, -1)
    )

    # Optional reference-normal gate.
    if use_ref_normal_gate:
        ref_dot = (b_normals * ref_normal_t.view(1, 1, 3)).sum(dim=-1).abs()
        ref_ok = ref_dot >= cos_threshold
        b_inliers = torch.where(ref_ok, b_inliers, torch.full_like(b_inliers, -1))

    best_b_iter = b_inliers.argmax(dim=1)
    best_b_count = b_inliers.gather(1, best_b_iter.unsqueeze(1)).squeeze(1)
    best_b_normal = b_normals.gather(
        1, best_b_iter.view(B, 1, 1).expand(B, 1, 3)
    ).squeeze(1)
    best_b_d = b_d_all.gather(1, best_b_iter.unsqueeze(1)).squeeze(1)
    boundary_has_plane = best_b_count > 0

    if not bool(boundary_has_plane.any()):
        return 0

    # -------------------------------------------------------------------
    # Step 2: Enumerate 26 neighbors of each boundary voxel; filter to
    # unprocessed voxels that the boundary plane actually crosses.
    # -------------------------------------------------------------------
    nb_idx, nb_hits = _resolve_neighbors(
        chunk, unique_voxel_keys_t, unique_packed_t, neigh_offsets_26, V,
    )  # (B, 26)
    nb_unprocessed = ~is_processed_voxel_t[nb_idx]
    nb_available = nb_hits & nb_unprocessed

    # Voxel centers in world coords: (shifted_key + vk_min + 0.5) * voxel_size.
    nb_centers = (
        unique_voxel_keys_t[nb_idx].to(torch.float32)
        + vk_min_t.view(1, 1, 3)
        + 0.5
    ) * voxel_size
    n_dot_c = (
        nb_centers * best_b_normal.view(B, 1, 3)
    ).sum(dim=-1) + best_b_d.view(B, 1)
    # Exact plane/AABB test: a plane n·x + d = 0 crosses an axis-aligned
    # cube of half-width h centered at c iff |n·c + d| ≤ h * (|nx|+|ny|+|nz|).
    n_l1 = best_b_normal.abs().sum(dim=-1)  # (B,)
    half_extent = 0.5 * voxel_size * n_l1   # (B,)
    plane_crosses = n_dot_c.abs() < half_extent.view(B, 1)
    is_cand_pair = (
        nb_available & plane_crosses & boundary_has_plane.view(B, 1)
    )

    if not bool(is_cand_pair.any()):
        return 0

    # Flatten to (boundary, candidate) pairs and process in sub-batches to
    # bound peak GPU memory (RANSAC tensors scale as C × max_p × 3·I).
    bi_full, ki_full = torch.nonzero(is_cand_pair, as_tuple=True)
    cand_vidx_full = nb_idx[bi_full, ki_full]
    cand_b_normal_full = best_b_normal[bi_full]
    cand_b_d_full = best_b_d[bi_full]
    C_total = int(cand_vidx_full.numel())

    accepted_chunks = []
    total_iters = I * _CANDIDATE_RANSAC_TRIES

    for c_start in range(0, C_total, _CANDIDATE_CHUNK):
        c_end = min(c_start + _CANDIDATE_CHUNK, C_total)
        cand_vidx = cand_vidx_full[c_start:c_end]
        cand_b_normal = cand_b_normal_full[c_start:c_end]
        cand_b_d = cand_b_d_full[c_start:c_end]
        C = int(cand_vidx.numel())

        # --- Step 3: gather ALL points in each candidate voxel ---
        cand_all_pts, cand_all_idx, _cand_all_cnt = _gather_all_points_in_voxels(
            cand_vidx, voxel_point_starts_t, voxel_point_indices_t, points_t, device,
        )
        if cand_all_pts.shape[1] == 0:
            continue
        max_p = int(cand_all_pts.shape[1])
        cand_slot_valid = cand_all_idx >= 0

        perp_b = (
            cand_all_pts * cand_b_normal.view(C, 1, 3)
        ).sum(dim=-1) + cand_b_d.view(C, 1)
        is_cand_pt = (perp_b.abs() < distance_threshold) & cand_slot_valid

        # Compact candidate points to the left and cap for RANSAC.
        order = (~is_cand_pt).to(torch.int64).argsort(dim=1, stable=True)
        pts_sorted = cand_all_pts.gather(
            1, order.unsqueeze(-1).expand(C, max_p, 3)
        )
        cand_pt_cnt = is_cand_pt.sum(dim=1).clamp(max=_CANDIDATE_RANSAC_POINTS_CAP)
        max_cand = int(cand_pt_cnt.max().item()) if C > 0 else 0
        if max_cand < 3:
            continue
        cand_pts = pts_sorted[:, :max_cand, :]
        cand_valid_pad = (
            torch.arange(max_cand, device=device).unsqueeze(0)
            < cand_pt_cnt.unsqueeze(1)
        )
        cand_pts = cand_pts * cand_valid_pad.unsqueeze(-1).to(cand_pts.dtype)

        # --- Step 4: candidate-voxel RANSAC with angle gate ---
        c_normals, c_d_all, c_inliers, _ = _batched_ransac(
            cand_pts, cand_pt_cnt, total_iters, ransac_inlier_threshold, device,
        )
        ang_dot = (c_normals * cand_b_normal.view(C, 1, 3)).sum(dim=-1).abs()
        angle_ok = ang_dot >= cos_threshold
        c_inliers = torch.where(
            angle_ok, c_inliers, torch.full_like(c_inliers, -1)
        )

        best_c_iter = c_inliers.argmax(dim=1)
        best_c_count = c_inliers.gather(1, best_c_iter.unsqueeze(1)).squeeze(1)
        best_c_normal = c_normals.gather(
            1, best_c_iter.view(C, 1, 1).expand(C, 1, 3)
        ).squeeze(1)
        best_c_d = c_d_all.gather(1, best_c_iter.unsqueeze(1)).squeeze(1)
        candidate_has_plane = best_c_count > 0
        if not bool(candidate_has_plane.any()):
            continue

        # --- Step 5: surface points = ALL points in the candidate voxel
        # within distance_threshold of the candidate plane ---
        perp_c = (
            cand_all_pts * best_c_normal.view(C, 1, 3)
        ).sum(dim=-1) + best_c_d.view(C, 1)
        is_surface_pt = (
            (perp_c.abs() < distance_threshold)
            & cand_slot_valid
            & candidate_has_plane.view(C, 1)
        )
        accepted_chunks.append(cand_all_idx[is_surface_pt])

        # Free large intermediates before the next sub-batch.
        del cand_all_pts, cand_all_idx, cand_slot_valid, perp_b, is_cand_pt
        del pts_sorted, cand_pts, c_normals, c_d_all, c_inliers, perp_c

    if not accepted_chunks:
        return 0
    accepted = torch.cat(accepted_chunks)
    if accepted.numel() == 0:
        return 0

    accepted_unique = torch.unique(accepted)
    newly = accepted_unique[~surface_mask_t[accepted_unique]]
    if newly.numel() == 0:
        return 0

    # --- Commit to GPU state ---
    surface_mask_t[newly] = True
    added_voxels = point_voxel_idx_t[newly]
    voxel_surface_counts_t.scatter_add_(
        0, added_voxels, torch.ones_like(newly)
    )
    changed_voxels = torch.unique(added_voxels)
    # A voxel is a "surface voxel" (and hence a future boundary candidate)
    # as soon as it contains any surface point.
    is_surface_voxel_t[changed_voxels] = True
    is_unassigned_voxel_t[changed_voxels] = (
        voxel_surface_counts_t[changed_voxels] < voxel_total_counts_t[changed_voxels]
    )

    return int(newly.numel())
