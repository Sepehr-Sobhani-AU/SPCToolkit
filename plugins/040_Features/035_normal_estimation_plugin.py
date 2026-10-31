"""
Plugin for estimating point cloud normals.

Computes surface normals for each point using local neighborhood analysis.
The KNN query and eigendecomposition run per spatial tile through the
constant-cube BatchProcessor, so peak memory stays bounded to a single tile
and the plugin scales to very large clouds (tens of millions of points) on
ordinary hardware — instead of materializing the full (N, k) neighbor arrays.

Normals are oriented parallel to +z (each normal flipped so its z-component is
non-negative). Richer orientation (viewpoint / MST) is deferred to a later step.

Also produces an eigenvalues branch since eigenvalues are always useful for
downstream analysis. Both are extracted from a single eigendecomposition pass —
one call to torch.linalg.eigh() yields both eigenvectors (normals) and
eigenvalues.
"""

import threading
from typing import Dict, Any
import numpy as np
import torch
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from core.entities.normals import Normals
from core.entities.eigenvalues import Eigenvalues
from core.services.batch_processor import BatchProcessor


class NormalEstimationPlugin(ActionPlugin):
    """
    Plugin for estimating surface normals from local point neighborhoods.

    Each spatial tile runs one KNN query + one batched eigendecomposition on
    GPU. torch.linalg.eigh() returns both eigenvalues and eigenvectors — the
    smallest eigenvector is the surface normal, and all three eigenvalues
    characterize local surface geometry. Normals are oriented parallel to +z.
    """

    def get_name(self) -> str:
        return "estimate_normals"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "k_neighbors": {
                "type": "int",
                "default": 30,
                "min": 5,
                "max": 200,
                "label": "Number of Neighbors (k)",
                "description": "Maximum number of nearest neighbors for normal estimation"
            },
            "max_radius": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 100000.0,
                "label": "Maximum Search Radius",
                "description": "Maximum radius for neighbor search (0 = KNN only, no radius limit)"
            },
            "target_points": {
                "type": "int",
                "default": 200000,
                "min": 1000,
                "max": 5000000,
                "label": "Target Points per Tile",
                "description": "Max points processed per spatial tile. The cloud is split "
                               "adaptively so every tile stays under this — dense areas split "
                               "into more tiles, sparse areas merge into fewer. May be reduced "
                               "automatically for large k to stay within GPU memory."
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        tree_widget = global_variables.global_tree_structure_widget

        # --- Validate: one branch selected (fast, main thread) ---
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a PointCloud branch first.")
            return
        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch at a time.")
            return

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)
        if node is None:
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Could not find the selected branch.")
            return

        # Reconstruct point cloud (fast if cached)
        point_cloud: PointCloud = controller.reconstruct(selected_uid)
        if point_cloud is None:
            QMessageBox.warning(main_window, "Reconstruction Failed",
                                "Could not reconstruct point cloud from selected branch.")
            return

        k = params["k_neighbors"]
        max_radius = params["max_radius"]
        target_points = params.get("target_points", 200000)

        # max_radius=0 means pure KNN (no radius limit)
        if max_radius <= 0:
            max_radius = float('inf')

        # --- Disable UI and show progress ---
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.show_progress(
            f"Estimating normals (k={k}, {point_cloud.size:,} points)..."
        )

        # --- Shared state for thread communication (GIL-safe) ---
        state = {
            'normals': None,
            'eigenvalues': None,
            'error': None,
            'done': False,
        }

        def _compute():
            """Tiled normals + eigenvalues via the adaptive BatchProcessor.

            Peak memory stays bounded to one spatial tile, so this scales to
            very large clouds. Normals are oriented parallel to +z.
            """
            try:
                # Free cached VRAM before heavy GPU work
                torch.cuda.empty_cache()
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except ImportError:
                    pass

                points = point_cloud.points.astype(np.float32)

                def _report(done, total):
                    percent = int((done / total) * 85)
                    global_variables.global_progress = (
                        percent, f"Estimating normals: tile {done}/{total}"
                    )

                normals_array, eigenvalues_array = _estimate_normals_batched(
                    points, k=k, max_radius=max_radius,
                    target_points=target_points, on_progress=_report
                )

                state['normals'] = normals_array
                state['eigenvalues'] = eigenvalues_array
            except Exception as e:
                state['error'] = str(e)
            finally:
                state['done'] = True

        # --- Launch background thread ---
        thread = threading.Thread(target=_compute, daemon=True)
        thread.start()

        # --- Poll for completion (processEvents keeps UI responsive) ---
        import time
        while not state['done']:
            percent, msg = global_variables.global_progress
            if msg:
                main_window.show_progress(msg, percent)
            QtWidgets.QApplication.processEvents()
            time.sleep(0.1)

        global_variables.global_progress = (None, "")

        # --- Error handling ---
        if state['error']:
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()
            QMessageBox.critical(
                main_window, "Normal Estimation Failed", state['error']
            )
            return

        # --- Add branches (main thread, Qt-safe) ---
        main_window.show_progress("Adding results...", 90)

        tree_widget.blockSignals(True)

        normals = Normals(state['normals'])
        parent_uid_str = str(node.uid)

        normals_uid = controller.add_analysis_result(
            normals, "normals", [node.uid], node, "estimate_normals", params
        )
        tree_widget.add_branch(
            normals_uid, parent_uid_str,
            "estimate_normals", tooltip=f"estimate_normals,{params}"
        )

        eigenvalues = Eigenvalues(state['eigenvalues'])

        eigen_uid = controller.add_analysis_result(
            eigenvalues, "eigenvalues", [node.uid], node,
            "compute_eigenvalues", params
        )
        tree_widget.add_branch(
            eigen_uid, parent_uid_str,
            "compute_eigenvalues", tooltip=f"compute_eigenvalues,{params}"
        )

        # Set visibility: hide parent, show normals, hide eigenvalues
        input_item = tree_widget.branches_dict.get(selected_uid)
        if input_item:
            input_item.setCheckState(0, Qt.Unchecked)
            tree_widget.visibility_status[selected_uid] = False

        tree_widget.visibility_status[normals_uid] = True

        eigen_item = tree_widget.branches_dict.get(eigen_uid)
        if eigen_item:
            eigen_item.setCheckState(0, Qt.Unchecked)
            tree_widget.visibility_status[eigen_uid] = False

        tree_widget.blockSignals(False)

        # Single render with correct visibility
        main_window.render_visible_data(zoom_extent=False)

        # Re-enable UI
        main_window.clear_progress()
        main_window.enable_menus()
        main_window.enable_tree()


def _estimate_normals_batched(
    points: np.ndarray,
    k: int,
    max_radius: float,
    target_points: int = 200000,
    smooth: bool = True,
    on_progress=None,
):
    """Compute normals + eigenvalues per spatial tile (memory-bounded).

    The cloud is partitioned into adaptive tiles (each <= the point budget) by
    BatchProcessor. Each tile runs its own KNN + eigendecomposition over its
    (primary + halo) points; results are written back for the tile's primary
    points only. Peak memory is one tile's neighbor array, never the global
    (N, k) array.

    Normals are oriented parallel to +z. Returns (normals (N,3), eigenvalues (N,3)).
    """
    points = points.astype(np.float32, copy=False)
    n_points = len(points)

    normals_array = np.zeros((n_points, 3), dtype=np.float32)
    eigenvalues_array = np.zeros((n_points, 3), dtype=np.float32)
    if n_points == 0:
        return normals_array, eigenvalues_array

    use_radius = np.isfinite(max_radius) and max_radius > 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    registry = global_variables.global_backend_registry
    knn_backend = registry.get_knn() if registry is not None else None

    # total_cap bounds the KNN batch (primary + halo) so a single tile can't exhaust
    # VRAM: it shrinks with k so the cuML KNN output (~batch*k) and the
    # eigendecomposition stay bounded. The primary budget is set a bit below the cap
    # so every tile keeps room for a halo within it; get_batch_for_grid_cell trims the
    # halo to the nearest points when a sparse tile borders a denser one.
    mem_cap = max(20000, 20_000_000 // max(k, 1))
    total_cap = min(target_points, mem_cap, n_points)
    primary_budget = max(1000, (total_cap * 4) // 5)
    processor = BatchProcessor(
        points=points,
        batch_size=primary_budget,
        overlap_percent=0.1,
        max_batch_size=total_cap,
    )
    unique_cells = np.unique(processor.grid_indices)
    total_cells = len(unique_cells)

    def _tile_knn(tile_points, kk):
        """Return (distances, indices) of kk neighbors within a single tile."""
        if knn_backend is not None:
            d, idx = knn_backend.query(tile_points, k=kk)
            return d.astype(np.float32), idx.astype(np.int64)
        from scipy.spatial import KDTree
        d, idx = KDTree(tile_points).query(tile_points, k=kk)
        d = np.atleast_2d(d).astype(np.float32)
        idx = np.atleast_2d(idx).astype(np.int64)
        return d, idx

    with torch.no_grad():
        for cell_count, cell_idx in enumerate(unique_cells, start=1):
            if global_variables.global_cancel_event.is_set():
                raise InterruptedError("Cancelled by user")

            batch_indices, is_primary = processor.get_batch_for_grid_cell(cell_idx)
            if len(batch_indices) == 0:
                continue

            tile_points = points[batch_indices]
            m = len(tile_points)
            kk = int(min(k, m))

            # Release torch's cached VRAM so the cuML KNN allocator (RMM) has room.
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            distances, neighbor_idx = _tile_knn(tile_points, kk)

            tile_torch = torch.from_numpy(tile_points).to(device)
            idx_torch = torch.from_numpy(neighbor_idx).to(device)
            dist_torch = torch.from_numpy(distances).to(device) if use_radius else None

            tile_normals = torch.empty((m, 3), dtype=torch.float32, device=device)
            tile_evals = torch.empty((m, 3), dtype=torch.float32, device=device)

            # Sub-batch the eigendecomposition: gathering (m, kk, 3) neighbors at once
            # can be gigabytes for large k, so process the tile's points in chunks.
            sub = max(1, 2_000_000 // max(kk, 1))
            for start in range(0, m, sub):
                end = min(start + sub, m)
                neighbors = tile_torch[idx_torch[start:end]]  # (chunk, kk, 3)

                if use_radius:
                    radius_mask = dist_torch[start:end] < max_radius
                    valid_counts = radius_mask.sum(dim=1)
                    fallback = valid_counts < 3
                    radius_mask[fallback] = True
                    valid_counts = torch.where(
                        fallback, torch.full_like(valid_counts, kk), valid_counts
                    ).float().unsqueeze(-1).unsqueeze(-1)
                    weights = radius_mask.float().unsqueeze(-1)
                    centroids = (neighbors * weights).sum(dim=1, keepdim=True) / valid_counts
                    centered = (neighbors - centroids) * weights
                    cov = torch.bmm(centered.transpose(1, 2), centered) / valid_counts.clamp(min=1)
                else:
                    centroids = neighbors.mean(dim=1, keepdim=True)
                    centered = neighbors - centroids
                    cov = torch.bmm(centered.transpose(1, 2), centered) / float(kk)

                evals_chunk, evecs_chunk = torch.linalg.eigh(cov)
                tile_evals[start:end] = evals_chunk

                # Normal = eigenvector of smallest eigenvalue, oriented parallel to +z
                normals_chunk = evecs_chunk[:, :, 0]
                normals_chunk = normals_chunk / \
                    torch.linalg.norm(normals_chunk, dim=1, keepdim=True).clamp(min=1e-8)
                flip = normals_chunk[:, 2] < 0
                normals_chunk[flip] = -normals_chunk[flip]
                tile_normals[start:end] = normals_chunk

            # Smooth eigenvalues within the tile (sub-batched neighbor gather)
            if smooth:
                smoothed = torch.empty_like(tile_evals)
                for start in range(0, m, sub):
                    end = min(start + sub, m)
                    smoothed[start:end] = tile_evals[idx_torch[start:end]].mean(dim=1)
                tile_evals = smoothed

            primary_mask = torch.from_numpy(is_primary).to(device)
            primary_global = batch_indices[is_primary]
            normals_array[primary_global] = tile_normals[primary_mask].cpu().numpy()
            eigenvalues_array[primary_global] = tile_evals[primary_mask].cpu().numpy()

            if on_progress is not None:
                on_progress(cell_count, total_cells)

    return normals_array, eigenvalues_array
