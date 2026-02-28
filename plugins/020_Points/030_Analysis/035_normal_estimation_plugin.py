"""
Plugin for estimating point cloud normals.

Computes surface normals for each point using local neighborhood analysis.
Supports hybrid KNN + radius search and multiple orientation methods
(viewpoint, custom point, MST-based consistent orientation).

Also produces an eigenvalues branch (same as compute_eigenvalues plugin)
since eigenvalues are always useful for downstream analysis.
"""

import threading
from typing import Dict, Any
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from core.entities.normals import Normals
from core.entities.eigenvalues import Eigenvalues


class NormalEstimationPlugin(ActionPlugin):
    """
    Plugin for estimating surface normals from local point neighborhoods.

    Uses Open3D's normal estimation (GPU or CPU backend selected automatically).
    The result is stored as a Normals branch with normal directions visualized
    as RGB colors, plus an Eigenvalues branch for downstream analysis.
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
            "orientation": {
                "type": "choice",
                "options": [
                    "Towards Viewpoint (0,0,0)",
                    "Towards Custom Point",
                    "Consistent (MST)",
                    "None"
                ],
                "default": "Towards Viewpoint (0,0,0)",
                "label": "Normal Orientation",
                "description": "Method to orient normals consistently"
            },
            "orient_x": {
                "type": "float",
                "default": 0.0,
                "min": -100000.0,
                "max": 100000.0,
                "label": "Orientation Point X",
                "description": "X coordinate for custom orientation point (only used with 'Towards Custom Point')"
            },
            "orient_y": {
                "type": "float",
                "default": 0.0,
                "min": -100000.0,
                "max": 100000.0,
                "label": "Orientation Point Y",
                "description": "Y coordinate for custom orientation point"
            },
            "orient_z": {
                "type": "float",
                "default": 0.0,
                "min": -100000.0,
                "max": 100000.0,
                "label": "Orientation Point Z",
                "description": "Z coordinate for custom orientation point"
            },
            "target_batch_size": {
                "type": "int",
                "default": 250000,
                "min": 10000,
                "max": 1000000,
                "label": "Batch Size",
                "description": "Number of points per batch. Smaller values use less memory."
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
        orientation = params["orientation"]
        batch_size = params.get("target_batch_size", 250000)

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

        backend = global_variables.global_backend_registry.get_normal_estimation()

        def _compute():
            """Heavy computation in background thread."""
            try:
                # Step 1: Estimate normals via backend
                global_variables.global_progress = (
                    None, f"Estimating normals (k={k}, {point_cloud.size:,} points)..."
                )
                normals_array = backend.estimate_normals(
                    point_cloud.points, k, max_radius, batch_size=batch_size
                )

                # Step 2: Orient normals
                global_variables.global_progress = (50, "Orienting normals...")
                if orientation == "Towards Viewpoint (0,0,0)":
                    normals_array = _orient_towards_point(
                        normals_array, point_cloud.points, np.array([0.0, 0.0, 0.0])
                    )
                elif orientation == "Towards Custom Point":
                    target = np.array([
                        params["orient_x"], params["orient_y"], params["orient_z"]
                    ], dtype=np.float32)
                    normals_array = _orient_towards_point(
                        normals_array, point_cloud.points, target
                    )
                elif orientation == "Consistent (MST)":
                    normals_array = _orient_mst(point_cloud.points, normals_array, k)

                # Step 3: Compute eigenvalues
                global_variables.global_progress = (
                    70, f"Computing eigenvalues (k={k}, {point_cloud.size:,} points)..."
                )
                from core.entities.point_cloud import _get_eigenvalue_utils
                analyser = _get_eigenvalue_utils()
                eigenvalues_array = analyser.compute_eigenvalues(
                    point_cloud.points, k=k
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


def _orient_towards_point(
    normals: np.ndarray, points: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Flip normals to face a target point. Fully vectorized."""
    direction = target - points  # (N, 3)
    dot_products = np.sum(normals * direction, axis=1)  # (N,)
    flip_mask = dot_products < 0
    normals[flip_mask] *= -1
    return normals


def _orient_mst(
    points: np.ndarray, normals: np.ndarray, k: int
) -> np.ndarray:
    """Orient normals consistently using Open3D MST propagation."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.orient_normals_consistent_tangent_plane(k)
    return np.asarray(pcd.normals, dtype=np.float32)
