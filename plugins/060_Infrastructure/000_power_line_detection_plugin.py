"""
Semi-automated power line (cable) detection plugin.

Workflow:
1. User selects a PointCloud branch and polygon-selects seed points on cables
2. DBSCAN groups picked points into per-cable seed clusters
3. Each seed cluster is traced in both directions using RANSAC + cylindrical region growing
4. Results are added as a Clusters child node with one cluster per cable
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.clusters import Clusters
from core.entities.masks import Masks
from core.entities.point_cloud import PointCloud
from core.services.power_line_tracer import PowerLineTracer


class PowerLineDetectionPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "power_line_detection"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "dbscan_eps": {
                "type": "float",
                "default": 0.10,
                "min": 0.001,
                "max": 20.0,
                "label": "Seed DBSCAN Eps",
                "description": "Max distance between picked points on the same cable (m)",
            },
            "dbscan_min_samples": {
                "type": "int",
                "default": 10,
                "min": 2,
                "max": 50,
                "label": "Seed DBSCAN Min Samples",
                "description": "Minimum picked points to form one cable seed",
            },
            "cylinder_radius": {
                "type": "float",
                "default": 0.03,
                "min": 0.001,
                "max": 5.0,
                "label": "Cylinder Radius",
                "description": "Search cylinder radius per growth step (m)",
            },
            "cylinder_length": {
                "type": "float",
                "default": 0.5,
                "min": 0.01,
                "max": 50.0,
                "label": "Cylinder Length",
                "description": "Search cylinder length per growth step (m)",
            },
            "min_points": {
                "type": "int",
                "default": 5,
                "min": 2,
                "max": 100,
                "label": "Min Points",
                "description": "Stop tracing if fewer points found in a cylinder",
            },
            "max_angle": {
                "type": "float",
                "default": 20.0,
                "min": 1.0,
                "max": 90.0,
                "label": "Max Angle (deg)",
                "description": "Max direction change per step — larger angles stop tracing (pole detection)",
            },
            "ransac_threshold": {
                "type": "float",
                "default": 0.03,
                "min": 0.001,
                "max": 5.0,
                "label": "RANSAC Threshold",
                "description": "RANSAC inlier distance threshold (m)",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget

        # --- Validate: one branch selected ---
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

        # --- Validate: enough points selected ---
        selected_indices = viewer_widget.picked_points_indices
        if not selected_indices or len(selected_indices) < 10:
            QMessageBox.warning(main_window, "Not Enough Points",
                                "Please select at least 10 seed points on cable(s) "
                                "using polygon selection (P key) or Shift+Click.")
            return

        # --- Reconstruct selected branch ---
        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{str(e)}")
            return

        pc_points = point_cloud.points

        # --- Map picked points to reconstructed point cloud via coordinate matching ---
        picked_coords = []
        for idx in selected_indices:
            if idx < len(viewer_widget.points):
                picked_coords.append(viewer_widget.points[idx, :3])
        if not picked_coords:
            QMessageBox.warning(main_window, "No Points",
                                "Could not retrieve coordinates for selected points.")
            return
        picked_coords = np.array(picked_coords, dtype=np.float32)

        tree_kd = cKDTree(pc_points)
        _, local_indices = tree_kd.query(picked_coords)
        seed_set = set(int(i) for i in local_indices)

        # Polygon re-test for full-resolution coverage
        polygon_mask = viewer_widget.retest_polygon_selection(pc_points)
        if polygon_mask is not None:
            polygon_indices = np.where(polygon_mask)[0]
            seed_set |= set(int(i) for i in polygon_indices)

        seed_indices = np.array(sorted(seed_set), dtype=np.intp)
        if len(seed_indices) < 10:
            QMessageBox.warning(main_window, "Not Enough Points",
                                f"Only {len(seed_indices)} seed points mapped. Need at least 10.")
            return

        # --- DBSCAN on seed points to separate per-cable clusters ---
        seed_points = pc_points[seed_indices]
        seed_pc = PointCloud(points=seed_points)
        eps = params.get("dbscan_eps", 1.5)
        min_samples = params.get("dbscan_min_samples", 3)

        try:
            seed_labels = seed_pc.dbscan(eps=eps, min_points=min_samples)
        except Exception:
            from sklearn.cluster import DBSCAN as SklearnDBSCAN
            db = SklearnDBSCAN(eps=eps, min_samples=min_samples)
            seed_labels = db.fit_predict(seed_points)

        unique_labels = set(seed_labels)
        unique_labels.discard(-1)  # remove noise

        if not unique_labels:
            QMessageBox.warning(main_window, "No Cable Seeds",
                                "DBSCAN found no valid clusters among picked points.\n"
                                "Try increasing DBSCAN Eps or selecting more points per cable.")
            return

        # --- Build shared KDTree and trace each cable ---
        tracer = PowerLineTracer(
            all_points=pc_points,
            kdtree=tree_kd,
            cylinder_radius=params.get("cylinder_radius", 0.5),
            cylinder_length=params.get("cylinder_length", 5.0),
            min_points=params.get("min_points", 5),
            max_angle_deg=params.get("max_angle", 15.0),
            ransac_threshold=params.get("ransac_threshold", 0.3),
        )

        # --- Trace each cable and collect only cable points ---
        all_cable_indices = []
        cable_assignments = []  # parallel list: cable_id for each index
        cluster_names = {}

        for cable_id, cluster_label in enumerate(sorted(unique_labels)):
            cable_seed_mask = seed_labels == cluster_label
            cable_seed_local = seed_indices[cable_seed_mask]

            cable_indices = tracer.trace_cable(cable_seed_local)
            all_cable_indices.append(cable_indices)
            cable_assignments.append(np.full(len(cable_indices), cable_id, dtype=np.int32))
            cluster_names[cable_id] = f"Cable {cable_id + 1}"

        if not all_cable_indices:
            QMessageBox.warning(main_window, "No Cable Points",
                                "Tracing did not find any cable points. "
                                "Try adjusting the parameters.")
            return

        # Merge and deduplicate (a point could be claimed by overlapping cables)
        combined_idx = np.concatenate(all_cable_indices)
        combined_labels = np.concatenate(cable_assignments)
        unique_idx, first_occ = np.unique(combined_idx, return_index=True)
        cable_labels = combined_labels[first_occ]

        n_cable_points = len(unique_idx)
        if n_cable_points == 0:
            QMessageBox.warning(main_window, "No Cable Points",
                                "Tracing did not find any cable points. "
                                "Try adjusting the parameters.")
            return

        # --- Add remaining-points mask as child of input branch ---
        remaining_mask = np.ones(len(pc_points), dtype=bool)
        remaining_mask[unique_idx] = False
        mask_data = Masks(mask=remaining_mask)

        mask_uid = controller.add_analysis_result(
            mask_data, "masks", [node.uid], node, "remaining_points", params
        )
        mask_node = controller.get_node(mask_uid)
        tree_widget.add_branch(
            mask_uid, str(node.uid),
            "remaining_points", tooltip=str(params)
        )

        # --- Build cable-only PointCloud ---
        cable_points = pc_points[unique_idx]
        cable_colors = point_cloud.colors[unique_idx] if point_cloud.colors is not None else None
        cable_pc = PointCloud(points=cable_points, colors=cable_colors)

        parent_uid_str = str(node.uid)
        pc_uid = controller.add_analysis_result(
            cable_pc, "point_cloud", [node.uid], node, "power_line_detection", params
        )
        pc_node = controller.get_node(pc_uid)
        tree_widget.add_branch(
            pc_uid, parent_uid_str,
            "power_line_detection", tooltip=str(params)
        )

        # --- Add Clusters as child of cable PointCloud ---
        clusters = Clusters(labels=cable_labels, cluster_names=cluster_names)
        clusters.set_random_color()

        cl_uid = controller.add_analysis_result(
            clusters, "cluster_labels", [pc_node.uid], pc_node, "power_line_cables", params
        )
        cl_node = controller.get_node(cl_uid)
        tree_widget.add_branch(
            cl_uid, str(pc_uid),
            "power_line_cables", tooltip=str(params)
        )

        # --- Turn off input branch, turn on remaining-points mask and clusters ---
        from PyQt5.QtCore import Qt
        tree_widget.blockSignals(True)
        # Hide the processed (input) branch
        input_item = tree_widget.branches_dict.get(selected_uid)
        if input_item:
            input_item.setCheckState(0, Qt.Unchecked)
            tree_widget.visibility_status[selected_uid] = False
        # Show the remaining-points mask branch
        mask_item = tree_widget.branches_dict.get(mask_uid)
        if mask_item:
            mask_item.setCheckState(0, Qt.Checked)
            tree_widget.visibility_status[mask_uid] = True
        # Show the clusters output branch
        cl_item = tree_widget.branches_dict.get(cl_uid)
        if cl_item:
            cl_item.setCheckState(0, Qt.Checked)
            tree_widget.visibility_status[cl_uid] = True
        tree_widget.blockSignals(False)

        # --- Render and clear selection ---
        main_window.render_visible_data(zoom_extent=False)
        viewer_widget.picked_points_indices.clear()
        viewer_widget._selection_polygons.clear()
        viewer_widget.update()

        n_cables = len(cluster_names)
        n_remaining = int(remaining_mask.sum())
        QMessageBox.information(
            main_window, "Power Line Detection Complete",
            f"Traced {n_cables} cable(s) — {n_cable_points:,} cable points, "
            f"{n_remaining:,} remaining points."
        )

