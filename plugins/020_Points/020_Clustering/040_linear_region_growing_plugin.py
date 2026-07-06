"""
Semi-automated linear-feature region growing plugin.

The linear counterpart to surface_region_growing: instead of growing a planar
surface, it grows a 1-D linear feature (cable, pipe, rail, kerb, edge) outward
from seed points the user picks in the viewer.

Workflow:
1. User selects a PointCloud branch and polygon-/Shift-selects seed points along
   one or more linear features.
2. The picked points are grouped with DBSCAN (points close together = one line),
   and each group is grown via the shared ``LinearRegionGrower`` using the chosen
   growth mode (axis trace, linearity-connected, or hybrid). Grown groups that
   turn out to be the same physical line are joined back together.
3. The result is one Clusters branch over the input cloud: label 0, 1, 2, … = the
   grown lines, -1 = everything else.

Several lines can be traced from a single selection. Optionally, each line's
joined centerline (a polyline) and search cylinders are added as their own
controllable branches.

Growth modes:
- **Axis Trace** — march a search cylinder along the fitted line; best for
  isolated thin features. Needs no upstream features.
- **Linearity-Connected** / **Hybrid** — gate growth by per-point linearity;
  best for edges/kerbs embedded in a surface. These require eigenvalues on the
  selected branch (run Compute Eigenvalues first) — they are consumed, not
  recomputed here.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.clusters import Clusters
from core.entities.point_cloud import PointCloud
from core.services.eigenvalue_utils import EigenvalueUtils
from core.services.linear_region_grower import (
    LinearRegionGrower,
    AXIS_TRACE,
    LINEARITY_CONNECTED,
    HYBRID,
    centerlines_to_vector_feature,
    cylinders_to_vector_feature,
)


_MODE_MAP = {
    "Axis Trace": AXIS_TRACE,
    "Linearity-Connected": LINEARITY_CONNECTED,
    "Hybrid": HYBRID,
}

_MIN_SEEDS = 2  # a line fit needs at least two points


class LinearRegionGrowingPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "linear_region_growing"

    def requires_selection(self) -> str:
        return "points"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "growth_mode": {
                "type": "choice",
                "options": ["Axis Trace", "Linearity-Connected", "Hybrid"],
                "default": "Axis Trace",
                "label": "Growth Mode",
                "description": "Axis Trace marches a cylinder along the line "
                               "(isolated features). Linearity-Connected / Hybrid "
                               "gate growth by per-point linearity (edges in a "
                               "surface) and require eigenvalues on the branch.",
            },
            "seed_eps": {
                "type": "float",
                "default": 0.10,
                "min": 0.001,
                "max": 10.0,
                "label": "Seed Group Distance (m)",
                "description": "Picked points closer than this are grouped as one "
                               "line to grow. Increase if one line is split into "
                               "several; decrease if separate lines get merged.",
            },
            "seed_min_samples": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 50,
                "label": "Min Seeds per Group",
                "description": "Fewest picked points needed to start a line group.",
            },
            "ransac_threshold": {
                "type": "float",
                "default": 0.03,
                "min": 0.001,
                "max": 5.0,
                "label": "RANSAC Threshold",
                "description": "RANSAC line inlier distance threshold (m)",
            },
            "ransac_iterations": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
                "label": "RANSAC Iterations",
                "description": "Max RANSAC hypotheses per line fit (higher = more robust, slower)",
            },
            "cylinder_radius": {
                "type": "float",
                "default": 0.03,
                "min": 0.001,
                "max": 5.0,
                "label": "Cylinder Radius",
                "description": "Axis-trace search cylinder radius per step (m)",
            },
            "cylinder_length": {
                "type": "float",
                "default": 0.5,
                "min": 0.01,
                "max": 50.0,
                "label": "Cylinder Length",
                "description": "Axis-trace search cylinder length per step (m)",
            },
            "cylinder_overlap": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 90.0,
                "label": "Cylinder Overlap (%)",
                "description": "Percent each step's cylinder overlaps the previous "
                               "(0 = end-to-end, 50 = half). Higher follows curves "
                               "better but is slower",
            },
            "min_points": {
                "type": "int",
                "default": 5,
                "min": 2,
                "max": 100,
                "label": "Min Points",
                "description": "Stop the axis march if fewer points found in a cylinder",
            },
            "max_angle": {
                "type": "float",
                "default": 20.0,
                "min": 1.0,
                "max": 90.0,
                "label": "Max Angle (deg)",
                "description": "Max direction change per step before the axis march stops",
            },
            "linearity_threshold": {
                "type": "float",
                "default": 0.4,
                "min": 0.0,
                "max": 1.0,
                "label": "Linearity Threshold",
                "description": "Linearity-Connected / Hybrid: accept a point only "
                               "if its linearity is above this",
            },
            "neighbor_k": {
                "type": "int",
                "default": 16,
                "min": 4,
                "max": 64,
                "label": "Neighbours (k)",
                "description": "Linearity-Connected: k-NN used to expand the region",
            },
            "show_cylinders": {
                "type": "bool",
                "default": False,
                "label": "Show Search Cylinders",
                "description": "Overlay the axis-trace search cylinders in the viewer (debug; axis-trace / hybrid only)",
            },
            "show_lines": {
                "type": "bool",
                "default": False,
                "label": "Show Centerlines",
                "description": "Overlay the traced centerline in the viewer (debug; axis-trace / hybrid only)",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget

        mode = _MODE_MAP.get(params.get("growth_mode", "Axis Trace"), AXIS_TRACE)
        needs_linearity = mode in (LINEARITY_CONNECTED, HYBRID)

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

        # --- Validate: enough seed points selected ---
        selected_indices = viewer_widget.picked_points_indices
        if not selected_indices or len(selected_indices) < _MIN_SEEDS:
            QMessageBox.warning(main_window, "Not Enough Points",
                                f"Please select at least {_MIN_SEEDS} seed points along "
                                "the linear feature using polygon selection (P key) "
                                "or Shift+Click.")
            return

        # --- Reconstruct selected branch ---
        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{str(e)}")
            return

        pc_points = point_cloud.points

        # --- Consume upstream linearity for the linearity-based modes ---
        linearity = None
        if needs_linearity:
            eigenvalues = point_cloud.attributes.get("eigenvalues")
            if eigenvalues is None:
                QMessageBox.warning(
                    main_window, "Eigenvalues Required",
                    f"'{params.get('growth_mode')}' mode needs per-point linearity.\n"
                    "Run Compute Eigenvalues on this branch first, then select the "
                    "eigenvalues node and re-run.")
                return
            linearity = EigenvalueUtils().compute_geometric_features(eigenvalues)["linearity"]

        # --- Map picked points to reconstructed cloud via coordinate matching ---
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
        if len(seed_indices) < _MIN_SEEDS:
            QMessageBox.warning(main_window, "Not Enough Points",
                                f"Only {len(seed_indices)} seed points mapped. "
                                f"Need at least {_MIN_SEEDS}.")
            return

        # --- Group the picked seeds into separate lines (DBSCAN) ---
        seed_pts = pc_points[seed_indices]
        seed_labels = np.asarray(PointCloud(points=seed_pts).dbscan(
            eps=params.get("seed_eps", 0.10),
            min_points=params.get("seed_min_samples", 2),
        ))
        seed_groups = [
            seed_indices[seed_labels == lbl]
            for lbl in sorted(set(int(l) for l in seed_labels))
            if lbl != -1 and np.count_nonzero(seed_labels == lbl) >= _MIN_SEEDS
        ]
        if not seed_groups:
            QMessageBox.warning(
                main_window, "No Seed Groups",
                "The picked points did not form any line group. Increase "
                "'Seed Group Distance' or pick more points along each line.")
            return

        # --- Grow one line per group, joining groups that are the same line ---
        grower = LinearRegionGrower(
            all_points=pc_points,
            kdtree=tree_kd,
            mode=mode,
            ransac_threshold=params.get("ransac_threshold", 0.03),
            max_iterations=params.get("ransac_iterations", 100),
            cylinder_radius=params.get("cylinder_radius", 0.03),
            cylinder_length=params.get("cylinder_length", 0.5),
            overlap=params.get("cylinder_overlap", 0.0) / 100.0,
            min_points=params.get("min_points", 5),
            max_angle_deg=params.get("max_angle", 20.0),
            linearity=linearity,
            linearity_threshold=params.get("linearity_threshold", 0.4),
            neighbor_k=params.get("neighbor_k", 16),
        )
        lines = grower.grow_lines(seed_groups)
        if not lines:
            QMessageBox.warning(main_window, "No Feature Points",
                                "Growing did not find any points. "
                                "Try adjusting the parameters.")
            return

        # --- Build one Clusters branch: label per line, -1 = rest ---
        labels = np.full(len(pc_points), -1, dtype=np.int32)
        cluster_names = {}
        for k, line in enumerate(lines):
            labels[line.indices] = k
            cluster_names[k] = f"Line {k + 1}"
        clusters = Clusters(labels=labels, cluster_names=cluster_names)
        clusters.set_random_color()

        result_uid = controller.add_analysis_result(
            clusters, "cluster_labels", [node.uid], node, "linear_region_growing", params
        )
        tree_widget.add_branch(
            result_uid, str(node.uid),
            "linear_region_growing", tooltip=f"linear_region_growing,{params}"
        )

        # --- Hide the input branch, show the result ---
        tree_widget.blockSignals(True)
        input_item = tree_widget.branches_dict.get(selected_uid)
        if input_item:
            input_item.setCheckState(0, Qt.Unchecked)
            tree_widget.visibility_status[selected_uid] = False
        result_item = tree_widget.branches_dict.get(result_uid)
        if result_item:
            result_item.setCheckState(0, Qt.Checked)
        tree_widget.visibility_status[result_uid] = True
        tree_widget.blockSignals(False)

        # --- Optional geometry: one centerlines branch + one cylinders branch,
        #     each holding all lines (gated by the show_* boxes) ---
        extras = []
        if params.get("show_lines"):
            vf = centerlines_to_vector_feature([line.centerline for line in lines])
            if vf is not None:
                vf.cluster_reference = result_uid
                extras.append(("centerlines", vf))
        if params.get("show_cylinders"):
            all_cylinders = [c for line in lines for c in line.cylinders]
            vf = cylinders_to_vector_feature(all_cylinders)
            if vf is not None:
                vf.cluster_reference = result_uid
                extras.append(("cylinders", vf))

        if extras:
            result_node = controller.get_node(result_uid)
            tree_widget.blockSignals(True)
            for name, feature in extras:
                vf_uid = controller.add_analysis_result(
                    feature, "vector_feature", [node.uid], result_node, name, params
                )
                tree_widget.add_branch(vf_uid, result_uid, name,
                                       tooltip=f"linear_region_growing,{params}")
                vf_item = tree_widget.branches_dict.get(vf_uid)
                if vf_item:
                    vf_item.setCheckState(0, Qt.Checked)
                tree_widget.visibility_status[vf_uid] = True
            tree_widget.blockSignals(False)

        # --- Render and clear selection ---
        main_window.render_visible_data(zoom_extent=False)
        viewer_widget.picked_points_indices.clear()
        viewer_widget._selection_polygons.clear()
        viewer_widget.update()

        n_feature = int((labels >= 0).sum())
        n_rest = len(labels) - n_feature
        QMessageBox.information(
            main_window, "Linear Region Growing Complete",
            f"Grew {len(lines)} line(s) — {n_feature:,} feature points, "
            f"{n_rest:,} remaining."
        )
