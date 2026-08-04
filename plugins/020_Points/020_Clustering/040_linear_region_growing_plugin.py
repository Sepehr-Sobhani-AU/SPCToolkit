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

import time
import threading

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox, QApplication
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
    lines_to_traces,
    STOP_REASONS,
)
from plugins.dialogs.line_extension_window import LineExtensionWindow
from application.selection_gate import picked_cloud_indices


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
                "description": "Length of the per-step fit window (m). This is where "
                               "the line is fit and how far the tip advances. Keep "
                               "it short on curved features (a long window fits a "
                               "chord and drifts outward); Search Reach handles gaps",
            },
            "reach_factor": {
                "type": "float",
                "default": 3.0,
                "min": 1.0,
                "max": 10.0,
                "label": "Search Reach ×",
                "description": "How far ahead the march looks for the next points, as "
                               "a multiple of Cylinder Length. >1 bridges gaps in "
                               "fragmented features; 1 = no bridging",
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
            "show_end_cylinders": {
                "type": "bool",
                "default": False,
                "label": "Show Stop Cylinders",
                "description": "Draw the last search cylinder at each end of every "
                               "line, split into one coloured branch per stop reason "
                               "(red=too few points, orange=sharp bend, "
                               "magenta=empty space, white=step cap) — shows where "
                               "and why growth stopped (axis-trace / hybrid only)",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget

        mode = _MODE_MAP.get(params.get("growth_mode", "Axis Trace"), AXIS_TRACE)

        # --- Validate + reconstruct the selected branch ---
        prep = self._validate_and_reconstruct(controller, viewer_widget, main_window, mode, params)
        if prep is None:
            return
        selected_uid, node, point_cloud, linearity = prep
        pc_points = point_cloud.points

        # --- Map the picked seeds and group them into separate lines ---
        seeds = self._resolve_seed_groups(viewer_widget, pc_points, params, main_window)
        if seeds is None:
            return
        seed_groups, tree_kd = seeds

        # --- Grow one line per group on a background thread (progress + cancel) ---
        grower = LinearRegionGrower(
            all_points=pc_points,
            kdtree=tree_kd,
            mode=mode,
            ransac_threshold=params.get("ransac_threshold", 0.03),
            max_iterations=params.get("ransac_iterations", 100),
            cylinder_radius=params.get("cylinder_radius", 0.03),
            cylinder_length=params.get("cylinder_length", 0.5),
            reach_factor=params.get("reach_factor", 3.0),
            overlap=params.get("cylinder_overlap", 0.0) / 100.0,
            min_points=params.get("min_points", 5),
            max_angle_deg=params.get("max_angle", 20.0),
            linearity=linearity,
            linearity_threshold=params.get("linearity_threshold", 0.4),
            neighbor_k=params.get("neighbor_k", 16),
        )
        lines, stopped_early = self._grow_threaded(main_window, grower, seed_groups)
        if lines is None:  # error during grow — message already shown
            return
        if not lines:
            QMessageBox.warning(main_window, "No Feature Points",
                                "Growing did not find any points. "
                                "Try adjusting the parameters." if not stopped_early
                                else "Cancelled before any line was grown.")
            return

        # --- Build the result Clusters branch and optional debug branches ---
        result_uid, labels = self._build_result_branch(
            controller, tree_widget, selected_uid, node, pc_points, lines, params
        )
        self._build_debug_branches(controller, tree_widget, node, result_uid, lines, params)

        # --- Render and clear selection ---
        main_window.render_visible_data(zoom_extent=False)
        viewer_widget.picked_points_indices.clear()
        viewer_widget._selection_polygons.clear()
        viewer_widget.update()

        self._show_summary(main_window, labels, lines, stopped_early)

        # --- Offer to walk the stops and extend the traces that fell short ---
        # Growth almost never reaches the end of every feature, and the fix is
        # cheapest right now while the grower and the picks are still to hand.
        # Declining is fine: the stops are persisted on the result branch, so
        # "Extend Traced Lines" reopens this on the saved branch at any time.
        self._offer_extension(main_window, result_uid, pc_points, lines, grower, params)

    def _offer_extension(self, main_window, result_uid, pc_points, lines, grower, params):
        """Open the guided-extension window if any line stopped somewhere worth
        looking at."""
        claimed = np.zeros(len(pc_points), dtype=bool)
        for line in lines:
            claimed[line.indices] = True
        promising = sum(
            1 for line in lines for stop in line.stops
            if grower.unclaimed_ahead(stop, claimed).size > 0
        )
        if promising == 0:
            return

        answer = QMessageBox.question(
            main_window, "Extend Traced Lines?",
            f"{promising} of the traced line ends have unclaimed points just "
            f"beyond them, so those features may continue further.\n\n"
            f"Step through them now and extend the ones that do?\n\n"
            f"(You can also do this later: select the result branch and run "
            f"Extend Traced Lines.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        window = LineExtensionWindow(result_uid, pc_points, lines, grower, params,
                                     parent=main_window)
        window.show()
        # Held on the main window so Python does not garbage-collect a modeless
        # dialog the moment this method returns.
        main_window._line_extension_window = window

    # ------------------------------------------------------------------ #
    # execute() steps                                                    #
    # ------------------------------------------------------------------ #

    def _validate_and_reconstruct(self, controller, viewer_widget, main_window, mode, params):
        """Validate the selection + picks, reconstruct the branch, and consume
        upstream linearity for the linearity modes.

        Returns ``(selected_uid, node, point_cloud, linearity)`` or ``None`` when
        a check fails (a QMessageBox has been shown).
        """
        # --- Validate: one branch selected ---
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a PointCloud branch first.")
            return None
        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch at a time.")
            return None

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)
        if node is None:
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Could not find the selected branch.")
            return None

        # --- Validate: enough seed points selected ---
        selected_indices = viewer_widget.picked_points_indices
        if not selected_indices or len(selected_indices) < _MIN_SEEDS:
            QMessageBox.warning(main_window, "Not Enough Points",
                                f"Please select at least {_MIN_SEEDS} seed points along "
                                "the linear feature using polygon selection (P key) "
                                "or Shift+Click.")
            return None

        # --- Reconstruct selected branch ---
        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{str(e)}")
            return None

        # --- Consume upstream linearity for the linearity-based modes ---
        linearity = None
        if mode in (LINEARITY_CONNECTED, HYBRID):
            eigenvalues = point_cloud.attributes.get("eigenvalues")
            if eigenvalues is None:
                QMessageBox.warning(
                    main_window, "Eigenvalues Required",
                    f"'{params.get('growth_mode')}' mode needs per-point linearity.\n"
                    "Run Compute Eigenvalues on this branch first, then select the "
                    "eigenvalues node and re-run.")
                return None
            linearity = EigenvalueUtils().compute_geometric_features(eigenvalues)["linearity"]

        return selected_uid, node, point_cloud, linearity

    def _resolve_seed_groups(self, viewer_widget, pc_points, params, main_window):
        """Map the picked viewer points to reconstructed-cloud indices (coord
        match + polygon re-test) and group them into separate lines with DBSCAN.

        Returns ``(seed_groups, tree_kd)`` — the KD-tree is reused by the grower —
        or ``None`` when no usable seed group is found (a QMessageBox is shown).
        """
        tree_kd = cKDTree(pc_points)
        seed_indices = picked_cloud_indices(viewer_widget, pc_points, tree_kd)
        if seed_indices is None:
            QMessageBox.warning(main_window, "No Points",
                                "Could not retrieve coordinates for selected points.")
            return None

        if len(seed_indices) < _MIN_SEEDS:
            QMessageBox.warning(main_window, "Not Enough Points",
                                f"Only {len(seed_indices)} seed points mapped. "
                                f"Need at least {_MIN_SEEDS}.")
            return None

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
            return None

        return seed_groups, tree_kd

    def _grow_threaded(self, main_window, grower, seed_groups):
        """Run ``grower.grow_lines`` on a daemon thread with a status-bar progress
        bar and cancel button (matching surface_region_growing's UX).

        Returns ``(lines, stopped_early)``. ``lines`` is ``None`` on error (a
        QMessageBox has been shown); on cancel it holds whatever was grown before
        the user stopped, and ``stopped_early`` is True.
        """
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.show_progress("Growing linear features...")
        main_window.show_cancel_button()  # clears any stale cancel flag

        cancel_event = global_variables.global_cancel_event
        state = {"lines": None, "error": None, "done": False}

        def _progress(done, total, message):
            percent = int(100 * done / total) if total else None
            global_variables.global_progress = (percent, message)

        def _work():
            try:
                state["lines"] = grower.grow_lines(
                    seed_groups, progress_cb=_progress, cancel_event=cancel_event
                )
            except Exception as e:
                state["error"] = str(e)
            finally:
                state["done"] = True

        thread = threading.Thread(target=_work, daemon=True)
        thread.start()

        while not state["done"]:
            percent, msg = global_variables.global_progress
            if msg:
                main_window.show_progress(msg, percent)
            QApplication.processEvents()
            time.sleep(0.1)

        # Read the cancel flag BEFORE hide_cancel_button clears it.
        global_variables.global_progress = (None, "")
        stopped_early = cancel_event.is_set()
        main_window.hide_cancel_button()
        main_window.clear_progress()
        main_window.enable_menus()
        main_window.enable_tree()

        if state["error"]:
            QMessageBox.critical(main_window, "Linear Region Growing Failed",
                                 state["error"])
            return None, stopped_early
        return state["lines"] or [], stopped_early

    def _build_result_branch(self, controller, tree_widget, selected_uid, node,
                             pc_points, lines, params):
        """Build the one Clusters branch (label per line, -1 = rest), register it,
        and toggle visibility (hide input, show result). Returns
        ``(result_uid, labels)``."""
        labels = np.full(len(pc_points), -1, dtype=np.int32)
        cluster_names = {}
        for k, line in enumerate(lines):
            labels[line.indices] = k
            cluster_names[k] = f"Line {k + 1}"
        # Carry the stops and centerlines on the result so a short trace can be
        # continued in a later session without re-growing it (see
        # Clusters.line_traces and the Extend Traced Lines plugin).
        clusters = Clusters(labels=labels, cluster_names=cluster_names,
                            line_traces=lines_to_traces(lines, params))
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

        return result_uid, labels

    def _build_debug_branches(self, controller, tree_widget, node, result_uid, lines, params):
        """Add the optional debug geometry branches (one centerlines branch, one
        cylinders branch, and one end-cylinder branch per stop reason), each
        gated by its ``show_*`` box and holding all lines."""
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
        if params.get("show_end_cylinders"):
            # One branch per stop reason, each in its own colour, so you can see
            # at a glance why every line ended where it did.
            cylinders_by_reason = {}
            for line in lines:
                for reason, cyl in line.end_cylinders:
                    cylinders_by_reason.setdefault(reason, []).append(cyl)
            for reason, cyls in cylinders_by_reason.items():
                label, color = STOP_REASONS.get(
                    reason, (reason, np.array([1.0, 1.0, 1.0], dtype=np.float32))
                )
                vf = cylinders_to_vector_feature(
                    cyls, color=color, symbol_type=f"Stop: {label}"
                )
                if vf is not None:
                    vf.cluster_reference = result_uid
                    extras.append((f"stop_{reason}", vf))

        if not extras:
            return

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

    def _show_summary(self, main_window, labels, lines, stopped_early):
        """Show the completion message: feature/rest counts and per-line stop
        reasons, noting if the user cancelled."""
        n_feature = int((labels >= 0).sum())
        n_rest = len(labels) - n_feature

        # Per-line stop reasons: why each end of each line stopped growing.
        stop_lines = []
        for k, line in enumerate(lines):
            reasons = [STOP_REASONS.get(r, (r, None))[0] for r, _ in line.end_cylinders]
            if reasons:
                stop_lines.append(f"Line {k + 1}: stopped on {', '.join(reasons)}")
        stop_summary = ("\n\nStop reasons:\n" + "\n".join(stop_lines)) if stop_lines else ""
        cancel_note = ("\n\nCancelled early — partial result saved."
                       if stopped_early else "")

        QMessageBox.information(
            main_window,
            "Linear Region Growing Cancelled" if stopped_early
            else "Linear Region Growing Complete",
            f"Grew {len(lines)} line(s) — {n_feature:,} feature points, "
            f"{n_rest:,} remaining." + stop_summary + cancel_note
        )
