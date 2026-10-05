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
   grown lines, -2 (UNASSIGNED) = everything else, drawn in the input's own
   colours. Each line is locked against selection.
4. Running again with that result branch selected grows MORE lines into it:
   pick seeds on the unassigned points (the locked lines cannot be picked), and
   the new lines are added after the existing ones. Growth sees the existing
   lines' points but never claims them. Clusters > Undo Cluster Edit reverts the
   last run.

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

import copy
import time
import threading

import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.clusters import Clusters, UNASSIGNED_LABEL, is_cluster
from core.entities.point_cloud import PointCloud
from core.services.eigenvalue_utils import EigenvalueUtils
from core.services.linear_region_grower import (
    LinearRegionGrower,
    AXIS_TRACE,
    LINEARITY_CONNECTED,
    HYBRID,
    centerlines_to_vector_feature,
    cylinders_to_vector_feature,
    frustums_to_vector_feature,
    lines_to_traces,
    merge_wireframes,
    resolved_stop_keys,
    traces_to_lines,
    STOP_REASONS,
)
from plugins.dialogs.line_extension_window import LineExtensionWindow
from application.selection_gate import selected_cloud_indices


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
                "description": "Max candidate lines tried per fit — at the seed and "
                               "at each march step (higher = more robust, slower)",
            },
            "cylinder_radius": {
                "type": "float",
                "default": 0.03,
                "min": 0.001,
                "max": 5.0,
                "label": "Tip Radius",
                "description": "Radius of the fit window at the tip (m). The "
                               "window widens forward at Max Angle so curves "
                               "stay in view. Also the width of the band fitted "
                               "around the chosen line each step",
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
            "min_angle": {
                "type": "float",
                "default": 5.0,
                "min": 0.0,
                "max": 90.0,
                "label": "Min Angle (deg)",
                "description": "Every step first searches a narrow cone at this "
                               "angle — little clutter, so the cleanest fit where "
                               "the line is clear. 0 = a plain cylinder",
            },
            "max_angle": {
                "type": "float",
                "default": 20.0,
                "min": 1.0,
                "max": 90.0,
                "label": "Max Angle (deg)",
                "description": "When the narrow cone loses the line (after trying "
                               "to bridge a gap), the cone opens step by step up "
                               "to this angle. Also the largest turn per step "
                               "before the march stops",
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
                "label": "Show Search Windows",
                "description": "Add two debug branches (axis-trace / hybrid only): "
                               "the cone-shaped window each step searched, and "
                               "the cylinder band each step then fitted",
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
                               "magenta=empty space, white=step cap, cyan=end of "
                               "your picks) — shows where and why growth stopped, "
                               "and is kept up to date by Extend Traced Lines "
                               "(axis-trace / hybrid only)",
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
        selected_uid, node, point_cloud, linearity, existing = prep
        pc_points = point_cloud.points

        # --- Map the picked seeds and group them into separate lines ---
        seeds = self._resolve_seed_groups(viewer_widget, pc_points, params,
                                          main_window, node=node)
        if seeds is None:
            return
        seed_groups = seeds

        # --- Grow one line per group on a background thread (progress + cancel) ---
        grower = LinearRegionGrower(
            all_points=pc_points,
            mode=mode,
            ransac_threshold=params.get("ransac_threshold", 0.03),
            max_iterations=params.get("ransac_iterations", 100),
            cylinder_radius=params.get("cylinder_radius", 0.03),
            cylinder_length=params.get("cylinder_length", 0.5),
            reach_factor=params.get("reach_factor", 3.0),
            overlap=params.get("cylinder_overlap", 0.0) / 100.0,
            min_points=params.get("min_points", 5),
            max_angle_deg=params.get("max_angle", 20.0),
            min_angle_deg=params.get("min_angle"),
            linearity=linearity,
            linearity_threshold=params.get("linearity_threshold", 0.4),
            neighbor_k=params.get("neighbor_k", 16),
        )
        # Every exit below frees what the grower's queries built on the GPU —
        # unless the extension window took the grower over, in which case the
        # window frees it when it closes (see LinearRegionGrower.release).
        handed_over = False
        try:
            # Growing into an existing result: its lines' points steer growth
            # but are never taken from them.
            blocked = None if existing is None else is_cluster(existing.labels)
            lines, stopped_early = self._grow_threaded(main_window, grower,
                                                       seed_groups, blocked)
            if lines is None:  # error during grow — message already shown
                return
            if not lines:
                QMessageBox.warning(main_window, "No Feature Points",
                                    "Growing did not find any points. "
                                    "Try adjusting the parameters." if not stopped_early
                                    else "Cancelled before any line was grown.")
                return

            # --- Write the lines: a new result branch, or into the selected one ---
            resolved = set()
            if existing is None:
                result_uid, labels = self._build_result_branch(
                    controller, tree_widget, selected_uid, node, pc_points, lines, params
                )
                all_lines = lines
                self._build_debug_branches(controller, tree_widget, node,
                                           result_uid, lines, params)
            else:
                result_uid = selected_uid
                all_lines, labels, resolved = self._expand_result_branch(
                    controller, node, lines, params)
                input_node = controller.get_node(str(node.parent_uid))
                self._update_debug_branches(controller, tree_widget, input_node,
                                            result_uid, all_lines, lines, params)

            # --- Render and clear selection ---
            main_window.render_visible_data(zoom_extent=False)
            viewer_widget.clear_selection()

            self._show_summary(main_window, labels, lines, stopped_early,
                               n_before=len(all_lines) - len(lines))

            # --- Offer to walk the stops and extend the traces that fell short ---
            # Growth almost never reaches the end of every feature, and the fix is
            # cheapest right now while the grower and the picks are still to hand.
            # Declining is fine: the stops are persisted on the result branch, so
            # "Extend Traced Lines" reopens this on the saved branch at any time.
            handed_over = self._offer_extension(
                main_window, result_uid, pc_points, all_lines, lines, grower,
                params, resolved)
        finally:
            if not handed_over:
                grower.release()

    def _offer_extension(self, main_window, result_uid, pc_points, all_lines,
                         new_lines, grower, params, resolved):
        """Open the guided-extension window if any line this run grew stopped
        somewhere worth looking at. The window gets ALL the branch's lines — it
        rewrites the branch from them — but only this run's are asked about.
        Returns True when the window was opened — it then owns the grower, and
        releases it on close."""
        claimed = np.zeros(len(pc_points), dtype=bool)
        for line in all_lines:
            claimed[line.indices] = True
        promising = sum(
            1 for line in new_lines for stop in line.stops
            if grower.unclaimed_ahead(stop, claimed).size > 0
        )
        if promising == 0:
            return False

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
            return False

        window = LineExtensionWindow(result_uid, pc_points, all_lines, grower,
                                     params, resolved=resolved,
                                     parent=main_window)
        window.show()
        # Held on the main window so Python does not garbage-collect a modeless
        # dialog the moment this method returns.
        main_window._line_extension_window = window
        return True

    # ------------------------------------------------------------------ #
    # execute() steps                                                    #
    # ------------------------------------------------------------------ #

    def _validate_and_reconstruct(self, controller, viewer_widget, main_window, mode, params):
        """Validate the selection + picks, reconstruct the branch, and consume
        upstream linearity for the linearity modes.

        When the selected branch is itself a linear-region-growing result, the
        run grows MORE lines into it: the cloud reconstructed is its input (the
        cloud its labels index, exactly as Extend Traced Lines does), and
        ``existing`` is its Clusters. Otherwise ``existing`` is None and a new
        result branch is made under the selected one.

        Returns ``(selected_uid, node, point_cloud, linearity, existing)`` or
        ``None`` when a check fails (a QMessageBox has been shown).
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
        if not viewer_widget.has_selection():
            QMessageBox.warning(main_window, "Not Enough Points",
                                f"Please select at least {_MIN_SEEDS} seed points along "
                                "the linear feature using polygon selection (P key) "
                                "or Shift+Click.")
            return None

        # --- Reconstruct the cloud to grow in ---
        existing = node.data if self._is_linear_result(node) else None
        source_uid = str(node.parent_uid) if existing is not None else selected_uid
        try:
            point_cloud = controller.reconstruct(source_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{str(e)}")
            return None
        if existing is not None and len(existing.labels) != len(point_cloud.points):
            QMessageBox.warning(
                main_window, "Branch Changed",
                f"This result has {len(existing.labels):,} labels but its input "
                f"cloud now reconstructs to {len(point_cloud.points):,} points, so "
                f"the two no longer line up. Run the growth on the input cloud.")
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

        return selected_uid, node, point_cloud, linearity, existing

    @staticmethod
    def _is_linear_result(node):
        """Whether *node* is a result of this plugin — a Clusters branch
        carrying line traces — and so the place a new run adds its lines."""
        if node is None or node.data_type != "cluster_labels":
            return False
        traces = getattr(node.data, "line_traces", None)
        return isinstance(traces, dict) and "lines" in traces

    def _resolve_seed_groups(self, viewer_widget, pc_points, params, main_window,
                             node=None):
        """Map the picked viewer points to reconstructed-cloud indices (coord
        match + polygon re-test) and group them into separate lines with DBSCAN.

        *node* supplies what may be picked. Without it the polygon re-test hands
        back every point the lasso encloses, locked clusters and noise included,
        because that widening happens in cloud-index space where the viewer's
        own filters no longer apply — the viewer would report 32 seeds while
        this produced thousands, and a bush would be traced as a line.

        Returns the seed groups, or ``None`` when no usable one is found (a
        QMessageBox is shown).
        """
        seed_indices = selected_cloud_indices(
            viewer_widget, node.uid, pc_points)
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

        return seed_groups

    def _grow_threaded(self, main_window, grower, seed_groups, blocked=None):
        """Run ``grower.grow_lines`` on a daemon thread with a status-bar progress
        bar and cancel button (matching surface_region_growing's UX).

        Returns ``(lines, stopped_early)``. ``lines`` is ``None`` on error (a
        QMessageBox has been shown); on cancel it holds whatever was grown before
        the user stopped, and ``stopped_early`` is True. *blocked* is passed
        through to ``grow_lines`` (points growth may see but not claim).
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
                    seed_groups, progress_cb=_progress, cancel_event=cancel_event,
                    blocked=blocked,
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
        """Build the one Clusters branch (label per line, UNASSIGNED = rest),
        register it, and toggle visibility (hide input, show result). Returns
        ``(result_uid, labels)``."""
        labels = self._labels_for(lines, len(pc_points))
        cluster_names = {k: f"Line {k + 1}" for k in range(len(lines))}
        # Carry the stops and centerlines on the result so a short trace can be
        # continued in a later session without re-growing it (see
        # Clusters.line_traces and the Extend Traced Lines plugin).
        clusters = Clusters(labels=labels, cluster_names=cluster_names,
                            line_traces=lines_to_traces(lines, params),
                            tint_locked=False)
        self._lock_lines(clusters, range(len(lines)))
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

    @staticmethod
    def _labels_for(lines, n_points):
        """Per-point labels: line k's points get k, everything else UNASSIGNED."""
        labels = np.full(n_points, UNASSIGNED_LABEL, dtype=np.int32)
        for k, line in enumerate(lines):
            labels[line.indices] = k
        return labels

    @staticmethod
    def _lock_lines(clusters, labels):
        """Lock these lines against selection, so the next seeds can only be
        picked from what is still unassigned. Not tinted (``tint_locked`` is off):
        every line carries the lock, and tinting them all would only wash the
        result out."""
        for label in labels:
            clusters.locked_clusters.setdefault(int(label), set()).add("select")

    def _expand_result_branch(self, controller, node, new_lines, params):
        """Add *new_lines* to the existing result branch *node*, in place.

        The lines already there keep their labels, names (a line classified as
        "Cable" stays "Cable"), colours and settled stops; the new ones follow
        them. The previous Clusters is kept for Clusters > Undo Cluster Edit, so
        a run that grew the wrong thing can be taken back in one step.

        Returns ``(all_lines, labels, resolved)`` — every line on the branch,
        the new labels, and the stops already dismissed as real ends.
        """
        clusters = node.data
        controller._cluster_undo[str(node.uid)] = copy.deepcopy(clusters)

        old_lines = traces_to_lines(clusters.line_traces, clusters.labels)
        resolved = resolved_stop_keys(clusters.line_traces)
        all_lines = old_lines + list(new_lines)
        added = range(len(old_lines), len(all_lines))

        clusters.labels = self._labels_for(all_lines, len(clusters.labels))
        for k in added:
            clusters.cluster_names[k] = f"Line {k + 1}"
        self._lock_lines(clusters, added)
        clusters.tint_locked = False
        # The traces carry ONE set of growth parameters, read back by Extend
        # Traced Lines to rebuild the grower — the latest run's, so extending
        # continues under the settings last chosen.
        clusters.line_traces = lines_to_traces(all_lines, params, resolved=resolved)
        clusters.set_random_color()

        controller.cache_service.invalidate(str(node.uid))
        controller.cache_service.invalidate_descendants(str(node.uid))
        return all_lines, clusters.labels, resolved

    def _build_debug_branches(self, controller, tree_widget, node, result_uid, lines, params):
        """Add the optional debug geometry branches (one centerlines branch, one
        search-windows and one cylinders branch, and one end-cylinder branch per
        stop reason), each
        gated by its ``show_*`` box and holding all lines."""
        extras = []
        if params.get("show_lines"):
            vf = centerlines_to_vector_feature([line.centerline for line in lines])
            if vf is not None:
                vf.cluster_reference = result_uid
                extras.append(("centerlines", vf))
        if params.get("show_cylinders"):
            vf = frustums_to_vector_feature(
                [w for line in lines for w in line.windows])
            if vf is not None:
                vf.cluster_reference = result_uid
                extras.append(("search_windows", vf))
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

        self._add_debug_branches(controller, tree_widget, node, result_uid,
                                 extras, params)

    def _add_debug_branches(self, controller, tree_widget, node, result_uid,
                            extras, params):
        """Add each ``(name, feature)`` in *extras* as a visible branch under the
        result. *node* is the input cloud the geometry depends on."""
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

    def _update_debug_branches(self, controller, tree_widget, input_node,
                               result_uid, all_lines, new_lines, params):
        """Bring the debug branches of an expanded result up to date.

        Centerlines and cylinders are rebuilt from every line (the traces carry
        both), so a branch already there always matches the lines. Search
        windows and stop markers are not saved, so the earlier runs' copies
        cannot be rebuilt — this run's geometry is appended to them instead.
        Branches that do not exist yet are created only when this run's dialog
        asked for them, and the appended kinds only grow when it did.
        """
        result_node = controller.get_node(result_uid)
        children = {child.params: (uid, child)
                    for uid, child in controller.data_nodes.data_nodes.items()
                    if child.parent_uid == result_node.uid}

        wanted = []
        if params.get("show_lines") or "centerlines" in children:
            wanted.append(("centerlines", centerlines_to_vector_feature(
                [line.centerline for line in all_lines])))
        if params.get("show_cylinders") or "cylinders" in children:
            wanted.append(("cylinders", cylinders_to_vector_feature(
                [c for line in all_lines for c in line.cylinders])))

        def appended(name, feature):
            old = children.get(name, (None, None))[1]
            return name, merge_wireframes(None if old is None else old.data, feature)

        if params.get("show_cylinders"):
            wanted.append(appended("search_windows", frustums_to_vector_feature(
                [w for line in new_lines for w in line.windows])))
        if params.get("show_end_cylinders"):
            by_reason = {}
            for line in new_lines:
                for reason, cyl in line.end_cylinders:
                    by_reason.setdefault(reason, []).append(cyl)
            for reason, cyls in by_reason.items():
                label, color = STOP_REASONS.get(
                    reason, (reason, np.array([1.0, 1.0, 1.0], dtype=np.float32)))
                wanted.append(appended(f"stop_{reason}", cylinders_to_vector_feature(
                    cyls, color=color, symbol_type=f"Stop: {label}")))

        new_branches = []
        for name, feature in wanted:
            if feature is None:
                continue
            feature.cluster_reference = result_uid
            if name in children:
                uid, child = children[name]
                child.data = feature
                controller.cache_service.invalidate(str(uid))
            else:
                new_branches.append((name, feature))
        self._add_debug_branches(controller, tree_widget, input_node, result_uid,
                                 new_branches, params)

    def _show_summary(self, main_window, labels, lines, stopped_early, n_before=0):
        """Show the completion message: feature/rest counts and per-line stop
        reasons for the lines this run grew, noting if the user cancelled.
        *n_before* is how many lines the branch already had, when this run
        added to an existing result."""
        n_feature = int(is_cluster(labels).sum())
        n_rest = len(labels) - n_feature

        # Per-line stop reasons: why each end of each line stopped growing.
        stop_lines = []
        for k, line in enumerate(lines):
            reasons = [STOP_REASONS.get(r, (r, None))[0] for r, _ in line.end_cylinders]
            if reasons:
                stop_lines.append(f"Line {n_before + k + 1}: stopped on "
                                  f"{', '.join(reasons)}")
        stop_summary = ("\n\nStop reasons:\n" + "\n".join(stop_lines)) if stop_lines else ""
        cancel_note = ("\n\nCancelled early — partial result saved."
                       if stopped_early else "")
        added_note = (f" Added to this branch's {n_before} existing line(s); "
                      f"Clusters > Undo Cluster Edit takes this run back."
                      if n_before else "")

        QMessageBox.information(
            main_window,
            "Linear Region Growing Cancelled" if stopped_early
            else "Linear Region Growing Complete",
            f"Grew {len(lines)} line(s) — {n_feature:,} feature points, "
            f"{n_rest:,} remaining." + added_note + stop_summary + cancel_note
        )
