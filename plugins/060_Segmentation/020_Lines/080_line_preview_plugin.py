"""
Line Preview - smooth traced lines and predict where they continue, without changing them.

Select a Linear Region Growing result and this runs every line's saved
centerline through ``core.services.path_tracker`` (a Kalman / extended Kalman
filter),
then adds branches to compare on screen against the original:

- ``smoothed_centerlines`` — each line smoothed by the tracker (forward pass +
  backward smoothing pass).
- ``predicted_paths`` — where the tracker expects each line to continue
  beyond both of its ends, e.g. across a gap.
- ``prediction_uncertainty`` — optional: a cone around each prediction, two sigma
  wide, showing how sure the tracker is as it reaches further.

Read-only: the result branch, its labels and its traces are not touched, and
linear growing itself does not use the tracker yet. Running again replaces the
preview branches it made last time, so settings can be tried side by side with
the original. See ``core/services/path_tracker.py`` for the method.
"""

from typing import Dict, Any

import numpy as np
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.services.path_tracker import track_polyline, STRAIGHT, CURVED
from core.services.linear_region_grower import (
    centerlines_to_vector_feature,
    frustums_to_vector_feature,
    traces_to_lines,
)


_MODEL_MAP = {
    "Curved": CURVED,
    "Straight": STRAIGHT,
}

_SMOOTHED_COLOR = np.array([1.0, 0.2, 0.8], dtype=np.float32)      # magenta
_PREDICTION_COLOR = np.array([1.0, 0.55, 0.0], dtype=np.float32)   # orange
_UNCERTAINTY_COLOR = np.array([1.0, 0.8, 0.5], dtype=np.float32)   # pale orange

# Uncertainty cones are drawn this many sigmas wide (two = about 95%).
_CONE_SIGMAS = 2.0

# Spacing of the samples along a predicted path (m).
_PREDICTION_STEP = 0.1


class LinePreviewPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "line_preview"

    def requires_selection(self) -> str:
        return "branches"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "model": {
                "type": "choice",
                "options": list(_MODEL_MAP),
                "default": "Curved",
                "label": "Model",
                "description": "Curved follows the line's direction and its "
                               "bend, and carries a bend on across a gap - for "
                               "sagging or curving lines.\n"
                               "Straight follows direction only - for straight "
                               "lines.",
            },
            "point_sigma": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 1.0,
                "decimals": 4,
                "label": "Point Noise (m)",
                "description": "How far a traced vertex may sit off the true "
                               "line. Larger = trust the trace less, smooth more.",
            },
            "flexibility": {
                "type": "float",
                "default": 0.05,
                "min": 0.0001,
                "max": 10.0,
                "decimals": 4,
                "label": "Flexibility",
                "description": "How freely the line may change as it goes — "
                               "direction (Straight) or bend (Curved). Larger = "
                               "follows changes faster, smooths less.",
            },
            "prediction_distance": {
                "type": "float",
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "label": "Prediction Distance (m)",
                "description": "How far to predict each line beyond its ends. "
                               "0 = no predictions.",
            },
            "show_uncertainty": {
                "type": "bool",
                "default": True,
                "label": "Show Uncertainty",
                "description": "Draw a cone around each prediction, two sigma "
                               "wide.",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        try:
            self._run(main_window, params)
        finally:
            main_window.clear_progress()

    def _run(self, main_window, params):
        controller = global_variables.global_application_controller
        tree_widget = global_variables.global_tree_structure_widget

        prepared = self._prepare(controller, main_window)
        if prepared is None:
            return
        result_uid, result_node, lines = prepared

        main_window.show_progress(f"Line preview: tracking {len(lines)} line(s)...")
        model = _MODEL_MAP.get(params.get("model"), CURVED)
        settings = {"point_sigma": params.get("point_sigma", 0.01)}
        flexibility = params.get("flexibility", 0.05)
        settings["curvature_noise" if model == CURVED else "direction_noise"] = flexibility
        distance = float(params.get("prediction_distance", 3.0))

        smoothed, predictions, cones, shifts = [], [], [], []
        for line in lines:
            if line.centerline is None or len(line.centerline) < 3:
                continue
            forward = track_polyline(line.centerline, model=model, **settings)
            if forward is None:
                continue
            smoothed.append(forward.smoothed)
            shifts.append(np.linalg.norm(
                forward.smoothed - forward.filtered, axis=1).mean())
            if distance <= 0:
                continue
            # Beyond the far end: onward from the forward estimate there.
            # Beyond the near end: the smoothed start, turned round.
            for state in (forward.end_state, forward.states[0].reversed()):
                positions, _tangents, sigma = forward.tracker.predict_path(
                    distance, _PREDICTION_STEP, state=state)
                predictions.append(positions)
                cones.extend(_cone_segments(positions, sigma))

        if not smoothed:
            QMessageBox.information(
                main_window, "Nothing to Track",
                "None of the lines on this branch has a centerline of three or "
                "more vertices to track.")
            return

        main_window.show_progress("Line preview: drawing...")
        features = [("smoothed_centerlines",
                     centerlines_to_vector_feature(smoothed, color=_SMOOTHED_COLOR))]
        if predictions:
            features.append(("predicted_paths", centerlines_to_vector_feature(
                predictions, color=_PREDICTION_COLOR)))
        if cones and params.get("show_uncertainty", True):
            features.append(("prediction_uncertainty", frustums_to_vector_feature(
                cones, color=_UNCERTAINTY_COLOR,
                symbol_type="Prediction Uncertainty")))
        self._show_branches(controller, tree_widget, result_uid, result_node,
                            features, params)
        main_window.render_visible_data(zoom_extent=False)
        main_window.clear_progress()

        QMessageBox.information(
            main_window, "Line Preview",
            f"Tracked {len(smoothed)} line(s) with the "
            f"{'curved' if model == CURVED else 'straight'} "
            f"model.\n\nSmoothing moved the centerline vertices "
            f"{np.mean(shifts) * 1000:.1f} mm on average from the forward "
            f"estimate.\n\nThe original lines are unchanged; toggle the smoothed_centerlines / predicted_paths "
            f"branches to compare.")

    # ------------------------------------------------------------------ #

    def _prepare(self, controller, main_window):
        """The selected linear-growing result and its lines, or None (a
        message has been shown)."""
        selected = controller.selected_branches
        if len(selected) != 1:
            QMessageBox.warning(main_window, "Select One Branch",
                                "Select exactly one Linear Region Growing result.")
            return None
        result_uid = selected[0]
        node = controller.get_node(result_uid)
        traces = getattr(getattr(node, "data", None), "line_traces", None)
        if (node is None or node.data_type != "cluster_labels"
                or not isinstance(traces, dict) or not traces.get("lines")):
            QMessageBox.warning(main_window, "Not a Traced Result",
                                "Select a Clusters branch produced by Linear "
                                "Region Growing — it carries the traced lines.")
            return None
        return result_uid, node, traces_to_lines(traces, node.data.labels)

    @staticmethod
    def _show_branches(controller, tree_widget, result_uid, result_node,
                       features, params):
        """Show each ``(name, feature)`` as a branch under the result,
        replacing the one a previous preview made rather than stacking a new
        one; drop old preview branches this run did not redraw."""
        children = {child.params: uid
                    for uid, child in controller.data_nodes.data_nodes.items()
                    if child.parent_uid == result_node.uid}
        drawn = set()
        tree_widget.blockSignals(True)
        for name, feature in features:
            if feature is None:
                continue
            drawn.add(name)
            feature.cluster_reference = result_uid
            if name in children:
                uid = str(children[name])
                controller.get_node(uid).data = feature
                controller.cache_service.invalidate(uid)
                continue
            uid = controller.add_analysis_result(
                feature, "vector_feature", [result_node.uid], result_node,
                name, params)
            tree_widget.add_branch(uid, result_uid, name,
                                   tooltip=f"line_preview,{params}")
            item = tree_widget.branches_dict.get(uid)
            if item:
                item.setCheckState(0, Qt.Checked)
            tree_widget.visibility_status[uid] = True
        for name in ("predicted_paths", "prediction_uncertainty"):
            if name in children and name not in drawn:
                uid = str(children[name])
                tree_widget.remove_branch(uid)
                controller.remove_node(uid)
        tree_widget.blockSignals(False)


def _cone_segments(positions, sigma):
    """Frustum segments along a predicted path, each ``_CONE_SIGMAS`` sigma
    wide at either end — the uncertainty cone, as drawable wireframe pieces."""
    segments = []
    for a, b, sa, sb in zip(positions[:-1], positions[1:], sigma[:-1], sigma[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length > 1e-9:
            segments.append((a, seg / length, _CONE_SIGMAS * sa,
                             _CONE_SIGMAS * sb, length))
    return segments
