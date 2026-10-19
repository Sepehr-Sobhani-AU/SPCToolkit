"""
Trace Lines - follow linear features from picked points.

Pick points along one or more lines (cable, pipe, kerb, edge, rail ...) and
run this: each line is followed in both directions from the picks until it
ends. Seed Group Distance and Line Radius are filled in from the picks when
the dialog opens (measured at the picks), and can be changed. Show Stop Points
marks where and why each line ended. It looks ahead, carries on through clutter and across gaps, finds the
line again after a kink, and traces each line once - a line that reaches
another line joins it there.

Writes the same result branch as Linear Region Growing (one label per line,
the rest unassigned, the traced centerlines carried on it), so:

- running again with the result branch selected adds more lines to it;
- Clusters > Undo Cluster Edit takes the last run back;
- Extend Traced Lines continues lines that stopped short.

The engine is ``core/services/kalman_line_tracer.py``; everything around it -
reading the cloud and the picks, writing and expanding the result, the
extension offered afterwards - is Linear Region Growing's
(``040_linear_region_growing_plugin.py``), so the two tools stay in step.
"""

import importlib
from typing import Dict, Any

from core.services.kalman_line_tracer import KalmanLineTracer
from core.services.linear_region_grower import LinearRegionGrower, AXIS_TRACE

# The module, not the class: importing the class into this module would
# register Linear Region Growing a second time under this file's menu path
# (PLUGIN_ARCHITECTURE.md §2, "Gotcha").
_lrg = importlib.import_module(
    "plugins.020_Points.020_Clustering.040_linear_region_growing_plugin")

# Shown when nothing is picked yet (the dialog fills in values worked out
# from the picks when there are some).
_DEFAULT_SEED_EPS = 0.5
_DEFAULT_LINE_RADIUS = 0.03


class TraceLinesPlugin(_lrg.LinearRegionGrowingPlugin):

    DISPLAY_NAME = "Trace lines"

    def get_name(self) -> str:
        return "trace_lines"

    def get_parameters(self) -> Dict[str, Any]:
        seed_eps, line_radius = self._from_selection()
        return {
            # --- Basic ----------------------------------------------------
            "seed_eps": {
                "type": "float",
                "default": seed_eps,
                "computed": True,
                "min": 0.001,
                "max": 100.0,
                "decimals": 3,
                "label": "Seed Group Distance (m)",
                "description": "Picks closer than this belong to one group. "
                               "Filled in from your picks when this opens - "
                               "change it if you like.\n"
                               "Raise it if one line's picks are split into "
                               "many groups (slower, same result).\n"
                               "Lower it if picks on different lines end up "
                               "together.",
            },
            "line_radius": {
                "type": "float",
                "default": line_radius,
                "computed": True,
                "min": 0.001,
                "max": 10.0,
                "decimals": 3,
                "label": "Line Radius (m)",
                "description": "How thick the line is: the radius that holds "
                               "most of its points, scanner noise included. "
                               "Measured at your picks when this opens - change "
                               "it if you like. Used for every line in this run.\n"
                               "Lower it if lines take points of lines beside "
                               "them.\n"
                               "Raise it if lines break up or stop on a clear "
                               "line.",
            },
            "max_gap": {
                "type": "float",
                "default": 20.0,
                "min": 0.0,
                "max": 500.0,
                "label": "Max Gap (m)",
                "description": "How far a line may disappear - behind a tree, "
                               "through a hole in the scan - and still be "
                               "followed to where it comes back.\n"
                               "Raise it if lines stop at gaps.\n"
                               "Lower it if lines reach across empty space "
                               "to other lines.\n"
                               "Typical: 5-20 m.",
            },
            "max_kink": {
                "type": "float",
                "default": 30.0,
                "min": 0.0,
                "max": 90.0,
                "label": "Max Kink (deg)",
                "description": "The sharpest sudden turn a line may take, e.g. "
                               "a cable at a pole or a pipe at an elbow.\n"
                               "Raise it if lines stop at corners.\n"
                               "Lower it if lines turn off onto other lines.\n"
                               "0 = never turn sharply. Typical: 20-45.",
            },
            "min_bend_radius": {
                "type": "float",
                "default": 50.0,
                "min": 0.1,
                "max": 100000.0,
                "label": "Min Bend Radius (m)",
                "description": "How tightly a line may curve smoothly (sharp "
                               "kinks are the setting above).\n"
                               "Large for straight or gently sagging lines "
                               "(cables, rails: 50+).\n"
                               "Small for curving ones (a kerb round a "
                               "corner: 2-5).\n"
                               "Lower it if lines stop on curves.",
            },
            "open_air": {
                "type": "bool",
                "default": True,
                "label": "Line in Open Air",
                "description": "On: the line stands clear of other points "
                               "(cable, pipe, rail) - anything crowding round "
                               "it, like leaves or a pole, is clutter and is "
                               "passed through.\n"
                               "Off: the line lies on or along a surface (kerb, "
                               "wall edge) - points around it are expected.",
            },
            # --- Advanced -------------------------------------------------
            "window": {
                "type": "float",
                "default": 1.0,
                "min": 0.05,
                "max": 50.0,
                "label": "Window (m) - advanced",
                "description": "How long a piece of line is looked at in each "
                               "step.\n"
                               "Raise it for sparse lines (few points per "
                               "metre) that stop early.\n"
                               "Lower it for tight curves.\n"
                               "Typical: 0.5-3 m.",
            },
            "step": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 25.0,
                "label": "Step (m) - advanced",
                "description": "How far the trace moves between looks at the "
                               "line. 0 = half the window.\n"
                               "Smaller follows bends more closely but is "
                               "slower.",
            },
            "min_points": {
                "type": "int",
                "default": 3,
                "min": 2,
                "max": 100,
                "label": "Min Points - advanced",
                "description": "Fewest points that count as seeing the line in "
                               "one window.\n"
                               "Raise it if lines wander onto stray points or "
                               "noise.\n"
                               "Lower it for very sparse lines that stop early.",
            },
            "gap_search_cone": {
                "type": "float",
                "default": 15.0,
                "min": 1.0,
                "max": 60.0,
                "label": "Gap Search Cone (deg) - advanced",
                "description": "How far off its direction a line is looked for "
                               "on the far side of a gap.\n"
                               "Raise it if lines do not pick up again after a "
                               "gap where they change direction.\n"
                               "Lower it if lines jump to other lines across "
                               "gaps.",
            },
            "flexibility": {
                "type": "float",
                "default": 0.02,
                "min": 0.0001,
                "max": 10.0,
                "decimals": 4,
                "label": "Bend Flexibility - advanced",
                "description": "How quickly a line's curve may change along it.\n"
                               "Raise it for lines whose bend keeps changing "
                               "(winding kerbs).\n"
                               "Lower it to keep straight lines straight "
                               "through noise.",
            },
            # --- Display --------------------------------------------------
            "show_lines": {
                "type": "bool",
                "default": False,
                "label": "Show Centerlines",
                "description": "Also add each line's traced centre line as a "
                               "branch you can show and hide.",
            },
            "show_end_cylinders": {
                "type": "bool",
                "default": False,
                "label": "Show Stop Points",
                "description": "Mark where each line ended and why, one branch "
                               "per reason: empty space (the gap was longer "
                               "than Max Gap), in clutter, unrelated line ahead "
                               "(the far side did not continue this line), "
                               "joined another line, length cap. The quickest "
                               "way to see which setting to change.",
            },
        }

    def _from_selection(self):
        """Seed Group Distance and Line Radius worked out from the current
        picks, to fill the dialog with; typical values when nothing is picked
        or the cloud cannot be read."""
        seed_eps, line_radius = _DEFAULT_SEED_EPS, _DEFAULT_LINE_RADIUS
        try:
            from config.config import global_variables
            from application.selection_gate import selected_cloud_indices
            controller = global_variables.global_application_controller
            viewer = global_variables.global_pcd_viewer_widget
            if controller is None or viewer is None or len(controller.selected_branches) != 1:
                return seed_eps, line_radius
            node = controller.get_node(controller.selected_branches[0])
            if node is None or not viewer.has_selection():
                return seed_eps, line_radius
            source = str(node.parent_uid) if self._is_linear_result(node) else str(node.uid)
            points = controller.reconstruct(source).points
            picks = selected_cloud_indices(viewer, node.uid, points)
            if picks is None or len(picks) < 2:
                return seed_eps, line_radius
            seed_eps = round(_lrg._auto_seed_eps(points[picks]), 3)
            tracer = KalmanLineTracer(points)
            try:
                measured = tracer.measure_radius(picks)
            finally:
                tracer.release()
            if measured:
                line_radius = round(measured, 3)
        except Exception as e:           # never block the dialog on this
            print(f"Trace Lines: could not work out settings from the picks: {e}")
        return seed_eps, line_radius

    def _make_growers(self, pc_points, params, mode, linearity):
        window = float(params.get("window", 1.0))
        step = float(params.get("step", 0.0)) or window / 2
        tracer = KalmanLineTracer(
            pc_points,
            step=step,
            window=window,
            max_gap=params.get("max_gap", 20.0),
            max_turn_deg=params.get("max_kink", 30.0),
            bridge_cone_deg=params.get("gap_search_cone", 15.0),
            min_bend_radius=params.get("min_bend_radius", 50.0),
            flexibility=params.get("flexibility", 0.02),
            open_air=params.get("open_air", True),
            radius=params.get("line_radius") or None,
            min_points=params.get("min_points", 3),
        )
        # The line extension offered after a run continues lines with the
        # axis march; its defaults are Linear Region Growing's.
        grower = LinearRegionGrower(all_points=pc_points, mode=AXIS_TRACE)
        return tracer, grower
