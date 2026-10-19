"""
Trace Lines - follow linear features from picked points.

Pick points along one or more lines (cable, pipe, kerb, edge, rail ...) and
run this: each line is followed in both directions from the picks until it
ends. The line's width is measured from its own points, so there is no width
to set. It looks ahead, carries on through clutter and across gaps, finds the
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


class TraceLinesPlugin(_lrg.LinearRegionGrowingPlugin):

    DISPLAY_NAME = "Trace lines"

    def get_name(self) -> str:
        return "trace_lines"

    def get_parameters(self) -> Dict[str, Any]:
        return {
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
                               "Typical: 0.5-3 m. The line's width is measured "
                               "from your picks - nothing to set.",
            },
            "show_lines": {
                "type": "bool",
                "default": False,
                "label": "Show Centerlines",
                "description": "Also add each line's traced centre line as a "
                               "branch you can show and hide.",
            },
        }

    def _make_growers(self, pc_points, params, mode, linearity):
        window = float(params.get("window", 1.0))
        tracer = KalmanLineTracer(
            pc_points,
            step=window / 2,
            window=window,
            max_gap=params.get("max_gap", 20.0),
            max_turn_deg=params.get("max_kink", 30.0),
            min_bend_radius=params.get("min_bend_radius", 50.0),
            open_air=params.get("open_air", True),
        )
        # The line extension offered after a run continues lines with the
        # axis march; its defaults are Linear Region Growing's.
        grower = LinearRegionGrower(all_points=pc_points, mode=AXIS_TRACE)
        return tracer, grower
