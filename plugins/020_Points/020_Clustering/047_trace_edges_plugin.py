"""
Trace Edges - follow an edge where two surfaces meet, from picked points.

Pick a few points on an edge - a kerb's top or bottom, a building corner, a
wall on a floor, a roof ridge, a window head - and run this: the edge is
followed in both directions until it ends. Picks closer than Seed Group
Distance are one edge; pick several edges at once by picking each in its own
spot. Each group of picks is one edge.

At each step the surfaces round the edge are fitted as planes, the slice is
cut square to where they cross, and the edge is read where the right two
surfaces meet - the pair whose shape (which way each surface leaves the
edge) matches the one learned at the picks. So on a kerb the top and the
bottom are told apart by where you pick. Where a surface is hidden (a car,
a tree) the trace coasts on and picks the edge up again past the gap.

Writes the same result branch as Trace Lines (one label per edge, the rest
unassigned, the traced edge lines carried on it); Show Edge Lines adds them
as a branch to see and export.

The engine is ``core/services/edge_line_tracer.py``; everything around it -
reading the cloud and the picks, writing and expanding the result - is
Linear Region Growing's (``040_linear_region_growing_plugin.py``), as for
Trace Lines.
"""

import importlib
from typing import Dict, Any

from core.services.edge_line_tracer import EdgeLineTracer
from core.services.linear_region_grower import LinearRegionGrower, AXIS_TRACE

# The module, not the class (PLUGIN_ARCHITECTURE.md §2, "Gotcha").
_lrg = importlib.import_module(
    "plugins.020_Points.020_Clustering.040_linear_region_growing_plugin")

# Shown when nothing is picked yet.
_DEFAULT_SEED_EPS = 0.5


class TraceEdgesPlugin(_lrg.LinearRegionGrowingPlugin):

    DISPLAY_NAME = "Trace edges"

    def get_name(self) -> str:
        return "trace_edges"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            # --- Basic ----------------------------------------------------
            "seed_eps": {
                "type": "float",
                "default": self._from_selection(),
                "computed": True,
                "min": 0.001,
                "max": 100.0,
                "decimals": 3,
                "label": "Seed Group Distance (m)",
                "description": "Picks closer than this belong to one edge. "
                               "Filled in from your picks when this opens - "
                               "change it if you like.\n"
                               "Lower it if picks on two edges close together "
                               "(a kerb's top and bottom) end up as one.",
            },
            "reach": {
                "type": "float",
                "default": 0.5,
                "min": 0.05,
                "max": 10.0,
                "label": "Look Across (m)",
                "description": "How far either side of the edge each reading "
                               "looks to see the two surfaces.\n"
                               "Far enough to see a good piece of both, not so "
                               "far it sees the next edge's surfaces too.\n"
                               "Lower it for small features (a low kerb, a "
                               "window reveal). Raise it for big, sparse ones "
                               "(a building corner in a sparse scan).\n"
                               "Typical: 0.3-1 m.",
            },
            "min_angle": {
                "type": "float",
                "default": 15.0,
                "min": 1.0,
                "max": 170.0,
                "label": "Min Edge Angle (deg)",
                "description": "The gentlest turn between the two surfaces "
                               "that still counts as an edge (a square corner "
                               "turns 90).\n"
                               "Lower it for shallow ridges and slight wall "
                               "bends.\n"
                               "Raise it if the trace follows faint folds in a "
                               "surface (a road's camber, a gutter).",
            },
            "max_gap": {
                "type": "float",
                "default": 5.0,
                "min": 0.0,
                "max": 500.0,
                "label": "Max Gap (m)",
                "description": "How far an edge may be hidden - behind a parked "
                               "car, a tree, a hole in the scan - and still be "
                               "followed to where it comes back.\n"
                               "Raise it if edges stop at gaps.\n"
                               "Lower it if edges run on where the edge has "
                               "really ended or turned.",
            },
            "min_bend_radius": {
                "type": "float",
                "default": 3.0,
                "min": 0.1,
                "max": 100000.0,
                "label": "Min Bend Radius (m)",
                "description": "How tightly an edge may curve.\n"
                               "Small for curving edges (a kerb round a "
                               "corner: 2-5).\n"
                               "Large to keep straight edges straight (a "
                               "building corner: 50+).",
            },
            # --- Advanced -------------------------------------------------
            "window": {
                "type": "float",
                "default": 1.0,
                "min": 0.05,
                "max": 50.0,
                "label": "Window (m) - advanced",
                "description": "How long a piece of edge is looked at in each "
                               "step.\n"
                               "Raise it for sparse scans that stop early.\n"
                               "Lower it for tight curves.\n"
                               "Typical: 0.5-2 m.",
            },
            "step": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 25.0,
                "label": "Step (m) - advanced",
                "description": "How far the trace moves between readings. "
                               "0 = half the window.\n"
                               "Smaller follows bends more closely but is "
                               "slower.",
            },
            "flexibility": {
                "type": "float",
                "default": 0.02,
                "min": 0.0001,
                "max": 10.0,
                "decimals": 4,
                "label": "Bend Flexibility - advanced",
                "description": "How quickly an edge's curve may change along "
                               "it.\n"
                               "Raise it for edges whose bend keeps changing.\n"
                               "Lower it to keep straight edges straight "
                               "through noise.",
            },
            # --- Display --------------------------------------------------
            "show_lines": {
                "type": "bool",
                "default": True,
                "label": "Show Edge Lines",
                "description": "Add the traced edge lines as a branch you can "
                               "show and hide.",
            },
            "show_end_cylinders": {
                "type": "bool",
                "default": False,
                "label": "Show Stop Points",
                "description": "Mark where each edge ended and why, one branch "
                               "per reason: empty space (no edge seen for "
                               "longer than Max Gap - the edge ended, turned "
                               "too sharply, or one surface stopped), joined "
                               "another edge, length cap.",
            },
        }

    def _from_selection(self):
        """Seed Group Distance worked out from the current picks, to fill the
        dialog with; a typical value when nothing is picked."""
        try:
            from config.config import global_variables
            from application.selection_gate import selected_cloud_indices
            controller = global_variables.global_application_controller
            viewer = global_variables.global_pcd_viewer_widget
            if controller is None or viewer is None or len(controller.selected_branches) != 1:
                return _DEFAULT_SEED_EPS
            node = controller.get_node(controller.selected_branches[0])
            if node is None or not viewer.has_selection():
                return _DEFAULT_SEED_EPS
            source = str(node.parent_uid) if self._is_linear_result(node) else str(node.uid)
            points = controller.reconstruct(source).points
            picks = selected_cloud_indices(viewer, node.uid, points)
            if picks is None or len(picks) < 2:
                return _DEFAULT_SEED_EPS
            return round(_lrg._auto_seed_eps(points[picks]), 3)
        except Exception as e:           # never block the dialog on this
            print(f"Trace Edges: could not work out settings from the picks: {e}")
            return _DEFAULT_SEED_EPS

    def _make_growers(self, pc_points, params, mode, linearity):
        window = float(params.get("window", 1.0))
        step = float(params.get("step", 0.0)) or window / 2
        tracer = EdgeLineTracer(
            pc_points,
            reach=params.get("reach", 0.5),
            min_angle_deg=params.get("min_angle", 15.0),
            step=step,
            window=window,
            max_gap=params.get("max_gap", 5.0),
            min_bend_radius=params.get("min_bend_radius", 3.0),
            flexibility=params.get("flexibility", 0.02),
        )
        # Not used for edges (see _offer_extension); the shared run frees it.
        grower = LinearRegionGrower(all_points=pc_points, mode=AXIS_TRACE)
        return tracer, grower

    def _offer_extension(self, main_window, result_uid, pc_points, all_lines,
                         new_lines, grower, params, resolved):
        # The guided extension continues lines with the thin-line axis march,
        # which does not read an edge. To carry an edge on, pick past its end
        # and run Trace Edges again on the result branch.
        return False
