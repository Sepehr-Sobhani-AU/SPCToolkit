"""
Trace Edges - follow an edge where two surfaces meet, from picked points.

Shift+click once near an edge - a kerb's top or bottom, a building corner,
a wall on a floor, a roof ridge, a window head - and run this: the edge is
followed in both directions until it ends. Each click is one edge, so
several clicks trace several edges in one run (Picking = One click per
edge; click on the side of the edge you mean, within about 10 cm). Or pick
points along each edge (Picking = Points along each edge): picks closer than
Seed Group Distance are one edge.

At each step the surfaces round the edge are fitted as planes, the slice is
cut square to where they cross, and the edge is read where the right two
surfaces meet - the pair whose shape (which way each surface leaves the
edge) matches the one learned at the picks. So on a kerb the top and the
bottom are told apart by where you pick. Where a surface is hidden (a car,
a tree) the trace coasts on and picks the edge up again past the gap.

On curves (a kerb round a corner): steps get shorter where the edge turns
(Dynamic Step), and a reading that fails is tried again on a half and then a
quarter of the window, which hardly bends - so tight kerb returns are
followed without a preset. Where a kerb's face is hidden in a scan shadow,
its top edge is read where the footpath ends. Where a kerb is lowered (a
driveway, a pram ramp) the top edge is bridged across; with Follow Lowered
Edges the bottom edge follows the gentle fold of the ramp.

Tracer = Line runs Trace Lines' thin-line engine instead, with Line Radius.

Writes the same result branch as Trace Lines (one label per edge, the rest
unassigned, the traced edge lines carried on it); Show Edge Lines adds them
as a branch to see and export, with the stretches not read as a normal edge
- lowered (orange) and bridged across a gap (magenta) - as their own
branches.

The engine is ``core/services/edge_line_tracer.py``; everything around it -
reading the cloud and the picks, writing and expanding the result - is
Linear Region Growing's (``040_linear_region_growing_plugin.py``), as for
Trace Lines.
"""

import importlib
from typing import Dict, Any

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from core.services.edge_line_tracer import EdgeLineTracer, MARK_LOWERED, MARK_BRIDGED
from core.services.kalman_line_tracer import KalmanLineTracer
from core.services.linear_region_grower import (
    LinearRegionGrower, AXIS_TRACE, centerlines_to_vector_feature)

# The module, not the class (PLUGIN_ARCHITECTURE.md §2, "Gotcha").
_lrg = importlib.import_module(
    "plugins.020_Points.020_Clustering.040_linear_region_growing_plugin")

# Shown when nothing is picked yet.
_DEFAULT_SEED_EPS = 0.5

PICK_CLICKS = "One click per edge"
PICK_ALONG = "Points along each edge"

TRACER_EDGE = "Edge - where two surfaces meet"
TRACER_LINE = "Line - as Trace Lines"

# Colours of the marked stretches.
_MARK_COLORS = {
    MARK_LOWERED: np.array([1.0, 0.55, 0.0], dtype=np.float32),   # orange
    MARK_BRIDGED: np.array([1.0, 0.0, 1.0], dtype=np.float32),    # magenta
}


class TraceEdgesPlugin(_lrg.LinearRegionGrowingPlugin):

    DISPLAY_NAME = "Trace edges"

    def get_name(self) -> str:
        return "trace_edges"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            # --- Basic ----------------------------------------------------
            "tracer": {
                "type": "dropdown",
                "options": {TRACER_EDGE: TRACER_EDGE, TRACER_LINE: TRACER_LINE},
                "default": TRACER_EDGE,
                "label": "Tracer",
                "description": "Edge: reads where two surfaces meet (kerb top "
                               "or bottom, building corner, wall on floor).\n"
                               "Line: reads a thin line of points as Trace Lines "
                               "does (cable, pipe, rail), with Line Radius - "
                               "Look Across, Min Edge Angle and Follow Lowered "
                               "Edges are not used.",
            },
            "line_radius": {
                "type": "float",
                "default": 0.03,
                "min": 0.001,
                "max": 10.0,
                "decimals": 3,
                "label": "Line Radius (m) - Line tracer",
                "description": "Only for Tracer = Line: how thick the line is "
                               "(as in Trace Lines). Not measured here: on a "
                               "surface the whole surface reads as one wide "
                               "line and the trace crawls.",
            },
            "picking": {
                "type": "dropdown",
                "options": {PICK_CLICKS: PICK_CLICKS, PICK_ALONG: PICK_ALONG},
                "default": PICK_CLICKS,
                "label": "Picking",
                "description": "One click per edge: Shift+click once near "
                               "each edge you want - on the side of it you mean, "
                               "within about 10 cm (a kerb's top and bottom are "
                               "close: click nearer the one you want). Each click "
                               "is one edge; several clicks trace several edges "
                               "in one run.\n"
                               "Points along each edge: select several points "
                               "along an edge (a lasso or clicks); picks closer "
                               "than Seed Group Distance are one edge.\n"
                               "Edge tracer only - the Line tracer needs points "
                               "along each line.",
            },
            "seed_eps": {
                "type": "float",
                "default": self._from_selection(),
                "computed": True,
                "min": 0.001,
                "max": 100.0,
                "decimals": 3,
                "label": "Seed Group Distance (m)",
                "description": "Points along each edge only. "
                               "Picks closer than this belong to one edge. "
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
            "follow_lowered": {
                "type": "bool",
                "default": False,
                "label": "Follow Lowered Edges",
                "description": "Where a kerb is lowered (a driveway, a pram "
                               "ramp) its face shrinks to nothing. On: the "
                               "BOTTOM edge follows the gentle fold where the "
                               "ramp meets the road (shown orange). Off: it "
                               "coasts across and picks the kerb up past the "
                               "ramp (shown magenta when it does).\n"
                               "A top edge always coasts across.\n"
                               "Check the orange stretches: on a ramp gentle "
                               "folds are everywhere.",
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
                               "0 = half the window. With Dynamic Step on, "
                               "this is the longest step.\n"
                               "Smaller follows bends more closely but is "
                               "slower.",
            },
            "dynamic_step": {
                "type": "bool",
                "default": True,
                "label": "Dynamic Step - advanced",
                "description": "Shorter steps where the edge turns - down to a "
                               "fifth of Step on a tight curve - and after a "
                               "reading that came in off where it was expected; "
                               "full steps on straights. Follows curves more "
                               "closely at nearly the same speed.\n"
                               "Off: the same step everywhere.",
            },
            "flexibility": {
                "type": "float",
                "default": 0.3,
                "min": 0.0001,
                "max": 10.0,
                "decimals": 4,
                "label": "Bend Flexibility - advanced",
                "description": "How quickly an edge's curve may change along "
                               "it - a kerb goes from straight into a corner "
                               "and out again within a few metres.\n"
                               "Raise it if edges run off where a curve starts "
                               "or ends.\n"
                               "Lower it to keep straight edges straight "
                               "through noise.",
            },
            # --- Display --------------------------------------------------
            "show_lines": {
                "type": "bool",
                "default": True,
                "label": "Show Edge Lines",
                "description": "Add the traced edge lines as a branch you can "
                               "show and hide, and the stretches not read as a "
                               "normal edge: lowered (orange) and bridged across "
                               "a gap (magenta).",
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

    def _resolve_seed_groups(self, viewer_widget, pc_points, params, main_window,
                             node=None):
        """One click per edge: each Shift+click on the branch is its own
        seed group, in click order. Points along each edge: as Trace Lines."""
        if params.get("picking", PICK_CLICKS) != PICK_CLICKS:
            return super()._resolve_seed_groups(viewer_widget, pc_points, params,
                                                main_window, node=node)
        if params.get("tracer", TRACER_EDGE) == TRACER_LINE:
            QMessageBox.warning(
                main_window, "Trace Edges",
                "The Line tracer needs points along each line.\n"
                "Set Picking to 'Points along each edge', or Tracer to 'Edge'.")
            return None
        rows = [row for uid, row in viewer_widget.picked_points
                if node is not None and uid == str(node.uid)]
        rows = [r for r in dict.fromkeys(rows) if 0 <= r < len(pc_points)]
        if not rows:
            QMessageBox.warning(
                main_window, "No Clicks",
                "One click per edge: Shift+click once near each edge you want "
                "to trace, then run Trace Edges.\n\n"
                "(A lasso selects many points without naming one: for a lasso "
                "set Picking to 'Points along each edge'.)")
            return None
        return [np.array([r], dtype=np.intp) for r in rows]

    def _make_growers(self, pc_points, params, mode, linearity):
        window = float(params.get("window", 1.0))
        step = float(params.get("step", 0.0)) or window / 2
        common = dict(
            step=step,
            window=window,
            max_gap=params.get("max_gap", 5.0),
            min_bend_radius=params.get("min_bend_radius", 3.0),
            flexibility=params.get("flexibility", 0.3),
        )
        if params.get("tracer", TRACER_EDGE) == TRACER_LINE:
            # Trace Lines' engine, for a line along a surface; its other
            # settings at Trace Lines' defaults.
            tracer = KalmanLineTracer(pc_points, open_air=False,
                                      radius=params.get("line_radius", 0.03), **common)
        else:
            tracer = EdgeLineTracer(
                pc_points,
                reach=params.get("reach", 0.5),
                min_angle_deg=params.get("min_angle", 15.0),
                follow_lowered=params.get("follow_lowered", False),
                dynamic_step=params.get("dynamic_step", True),
                **common,
            )
        # Not used for edges (see _offer_extension); the shared run frees it.
        grower = LinearRegionGrower(all_points=pc_points, mode=AXIS_TRACE)
        return tracer, grower

    def _build_debug_branches(self, controller, tree_widget, node, result_uid, lines, params):
        super()._build_debug_branches(controller, tree_widget, node, result_uid,
                                      lines, params)
        if not params.get("show_lines"):
            return
        extras = []
        for kind, name in ((MARK_LOWERED, "lowered_stretches"),
                           (MARK_BRIDGED, "bridged_gaps")):
            pieces = [v for line in lines for k, v in getattr(line, "marks", ()) if k == kind]
            vf = centerlines_to_vector_feature(pieces, color=_MARK_COLORS[kind])
            if vf is not None:
                vf.cluster_reference = result_uid
                extras.append((name, vf))
        self._add_debug_branches(controller, tree_widget, node, result_uid,
                                 extras, params)

    def _offer_extension(self, main_window, result_uid, pc_points, all_lines,
                         new_lines, grower, params, resolved):
        # The guided extension continues lines with the thin-line axis march,
        # which does not read an edge. To carry an edge on, pick past its end
        # and run Trace Edges again on the result branch.
        return False
