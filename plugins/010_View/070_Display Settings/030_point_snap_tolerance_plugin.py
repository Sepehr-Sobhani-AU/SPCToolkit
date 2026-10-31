"""
Plugin for setting the point snap tolerance used when picking points.

Point picking (Shift + Left Click, Identify, Distance) snaps the click to the
nearest point within a distance threshold. That threshold is computed as
``max_extent * factor`` — this plugin exposes the ``factor`` so the user can
tighten snapping (small factor: only snap when clicking almost exactly on a
point) or loosen it (large factor: snap even from farther away).

Workflow:
1. User clicks View > Point Snap Tolerance
2. Dialog shows the current factor with a float spinner
3. Plugin sets the new value on the viewer widget
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class PointSnapTolerancePlugin(ActionPlugin):
    """Action plugin for adjusting the point-picking snap tolerance."""

    def get_name(self) -> str:
        return "point_snap_tolerance"

    def get_parameters(self) -> Dict[str, Any]:
        viewer_widget = global_variables.global_pcd_viewer_widget
        current = viewer_widget.picking_point_threshold_factor if viewer_widget else 1.0
        return {
            "snap_tolerance": {
                "type": "float",
                "default": current,
                "min": 0.001,
                "max": 5.0,
                "decimals": 3,
                "label": "Snap Tolerance",
                "description": (
                    "How close a click must be to snap to a point, as a factor of "
                    "the scene extent. Smaller = tighter (must click nearly on the "
                    "point); larger = looser."
                ),
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer_widget = global_variables.global_pcd_viewer_widget
        if viewer_widget is not None:
            viewer_widget.picking_point_threshold_factor = params["snap_tolerance"]
