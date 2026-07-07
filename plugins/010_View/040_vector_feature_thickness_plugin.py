"""
Plugin for setting the rendered thickness of vector-feature line geometry.

Vector features (mesh wireframes, CAD polylines) are drawn as GL lines, which
render 1px thin by default and are easy to lose against a dense point cloud.
This plugin sets the line width the viewer uses to draw that geometry.

Workflow:
1. User clicks View > Vector Feature Thickness
2. Dialog shows the current line width with a float spinner
3. Plugin sets the new value on the viewer widget
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class VectorFeatureThicknessPlugin(ActionPlugin):
    """Action plugin for adjusting the vector-feature line width in the viewer."""

    def get_name(self) -> str:
        return "vector_feature_thickness"

    def get_parameters(self) -> Dict[str, Any]:
        viewer_widget = global_variables.global_pcd_viewer_widget
        current = viewer_widget.line_width if viewer_widget else 1.0
        return {
            "line_width": {
                "type": "float",
                "default": current,
                "min": 1.0,
                "max": 20.0,
                "decimals": 1,
                "label": "Line Thickness",
                "description": "Width of rendered vector-feature lines (wireframes, polylines)",
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer_widget = global_variables.global_pcd_viewer_widget
        if viewer_widget is not None:
            viewer_widget.line_width = params["line_width"]
