"""
Plugin for setting the rendered point size in the 3D viewer.

Workflow:
1. User clicks View > Point Size
2. Dialog shows current point size with a float spinner (0.5 - 20.0)
3. Plugin sets the new value on the viewer widget
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class PointSizePlugin(ActionPlugin):
    """Action plugin for adjusting the 3D viewer point size."""

    def get_name(self) -> str:
        return "point_size"

    def get_parameters(self) -> Dict[str, Any]:
        viewer_widget = global_variables.global_pcd_viewer_widget
        current = viewer_widget.point_size if viewer_widget else 0.5
        return {
            "point_size": {
                "type": "float",
                "default": current,
                "min": 0.5,
                "max": 20.0,
                "decimals": 1,
                "label": "Point Size",
                "description": "Size of rendered points in the 3D viewer"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer_widget = global_variables.global_pcd_viewer_widget
        viewer_widget.point_size = params["point_size"]
