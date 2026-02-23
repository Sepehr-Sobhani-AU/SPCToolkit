"""
Plugin for zooming into a user-drawn rectangular region.

Workflow:
1. User clicks View > Zoom Window (or presses Z)
2. Cursor changes to crosshair; user drags a rectangle over the region of interest
3. Camera zooms to fit the 3D points within that rectangle, preserving the current rotation
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class ZoomWindowPlugin(ActionPlugin):
    """Action plugin that activates zoom window mode in the 3D viewer."""

    def get_name(self) -> str:
        return "zoom_window"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer_widget = global_variables.global_pcd_viewer_widget
        if viewer_widget is not None:
            viewer_widget.enter_zoom_window_mode()
