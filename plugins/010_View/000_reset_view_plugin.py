# plugins/View/reset_view_plugin.py
"""
Plugin for resetting the camera to its default position and orientation.

Workflow:
1. User clicks View → Reset View
2. Plugin calls viewer_widget.reset_view()
3. Camera returns to default zoom, rotation, pan, and distance
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class ResetViewPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "reset_view"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer_widget = global_variables.global_pcd_viewer_widget
        viewer_widget.reset_view()