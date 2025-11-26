# plugins/View/zoom_to_extent_plugin.py
"""
Plugin for zooming the camera to fit all visible points in the viewport.

This action plugin provides a convenient menu-based way to frame the entire
point cloud dataset optimally in the 3D viewer.

Workflow:
1. User clicks View → Zoom To Extent
2. Plugin calls viewer_widget.zoom_to_extent()
3. Camera adjusts to show all visible points with optimal framing
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class ZoomToExtentPlugin(ActionPlugin):
    """
    Action plugin for zooming camera to fit all visible points.

    This plugin provides a menu action that frames the entire point cloud
    in the viewport by calculating optimal camera distance and centering
    the view on the data.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "zoom_to_extent"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - directly executes zoom operation.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the zoom to extent action.

        Args:
            main_window: The main application window
            params: Not used for this plugin (empty dict)
        """
        # Get viewer widget from global variables
        viewer_widget = global_variables.global_pcd_viewer_widget

        # Call zoom to extent method
        viewer_widget.zoom_to_extent()
