from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class ZoomToExtentPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "zoom_to_extent"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer_widget = global_variables.global_pcd_viewer_widget
        viewer_widget.zoom_to_extent(preserve_rotation=True)
