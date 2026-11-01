"""
Plugin for choosing how a left drag turns the view.

Orbit (default): left/right spins the view around the vertical Z axis and
up/down tilts it, so Z always stays pointing up — like walking around a model
on a turntable. Free: the older behaviour, turning around the screen's X and Y
axes, where Z can tip over.

Workflow:
1. User clicks View > Display Settings > Rotation Style
2. Dialog shows the current style
3. Plugin sets the chosen style on the viewer widget
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from config.config import global_variables

_LABELS = {"orbit": "Orbit", "free": "Free"}


class RotationStylePlugin(ActionPlugin):
    """Action plugin for switching between orbit and free rotation."""

    def get_name(self) -> str:
        return "rotation_style"

    def get_parameters(self) -> Dict[str, Any]:
        viewer_widget = global_variables.global_pcd_viewer_widget
        current = viewer_widget.rotation_style if viewer_widget else "orbit"
        return {
            "style": {
                "type": "choice",
                "options": list(_LABELS.values()),
                "default": _LABELS.get(current, "Orbit"),
                "label": "Rotation Style",
                "description": (
                    "Orbit: left/right spins around the vertical Z axis, up/down "
                    "tilts; Z stays up. Free: turns around the screen's X and Y axes."
                ),
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer_widget = global_variables.global_pcd_viewer_widget
        if viewer_widget is None:
            return
        by_label = {label: key for key, label in _LABELS.items()}
        viewer_widget.rotation_style = by_label.get(params["style"], "orbit")
