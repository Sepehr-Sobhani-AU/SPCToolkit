# plugins/100_Help/010_Keyboard and Mouse Handling/010_mouse_controls_plugin.py
"""
Plugin for displaying the mouse controls reference.

The bindings listed here are the ones handled by
`gui/widgets/pcd_viewer/_mouse_input.py` (MouseInputMixin), the viewer's zoom
window and polygon modes, and `gui/widgets/tree_structure_widget.py`.
"""

from typing import Any, Dict

from plugins.dialogs.shortcuts_dialog import ShortcutsDialog
from plugins.interfaces import ActionPlugin

# (combination, action) rows grouped into titled sections.
MOUSE_SECTIONS = [
    ("Camera — 3D viewer", [
        ("Left drag", "Rotate around the X and Y axes."),
        ("Ctrl + Left drag", "Rotate around the Z axis."),
        ("Right or Middle drag", "Pan along the X and Y axes."),
        ("Ctrl + Right/Middle drag", "Pan along the Z axis."),
        ("Wheel", "Zoom in and out."),
        ("Double Left click", "Move the centre of rotation to the clicked point."),
    ]),
    ("Selection — 3D viewer", [
        ("Shift + Left click", "Select the point under the cursor."),
        ("Shift + Right click", "Deselect the point under the cursor."),
        ("Ctrl + Shift + Right click", "Deselect the whole cluster under the cursor "
                                       "(matched by colour)."),
    ]),
    ("Zoom window mode (press Z)", [
        ("Left drag", "Draw the zoom rectangle; releasing zooms to it."),
        ("Right click", "Cancel zoom window mode."),
    ]),
    ("Polygon mode (press P or Shift + P)", [
        ("Left click", "Add a polygon vertex."),
        ("Right click", "Close the polygon and select (or deselect) the points inside."),
        ("Double Left click", "Close the polygon and select (or deselect) the points inside."),
    ]),
    ("Tree panel", [
        ("Left click", "Select a single branch, clearing any other selection."),
        ("Ctrl + Left click", "Add or remove a branch from the selection (multi-select)."),
        ("Click the Branch checkbox", "Show or hide the branch in the viewer."),
        ("Click the Cache checkbox", "Cache or uncache the branch. Root branches stay cached."),
    ]),
    ("Notes", [
        ("Camera in modes", "Camera drag is disabled while zoom window or polygon mode is "
                            "active. Press Esc to leave the mode."),
    ]),
]


class MouseControlsPlugin(ActionPlugin):
    """Action plugin showing the mouse controls reference dialog."""

    def get_name(self) -> str:
        """Return the plugin name."""
        return "mouse_controls"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - directly shows the reference dialog.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Show the mouse controls reference.

        Args:
            main_window: The main application window
            params: Not used for this plugin (empty dict)
        """
        dialog = ShortcutsDialog(main_window, "Mouse Controls", MOUSE_SECTIONS)
        dialog.exec_()
