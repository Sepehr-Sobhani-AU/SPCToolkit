# plugins/100_Help/010_Keyboard and Mouse Handling/000_keyboard_shortcuts_plugin.py
"""
Plugin for displaying the keyboard shortcuts reference.

The bindings listed here are the ones handled by
`gui/widgets/pcd_viewer/_key_input.py` (KeyInputMixin). The application has no
menu accelerators, so the 3D viewer must have focus for any of them to fire.
"""

from typing import Any, Dict

from plugins.dialogs.shortcuts_dialog import ShortcutsDialog
from plugins.interfaces import ActionPlugin

# (combination, action) rows grouped into titled sections.
KEYBOARD_SECTIONS = [
    ("View — 3D viewer", [
        ("F", "Zoom to extent (fit all visible points in the viewport)."),
        ("Ctrl + R", "Reset the camera to its default view."),
        ("Z", "Toggle zoom window mode, then drag a rectangle to zoom into it."),
    ]),
    ("Selection — 3D viewer", [
        ("P", "Toggle polygon selection mode."),
        ("Shift + P", "Toggle polygon deselect mode (removes points from the selection)."),
        ("Esc", "Cancel zoom window or polygon mode. Otherwise, deselect all points "
                "after confirmation."),
    ]),
    ("Display size — 3D viewer", [
        ("+ or =", "Increase point size (x1.2)."),
        ("- or _", "Decrease point size (/1.2)."),
        ("Shift + +", "Increase vector feature line width (x1.2)."),
        ("Shift + -", "Decrease vector feature line width (/1.2)."),
    ]),
    ("Clusters — 3D viewer", [
        ("C", "Split the selected clusters (runs the Split Clusters plugin)."),
        ("M", "Merge the selected clusters (runs the Merge Clusters plugin)."),
        ("Delete", "Remove the selected clusters (runs the Remove Clusters plugin)."),
    ]),
    ("Notes", [
        ("Focus", "The 3D viewer must have keyboard focus. Click in the viewer first if a "
                  "key does nothing."),
        ("Menus", "Menu items have no keyboard accelerators. Every shortcut above belongs "
                  "to the viewer."),
    ]),
]


class KeyboardShortcutsPlugin(ActionPlugin):
    """Action plugin showing the keyboard shortcut reference dialog."""

    def get_name(self) -> str:
        """Return the plugin name."""
        return "keyboard_shortcuts"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - directly shows the reference dialog.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Show the keyboard shortcut reference.

        Args:
            main_window: The main application window
            params: Not used for this plugin (empty dict)
        """
        dialog = ShortcutsDialog(main_window, "Keyboard Shortcuts", KEYBOARD_SECTIONS)
        dialog.exec_()
