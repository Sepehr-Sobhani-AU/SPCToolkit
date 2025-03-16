# plugins/menus/view_menu_plugin.py
"""
View menu plugin that adds view control operations to the SPCToolkit.

This plugin creates a View menu with options for controlling the viewport and camera,
including zooming to fit all visible points in the view.
"""
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin

from config.config import global_variables


class ViewMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds view-related menu items.

    This plugin adds a "View" menu with options for controlling the viewport and camera,
    including functionality like zooming to fit all visible points in the view.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "View"
        """
        return "View"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for viewport operations.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Zoom Extent",
                "action": "zoom_extent",
                "tooltip": "Fit all visible point cloud data in view",
            },
            # Additional view-related operations can be added here in the future
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        global_variables.global_pcd_viewer_widget.zoom_extent()
