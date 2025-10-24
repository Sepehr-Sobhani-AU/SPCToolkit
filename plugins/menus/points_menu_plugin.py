# plugins/menus/points_menu_plugin.py
"""
Points menu plugin that adds point cloud operations to the SPCToolkit.

This plugin creates a Points menu with various point cloud manipulation operations.
"""
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class PointsMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds point-related menu items.

    This plugin adds a "Points" menu with operations for manipulating
    point clouds, such as subsampling and draping operations.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "Points"
        """
        return "Points"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for point operations.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Subsample",
                "action": "subsample",
                "tooltip": "Randomly subsample points from the point cloud",
            },
            {
                "name": "Density Subsampling",
                "action": "density_subsampling",
                "tooltip": "Subsample points based on voxel density",
            },
            {
                "name": "Drape on XY Plane",
                "action": "drape_on_xy_plane",
                "tooltip": "Project all points to a constant Z height",
            },
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        # Open the dialog box for the selected analysis type
        main_window.open_dialog_box(action_name)