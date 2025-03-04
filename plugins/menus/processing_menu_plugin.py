# plugins/menus/processing_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class ProcessingMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds data processing menu items.

    This plugin adds items for processing operations to the "Action" menu.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "Action"
        """
        return "Action"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for data processing operations.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Subsampling",
                "action": "_submenu_",  # Special marker for a submenu
                "submenu": [
                    {
                        "name": "Random Subsampling",
                        "action": "subsampling",
                        "tooltip": "Reduce the number of points by random selection"
                    },
                    {
                        "name": "Density-Based Subsampling",
                        "action": "density_subsampling",
                        "tooltip": "Reduce the number of points based on spatial density (voxel-based)"
                    }
                ]
            },
            {
                "name": "Filtering",
                "action": "filtering",
                "tooltip": "Filter points based on a condition"
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        # Open the appropriate dialog box for the selected operation
        main_window.open_dialog_box(action_name)