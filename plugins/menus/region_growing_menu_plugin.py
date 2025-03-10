# plugins/menus/region_growing_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class RegionGrowingMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds region growing related menu items.

    This plugin adds a "Region Growing" submenu under the "Action" menu with
    items for various region growing segmentation techniques.
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
        Return the menu items for the Region Growing submenu.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "action": "_submenu_",
                "name": "Region Growing",
                "submenu": [
                    {
                        "name": "Surface Region Growing",
                        "action": "surface_region_growing",
                        "tooltip": "Segment point cloud into regions based on surface continuity"
                    }
                    # Additional region growing methods can be added here in the future
                ]
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        # Open the appropriate dialog box for the selected region growing algorithm
        main_window.open_dialog_box(action_name)