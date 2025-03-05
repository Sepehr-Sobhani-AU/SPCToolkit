# plugins/menus/points_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class PointsMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds point-related operations.

    This plugin adds a "Points" menu with submenus for different point operations
    such as augmentation techniques.
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
        Return the menu items for points operations.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Augmentation",
                "action": "_submenu_",  # Special marker for a submenu
                "submenu": [
                    {
                        "name": "Moving Least Squares (MLS)",
                        "action": "mls_augmentation",
                        "tooltip": "Smooth and upsample the point cloud using Moving Least Squares"
                    }
                    # More augmentation methods will be added here later
                ]
            }
            # More point operations can be added here
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