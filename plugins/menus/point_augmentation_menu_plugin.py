# plugins/menus/point_augmentation_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class PointAugmentationMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds point augmentation related menu items.

    This plugin adds a "Point Augmentation" submenu under the "Action" menu with
    items for various point augmentation techniques, such as MLS density augmentation.
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
        Return the menu items for the Point Augmentation submenu.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "action": "_submenu_",
                "name": "Point Augmentation",
                "submenu": [
                    {
                        "name": "MLS Density Augmentation",
                        "action": "mls_augmentation",
                        "tooltip": "Add points to create more uniform density using Moving Least Squares"
                    }
                    # Additional augmentation methods can be added here in the future
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
        # Open the appropriate dialog box for the selected point augmentation algorithm
        main_window.open_dialog_box(action_name)