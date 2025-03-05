# plugins/menus/noise_removal_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class NoiseRemovalMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds noise removal related menu items.

    This plugin adds a "Noise Removal" submenu under the "Action" menu with
    items for various noise filtering algorithms.
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
        Return the menu items for the Noise Removal submenu.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "action": "_submenu_",
                "name": "Noise Removal",
                "submenu": [
                    {
                        "name": "Statistical Outlier Removal (SOR)",
                        "action": "sor",
                        "tooltip": "Remove noise points based on statistical analysis of neighbor distances"
                    }
                    # Additional noise removal methods can be added here in the future
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
        # Open the appropriate dialog box for the selected noise removal algorithm
        main_window.open_dialog_box(action_name)