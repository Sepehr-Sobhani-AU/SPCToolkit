# plugins/menus/selection_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class SelectionMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds selection-related menu items.

    This plugin adds items for separating selected points and clusters
    to the "Selection" menu.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "Selection"
        """
        return "Selection"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for selection operations.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Separate Selected Points",
                "action": "separate_selected_points",
                "tooltip": "Create a new branch containing only the selected points"
            },
            {
                "name": "Separate Selected Clusters",
                "action": "separate_selected_clusters",
                "tooltip": "Create a new branch containing only the selected clusters"
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