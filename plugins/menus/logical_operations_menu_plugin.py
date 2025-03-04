# plugins/menus/logical_operations_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class LogicalOperationsMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds logical operations menu items.

    This plugin adds a "Logical Operations" submenu under the "Action" menu with
    items for various logical operations between point clouds.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "Action/Logical Operations"
        """
        return "Action/Logical Operations"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for logical operations.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Subtract",
                "action": "subtract",
                "tooltip": "Subtract one point cloud from another"
            },
            {
                "name": "Union",
                "action": "union",
                "tooltip": "Union of two point clouds"
            },
            {
                "name": "Intersect",
                "action": "intersect",
                "tooltip": "Intersection of two point clouds"
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        # Open the appropriate dialog box for the selected logical operation
        main_window.open_dialog_box(action_name)