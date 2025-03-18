# plugins/menus/ground_menu_plugin.py (modified)
"""
Menu plugin that adds ground-related analysis menu items.
"""

from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class GroundMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds ground-related analysis menu items.

    This plugin adds a "Ground" submenu under the "Action" menu with
    items for ground surface analysis operations such as vertical distance
    calculation and slicing.
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
        Return the menu items for the Ground submenu.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "action": "_submenu_",
                "name": "Ground",
                "submenu": [
                    {
                        "name": "Calculate Distance to Ground",
                        "action": "dist_to_ground",
                        "tooltip": "Calculate vertical distance from points to nearest ground points"
                    },
                    {
                        "name": "Slice by Distance to Ground",
                        "action": "ground_slice",
                        "tooltip": "Extract points within a specific distance range from ground"
                    }
                    # Additional ground-related methods can be added here in the future
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
        # Open the appropriate dialog box for the selected ground analysis algorithm
        main_window.open_dialog_box(action_name)
