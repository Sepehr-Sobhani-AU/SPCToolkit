# plugins/menus/point_analysis_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class PointAnalysisMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds point analysis related menu items.

    This plugin adds a "Point Analysis" submenu under the "Action" menu with
    items for various point-based analysis algorithms, including average distance
    calculation and other point metrics.
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
        Return the menu items for the Point Analysis submenu.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "action": "_submenu_",
                "name": "Point Analysis",
                "submenu": [
                    {
                        "name": "Average Distance to k-NN",
                        "action": "average_distance",
                        "tooltip": "Calculate the average distance of each point to its k nearest neighbors"
                    }
                    # Additional point analysis methods can be added here in the future
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
        # Open the appropriate dialog box for the selected point analysis algorithm
        main_window.open_dialog_box(action_name)