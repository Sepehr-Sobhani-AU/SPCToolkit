# plugins/menus/eigenvalue_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class EigenvalueMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds eigenvalue analysis menu items.

    This plugin adds an "Eigenvalue Analysis" submenu under the "Action/Surface Analysis" menu with
    items for calculating, visualizing, analyzing, and filtering eigenvalues.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "Action/Surface Analysis"
        """
        return "Action/Surface Analysis"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for eigenvalue analysis.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Calculate Eigenvalues",
                "action": "eigenvalue_calculation",
                "tooltip": "Calculate eigenvalues of local neighbourhoods in the point cloud"
            },
            {
                "name": "Visualize Eigenvalues",
                "action": "eigenvalue_visualization",
                "tooltip": "Visualize eigenvalue properties using colour mapping"
            },
            {
                "name": "Analyze Eigenvalues",
                "action": "eigenvalue_analysis",
                "tooltip": "Perform geometric analysis based on eigenvalues (find planes, edges, corners)"
            },
            {
                "name": "Filter by Eigenvalues",
                "action": "eigenvalue_filtering",
                "tooltip": "Filter points based on eigenvalue thresholds or geometric properties"
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        # Open the appropriate dialog box for the selected action
        main_window.open_dialog_box(action_name)