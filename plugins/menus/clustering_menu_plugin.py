
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class ClusteringMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds clustering-related menu items.

    This plugin adds a "Clustering" submenu under the "Action" menu with
    items for various clustering algorithms.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "Action/Clustering"
        """
        return "Action/Clustering"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for clustering algorithms.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "DBSCAN",
                "action": "dbscan",
                "tooltip": "Density-Based Spatial Clustering of Applications with Noise"
            },
            {
                "name": "K-Means",
                "action": "kmeans",
                "tooltip": "K-Means clustering algorithm"
            },
            {
                "name": "Region Growing",
                "action": "region_growing",
                "tooltip": "Region growing segmentation algorithm"
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        # Open the appropriate dialog box for the selected clustering algorithm
        main_window.open_dialog_box(action_name)