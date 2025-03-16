# plugins/menus/cluster_size_filter_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class ClusterSizeFilterMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds cluster size filtering functionality.

    This plugin adds a "Filter by Cluster Size" menu item to the "Action/Clustering"
    submenu, allowing users to filter clusters based on the number of points they
    contain. This is particularly useful after running clustering algorithms like
    DBSCAN to focus on significant clusters and remove small, potentially noisy ones.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Places the menu item in the Clustering submenu under the Action menu.

        Returns:
            str: The menu path "Action/Clustering"
        """
        return "Action/Clustering"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for cluster size filtering.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Filter by Cluster Size",
                "action": "cluster_size_filter",
                "tooltip": "Filter clusters based on minimum number of points"
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when the menu item is clicked.

        Args:
            action_name (str): The action identifier for the clicked menu item
            main_window: A reference to the main application window
        """
        # When the menu item is clicked, open the dialog box for cluster size filtering
        main_window.open_dialog_box(action_name)