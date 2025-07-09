# plugins/menus/feature_detection_menu_plugin.py
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class FeatureDetectionMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds the Feature Detection menu with classification options.

    This plugin creates a top-level "Feature Detection" menu in the application
    menu bar and adds submenu items for different feature detection operations,
    including manual classification.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "Feature Detection" as a top-level menu
        """
        return "Feature Detection"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for the Feature Detection menu.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions
        """
        return [
            {
                "name": "Classification",
                "action": "_submenu_",  # Special marker for a submenu
                "submenu": [
                    {
                        "name": "Manual Classification",
                        "action": "manual_classification",
                        "tooltip": "Manually classify clusters for feature detection"
                    }
                    # More classification options can be added here later
                ]
            }
            # More feature detection categories can be added here
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
