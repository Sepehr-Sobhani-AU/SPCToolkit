# plugins/menus/draw_menu_plugin.py
"""
Draw menu plugin that adds drawing-related functionality to SPCToolkit.

This plugin creates a Draw menu with polygon drawing options including
Convex Hull, Concave Hull, and Alpha Shape algorithms for creating
boundaries around point cloud data.
"""
from typing import Dict, Any, List

from plugins.interfaces import MenuPlugin


class DrawMenuPlugin(MenuPlugin):
    """
    Menu plugin that adds drawing-related menu items.

    This plugin adds a "Draw" menu to the main menu bar with a "Polygon"
    submenu containing various polygon generation algorithms. Currently,
    only Alpha Shape is enabled while Convex Hull and Concave Hull are
    disabled pending implementation.
    """

    def get_menu_location(self) -> str:
        """
        Return the menu location for this plugin's items.

        Returns:
            str: The menu path "Draw" to create a new top-level menu
        """
        return "Draw"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return the menu items for drawing operations.

        This method defines the structure of the Draw menu, including
        the Polygon submenu with three polygon generation options.
        The items are configured with their enable/disable states as
        specified in the requirements.

        Returns:
            List[Dict[str, Any]]: List of menu item definitions with
                                  the Polygon submenu and its items
        """
        return [
            {
                "name": "Polygon",
                "action": "_submenu_",  # Special marker for a submenu
                "submenu": [
                    {
                        "name": "Convex Hull",
                        "action": "convex_hull",
                        "tooltip": "Generate convex hull polygon around selected points"
                    },
                    {
                        "name": "Concave Hull",
                        "action": "concave_hull",
                        "tooltip": "Generate concave hull polygon around selected points"
                    },
                    {
                        "name": "Alpha Shape",
                        "action": "alpha_shape",
                        "tooltip": "Generate alpha shape polygon around selected points"
                    }
                ]
            }
        ]

    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is clicked.

        This method processes the action triggered by clicking on enabled
        menu items. Currently, only the alpha_shape action is handled since
        it's the only enabled item. The method opens the appropriate dialog
        box for parameter input when alpha_shape is selected.

        Args:
            action_name (str): The action identifier for the clicked menu item.
                             Expected values: "alpha_shape" (only enabled action)
            main_window: A reference to the main application window, used for
                        opening dialog boxes and accessing application state

        Note:
            Disabled menu items (convex_hull, concave_hull) won't trigger
            this method as they are not clickable in the UI.
        """
        # Only handle enabled actions
        if action_name == "alpha_shape":
            # Open the dialog box for alpha shape parameters
            main_window.open_dialog_box(action_name)
        # Note: convex_hull and concave_hull actions won't be triggered
        # since they are disabled, but we could add handling here for
        # future implementation when they are enabled
