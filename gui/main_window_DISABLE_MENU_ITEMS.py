# gui/main_window.py
"""
  Create main window for the PCD Toolkit application with dynamic menus from plugins.
"""
# define the global variables
from config.config import global_variables

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox
from gui.widgets import PCDViewerWidget, TreeStructureWidget
from core.data_manager import DataManager
from services.file_manager import FileManager
from gui.dialog_boxes.dialog_boxes_manager import DialogBoxesManager
from plugins.plugin_manager import PluginManager


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the SPCToolkit, with a tree structure widget as a left pane.

    This version uses plugins to dynamically build its menu structure.
    """

    def __init__(self, plugin_manager: PluginManager):
        super().__init__()
        self.plugin_manager = plugin_manager

        # Set up the main window components
        self.menus = {}  # Dictionary to store menu/submenu references
        self.actions = {}  # Dictionary to store action references

        # Standard components as before
        self.setWindowTitle("PCD Toolkit")
        self.resize(1600, 1200)

        # Create an instance of FileManager
        self.file_manager = FileManager()
        global_variables.global_file_manager = self.file_manager

        # Create global instances of the tree structure widget, PCD viewer widget, dialog boxes manager, and data manager
        self.tree_widget = TreeStructureWidget()
        global_variables.global_tree_structure_widget = self.tree_widget

        self.pcd_viewer_widget = PCDViewerWidget()
        global_variables.global_pcd_viewer_widget = self.pcd_viewer_widget

        self.dialog_boxes_manager = DialogBoxesManager(plugin_manager)
        self.data_manager = DataManager(
            self.file_manager,
            self.tree_widget,
            self.pcd_viewer_widget,
            self.dialog_boxes_manager,
            self.plugin_manager
        )
        global_variables.global_data_manager = self.data_manager

        # Set up the UI components
        self.setup_ui()

    def setup_ui(self):
        """Sets up the main window UI components with dynamically built menus."""

        # Create the QSplitter for the main layout with left and right panes
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left side: TreeStructureWidget
        self.splitter.addWidget(self.tree_widget)

        # Right side: PCDViewerWidget
        self.splitter.addWidget(self.pcd_viewer_widget)

        # Initial sizes for the left and right panes
        self.splitter.setSizes([200, 600])

        # Create basic menu structure
        self.setup_base_menus()

        # Populate menus from plugins
        self.populate_menus_from_plugins()

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

    def setup_base_menus(self):
        """Set up the base menu structure for the application."""
        self.menubar = self.menuBar()

        # Create base menus in the correct order
        base_menus = ["File", "View", "Points", "Selection", "Draw", "Action", "Feature Detection", "Help"]
        for menu_name in base_menus:
            self.menus[menu_name] = self.menubar.addMenu(menu_name)

        # Add "Import Point Cloud" action to File menu (renamed from "Open")
        import_action = QtWidgets.QAction("Import Point Cloud", self)
        import_action.triggered.connect(self.open_file_dialog)
        self.menus["File"].addAction(import_action)
        self.actions["import_point_cloud"] = import_action

    def populate_menus_from_plugins(self):
        """Populate menus from available menu plugins."""
        # Get available analysis plugins for checking implementation status
        analysis_plugins = self.plugin_manager.get_analysis_plugins()

        for plugin in self.plugin_manager.get_menu_plugins():
            menu_location = plugin.get_menu_location()

            # Create submenu if needed
            if "/" in menu_location:
                parent_menu, submenu_name = menu_location.split("/", 1)

                # Verify parent menu exists
                if parent_menu not in self.menus:
                    print(f"Warning: Parent menu '{parent_menu}' does not exist. Creating it.")
                    self.menus[parent_menu] = self.menubar.addMenu(parent_menu)

                # Create submenu if it doesn't exist
                if menu_location not in self.menus:
                    submenu = QtWidgets.QMenu(submenu_name, self)
                    self.menus[parent_menu].addMenu(submenu)
                    self.menus[menu_location] = submenu

            # Add menu items
            for item in plugin.get_menu_items():
                # Check if this is a submenu
                if item.get("action") == "_submenu_" and "submenu" in item:
                    # Create a submenu
                    submenu_name = item["name"]
                    submenu_path = f"{menu_location}/{submenu_name}"

                    # Create the submenu
                    submenu = QtWidgets.QMenu(submenu_name, self)
                    self.menus[menu_location].addMenu(submenu)
                    self.menus[submenu_path] = submenu

                    # Add items to the submenu
                    for subitem in item["submenu"]:
                        subaction = QtWidgets.QAction(subitem["name"], self)

                        # Check if explicitly enabled/disabled in menu definition
                        explicitly_enabled = subitem.get("enabled", True)

                        # Check if corresponding analysis plugin exists
                        action_name = subitem["action"]
                        plugin_exists = action_name in analysis_plugins

                        # Determine final enabled state
                        # Item is enabled only if explicitly enabled AND plugin exists
                        should_enable = explicitly_enabled and plugin_exists
                        subaction.setEnabled(should_enable)

                        # Set tooltip and status tip based on status
                        base_tooltip = subitem.get("tooltip", "")

                        if not explicitly_enabled:
                            # Explicitly disabled
                            status_text = f"{base_tooltip} (Disabled)" if base_tooltip else "Disabled"
                            # Also modify the display name to show it's disabled
                            subaction.setText(f"{subitem['name']} [Disabled]")
                        elif not plugin_exists:
                            # No implementation yet
                            status_text = f"{base_tooltip} (Not yet implemented)" if base_tooltip else "Not yet implemented"
                            # Also modify the display name to show it's not implemented
                            subaction.setText(f"{subitem['name']} [Not Implemented]")
                        else:
                            status_text = base_tooltip

                        # Set status tip - this shows in the status bar
                        if status_text:
                            subaction.setStatusTip(status_text)

                        # Connect action to handle_action method
                        subaction.triggered.connect(
                            lambda checked, p=plugin, a=action_name: self._handle_menu_action(p, a)
                        )

                        # Add action to the submenu
                        submenu.addAction(subaction)

                        # Store reference to the action
                        action_id = f"{submenu_path}/{action_name}"
                        self.actions[action_id] = subaction
                else:
                    # Regular menu item
                    action = QtWidgets.QAction(item["name"], self)

                    # Check if explicitly enabled/disabled in menu definition
                    explicitly_enabled = item.get("enabled", True)

                    # Check if corresponding analysis plugin exists
                    action_name = item["action"]
                    plugin_exists = action_name in analysis_plugins

                    # Determine final enabled state
                    should_enable = explicitly_enabled and plugin_exists
                    action.setEnabled(should_enable)

                    # Set tooltip and status tip based on status
                    base_tooltip = item.get("tooltip", "")

                    if not explicitly_enabled:
                        # Explicitly disabled
                        status_text = f"{base_tooltip} (Disabled)" if base_tooltip else "Disabled"
                        # Also modify the display name to show it's disabled
                        action.setText(f"{item['name']} [Disabled]")
                    elif not plugin_exists:
                        # No implementation yet
                        status_text = f"{base_tooltip} (Not yet implemented)" if base_tooltip else "Not yet implemented"
                        # Also modify the display name to show it's not implemented
                        action.setText(f"{item['name']} [Not Implemented]")
                    else:
                        status_text = base_tooltip

                    # Set status tip - this shows in the status bar
                    if status_text:
                        action.setStatusTip(status_text)

                    # Connect action to handle_action method
                    action.triggered.connect(
                        lambda checked, p=plugin, a=action_name: self._handle_menu_action(p, a)
                    )

                    # Add action to the menu
                    target_menu = self.menus[menu_location]
                    target_menu.addAction(action)

                    # Store reference to the action
                    action_id = f"{menu_location}/{action_name}"
                    self.actions[action_id] = action

            print(f"Added menu items from plugin {plugin.__class__.__name__} to {menu_location}")

    def _handle_menu_action(self, plugin, action_name):
        """
        Handle menu action triggers with safety checks.

        This method is called when a menu item is clicked. It verifies
        that the action has an implementation before processing.

        Args:
            plugin: The plugin that handles this action
            action_name (str): The name of the action that was triggered
        """
        # Double-check if the action has an implementation
        analysis_plugins = self.plugin_manager.get_analysis_plugins()

        if action_name not in analysis_plugins:
            # Show informative message to user
            QMessageBox.information(
                self,
                "Feature Not Available",
                f"The '{action_name}' feature is not yet implemented.\n"
                f"This functionality will be available in a future update.",
                QMessageBox.Ok
            )
            return

        # Action has implementation, proceed normally
        plugin.handle_action(action_name, self)

    def open_file_dialog(self):
        """Handler for 'Open' action to open and display a point cloud file."""
        self.file_manager.open_point_cloud_file(self)

    def open_dialog_box(self, analysis_type):
        """
        Open a dialog box for parameter input for an analysis type.

        This method checks if an analysis plugin exists before attempting
        to open a dialog box. If no plugin exists, it shows an information
        message to the user.

        Args:
            analysis_type (str): The type of analysis to open a dialog for
        """
        # Check if the analysis plugin exists
        analysis_plugins = self.plugin_manager.get_analysis_plugins()

        if analysis_type not in analysis_plugins:
            # No plugin found, show a message to the user
            QMessageBox.information(
                self,
                "Feature Not Available",
                f"The '{analysis_type}' feature is not yet implemented.\n"
                f"This functionality will be available in a future update.",
                QMessageBox.Ok
            )
            return

        # Plugin exists, proceed with opening the dialog
        self.dialog_boxes_manager.open_dialog_box(analysis_type)