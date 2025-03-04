# gui/main_window.py
"""
  Create main window for the PCD Toolkit application with dynamic menus from plugins.
"""
# define the global variables
from config.config import global_variables

from PyQt5 import QtWidgets, QtCore
from gui.widgets import PCDViewerWidget, TreeStructureWidget
from core.data_manager import DataManager
from services.file_manager import FileManager
from gui.dialog_boxes.dialog_boxes_manager import DialogBoxesManager
from plugins.plugin_manager import PluginManager


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the PCD Toolkit, with a tree structure widget as a left pane.

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

        # Create base menus
        base_menus = ["File", "Selection", "Action", "Help"]
        for menu_name in base_menus:
            self.menus[menu_name] = self.menubar.addMenu(menu_name)

        # Add basic "Open" action to File menu
        open_action = QtWidgets.QAction("Open", self)
        open_action.triggered.connect(self.open_file_dialog)
        self.menus["File"].addAction(open_action)
        self.actions["open"] = open_action

    def populate_menus_from_plugins(self):
        """Populate menus from available menu plugins."""
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

                        # Set tooltip if provided
                        if "tooltip" in subitem:
                            subaction.setToolTip(subitem["tooltip"])
                            subaction.setStatusTip(subitem["tooltip"])

                        # Connect action to handle_action method
                        subaction.triggered.connect(
                            lambda checked, p=plugin, a=subitem["action"]: p.handle_action(a, self)
                        )

                        # Add action to the submenu
                        submenu.addAction(subaction)

                        # Store reference to the action
                        action_id = f"{submenu_path}/{subitem['action']}"
                        self.actions[action_id] = subaction
                else:
                    # Regular menu item
                    action = QtWidgets.QAction(item["name"], self)

                    # Set tooltip if provided
                    if "tooltip" in item:
                        action.setToolTip(item["tooltip"])
                        action.setStatusTip(item["tooltip"])

                    # Connect action to handle_action method
                    action.triggered.connect(
                        lambda checked, p=plugin, a=item["action"]: p.handle_action(a, self)
                    )

                    # Add action to the menu
                    target_menu = self.menus[menu_location]
                    target_menu.addAction(action)

                    # Store reference to the action
                    action_id = f"{menu_location}/{item['action']}"
                    self.actions[action_id] = action

            print(f"Added menu items from plugin {plugin.__class__.__name__} to {menu_location}")

    def open_file_dialog(self):
        """Handler for 'Open' action to open and display a point cloud file."""
        self.file_manager.open_point_cloud_file(self)

    def open_dialog_box(self, analysis_type):
        """Open a dialog box for parameter input for an analysis type."""
        self.dialog_boxes_manager.open_dialog_box(analysis_type)