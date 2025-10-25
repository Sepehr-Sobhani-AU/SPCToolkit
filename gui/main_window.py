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
        # All menus are now dynamically created from plugin folder structure!

    def populate_menus_from_plugins(self):
        """
        Populate menus from folder-based plugin structure.

        The folder structure automatically defines the menu hierarchy:
        - plugins/Points/Clustering/dbscan.py -> Menu: Points > Clustering > DBSCAN
        - plugins/Processing/subtract.py      -> Menu: Processing > Subtract
        """
        print("Building menus from folder structure...")

        # Get the menu structure from plugin manager
        menu_structure = self.plugin_manager.get_menu_structure()

        if not menu_structure:
            print("No plugins found in folder structure.")
            return

        # Sort menu paths for consistent ordering
        sorted_menu_paths = sorted(menu_structure.keys())

        for menu_path in sorted_menu_paths:
            plugin_names = menu_structure[menu_path]

            # Create menu hierarchy from path
            # Example: "Points/Clustering" -> Create "Points" menu, then "Clustering" submenu
            self._create_menu_hierarchy(menu_path)

            # Get the target menu for this path
            target_menu = self._get_menu_by_path(menu_path)

            # Add each plugin as a menu action
            for plugin_name in plugin_names:
                self._add_plugin_menu_action(target_menu, plugin_name, menu_path)

        print(f"Created {len(self.menus)} menus with {len(self.actions)} actions")

    def _create_menu_hierarchy(self, menu_path: str):
        """
        Create menu hierarchy from a path like "Points/Clustering/Advanced".

        Args:
            menu_path: Menu path with forward slashes (e.g., "Points/Clustering")
        """
        parts = menu_path.split('/')
        current_path = ""
        parent_menu = None

        for i, part in enumerate(parts):
            # Build the path incrementally
            if current_path:
                current_path += f"/{part}"
            else:
                current_path = part

            # Check if menu already exists
            if current_path in self.menus:
                parent_menu = self.menus[current_path]
                continue

            # Create the menu or submenu
            if i == 0:
                # Top-level menu
                menu = self.menubar.addMenu(part)
                self.menus[current_path] = menu
                parent_menu = menu
            else:
                # Submenu
                submenu = QtWidgets.QMenu(part, self)
                parent_menu.addMenu(submenu)
                self.menus[current_path] = submenu
                parent_menu = submenu

    def _get_menu_by_path(self, menu_path: str):
        """
        Get a menu by its path.

        Args:
            menu_path: Menu path (e.g., "Points/Clustering")

        Returns:
            QMenu object
        """
        return self.menus.get(menu_path)

    def _add_plugin_menu_action(self, menu, plugin_name: str, menu_path: str):
        """
        Add a plugin as a menu action.

        Args:
            menu: The QMenu to add the action to
            plugin_name: The name of the plugin
            menu_path: The menu path for this plugin
        """
        if not menu:
            print(f"Warning: Could not find menu for path '{menu_path}'")
            return

        # Get plugin class
        plugin_class = self.plugin_manager.get_plugin(plugin_name)
        if not plugin_class:
            print(f"Warning: Plugin '{plugin_name}' not found")
            return

        # Create an instance to get display name
        try:
            plugin_instance = plugin_class()
            display_name = plugin_instance.get_name()

            # Convert snake_case to Title Case for better display
            # e.g., "dbscan_clustering" -> "DBSCAN Clustering"
            display_name = self._format_plugin_name(display_name)

        except Exception as e:
            print(f"Error instantiating plugin '{plugin_name}': {e}")
            display_name = plugin_name

        # Create menu action
        action = QtWidgets.QAction(display_name, self)

        # Connect to plugin execution
        action.triggered.connect(
            lambda checked, pname=plugin_name: self.open_dialog_box(pname)
        )

        # Add to menu
        menu.addAction(action)

        # Store reference
        action_id = f"{menu_path}/{plugin_name}"
        self.actions[action_id] = action

        print(f"  Added action: {menu_path} > {display_name}")

    def _format_plugin_name(self, name: str) -> str:
        """
        Format plugin name for display in menu.

        Converts "dbscan_clustering" to "DBSCAN Clustering"

        Args:
            name: Plugin name

        Returns:
            Formatted display name
        """
        # Replace underscores with spaces
        formatted = name.replace('_', ' ')

        # Title case each word
        words = formatted.split()
        formatted_words = []

        for word in words:
            # Keep acronyms uppercase (like DBSCAN, SOR, MLS)
            if word.upper() in ['DBSCAN', 'HDBSCAN', 'SOR', 'MLS', 'PCA', 'ICP']:
                formatted_words.append(word.upper())
            else:
                formatted_words.append(word.capitalize())

        return ' '.join(formatted_words)

    def open_file_dialog(self):
        """Handler for 'Open' action to open and display a point cloud file."""
        self.file_manager.open_point_cloud_file(self)

    def open_dialog_box(self, plugin_name):
        """
        Open a dialog box for parameter input or execute action directly.

        Routes to appropriate handler based on plugin type (data processing vs action).
        """
        # Check if this is an action plugin
        if self.plugin_manager.is_action_plugin(plugin_name):
            # Handle action plugin directly
            self.execute_action_plugin(plugin_name)
        else:
            # Handle data processing plugin through dialog manager
            self.dialog_boxes_manager.open_dialog_box(plugin_name)

    def execute_action_plugin(self, plugin_name: str):
        """
        Execute an action plugin.

        Args:
            plugin_name: The name of the action plugin to execute
        """
        plugin_class = self.plugin_manager.get_plugin(plugin_name)
        if not plugin_class:
            print(f"Error: Action plugin '{plugin_name}' not found")
            return

        try:
            plugin_instance = plugin_class()
            parameter_schema = plugin_instance.get_parameters()

            # If no parameters needed, execute immediately
            if not parameter_schema or len(parameter_schema) == 0:
                plugin_instance.execute(self, {})
            else:
                # Import DynamicDialog here to avoid circular imports
                from gui.dialog_boxes.dynamic_dialog import DynamicDialog

                # Create and open a dynamic dialog for parameter input
                dialog = DynamicDialog(f"{plugin_name.title()} Parameters", parameter_schema)
                if dialog.exec_():
                    # If the user clicked OK, get the parameters and execute
                    params = dialog.get_parameters()
                    plugin_instance.execute(self, params)

        except Exception as e:
            print(f"Error executing action plugin '{plugin_name}': {str(e)}")
