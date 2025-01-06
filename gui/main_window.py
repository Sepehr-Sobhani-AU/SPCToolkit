"""
  Create main window for the PCD Toolkit application with a tree structure widget and PCD viewer widget.
"""

from PyQt5 import QtWidgets, QtCore
from gui.widgets import PCDViewerWidget, TreeStructureWidget
from core.data_manager import DataManager
from services.file_manager import FileManager


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the PCD Toolkit, with a tree structure widget as a left pane.
    """

    def __init__(self):
        super().__init__()

        # Set up the main window
        self.menuHelp = None
        self.menuFile = None
        self.menubar = None
        self.pcd_viewer_widget = None
        self.tree_widget = None
        self.splitter = None
        self.actionOpen = None
        self.statusbar = None
        self.setWindowTitle("PCD Toolkit")
        self.resize(800, 600)

        # Create an instance of FileManager
        self.file_manager = FileManager()
        self.tree_widget = TreeStructureWidget()
        self.pcd_viewer_widget = PCDViewerWidget()
        self.data_manager = DataManager(self.file_manager, self.tree_widget, self.pcd_viewer_widget)

        # Set up the UI components
        self.setup_ui()

    def setup_ui(self):
        """Sets up the main window UI components, including the left tree structure pane and right PCD viewer pane."""

        # Create the QSplitter for the main layout with left and right panes
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left side: TreeStructureWidget for tree structure display
        self.splitter.addWidget(self.tree_widget)

        # Right side: PCDViewerWidget for visualisation
        self.splitter.addWidget(self.pcd_viewer_widget)

        # Initial sizes for the left and right panes
        self.splitter.setSizes([200, 600])

        # Menu bar
        self.menubar = self.menuBar()
        self.menuFile = self.menubar.addMenu("File")
        self.menuHelp = self.menubar.addMenu("Help")

        # "Open" action
        self.actionOpen = QtWidgets.QAction("Open", self)
        self.menuFile.addAction(self.actionOpen)
        # Connect the Open action
        self.actionOpen.triggered.connect(self.open_file_dialog)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

    def open_file_dialog(self):
        """Handler for 'Open' action to open and display a point cloud file."""

        self.file_manager.open_point_cloud_file(self)



