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
        self.menuAction = None
        self.menubar = None
        self.pcd_viewer_widget = None
        self.tree_widget = None
        self.splitter = None
        self.actionOpen = None
        self.actionSubsample = None
        self.actionCluster = None
        self.actionFilter = None
        self.statusbar = None
        self.setWindowTitle("PCD Toolkit")
        self.resize(1600, 1200)

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
        self.menuAction = self.menubar.addMenu("Action")

        # "Open" action
        self.actionOpen = QtWidgets.QAction("Open", self)
        self.menuFile.addAction(self.actionOpen)
        # Connect the Open action
        self.actionOpen.triggered.connect(self.open_file_dialog)

        # "Subsample" action
        self.actionSubsample = QtWidgets.QAction("Subsampling", self)
        self.menuAction.addAction(self.actionSubsample)
        # Connect the Subsample action
        self.actionSubsample.triggered.connect(lambda: self.apply_analysis('subsampling'))

        # "Filter" action
        self.actionFilter = QtWidgets.QAction("Filtering", self)
        self.menuAction.addAction(self.actionFilter)
        # Connect the Filter action
        self.actionFilter.triggered.connect(lambda: self.apply_analysis('filtering'))

        # "Clusters" action
        self.actionCluster = QtWidgets.QAction("Clustering", self)
        self.menuAction.addAction(self.actionCluster)
        # Connect the Clusters action
        self.actionCluster.triggered.connect(lambda: self.apply_analysis('clustering'))

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

    def open_file_dialog(self):
        """Handler for 'Open' action to open and display a point cloud file."""

        self.file_manager.open_point_cloud_file(self)

    def apply_analysis(self, analysis_type):
        self.data_manager.apply_analysis(analysis_type)




