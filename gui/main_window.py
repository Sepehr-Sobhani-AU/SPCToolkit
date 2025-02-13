"""
  Create main window for the PCD Toolkit application with a tree structure widget and PCD viewer widget.
"""
# define the global variables
from config.config import global_variables

from PyQt5 import QtWidgets, QtCore
from gui.widgets import PCDViewerWidget, TreeStructureWidget
from core.data_manager import DataManager
from services.file_manager import FileManager
from gui.dialog_boxes.dialog_boxes_manager import DialogBoxesManager


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the PCD Toolkit, with a tree structure widget as a left pane.
    """


    def __init__(self):
        super().__init__()

        # Set up the main window components
        self.actionSeparateSelectedClusters = None
        self.actionSeparateSelectedPoints = None
        self.menuCluster = None
        self.menuFile = None
        self.menuSelection = None
        self.menuAction = None
        self.menuHelp = None
        self.menubar = None
        self.pcd_viewer_widget = None
        self.tree_widget = None
        self.splitter = None
        self.actionOpen = None
        self.actionSelectClusters = None
        self.actionSubsample = None
        self.actionCluster = None
        self.actionFilter = None
        self.actionRegionGrowing = None
        self.actionDBSCAN = None
        self.actionKMeans = None
        self.statusbar = None
        self.setWindowTitle("PCD Toolkit")
        self.resize(1600, 1200)

        # Create an instance of FileManager
        self.file_manager = FileManager()
        self.tree_widget = TreeStructureWidget()
        # Set the global_tree_structure_widget as a global variable for easy access from other modules
        global_variables.global_tree_structure_widget = self.tree_widget
        global_tree_widget = global_variables.global_tree_structure_widget

        self.pcd_viewer_widget = PCDViewerWidget()
        # Set the global_pcd_viewer_widget as a global variable for easy access from other modules
        global_variables.global_pcd_viewer_widget = self.pcd_viewer_widget
        global_pcd_viewer_widget = global_variables.global_pcd_viewer_widget

        self.dialog_boxes_manager = DialogBoxesManager()
        self.data_manager = DataManager(self.file_manager, self.tree_widget, self.pcd_viewer_widget, self.dialog_boxes_manager)

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
        self.menuSelection = self.menubar.addMenu("Selection")
        self.menuAction = self.menubar.addMenu("Action")
        self.menuHelp = self.menubar.addMenu("Help")

        # "Open" action
        self.actionOpen = QtWidgets.QAction("Open", self)
        self.menuFile.addAction(self.actionOpen)
        # Connect the Open action
        self.actionOpen.triggered.connect(self.open_file_dialog)

        # "Separate Selected Points" action
        self.actionSeparateSelectedPoints = QtWidgets.QAction("Separate Selected Points", self)
        self.menuSelection.addAction(self.actionSeparateSelectedPoints)
        # Connect the Select Clusters action
        self.actionSeparateSelectedPoints.triggered.connect(lambda: self.open_dialog_box('separate_selected_points'))

        # "Separate Selected Clusters" action
        self.actionSeparateSelectedClusters = QtWidgets.QAction("Separate Selected Clusters", self)
        self.menuSelection.addAction(self.actionSeparateSelectedClusters)
        # Connect the Select Clusters action
        self.actionSeparateSelectedClusters.triggered.connect(lambda: self.open_dialog_box('separate_selected_clusters'))

        # "Subsample" action
        self.actionSubsample = QtWidgets.QAction("Subsampling", self)
        self.menuAction.addAction(self.actionSubsample)
        # Connect the Subsample action
        self.actionSubsample.triggered.connect(lambda: self.open_dialog_box('subsampling'))

        # "Filter" action
        self.actionFilter = QtWidgets.QAction("Filtering", self)
        self.menuAction.addAction(self.actionFilter)
        # Connect the Filter action
        self.actionFilter.triggered.connect(lambda: self.open_dialog_box('filtering'))

        # Create a "Dbscan" submenu under "Action"
        self.menuCluster = QtWidgets.QMenu("Clustering", self)
        self.menuAction.addMenu(self.menuCluster)  # Add clustering submenu under Action

        # Add specific clustering methods to the submenu
        self.actionKMeans = QtWidgets.QAction("K-Means", self)
        self.actionDBSCAN = QtWidgets.QAction("DBSCAN", self)
        self.actionRegionGrowing = QtWidgets.QAction("Region Growing", self)

        self.menuCluster.addAction(self.actionKMeans)
        self.menuCluster.addAction(self.actionDBSCAN)
        self.menuCluster.addAction(self.actionRegionGrowing)

        # Connect clustering actions
        self.actionKMeans.triggered.connect(lambda: self.open_dialog_box('kmeans'))
        self.actionDBSCAN.triggered.connect(lambda: self.open_dialog_box('dbscan'))
        self.actionRegionGrowing.triggered.connect(lambda: self.open_dialog_box('region_growing'))

        # "Logical Operations" action
        self.actionLogicalOperations = QtWidgets.QAction("Logical Operations", self)
        self.menuAction.addAction(self.actionLogicalOperations)
        # Connect the Filter action
        self.actionLogicalOperations.triggered.connect(lambda: self.open_dialog_box('logical_operations'))

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

    def open_file_dialog(self):
        """Handler for 'Open' action to open and display a point cloud file."""

        self.file_manager.open_point_cloud_file(self)

    # TODO: Docstrings
    # TODO: Validate the parameters
    def open_dialog_box(self, analysis_type):
        self.dialog_boxes_manager.open_dialog_box(analysis_type)
