"""
  Create main window for the PCD Toolkit application with a tree structure widget and PCD viewer widget.
"""
import numpy as np
from PyQt5 import QtWidgets, QtCore
from gui.widgets import PCDViewerWidget, TreeStructureWidget
from services.file_manager import FileManager
from core.point_cloud import PointCloud
from core.point_clouds import PointClouds


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

        # Create an instance of PointClouds to manage point cloud data
        self.point_clouds = PointClouds()

        # Create an instance of FileManager
        self.file_manager = FileManager()

        # Connect the signal to a slot in MainWindow
        self.file_manager.point_cloud_loaded.connect(self.on_point_cloud_loaded)

        # Set up the UI components
        self.setup_ui()

    def setup_ui(self):
        """Sets up the main window UI components, including the left tree structure pane and right PCD viewer pane."""

        # Create the QSplitter for the main layout with left and right panes
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left side: TreeStructureWidget for tree structure display
        self.tree_widget = TreeStructureWidget(self.point_clouds)
        self.splitter.addWidget(self.tree_widget)

        # Right side: PCDViewerWidget for visualisation
        self.pcd_viewer_widget = PCDViewerWidget()
        self.tree_widget.branch_visibility_changed.connect(self.on_point_cloud_visibility_changed)
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

    def on_point_cloud_loaded(self, file_name, point_cloud: PointCloud):
        """Slot to handle the file loaded signal and update the tree structure."""

        if point_cloud is not None:

            # PointCloud instance added to the dictionary and returns an uuid
            self.point_clouds.add_point_cloud(point_cloud)

            # TODO: Loading point cloud data will change the PointClouds dictionary, which will trigger the signal
            # Update the point cloud data in the OpenGL viewer
            self.pcd_viewer_widget.set_points(point_cloud.points, point_cloud.colors)

    def on_point_cloud_visibility_changed(self, point_clouds_visibility_status):
        """Slot to handle the visibility change signal and update the point cloud visibility in the viewer."""
        # Iterate through all items in the point_cloud_visibility_status dictionary and:
        # - Merge the points if the visibility status is True
        # - Merge the colors if the visibility status is true

        # Properly initialize `points` and `colors` as empty arrays with the correct shape (Nx3)
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.float32)

        for point_cloud_uuid, visibility_status in point_clouds_visibility_status.items():
            if visibility_status:
                point_cloud = self.point_clouds.get_point_cloud(point_cloud_uuid)
                points = np.vstack((points, point_cloud.points))
                if point_cloud.colors is not None:
                    colors = np.vstack((colors, point_cloud.colors))
                else:
                    colors = None

        if len(points) == 0:
            points = None

        self.pcd_viewer_widget.set_points(points, colors)


    # def toggle_point_cloud_visibility(self, point_cloud_uuid, visibility_status):
    #     """Slot to toggle visibility of a point cloud in the viewer."""
    #     if visibility_status:
    #         print(f"Showing point cloud with UUID: {point_cloud_uuid}")
    #         self.pcd_viewer_widget.visible = True
    #         self.pcd_viewer_widget.show_point_cloud(True)
    #     else:
    #         print(f"Hiding point cloud with UUID: {point_cloud_uuid}")
    #         self.pcd_viewer_widget.visible = False
    #         self.pcd_viewer_widget.show_point_cloud(False)

