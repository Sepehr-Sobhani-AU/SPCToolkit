from PyQt5 import QtWidgets, QtCore
from gui.widgets import PCDViewerWidget, TreeStructureWidget
from services.file_manager import FileManager


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the PCD Toolkit, with a tree structure widget as a left pane.
    """

    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("PCD Toolkit")
        self.resize(800, 600)

        # Set up the UI components
        self.setup_ui()

    def setup_ui(self):
        """Sets up the main window UI components, including the left tree structure pane and right PCD viewer pane."""

        # Create the QSplitter for the main layout with left and right panes
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left side: TreeStructureWidget for tree structure display
        self.tree_widget = TreeStructureWidget()
        self.splitter.addWidget(self.tree_widget)

        # Right side: PCDViewerWidget for visualisation
        self.pcd_viewer_widget = PCDViewerWidget()
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

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

        # Connect the Open action
        self.actionOpen.triggered.connect(self.open_file_dialog)

    def open_file_dialog(self):
        """Handler for 'Open' action to open and display a point cloud file."""

        points, colors = FileManager.open_point_cloud_file(self)
        if points is not None:
            self.pcd_viewer_widget.set_points(points, colors)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
