from PyQt5 import QtWidgets
from gui.widgets import PCDViewerWidget
from services.file_manager import FileManager


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the PCD Visualiser application.
    """

    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("PCD Toolkit")
        self.resize(800, 600)

        # Set up the UI components
        self.setup_ui()

    def setup_ui(self):
        """Sets up the main window UI components."""

        # Central widget and layout
        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # PCDViewerWidget
        self.pcd_viewer_widget = PCDViewerWidget(self.centralwidget)
        self.main_layout.addWidget(self.pcd_viewer_widget)

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

