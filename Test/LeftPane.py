from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QTreeWidget, QTreeWidgetItem, QSizePolicy,
    QVBoxLayout, QWidget, QSplitter, QLabel, QPushButton, QMenu, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QPoint
import sys


class SPCToolkitGUI(QMainWindow):
    """Main GUI Window for SPCToolkit with a Tree Structure including checkboxes and menus."""

    def __init__(self):
        super().__init__()

        # Window settings
        self.setWindowTitle("SPCToolkit with Advanced Tree Structure")
        self.setGeometry(100, 100, 800, 600)

        # Main layout using QSplitter for left and right panes
        self.splitter = QSplitter(Qt.Horizontal)
        #self.setCentralWidget(self.splitter)

        # Left pane: Tree structure with checkboxes and menus
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Point Cloud Structure", ""])

        self.populate_tree()
        self.splitter.addWidget(self.tree_widget)

        # Right pane: Placeholder for processing options
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.addWidget(QLabel("Processing Options"))
        self.splitter.addWidget(self.right_widget)

        # Set splitter as central widget
        self.setCentralWidget(self.splitter)

    def populate_tree(self):
        """Populates the tree widget with custom point cloud structure including checkboxes and menu buttons."""

        # Main branch for point cloud
        main_point_cloud = QTreeWidgetItem(self.tree_widget)
        main_point_cloud.setText(0, "Main Point Cloud")
        main_point_cloud.setCheckState(0, Qt.Unchecked)
        self.add_menu_button(main_point_cloud)

        # Sub-branch for ground points
        ground_points = QTreeWidgetItem(main_point_cloud)
        ground_points.setText(0, "Ground Points")
        ground_points.setCheckState(0, Qt.Unchecked)
        self.add_menu_button(ground_points)

        # Sub-branch for non-ground points
        non_ground_points = QTreeWidgetItem(main_point_cloud)
        non_ground_points.setText(0, "Non-Ground Points")
        non_ground_points.setCheckState(0, Qt.Unchecked)
        self.add_menu_button(non_ground_points)

        # Example sub-items under ground points
        ground_point_1 = QTreeWidgetItem(ground_points)
        ground_point_1.setText(0, "Ground Point 1")
        ground_point_1.setCheckState(0, Qt.Unchecked)
        self.add_menu_button(ground_point_1)

        ground_point_2 = QTreeWidgetItem(ground_points)
        ground_point_2.setText(0, "Ground Point 2")
        ground_point_2.setCheckState(0, Qt.Unchecked)
        self.add_menu_button(ground_point_2)

        # Example sub-items under non-ground points
        non_ground_point_1 = QTreeWidgetItem(non_ground_points)
        non_ground_point_1.setText(0, "Non-Ground Point 1")
        non_ground_point_1.setCheckState(0, Qt.Unchecked)
        self.add_menu_button(non_ground_point_1)

        non_ground_point_2 = QTreeWidgetItem(non_ground_points)
        non_ground_point_2.setText(0, "Non-Ground Point 2sdrgsxdhb")
        non_ground_point_2.setCheckState(0, Qt.Unchecked)
        self.add_menu_button(non_ground_point_2)

        # Expand the tree by default
        self.tree_widget.expandAll()

        # Resize columns to fit content
        self.tree_widget.resizeColumnToContents(0)
        self.tree_widget.resizeColumnToContents(1)

    def reset_tree_widget_size(self):
        # Calculate the total width needed by summing column widths
        total_width = self.tree_widget.columnWidth(0) + 45

        # Set splitter as central widget
        self.setCentralWidget(self.splitter)
        # Set initial sizes for splitter
        self.splitter.setSizes([total_width, self.width()-total_width])  # Allocate initial sizes

        self.tree_widget.setMaximumWidth(total_width)

    def add_menu_button(self, item):
        """Adds a three-dot menu button to each tree item in the second column."""
        menu_button = QPushButton("...")
        menu_button.setFixedSize(30, 10)
        menu_button.setStyleSheet("border: none;")
        menu_button.clicked.connect(lambda: self.show_menu(menu_button))

        # Set the button in the second column
        self.tree_widget.setItemWidget(item, 1, menu_button)

    def show_menu(self, button):
        """Shows a dropdown menu when the three-dot button is clicked."""
        menu = QMenu()

        # Example actions for dropdown menu
        toggle_visibility_action = menu.addAction("Toggle Visibility")
        other_option_action = menu.addAction("Other Option")

        # Connect actions to functions
        toggle_visibility_action.triggered.connect(lambda: print("Toggled visibility"))
        other_option_action.triggered.connect(lambda: print("Other option selected"))

        # Show the menu at the position of the button
        menu.exec_(button.mapToGlobal(QPoint(0, button.height())))


def main():
    """Main function to run the SPCToolkit GUI application."""
    app = QApplication(sys.argv)
    gui = SPCToolkitGUI()
    gui.reset_tree_widget_size()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
