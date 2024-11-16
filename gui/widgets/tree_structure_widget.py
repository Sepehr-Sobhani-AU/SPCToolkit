
# Standard library imports
import uuid

# Third party imports
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QMenu, QPushButton, QWidget
)


UUIDRole = Qt.UserRole + 1  # Custom role specifically for UUIDs


class TreeStructureWidget(QTreeWidget):
    """Custom Tree Widget with dynamic branch management, visibility toggling, and context menus."""

    # Define the signal to indicate visibility change
    branch_visibility_changed = pyqtSignal(dict)

    def __init__(self, point_clouds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setHeaderLabels(["Point Cloud Structure", "Actions"])
        self.current_branch = None  # Track the currently selected branch
        self.point_clouds = point_clouds

        # Store the visibility status of branches
        # This dictionary will store the UUIDs of branches and their visibility status (True/False)
        # {branch_uuid: visibility_status}
        self.branches_visibility_status = {}

        # Connect checkbox toggle and item selection
        self.itemChanged.connect(self.on_item_checked)
        self.itemClicked.connect(self.on_item_selected)

        # Connect signal to slot for visibility tracking
        self.point_clouds.point_clouds_updated.connect(self.update_branches)

    def add_branch(self, branch_uuid: uuid.UUID, parent_uuid: uuid.UUID = None, text="New Branch", checkable=True):
        """Adds a new branch or sub-branch to the tree."""
        item = QTreeWidgetItem(parent_uuid)
        item.setText(0, text)

        # Store the UUID with the item using a custom role
        item.setData(0, UUIDRole, str(branch_uuid))

        if checkable:
            item.setCheckState(0, Qt.Checked)  # Make branch checkable for visibility toggle

        # Add a menu button to the second column of the item
        self.add_menu_button(item)

        # If no parent, add it as a top-level item
        if parent_uuid is None:
            self.addTopLevelItem(item)

        return item  # Return the item to allow further customisation if needed

    def remove_branch(self, item):
        """Removes a branch or sub-branch from the tree."""

        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            self.takeTopLevelItem(self.indexOfTopLevelItem(item))

        branch_uuid = item.data(0, UUIDRole)
        return branch_uuid

    def add_menu_button(self, item):
        """Adds a menu button to a branch or sub-branch for context actions."""
        menu_button = QPushButton("...")
        menu_button.setFixedSize(30, 20)
        menu_button.setStyleSheet("border: none;")

        # Connect button to open context menu
        menu_button.clicked.connect(lambda: self.show_menu(item, menu_button))
        self.setItemWidget(item, 1, menu_button)  # Place button in the second column

    def show_menu(self, item, button):
        """Shows a customisable context menu when the menu button is clicked."""
        menu = QMenu()

        # Example dynamic actions - can add/remove actions as needed
        action_1 = menu.addAction("Toggle Visibility")
        action_2 = menu.addAction("Delete Branch")
        custom_action = menu.addAction("Custom Action")

        # Connect actions to specific methods
        action_1.triggered.connect(lambda: self.toggle_visibility(item))
        action_2.triggered.connect(lambda: self.remove_branch(item))
        custom_action.triggered.connect(lambda: self.custom_action(item))

        # Show the menu at the button position
        menu.exec_(button.mapToGlobal(QPoint(0, button.height())))

    def custom_action(self, item):
        """Example of a custom action for a branch."""
        print(f"Custom action performed on branch: {item.text(0)}")

    def on_item_checked(self, item):
        """Emits a signal when an item is checked or unchecked to toggle visibility."""

        # Retrieve the UUID associated with the item
        branch_uuid = item.data(0, UUIDRole)

        # Check if the item is checked or unchecked
        visibility_status = item.checkState(0) == Qt.Checked

        # Update the visibility status of all branches in the dictionary
        # Iterate through all items in the tree (branches and sub-branches) and update the visibility status
        for i in range(self.topLevelItemCount()):
            top_level_item = self.topLevelItem(i)
            self.branches_visibility_status[top_level_item.data(0, UUIDRole)] = top_level_item.checkState(0) == Qt.Checked
            for j in range(top_level_item.childCount()):
                child_item = top_level_item.child(j)
                self.branches_visibility_status[child_item.data(0, UUIDRole)] = child_item.checkState(0) == Qt.Checked

        # Emit the signal with the UUID and visibility status
        self.branch_visibility_changed.emit(self.branches_visibility_status)

    def on_item_selected(self, item):
        """Tracks the currently selected branch or sub-branch."""
        self.current_branch = item
        print(f"Selected branch: {item.text(0)}")

    def get_selected_branch(self):
        """Returns the currently selected branch or sub-branch."""
        return self.current_branch

    def update_branches(self, point_clouds: dict):
        """Updates the tree structure with a list of branches."""
        self.clear()
        for point_cloud_uuid, point_cloud in point_clouds.items():
            self.add_branch(point_cloud_uuid, parent_uuid=point_cloud.parent_uuid, text=point_cloud.name, checkable=True)
