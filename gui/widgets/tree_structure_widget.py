from PyQt5.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QMenu, QPushButton, QWidget
)
from PyQt5.QtCore import Qt, QPoint


class TreeStructureWidget(QTreeWidget):
    """Custom Tree Widget with dynamic branch management, visibility toggling, and context menus."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setHeaderLabels(["Point Cloud Structure", "Actions"])
        self.current_branch = None  # Track the currently selected branch

        # Connect checkbox toggle and item selection
        self.itemChanged.connect(self.on_item_checked)
        self.itemClicked.connect(self.on_item_selected)

    def add_branch(self, parent=None, text="New Branch", checkable=True):
        """Adds a new branch or sub-branch to the tree."""
        item = QTreeWidgetItem(parent)
        item.setText(0, text)
        if checkable:
            item.setCheckState(0, Qt.Unchecked)  # Make branch checkable for visibility toggle

        # Add a menu button to the second column of the item
        self.add_menu_button(item)

        # If no parent, add it as a top-level item
        if parent is None:
            self.addTopLevelItem(item)

        return item  # Return the item to allow further customisation if needed

    def remove_branch(self, item):
        """Removes a branch or sub-branch from the tree."""
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            self.takeTopLevelItem(self.indexOfTopLevelItem(item))

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

    def toggle_visibility(self, item):
        """Toggles the visibility of the points associated with the branch."""
        checked = item.checkState(0) == Qt.Checked
        item.setCheckState(0, Qt.Unchecked if checked else Qt.Checked)
        self.on_item_checked(item)  # Manually trigger visibility logic

    def custom_action(self, item):
        """Example of a custom action for a branch."""
        print(f"Custom action performed on branch: {item.text(0)}")

    def on_item_checked(self, item):
        """Handles visibility toggle for each branch when checkbox is clicked."""
        if item.checkState(0) == Qt.Checked:
            print(f"{item.text(0)} is now visible")
        else:
            print(f"{item.text(0)} is now hidden")

    def on_item_selected(self, item):
        """Tracks the currently selected branch or sub-branch."""
        self.current_branch = item
        print(f"Selected branch: {item.text(0)}")

    def get_selected_branch(self):
        """Returns the currently selected branch or sub-branch."""
        return self.current_branch
