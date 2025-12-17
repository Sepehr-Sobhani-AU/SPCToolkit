from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QApplication
from PyQt5.QtCore import pyqtSignal, Qt


class TreeStructureWidget(QTreeWidget):
    """
    A widget to display and manage a tree structure of branches (DataNodes).
    Handles user interactions such as adding, moving, deleting, and toggling visibility of branches.

    Attributes:
        branch_visibility_changed (pyqtSignal): Signal emitted when branch visibility is toggled.
        branch_hierarchy_updated (pyqtSignal): Signal emitted when branch hierarchy is modified.
        selected_branches_changed (pyqtSignal): Signal emitted when branches are selected.
    """

    # Signals
    branch_added = pyqtSignal(dict)
    branch_visibility_changed = pyqtSignal(dict)
    branch_hierarchy_updated = pyqtSignal(dict)
    branch_selection_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Internal dictionaries to manage tree branches
        self.branches_dict = {}
        self.visibility_status = {}

        # Configure tree widget properties
        self.setColumnCount(2)  # Column 0: Branch name/visibility, Column 1: Cache
        self.setHeaderLabels(["Branch", "Cache"])
        self.setSelectionMode(QTreeWidget.MultiSelection)

        # Connect signals
        self.itemChanged.connect(self.on_item_checked)
        self.itemSelectionChanged.connect(self.on_selection_changed)

    def add_branch(self, uuid: str, parent_uuid: str, name: str, is_root: bool = False):
        """
        Adds a new branch to the tree.

        Args:
            uuid (str): Unique identifier for the branch.
            parent_uuid (str): Unique identifier of the parent branch. None for top-level branches.
            name (str): Name of the branch.
            is_root (bool): Whether this is a root PointCloud node (always cached).
        """
        # Create a new tree item for the branch
        item = QTreeWidgetItem([name, ""])  # Two columns: name and cache icon
        item.setData(0, Qt.UserRole, uuid)

        # Column 0: Visibility checkbox
        item.setCheckState(0, Qt.Checked)
        item.setFlags(item.flags() | Qt.ItemIsEditable)

        # Column 1: Cache checkbox
        if is_root:
            # Root nodes are always "cached" (data is in memory)
            item.setCheckState(1, Qt.Checked)
            # Make cache checkbox non-editable for root nodes
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
        else:
            item.setCheckState(1, Qt.Unchecked)

        # If a parent UUID is provided, find the parent and add as a child
        if parent_uuid and parent_uuid in self.branches_dict:
            parent_item = self.branches_dict[parent_uuid]
            parent_item.addChild(item)
            parent_item.setExpanded(True)
        else:
            self.addTopLevelItem(item)

        # Update internal dictionaries
        self.branches_dict[uuid] = item
        self.visibility_status[uuid] = True

        # Emit the visibility_status update signal
        self.branch_added.emit(self.visibility_status)

    # def remove_branch(self, uids: list[str]):
    #     """
    #     Removes branches from the tree.
    #
    #     Args:
    #         uids (list[str]): List of unique identifiers for the branches to remove.
    #     """
    #     # Store removed branches to emit hierarchy update signal
    #     removed_branches = []
    #
    #     for uid in uids:
    #         if uid in self.branches_dict:
    #             item = self.branches_dict.pop(uid)
    #             removed_branches.append(uid)
    #             parent_item = item.parent()
    #             if parent_item:
    #                 parent_item.removeChild(item)
    #                 child_uids = [item.data(0, Qt.UserRole) for item in item.takeChildren()]
    #                 removed_branches.append(uid)
    #             else:
    #                 self.takeTopLevelItem(self.indexOfTopLevelItem(item))
    #             del self.visibility_status[uid]

    # def move_branch(self, uids: list[str], new_parent_uuid: str):
    #     """
    #     Moves branches to a new parent.
    #
    #     Args:
    #         uids (list[str]): List of unique identifiers for the branches to move.
    #         new_parent_uuid (str): Unique identifier of the new parent branch.
    #     """
    #     for uid in uids:
    #         if uid in self.branches_dict:
    #             item = self.branches_dict[uid]
    #             parent_item = item.parent()
    #
    #             # Remove from current parent
    #             if parent_item:
    #                 parent_item.removeChild(item)
    #             else:
    #                 self.takeTopLevelItem(self.indexOfTopLevelItem(item))
    #
    #             # Add to the new parent
    #             if new_parent_uuid in self.branches_dict:
    #                 new_parent_item = self.branches_dict[new_parent_uuid]
    #                 new_parent_item.addChild(item)
    #             else:
    #                 self.addTopLevelItem(item)
    #
    #     # Emit the hierarchy update signal
    #     self.branch_hierarchy_updated.emit(self._get_hierarchy())

    # def toggle_visibility(self, uids: list[str]):
    #     """
    #     Toggles visibility of branches programmatically.
    #
    #     Args:
    #         uids (list[str]): List of unique identifiers for the branches to toggle.
    #     """
    #     for uid in uids:
    #         if uid in self.branches_dict:
    #             current_status = self.visibility_status.get(uid, False)
    #             new_status = not current_status
    #             self.visibility_status[uid] = new_status
    #             self.branches_dict[uid].setCheckState(0, Qt.Checked if new_status else Qt.Unchecked)

    def on_item_checked(self, item, column):
        """
        Handles item check state changes by user and updates visibility or cache status.

        Args:
            item: The tree item that was checked/unchecked.
            column: The column index (0 for visibility, 1 for cache).
        """
        from config.config import global_variables

        uid = item.data(0, Qt.UserRole)
        if uid:
            if column == 0:
                # Visibility checkbox changed
                self.visibility_status[uid] = item.checkState(0) == Qt.Checked
                self.branch_visibility_changed.emit(self.visibility_status)
            elif column == 1:
                # Cache checkbox changed - use singleton pattern (NO signal!)
                is_cached = item.checkState(1) == Qt.Checked
                data_manager = global_variables.global_data_manager
                if data_manager:
                    if is_cached:
                        data_manager.cache_branch(uid)
                    else:
                        data_manager.uncache_branch(uid)

    # TODO: Fix selecting multiple items using Ctrl key
    def on_selection_changed(self):
        """
        Handles item selection changes and emits the selected branches.
        """
        selected_items = self.selectedItems()
        last_selected_item = selected_items[-1] if selected_items else None

        if last_selected_item:
            # Check if Ctrl is pressed
            if not (QApplication.keyboardModifiers() & Qt.ControlModifier):
                # Block signals to prevent recursive calls
                self.blockSignals(True)
                self.clearSelection()
                last_selected_item.setSelected(True)
                self.blockSignals(False)

                # Update selected_items to reflect the new selection
                selected_items = self.selectedItems()  # <-- Fix here

        selected_uids = [item.data(0, Qt.UserRole) for item in selected_items if item.data(0, Qt.UserRole)]
        self.branch_selection_changed.emit(selected_uids)

    def get_all_items(self):
        """
        Retrieves all items (branches) in the tree structure, including children.

        Returns:
            dict: Dictionary mapping branch UUIDs to their corresponding QTreeWidgetItem instances.
        """

        return self.branches_dict

    def _get_all_children(self, parent_item, all_items):
        """
        Recursively retrieves all children of a given parent item.

        Args:
            parent_item (QTreeWidgetItem): The parent tree item.
            all_items (list): The list to store all tree items.
        """
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            all_items.append(child)
            self._get_all_children(child, all_items)  # Recursive call

    def _get_hierarchy(self):
        """
        Constructs a dictionary representing the current branch hierarchy.

        Returns:
            dict: Dictionary mapping branch UUIDs to their parent UUIDs.
        """
        hierarchy = {}
        for uid, item in self.branches_dict.items():
            parent_item = item.parent()
            parent_uuid = parent_item.data(0, Qt.UserRole) if parent_item else None
            hierarchy[uid] = parent_uuid
        return hierarchy

    def update_cache_tooltip(self, uid: str, memory_usage: str = None):
        """
        Update the tooltip for the cache column to show memory usage.

        Args:
            uid (str): UUID of the branch.
            memory_usage (str, optional): Memory usage string (e.g., "12.34 MB").
                                         If None, shows "Not cached".
        """
        item = self.branches_dict.get(uid)
        if item:
            if memory_usage:
                item.setToolTip(1, f"Cached: {memory_usage}")
            else:
                item.setToolTip(1, "Not cached")
