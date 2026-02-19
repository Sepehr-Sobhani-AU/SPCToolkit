import logging
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QApplication, QHeaderView
from PyQt5.QtCore import pyqtSignal, Qt

logger = logging.getLogger(__name__)


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

        # Track Ctrl key state for multi-selection
        self._ctrl_held = False

        # Configure tree widget properties
        self.setColumnCount(2)  # Column 0: Branch name/visibility, Column 1: Cache
        self.setHeaderLabels(["Branch", "Cache"])
        # Column 0 (Branch) stretches to fill available space
        # Column 1 (Cache) sizes to fit its content
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.setSelectionMode(QTreeWidget.MultiSelection)

        # Connect signals
        self.itemChanged.connect(self.on_item_checked)
        self.itemSelectionChanged.connect(self.on_selection_changed)

    def mousePressEvent(self, event):
        """
        Capture Ctrl modifier state at click time for multi-selection support.
        """
        try:
            self._ctrl_held = bool(event.modifiers() & Qt.ControlModifier)
            logger.debug(f"mousePressEvent: Ctrl held = {self._ctrl_held}")
        except Exception as e:
            logger.error(f"Error in mousePressEvent capturing Ctrl state: {e}")
            self._ctrl_held = False

        # Always call parent event handler
        super().mousePressEvent(event)

    def add_branch(self, uuid: str, parent_uuid: str, name: str, is_root: bool = False, tooltip: str = None):
        """
        Adds a new branch to the tree.

        Args:
            uuid (str): Unique identifier for the branch.
            parent_uuid (str): Unique identifier of the parent branch. None for top-level branches.
            name (str): Name of the branch.
            is_root (bool): Whether this is a root PointCloud node (always cached).
            tooltip (str): Optional tooltip text for the branch name column.
        """
        logger.debug(f"TreeStructureWidget.add_branch() called")
        logger.debug(f"  uuid: {uuid[:8] if uuid else 'None'}...")
        logger.debug(f"  parent_uuid: {parent_uuid[:8] if parent_uuid else 'None'}...")
        logger.debug(f"  name: {name}")
        logger.debug(f"  is_root: {is_root}")
        logger.debug(f"  Total branches before: {len(self.branches_dict)}")

        try:
            # Create a new tree item for the branch
            item = QTreeWidgetItem([name, ""])  # Two columns: name and cache icon
            item.setData(0, Qt.UserRole, uuid)
            item.setData(0, Qt.UserRole + 1, name)  # Store name for rename detection
            if tooltip:
                item.setToolTip(0, tooltip)

            # Column 0: Visibility checkbox
            item.setCheckState(0, Qt.Checked)
            item.setFlags(item.flags() | Qt.ItemIsEditable)

            # Column 1: Cache checkbox
            if is_root:
                # Root nodes are always "cached" (data is in memory)
                item.setCheckState(1, Qt.Checked)
                # Store that this is a root node (we'll prevent unchecking in on_item_checked)
                item.setData(1, Qt.UserRole, "root_cached")
            else:
                item.setCheckState(1, Qt.Unchecked)

            # If a parent UUID is provided, find the parent and add as a child
            if parent_uuid and parent_uuid in self.branches_dict:
                parent_item = self.branches_dict[parent_uuid]
                parent_item.addChild(item)
                parent_item.setExpanded(True)
                logger.debug(f"  Added as child to parent: {parent_uuid[:8]}...")
            else:
                self.addTopLevelItem(item)
                logger.debug(f"  Added as top-level item")

            # Update internal dictionaries
            self.branches_dict[uuid] = item
            self.visibility_status[uuid] = True

            logger.debug(f"  Total branches after: {len(self.branches_dict)}")

            # Emit the visibility_status update signal
            logger.debug(f"  Emitting branch_added signal...")
            self.branch_added.emit(self.visibility_status)
            logger.debug(f"  branch_added signal emitted")

        except Exception as e:
            logger.error(f"  Error in add_branch(): {e}")
            import traceback
            logger.error(f"  Traceback:\n{traceback.format_exc()}")
            raise

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

    def remove_branch(self, uid: str) -> bool:
        """
        Remove a single branch from the tree widget UI.

        Args:
            uid (str): Unique identifier of the branch to remove.

        Returns:
            bool: True if the branch was removed, False if not found.
        """
        if uid not in self.branches_dict:
            return False

        item = self.branches_dict[uid]
        parent = item.parent()

        if parent:
            parent.removeChild(item)
        else:
            index = self.indexOfTopLevelItem(item)
            self.takeTopLevelItem(index)

        del self.branches_dict[uid]
        if uid in self.visibility_status:
            del self.visibility_status[uid]

        return True

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
                # Detect text rename vs visibility toggle
                stored_name = item.data(0, Qt.UserRole + 1)
                current_text = item.text(0)
                if stored_name is not None and current_text != stored_name:
                    # Text was edited — persist rename to DataNode.alias
                    item.setData(0, Qt.UserRole + 1, current_text)
                    controller = global_variables.global_application_controller
                    if controller:
                        node = controller.get_node(uid)
                        if node:
                            node.alias = current_text
                else:
                    # Visibility checkbox changed
                    self.visibility_status[uid] = item.checkState(0) == Qt.Checked
                    self.branch_visibility_changed.emit(self.visibility_status)
            elif column == 1:
                # Cache checkbox changed - use singleton pattern (NO signal!)

                # Check if this is a root node (always cached)
                is_root = item.data(1, Qt.UserRole) == "root_cached"
                if is_root and item.checkState(1) == Qt.Unchecked:
                    # Prevent unchecking root nodes - restore checked state
                    item.setCheckState(1, Qt.Checked)
                    return

                is_cached = item.checkState(1) == Qt.Checked
                controller = global_variables.global_application_controller
                if controller:
                    if is_cached:
                        controller.cache_node(uid)
                    else:
                        controller.uncache_node(uid)
                    # Update memory tooltip
                    memory_usage = controller.get_cache_memory_usage(uid)
                    self.update_cache_tooltip(uid, memory_usage)

    def on_selection_changed(self):
        """
        Handles item selection changes and emits the selected branches.

        Uses _ctrl_held captured at mouse press time instead of checking
        keyboard modifiers here (which may have changed by the time this is called).
        """
        try:
            logger.debug(f"on_selection_changed: _ctrl_held = {self._ctrl_held}")
            selected_items = self.selectedItems()
            last_selected_item = selected_items[-1] if selected_items else None

            if last_selected_item:
                # Use stored Ctrl state from mousePressEvent for reliable multi-select
                if not self._ctrl_held:
                    # Single selection mode - clear others
                    self.blockSignals(True)
                    self.clearSelection()
                    last_selected_item.setSelected(True)
                    self.blockSignals(False)

                    # Update selected_items to reflect the new selection
                    selected_items = self.selectedItems()

            selected_uids = [item.data(0, Qt.UserRole) for item in selected_items if item.data(0, Qt.UserRole)]
            logger.debug(f"on_selection_changed: emitting {len(selected_uids)} selected UIDs")
            self.branch_selection_changed.emit(selected_uids)
        except Exception as e:
            logger.error(f"Error in on_selection_changed: {e}", exc_info=True)

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
        Update the cache column to show memory usage as text label.

        Args:
            uid (str): UUID of the branch.
            memory_usage (str, optional): Memory usage string (e.g., "12.34 MB").
                                         If None, shows empty string.
        """
        item = self.branches_dict.get(uid)
        if item:
            if memory_usage:
                # Set the text label in the cache column to show memory usage
                item.setText(1, memory_usage)
                item.setToolTip(1, f"Memory usage: {memory_usage}")
            else:
                # Clear the text if no memory usage info
                item.setText(1, "")
                item.setToolTip(1, "")
