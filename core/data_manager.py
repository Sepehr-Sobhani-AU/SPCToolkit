"""Centralised manager for handling data operations, analysis, and interactions"""

# TODO: apply_analysis method is incomplete and needs to be implemented

import uuid

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from services.file_manager import FileManager
from core.point_cloud import PointCloud
from core.data_node import DataNode
from core.data_nodes import DataNodes
from core.anaysis_manager import AnalysisManager
from gui.widgets.tree_structure_widget import TreeStructureWidget
from gui.widgets.pcd_viewer_widget import PCDViewerWidget


class DataManager(QObject):
    """
    Centralised manager for handling data operations, analysis, and interactions
    between widgets and data nodes.

    Responsibilities:
    - Manage data nodes (add, remove, update, validate dependencies).
    - Coordinate rendering in PCDViewerWidget.
    - Synchronise tree structure and handle user interactions.
    - Manage analysis workflows and derived data.
    """

    # Signals for UI updates
    visibility_changed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, file_manager: FileManager, tree_widget: TreeStructureWidget, viewer_widget: PCDViewerWidget, parent=None):
        super().__init__(parent)
        self.file_manager = file_manager
        self.tree_widget = tree_widget
        self.viewer_widget = viewer_widget
        self.data_nodes = DataNodes()
        self.analysis_manager = AnalysisManager()

        # Connect signals
        self.file_manager.point_cloud_loaded.connect(self._on_point_cloud_loaded)
        self.tree_widget.branch_visibility_changed.connect(self._on_branch_visibility_changed)
        self.tree_widget.branch_added.connect(self._on_branch_added)
        self.tree_widget.branch_selection_changed.connect(self._on_branch_selection_changed)
        # self.tree_widget.branch_deleted.connect(self.on_branch_deleted)
        # self.tree_widget.branch_moved.connect(self.on_branch_moved)

    def _on_point_cloud_loaded(self, file_path: str, point_cloud: PointCloud):
        """
        Handle the point cloud loaded signal from FileManager.

        Args:
            file_path (str): The path of the loaded file.
            point_cloud (PointCloud): The loaded point cloud instance.
        """
        try:
            # Create a DataNode from the loaded PointCloud
            data_node = DataNode(
                data=point_cloud,
                name=point_cloud.name,
                parent_uid=None,  # Top-level branch
                depends_on=None
            )
            # Add the DataNode to the DataNodes manager
            uid = self.data_nodes.add_data(data_node)

            # Update the TreeStructureWidget
            self.tree_widget.add_branch(str(uid), None, point_cloud.name)

        except Exception as e:
            self.error_occurred.emit(f"Failed to load point cloud: {file_path}. Error: {str(e)}")

    # def delete_branch(self, uids: list[str]):
    #     """
    #     Delete branches and their corresponding data nodes.
    #
    #     Args:
    #         uids (list[str]): List of branch UUIDs to delete.
    #     """
    #
    #     for uid in uids:
    #         if not self.validate_dependency(uid):
    #             self.error_occurred.emit(f"Cannot delete branch {uid}. It is a dependency.")
    #         else:
    #             self.data_nodes.remove_data(uuid.UUID(uid))
    #             self.tree_widget.remove_branch(uids)
    #             self.viewer_widget.set_data_nodes(self.data_nodes.data_nodes)
    #             self.viewer_widget.update()

    # def move_branch(self, uids: list[str], new_parent_uuid: str):
    #     """
    #     Move branches to a new parent and validate dependencies.
    #
    #     Args:
    #         uids (list[str]): Branch UUIDs to move.
    #         new_parent_uuid (str): New parent UUID.
    #     """
    #     for uid in uids:
    #         if not self.validate_dependency(uid):
    #             self.error_occurred.emit(f"Cannot move branch {uid}. It is a dependency.")
    #
    #         else:
    #             self.data_nodes.update_parent(uuid.UUID(uid), uuid.UUID(new_parent_uuid))
    #             self.tree_widget.move_branch([uid], new_parent_uuid)

    # TODO: I think this method is not needed
    # def toggle_visibility(self, uids: list[str], visibility: bool):
    #     """
    #     Toggle the visibility of specified branches.
    #
    #     Args:
    #         uids (list[str]): Branch UUIDs to toggle visibility.
    #         visibility (bool): Desired visibility state.
    #     """
    #     visibility_status = {uid: visibility for uid in uids}
    #     self.viewer_widget.update_visibility(set(uuid for uid, vis in visibility_status.items() if vis))

    # def apply_analysis(self, uuids: list[str], analysis_type: str, params: dict):
    #     """
    #     Apply an analysis task on selected branches.
    #
    #     Args:
    #         uuids (list[str]): Branch UUIDs to analyse.
    #         analysis_type (str): Type of analysis to perform.
    #         params (dict): Analysis-specific parameters.
    #     """
    #     try:
    #         results = self.analysis_manager.apply_analysis(uuids, analysis_type, params)
    #         for result in results:
    #             result_uuid = self.data_nodes.add_data(result)
    #             self.tree_widget.add_branch(result_uuid, result.depends_on, result.name)
    #         self.viewer_widget.set_data_nodes(self.data_nodes.data_nodes)
    #         self.viewer_widget.update()
    #     except Exception as e:
    #         self.error_occurred.emit(f"Failed to apply analysis. Error: {str(e)}")

    # def validate_dependency(self, uid: str) -> bool:
    #     """
    #     Validate if branch can be safely moved or deleted.
    #
    #     Args:
    #         uid (str): Branch UUID to validate.
    #
    #     Returns:
    #         bool: Validation result.
    #     """
    #     return self.data_nodes.validate_dependency(uuid.UUID(uid))

    def _on_branch_visibility_changed(self, visibility_status: dict):
        """
        Handle visibility changes from the tree structure widget.

        Args:
            visibility_status (dict): Dictionary of UUIDs to visibility states.
        """
        self._render_visible_data(visibility_status, zoom_extent=False)

    def _on_branch_selection_changed(self, uids: list[str]):
        """
        Handle branch selection changes from the tree structure widget.

        Args:
            uids (list[str]): List of selected branch UUIDs.
        """
        self.selected_branches = uids
        for uid in uids:
            print(uid)
        print()

    def _on_branch_added(self, visibility_status: dict):
        """
        Handle branch additions from the tree structure widget.

        Args:
            visibility_status (dict): Dictionary of UUIDs to visibility states.
        """
        self._render_visible_data(visibility_status, zoom_extent=True)

    def _render_visible_data(self, visibility_status: dict, zoom_extent: bool = False):
        """
        Handle visibility toggles from the tree structure widget.

        Args:
            visibility_status (dict): Dictionary of UUIDs to visibility states.
            zoom_extent (bool): Whether to zoom to the extent of the visible data
        """
        points_to_show = np.empty((0, 3), dtype=np.float32)
        uids_to_show = [uid for uid, vis in visibility_status.items() if vis]
        # TODO: Update to handle multiple data types, point clouds, derived data, etc.
        if uids_to_show:
            nodes_to_show = [self.data_nodes.get_data(uuid.UUID(uid)).data for uid in uids_to_show]
            for node in nodes_to_show:
                if isinstance(node, PointCloud):
                    points_to_show = np.append(points_to_show, node.points, axis=0)
            #points_to_show = node.points for node in nodes_to_show if isinstance(node, PointCloud)
            colors_to_show = np.array([node.colors for node in nodes_to_show if isinstance(node, PointCloud)])
            # TODO: Handle colors
            self.viewer_widget.set_points(points_to_show)
        else:
            self.viewer_widget.set_points(None)

        self.viewer_widget.update()

        if zoom_extent:
            self.viewer_widget.zoom_extent()
    # def on_branch_deleted(self, uids: list[str]):
    #     """
    #     Handle branch deletion from the tree structure widget.
    #
    #     Args:
    #         uids (list[str]): UUIDs of branches to delete.
    #     """
    #     self.delete_branch(uids)

    # def on_branch_moved(self, uids: list[str], new_parent_uid: str):
    #     """
    #     Handle branch movement in the tree structure widget.
    #
    #     Args:
    #         uids (list[str]): UUIDs of branches to move.
    #         new_parent_uid (str): UUID of the new parent branch.
    #     """
    #     self.move_branch(uids, new_parent_uid)

    # Render visible data nodes in the viewer widget

    # Convert list of valid UUID strings to UUID objects
    @staticmethod
    def _parse_uuids(uuids: list[str]) -> list[uuid.UUID]:
        return [uuid.UUID(uid) for uid in uuids]
