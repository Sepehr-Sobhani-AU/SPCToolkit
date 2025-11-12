# core/data_manager.py
"""Centralised manager for handling data operations, analysis, and interactions"""
from config.config import global_variables
import uuid

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from core.node_reconstruction_manager import NodeReconstructionManager
from services.file_manager import FileManager
from core.point_cloud import PointCloud
from core.data_node import DataNode
from core.data_nodes import DataNodes
from core.anaysis_manager import AnalysisManager
from plugins.plugin_manager import PluginManager

from gui.dialog_boxes.dialog_boxes_manager import DialogBoxesManager
from gui.widgets.tree_structure_widget import TreeStructureWidget
from gui.widgets.pcd_viewer_widget import PCDViewerWidget


class DataManager(QObject):
    """
    Centralised manager for handling data operations, analysis, and interactions
    between widgets and data nodes.

    This version has been updated to use the plugin-based AnalysisManager.
    """

    # Signals for UI updates
    visibility_changed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self,
                 file_manager: FileManager,
                 tree_widget: TreeStructureWidget,
                 viewer_widget: PCDViewerWidget,
                 dialog_boxes_manager: DialogBoxesManager,
                 plugin_manager: PluginManager):
        super().__init__()
        self.file_manager = file_manager
        self.tree_widget = tree_widget
        self.viewer_widget = viewer_widget
        self.dialog_boxes_manager = dialog_boxes_manager
        self.plugin_manager = plugin_manager
        self.data_nodes = DataNodes()

        # Set the data nodes instance in the global variables for easy access
        global_variables.global_data_nodes = self.data_nodes

        # Create the analysis manager with the plugin manager
        self.analysis_manager = AnalysisManager(plugin_manager)
        self.node_reconstruction_manager = NodeReconstructionManager()
        self.selected_branches = []

        # Connect signals
        self.file_manager.point_cloud_loaded.connect(self._on_point_cloud_loaded)
        self.analysis_manager.analysis_completed.connect(self._on_analysis_completed)
        self.tree_widget.branch_visibility_changed.connect(self._on_branch_visibility_changed)
        self.tree_widget.branch_added.connect(self._on_branch_added)
        self.tree_widget.branch_selection_changed.connect(self._on_branch_selection_changed)
        self.dialog_boxes_manager.analysis_params.connect(self.apply_analysis)
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
            data_node = DataNode(params=point_cloud.name, data=point_cloud, data_type="point_cloud", parent_uid=None,
                                 depends_on=None, tags=[])
            # Add the DataNode to the DataNodes manager
            uid = self.data_nodes.add_node(data_node)

            # Update the TreeStructureWidget
            self.tree_widget.add_branch(str(uid), "", point_cloud.name)

        except Exception as e:
            self.error_occurred.emit(f"Failed to load point cloud: {file_path}. Error: {str(e)}")

    def _on_analysis_completed(self, result: any, result_type: str, dependencies: list, parent: DataNode, analysis_type: str, params: dict):

        try:
            # Create a DataNode from the analysis result
            data_node = DataNode(f"{analysis_type}" + f",{params}", data=result, data_type=result_type,
                                 parent_uid=parent.uid, depends_on=dependencies, tags=[analysis_type, params])

            # Add the DataNode to the DataNodes manager
            uid = self.data_nodes.add_node(data_node)

            # Update the TreeStructureWidget
            self.tree_widget.add_branch(str(uid), str(parent.uid), data_node.params)

        except Exception as e:
            self.error_occurred.emit(f"Failed to apply analysis: {analysis_type}. Error: {str(e)}")

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

    def apply_analysis(self, analysis_type: str, params: dict):
        """
        Apply an analysis to the selected branches using threaded execution.

        Args:
            analysis_type (str): The type of analysis to apply.
            params (dict): Parameters for the analysis.
        """

        # Get global instances via singleton pattern
        main_window = global_variables.global_main_window
        thread_manager = global_variables.global_analysis_thread_manager

        # Show overlay and disable menus and tree during processing
        main_window.tree_overlay.position_over(self.tree_widget)
        main_window.tree_overlay.show_processing(f"Running {analysis_type}...")
        main_window.disable_menus()
        main_window.disable_tree()

        # Get plugin class
        plugins = self.plugin_manager.get_analysis_plugins()
        if analysis_type not in plugins:
            print(f"Error: Plugin '{analysis_type}' not found")
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            return

        plugin_class = plugins[analysis_type]

        # Process each selected branch
        for uid in self.selected_branches:
            data_node = self.data_nodes.get_node(uuid.UUID(uid))

            # Note: Reconstruction will happen in the background thread if needed
            # We pass the original data_node and the thread will handle reconstruction

            # Start threaded analysis - uses singleton pattern, no callback!
            # Pass the UID so thread can do reconstruction if needed
            thread_manager.start_analysis(plugin_class, data_node, params, analysis_type, str(data_node.uid))

        # Start polling for completion
        self._start_completion_polling()

    def _start_completion_polling(self):
        """Start QTimer to poll for thread completion."""
        if not hasattr(self, '_completion_timer'):
            from PyQt5.QtCore import QTimer
            self._completion_timer = QTimer()
            self._completion_timer.timeout.connect(self._check_thread_completion)

        self._completion_timer.start(100)  # Poll every 100ms

    def _check_thread_completion(self):
        """
        Check if analysis thread completed.
        Called periodically by QTimer. Thread manager will process results via singleton pattern.
        """
        thread_manager = global_variables.global_analysis_thread_manager

        # Thread manager processes completion and calls handle_analysis_result() via singleton
        if thread_manager.check_and_process_completion():
            self._completion_timer.stop()

    def handle_analysis_result(self, result, result_type, dependencies, data_node, analysis_type, params):
        """
        Handle completed analysis result.

        Called by AnalysisThreadManager via singleton pattern when analysis completes.

        Args:
            result: The analysis result (PointCloud, Masks, Clusters, etc.)
            result_type: Type identifier for the result
            dependencies: List of dependency UIDs
            data_node: The DataNode that was analyzed
            analysis_type: Name of the analysis that was performed
            params: Parameters used for the analysis
        """
        # Create DataNode for the result
        data_node_result = DataNode(
            f"{analysis_type},{params}",
            data=result,
            data_type=result_type,
            parent_uid=data_node.uid,
            depends_on=dependencies,
            tags=[analysis_type, params]
        )

        # Add to data nodes collection
        uid = self.data_nodes.add_node(data_node_result)

        # Update tree widget
        self.tree_widget.add_branch(str(uid), str(data_node.uid), data_node_result.params)

    # TODO: Docstrings
    # TODO: Validations
    def reconstruct_branch(self, uid) -> PointCloud:
        # Step up the hierarchy until the root data node is reached and create a list of data node UUIDs
        # in the hierarchy.
        # TODO: There is inconsistancy in type of uid, it is str in some places and uuid.UUID in others.
        self.data_node_uids = []
        if isinstance(uid, str):
            uid = uuid.UUID(uid)
        data_node = self.data_nodes.get_node(uid)
        while data_node.parent_uid is not None:
            self.data_node_uids.append(data_node.uid)
            parent_uid = data_node.parent_uid
            data_node = self.data_nodes.get_node(parent_uid)

        self.data_node_uids.append(data_node.uid)


        # Reverse the list of data node UUIDs to step back down the hierarchy from the root data node to the current data node.
        self.data_node_uids.reverse()

        # Apply the analysis reconstruction to 'data_node_uuids' list of data nodes recursively.
        # The first data node in the list is the root and is a PointCloud instance.
        # The AnalysisReconstruction class will be used as a recursive function, it will apply the analysis reconstruction
        # to all data nodes in the hierarchy.
        # It will get a PointCloud instance and a DataNode instance as input and return a PointCloud instance.
        # So, each time the AnalysisReconstruction class is called, it will return a PointCloud instance that will be used
        # as input for the next call. Also, the DataNode instance will be the next item in 'data_node_uuids' list.

        uid = self.data_node_uids[0]
        data_node = self.data_nodes.get_node(uid)
        point_cloud = data_node.data
        for uid in self.data_node_uids[1:]:
            data_node = self.data_nodes.get_node(uid)
            if data_node.data_type == "point_cloud":
                point_cloud = data_node.data
            else:
                point_cloud = self.node_reconstruction_manager.reconstruct_node(point_cloud, data_node)
        return point_cloud

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
        # Get main window reference
        main_window = global_variables.global_main_window

        # Show overlay and disable menus and tree during processing
        main_window.tree_overlay.position_over(self.tree_widget)
        main_window.tree_overlay.show_processing("Updating visibility...")
        main_window.disable_menus()
        main_window.disable_tree()

        try:
            self._render_visible_data(visibility_status, zoom_extent=False)
        finally:
            # Hide overlay and re-enable menus and tree
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

    def _on_branch_selection_changed(self, uids: list[str]):
        """
        Handle branch selection changes from the tree structure widget.

        Args:
            uids (list[str]): List of selected branch UUIDs.
        """
        self.selected_branches = uids

    def _on_branch_added(self, visibility_status: dict):
        """
        Handle branch additions from the tree structure widget.

        Args:
            visibility_status (dict): Dictionary of UUIDs to visibility states.
        """
        # Get main window reference
        main_window = global_variables.global_main_window

        # Show overlay and disable menus and tree during processing
        main_window.tree_overlay.position_over(self.tree_widget)
        main_window.tree_overlay.show_processing("Rendering new branch...")
        main_window.disable_menus()
        main_window.disable_tree()

        try:
            self._render_visible_data(visibility_status, zoom_extent=True)
        finally:
            # Hide overlay and re-enable menus and tree
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

    def _render_visible_data(self, visibility_status: dict, zoom_extent: bool = False):
        """
        Handle visibility toggles from the tree structure widget.

        Args:
            visibility_status (dict): Dictionary of UUIDs to visibility states.
            zoom_extent (bool): Whether to zoom to the extent of the visible data
        """
        points_to_show = np.empty((0, 3), dtype=np.float32)
        colors_to_show = np.empty((0, 3), dtype=np.float32)
        uids_to_show = [uid for uid, vis in visibility_status.items() if vis]
        # TODO: Update to handle multiple data types, point clouds, derived data, etc.
        if uids_to_show:
            for uid in uids_to_show:

                node = self.data_nodes.get_node(uuid.UUID(uid))
                node_type = node.data_type
                # Render point clouds

                point_cloud = self.reconstruct_branch(uid)

                points_to_show = np.append(points_to_show, point_cloud.points, axis=0)
                if point_cloud.colors is not None:
                    colors_to_show = np.append(colors_to_show, point_cloud.colors, axis=0)
                else:
                    # If no colors are present, use white
                    colors_to_show = np.append(colors_to_show, np.ones((point_cloud.size, 3), dtype=np.float32), axis=0)

            self.viewer_widget.set_points(points=points_to_show, colors=colors_to_show)
        else:
            self.viewer_widget.set_points(None)

        self.viewer_widget.update()

        if zoom_extent:
            self.viewer_widget.zoom_to_extent()
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
