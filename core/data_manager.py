# core/data_manager.py
"""Centralised manager for handling data operations, analysis, and interactions"""
from config.config import global_variables
import uuid

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, Qt

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

            # Update the TreeStructureWidget - root nodes are always "cached"
            self.tree_widget.add_branch(str(uid), "", point_cloud.name, is_root=True)

            # Set tooltip showing memory usage for root node
            memory_usage = self.get_cache_memory_usage(str(uid))
            self.tree_widget.update_cache_tooltip(str(uid), memory_usage)

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

        # Update UI to show cache status (thread may have cached the parent during analysis)
        # The thread auto-caches if it had to reconstruct, so we just update the UI here
        if data_node.is_cached:
            # Update UI to show cache checkbox as checked
            # Block signals to prevent triggering on_item_checked which would call cache_branch()
            item = self.tree_widget.branches_dict.get(str(data_node.uid))
            if item:
                self.tree_widget.blockSignals(True)
                item.setCheckState(1, Qt.Checked)
                self.tree_widget.blockSignals(False)

            # Update tooltip
            memory_usage = self.get_cache_memory_usage(str(data_node.uid))
            self.tree_widget.update_cache_tooltip(str(data_node.uid), memory_usage)

            print(f"[CACHE STATUS] Parent is cached: {data_node.params} ({memory_usage})")

        # Hide parent and show only the new child result
        parent_uid_str = str(data_node.uid)
        child_uid_str = str(uid)

        # Update visibility status
        if parent_uid_str in self.tree_widget.visibility_status:
            self.tree_widget.visibility_status[parent_uid_str] = False
        self.tree_widget.visibility_status[child_uid_str] = True

        # Update UI checkboxes
        parent_item = self.tree_widget.branches_dict.get(parent_uid_str)
        if parent_item:
            parent_item.setCheckState(0, Qt.Unchecked)

        child_item = self.tree_widget.branches_dict.get(child_uid_str)
        if child_item:
            child_item.setCheckState(0, Qt.Checked)

        # Trigger visibility update to render the child
        self.tree_widget.branch_visibility_changed.emit(self.tree_widget.visibility_status)

    # TODO: Docstrings
    # TODO: Validations
    def reconstruct_branch(self, uid) -> PointCloud:
        """
        Reconstruct a branch by applying transformations from nearest cached ancestor to target.

        This method steps up the hierarchy checking each ancestor for a cache.
        When a cached ancestor is found (or root is reached), reconstruction starts from there.
        This is more efficient than the previous approach of always building the full hierarchy.

        Args:
            uid: UUID of the branch to reconstruct (str or uuid.UUID).

        Returns:
            PointCloud: The reconstructed point cloud.

        Raises:
            ValueError: If the data node is not found.
        """
        # TODO: There is inconsistancy in type of uid, it is str in some places and uuid.UUID in others.
        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        # Get target node
        target_node = self.data_nodes.get_node(uid)
        if target_node is None:
            raise ValueError(f"DataNode with UID {uid} not found")

        # CRITICAL: If the target itself is cached, return it immediately!
        # No need to reconstruct anything if we already have the result
        if target_node.is_cached and target_node.cached_point_cloud is not None:
            print(f"[CACHE DIRECT HIT] Returning cached target: {target_node.params}")
            return target_node.cached_point_cloud

        # Build path from target back to root (or first cached ancestor)
        path = [target_node]
        current_node = target_node

        # Step up hierarchy, checking for cached ancestors
        while current_node.parent_uid is not None:
            parent_node = self.data_nodes.get_node(current_node.parent_uid)
            if parent_node is None:
                break

            path.append(parent_node)

            # Check if this parent is cached (or is root PointCloud with data in memory)
            if parent_node.is_cached and parent_node.cached_point_cloud is not None:
                # Found cached ancestor - stop here
                break
            elif parent_node.data_type == "point_cloud" and parent_node.parent_uid is None:
                # This is root PointCloud - stop here (data always in memory)
                break

            current_node = parent_node

        # Reverse path to go from start point (cache/root) to target
        path.reverse()

        # Start from the first node (either cached ancestor or root)
        start_node = path[0]
        if start_node.is_cached and start_node.cached_point_cloud is not None:
            # Start from cached reconstruction
            point_cloud = start_node.cached_point_cloud
            print(f"[CACHE HIT] Starting from cached node: {start_node.params} | Path length: {len(path)} | Steps saved: {len(path)-1}")
        else:
            # Start from root PointCloud data
            point_cloud = start_node.data
            print(f"[CACHE MISS] Starting from root: {start_node.params} | Path length: {len(path)} | Full reconstruction")

        # Apply transformations from start to target
        for node in path[1:]:
            if node.data_type == "point_cloud":
                # Direct PointCloud (shouldn't happen in normal hierarchy, but handle it)
                point_cloud = node.data
            else:
                # Apply transformation
                print(f"[RECONSTRUCTION] Applying {node.data_type} transformation for {node.params}")
                point_cloud = self.node_reconstruction_manager.reconstruct_node(point_cloud, node)

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
        import time

        points_to_show = np.empty((0, 3), dtype=np.float32)
        colors_to_show = np.empty((0, 3), dtype=np.float32)
        uids_to_show = [uid for uid, vis in visibility_status.items() if vis]

        # TODO: Update to handle multiple data types, point clouds, derived data, etc.
        if uids_to_show:
            for uid in uids_to_show:
                node = self.data_nodes.get_node(uuid.UUID(uid))
                node_type = node.data_type

                # Reconstruct the branch (will use cache if available)
                point_cloud = self.reconstruct_branch(uid)

                # Calculate memory usage for display (works for any point cloud)
                memory_usage = self._calculate_point_cloud_memory(point_cloud)

                # Auto-cache visible branches since they're already in memory
                # If not already cached, cache it now
                if not node.is_cached:
                    node.cached_point_cloud = point_cloud
                    node.is_cached = True
                    node.cache_timestamp = time.time()

                    # Update UI to show cache checkbox as checked
                    # Block signals to prevent triggering on_item_checked which would call cache_branch()
                    item = self.tree_widget.branches_dict.get(uid)
                    if item:
                        self.tree_widget.blockSignals(True)
                        item.setCheckState(1, Qt.Checked)
                        self.tree_widget.blockSignals(False)

                    print(f"[AUTO-CACHE] Cached visible branch: {node.params} ({memory_usage})")

                # Always display memory usage (whether cached or not)
                self.tree_widget.update_cache_tooltip(uid, memory_usage)

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

    def cache_branch(self, uid: str):
        """
        Cache the reconstruction result for a branch.

        Args:
            uid (str): UUID of the branch to cache.
        """
        import time

        node = self.data_nodes.get_node(uuid.UUID(uid))
        if node is None:
            print(f"Warning: Cannot cache branch {uid}, node not found")
            return

        # Check if already cached - if so, just update UI and return
        if node.is_cached and node.cached_point_cloud is not None:
            print(f"[CACHE] Branch already cached: {node.params}")
            # Update UI tooltip (in case it wasn't set)
            memory_usage = self.get_cache_memory_usage(uid)
            self.tree_widget.update_cache_tooltip(uid, memory_usage)
            return

        # Reconstruct the branch (only if not already cached)
        point_cloud = self.reconstruct_branch(uid)

        # Store cache in the node
        node.cached_point_cloud = point_cloud
        node.is_cached = True
        node.cache_timestamp = time.time()

        # Update UI tooltip
        memory_usage = self.get_cache_memory_usage(uid)
        self.tree_widget.update_cache_tooltip(uid, memory_usage)

        print(f"[CACHE] Cached branch: {node.params} ({memory_usage})")

    def uncache_branch(self, uid: str):
        """
        Remove cached data for a branch.

        Args:
            uid (str): UUID of the branch to uncache.
        """
        node = self.data_nodes.get_node(uuid.UUID(uid))
        if node is None:
            print(f"Warning: Cannot uncache branch {uid}, node not found")
            return

        # Get memory usage before clearing cache (so we can still show it)
        memory_usage = self.get_cache_memory_usage(uid)

        # Clear cache
        node.cached_point_cloud = None
        node.is_cached = False
        node.cache_timestamp = None

        # Keep showing memory usage even after uncaching (helps user decide)
        # The size remains the same, just not taking up RAM anymore
        self.tree_widget.update_cache_tooltip(uid, memory_usage)

        print(f"Uncached branch: {node.params} (was {memory_usage})")

    def _calculate_point_cloud_memory(self, point_cloud) -> str:
        """
        Calculate approximate memory usage of a point cloud.

        Args:
            point_cloud: PointCloud instance to calculate memory for.

        Returns:
            str: Memory usage in human-readable format (e.g., "12.34 MB").
        """
        if point_cloud is None:
            return "0 MB"

        bytes_used = 0

        # Calculate memory for points
        if hasattr(point_cloud, 'points') and point_cloud.points is not None:
            bytes_used += point_cloud.points.nbytes

        # Calculate memory for colors
        if hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
            bytes_used += point_cloud.colors.nbytes

        # Calculate memory for normals
        if hasattr(point_cloud, 'normals') and point_cloud.normals is not None:
            bytes_used += point_cloud.normals.nbytes

        # Calculate memory for attributes
        if hasattr(point_cloud, 'attributes') and point_cloud.attributes is not None:
            for key, value in point_cloud.attributes.items():
                if hasattr(value, 'nbytes'):
                    bytes_used += value.nbytes

        mb_used = bytes_used / (1024 * 1024)
        return f"{mb_used:.2f} MB"

    def get_cache_memory_usage(self, uid: str) -> str:
        """
        Calculate approximate memory usage of cached point cloud.

        Args:
            uid (str): UUID of the branch.

        Returns:
            str: Memory usage in human-readable format (e.g., "12.34 MB").
        """
        node = self.data_nodes.get_node(uuid.UUID(uid))
        if node is None or not node.is_cached or node.cached_point_cloud is None:
            return "0 MB"

        return self._calculate_point_cloud_memory(node.cached_point_cloud)

    def invalidate_descendant_caches(self, uid: str):
        """
        Invalidate all caches that depend on this branch.

        This should be called when a branch is modified to ensure descendant
        caches are rebuilt. Currently not used since nodes are immutable (only
        created, not modified), but useful for future features.

        Args:
            uid (str): UUID of the branch that was modified.
        """
        nodes_to_invalidate = []
        uid_obj = uuid.UUID(uid)

        # Find all descendants that have caches
        for node_uid, node in self.data_nodes.data_nodes.items():
            if self._is_descendant_of(node, uid_obj) and node.is_cached:
                nodes_to_invalidate.append(str(node_uid))

        # Clear their caches
        for node_uid in nodes_to_invalidate:
            node = self.data_nodes.get_node(uuid.UUID(node_uid))
            node.cached_point_cloud = None
            node.cache_timestamp = None
            # Note: We keep is_cached=True so user preference is preserved
            # The cache will be rebuilt next time it's accessed

            # Update UI to show cache is stale/needs rebuild
            item = self.tree_widget.branches_dict.get(node_uid)
            if item:
                item.setCheckState(1, Qt.Unchecked)

        if nodes_to_invalidate:
            print(f"Invalidated {len(nodes_to_invalidate)} descendant caches")

    def _is_descendant_of(self, node: DataNode, ancestor_uid: uuid.UUID) -> bool:
        """
        Check if a node is a descendant of the specified ancestor.

        Args:
            node: The node to check.
            ancestor_uid: UUID of the potential ancestor.

        Returns:
            bool: True if node is a descendant of ancestor_uid.
        """
        current = node
        while current.parent_uid is not None:
            if current.parent_uid == ancestor_uid:
                return True
            current = self.data_nodes.get_node(current.parent_uid)
            if current is None:
                break
        return False

    # Render visible data nodes in the viewer widget

    # Convert list of valid UUID strings to UUID objects
    @staticmethod
    def _parse_uuids(uuids: list[str]) -> list[uuid.UUID]:
        return [uuid.UUID(uid) for uid in uuids]
