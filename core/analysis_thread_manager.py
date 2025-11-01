"""
AnalysisThreadManager - Manages background analysis execution using threading.

This module provides thread-based execution of analysis plugins to keep the UI responsive
during long-running operations. Uses singleton pattern (global_variables) instead of callbacks
for communication.
"""

import threading
from typing import Optional, Type, Dict, Any
from config.config import global_variables


class AnalysisThreadManager:
    """
    Manages background analysis thread execution.

    Uses singleton pattern to communicate with other components via global_variables.
    When analysis completes, directly calls methods on global instances rather than
    using callbacks or signals.

    Attributes:
        active_thread: Currently running AnalysisThread instance
        is_completed: Flag indicating if analysis has completed
        result_data: Dictionary containing successful analysis results
        error_data: Dictionary containing error information if analysis failed
    """

    def __init__(self):
        """Initialize the AnalysisThreadManager."""
        self.active_thread: Optional['AnalysisThread'] = None
        self.is_completed: bool = False
        self.result_data: Optional[Dict[str, Any]] = None
        self.error_data: Optional[Dict[str, Any]] = None

    def start_analysis(self, plugin_class: Type, data_node, params: Dict[str, Any], analysis_type: str, node_uid: str):
        """
        Start analysis in background thread.

        Uses singleton pattern - no callback parameter needed. Results are processed
        by calling methods on global_variables instances.

        Args:
            plugin_class: The plugin class to instantiate and execute
            data_node: DataNode containing the data to analyze
            params: Dictionary of parameters for the plugin
            analysis_type: Name/type of the analysis being performed
            node_uid: UID of the node (for reconstruction if needed)
        """
        # Reset state
        self.is_completed = False
        self.result_data = None
        self.error_data = None

        # Create and start thread
        thread = AnalysisThread(plugin_class, data_node, params, analysis_type, node_uid, self)
        self.active_thread = thread
        thread.start()

        print(f"Started analysis thread for '{analysis_type}'")

    def is_running(self) -> bool:
        """
        Check if analysis thread is currently running.

        Returns:
            bool: True if thread is active and running, False otherwise
        """
        return self.active_thread is not None and self.active_thread.is_alive()

    def check_and_process_completion(self) -> bool:
        """
        Check if analysis completed and process results using singleton pattern.

        This method is called periodically by QTimer from the main thread. When analysis
        is complete, it directly calls methods on global instances to update the UI
        and data structures.

        Returns:
            bool: True if processing occurred (analysis was completed), False otherwise
        """
        if not self.is_completed:
            return False

        # Get global instances via singleton pattern
        main_window = global_variables.global_main_window
        data_manager = global_variables.global_data_manager

        # Hide overlay and enable menus (UI updates)
        main_window.tree_overlay.hide_processing()
        main_window.enable_menus()

        if self.error_data:
            # Handle error case
            error_msg = self.error_data.get('error', 'Unknown error')
            analysis_type = self.error_data.get('analysis_type', 'Unknown')
            print(f"Analysis '{analysis_type}' failed: {error_msg}")

            self._cleanup()
            return True

        # Process successful result - call DataManager method directly via singleton
        data_manager.handle_analysis_result(
            self.result_data['result'],
            self.result_data['result_type'],
            self.result_data['dependencies'],
            self.result_data['data_node'],
            self.result_data['analysis_type'],
            self.result_data['params']
        )

        print(f"Analysis '{self.result_data['analysis_type']}' completed successfully")

        self._cleanup()
        return True

    def _cleanup(self):
        """Clean up manager state after processing completion."""
        self.is_completed = False
        self.result_data = None
        self.error_data = None
        self.active_thread = None

    def _mark_completed(self, result_data: Optional[Dict], error_data: Optional[Dict]):
        """
        Mark analysis as completed (called by AnalysisThread).

        This method is thread-safe and can be called from the background thread.

        Args:
            result_data: Dictionary containing result, result_type, dependencies, etc.
            error_data: Dictionary containing error information (if failed)
        """
        self.result_data = result_data
        self.error_data = error_data
        self.is_completed = True


class AnalysisThread(threading.Thread):
    """
    Background thread that executes analysis plugin.

    Runs plugin.execute() in a separate thread to avoid blocking the UI.
    Deep copies data before execution to ensure thread safety.
    Handles reconstruction if data_node is not a point_cloud type.
    """

    def __init__(self, plugin_class: Type, data_node, params: Dict[str, Any],
                 analysis_type: str, node_uid: str, manager: AnalysisThreadManager):
        """
        Initialize the AnalysisThread.

        Args:
            plugin_class: The plugin class to instantiate
            data_node: DataNode to process
            params: Plugin parameters
            analysis_type: Name of the analysis
            node_uid: UID of the node (for reconstruction)
            manager: AnalysisThreadManager instance to notify when complete
        """
        super().__init__(daemon=True)
        self.plugin_class = plugin_class
        self.params = params  # No need to copy - plugins don't modify params
        self.analysis_type = analysis_type
        self.node_uid = node_uid
        self.manager = manager

        # Store reference directly (no deep copy needed)
        # Plugins only READ from data, they don't modify it
        # This is memory efficient and thread-safe for read-only access
        self.data_node = data_node

    def run(self):
        """
        Execute plugin in background thread.

        This method runs in a separate thread. It handles reconstruction if needed,
        creates a plugin instance, executes it, and notifies the manager when complete.

        Note: No deep copy is performed. Plugins only read data (thread-safe) and
        return new objects. This is memory efficient.
        """
        try:
            # Use data_node directly (no copy needed - read-only access is thread-safe)
            data_node_to_process = self.data_node

            # Handle reconstruction in background thread if data is not a point_cloud
            if data_node_to_process.data_type != "point_cloud":
                print(f"Reconstructing branch in background thread...")
                # Reconstruct the branch in the background thread
                from core.data_node import DataNode
                point_cloud = self._reconstruct_branch_in_thread()
                # Create a new temporary DataNode for the reconstructed point cloud
                reconstructed_data_node = DataNode(
                    params=data_node_to_process.params,
                    data=point_cloud,
                    data_type="point_cloud"
                )
                reconstructed_data_node.uid = data_node_to_process.uid
                data_node_to_process = reconstructed_data_node

            # Create plugin instance in this thread
            plugin_instance = self.plugin_class()

            print(f"Executing plugin '{self.analysis_type}' in background thread...")

            # Execute analysis (plugin only reads data, doesn't modify it)
            result, result_type, dependencies = plugin_instance.execute(
                data_node_to_process,
                self.params
            )

            # Package result data
            result_data = {
                'result': result,
                'result_type': result_type,
                'dependencies': dependencies,
                'data_node': data_node_to_process,
                'analysis_type': self.analysis_type,
                'params': self.params
            }

            # Mark completed with success (thread-safe)
            self.manager._mark_completed(result_data, None)

        except Exception as e:
            # Mark completed with error (thread-safe)
            error_data = {
                'error': str(e),
                'analysis_type': self.analysis_type
            }
            self.manager._mark_completed(None, error_data)

            print(f"Exception in analysis thread: {str(e)}")

    def _reconstruct_branch_in_thread(self):
        """
        Reconstruct branch in background thread.
        This is a copy of DataManager.reconstruct_branch() but runs in the background.
        """
        import uuid as uuid_module
        from core.node_reconstruction_manager import NodeReconstructionManager

        # Get data_nodes from global_variables
        data_nodes = global_variables.global_data_nodes
        reconstruction_manager = NodeReconstructionManager()

        # Build hierarchy list
        data_node_uids = []
        uid = uuid_module.UUID(self.node_uid) if isinstance(self.node_uid, str) else self.node_uid
        data_node = data_nodes.get_node(uid)

        while data_node.parent_uid is not None:
            data_node_uids.append(data_node.uid)
            parent_uid = data_node.parent_uid
            data_node = data_nodes.get_node(parent_uid)

        data_node_uids.append(data_node.uid)
        data_node_uids.reverse()

        # Reconstruct from root to target
        uid = data_node_uids[0]
        data_node = data_nodes.get_node(uid)
        point_cloud = data_node.data

        for uid in data_node_uids[1:]:
            data_node = data_nodes.get_node(uid)
            if data_node.data_type == "point_cloud":
                point_cloud = data_node.data
            else:
                point_cloud = reconstruction_manager.reconstruct_node(point_cloud, data_node)

        return point_cloud
