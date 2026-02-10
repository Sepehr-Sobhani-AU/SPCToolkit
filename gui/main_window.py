# gui/main_window.py
"""
  Create main window for the PCD Toolkit application with dynamic menus from plugins.
"""
import logging
import traceback

from config.config import global_variables

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from gui.widgets import PCDViewerWidget, TreeStructureWidget, ProcessOverlayWidget
from services.file_manager import FileManager
from gui.dialog_boxes.dialog_boxes_manager import DialogBoxesManager
from plugins.plugin_manager import PluginManager
from infrastructure.hardware_detector import HardwareDetector

# Application layer
from application.application_controller import ApplicationController

logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window for the PCD Toolkit, with a tree structure widget as a left pane.

    This version uses plugins to dynamically build its menu structure.
    """

    def __init__(self, plugin_manager: PluginManager):
        super().__init__()
        self.plugin_manager = plugin_manager

        # Set up the main window components
        self.menus = {}  # Dictionary to store menu/submenu references
        self.actions = {}  # Dictionary to store action references

        # Standard components as before
        self.setWindowTitle("PCD Toolkit")
        self.resize(1600, 1200)

        # Create an instance of FileManager
        self.file_manager = FileManager()
        global_variables.global_file_manager = self.file_manager

        # Create global instances of the tree structure widget, PCD viewer widget, dialog boxes manager, and data manager
        self.tree_widget = TreeStructureWidget()
        global_variables.global_tree_structure_widget = self.tree_widget

        self.pcd_viewer_widget = PCDViewerWidget()
        global_variables.global_pcd_viewer_widget = self.pcd_viewer_widget

        self.dialog_boxes_manager = DialogBoxesManager(plugin_manager)

        # Store reference to main window in global variables
        global_variables.global_main_window = self

        # === Application Layer Setup ===
        self.controller = ApplicationController.create(plugin_manager, self.file_manager)
        global_variables.global_application_controller = self.controller
        global_variables.global_data_nodes = self.controller.data_nodes

        # Connect signals to MainWindow handlers (which use ApplicationController)
        self.file_manager.point_cloud_loaded.connect(self._on_point_cloud_loaded)
        self.tree_widget.branch_visibility_changed.connect(self._on_branch_visibility_changed)
        self.tree_widget.branch_added.connect(self._on_branch_added)
        self.tree_widget.branch_selection_changed.connect(self._on_branch_selection_changed)

        # LOD state (synced with rendering_coordinator)
        self._current_sample_rate = 1.0

        # Backward compat: overlay references (kept for plugins that may use them)
        self.tree_overlay = ProcessOverlayWidget(parent=self)
        self.window_overlay = ProcessOverlayWidget(parent=self)

        # Set up the UI components
        self.setup_ui()

    def setup_ui(self):
        """Sets up the main window UI components with dynamically built menus."""

        # Create the QSplitter for the main layout with left and right panes
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left side: TreeStructureWidget
        self.splitter.addWidget(self.tree_widget)

        # Right side: PCDViewerWidget
        self.splitter.addWidget(self.pcd_viewer_widget)

        # Initial sizes for the left and right panes
        self.splitter.setSizes([200, 600])

        # Create basic menu structure
        self.setup_base_menus()

        # Populate menus from plugins
        self.populate_menus_from_plugins()

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

        # Processing progress bar (hidden by default, shown during operations)
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setMaximumHeight(16)
        self._progress_bar.setRange(0, 0)  # Indeterminate by default
        self._progress_bar.hide()
        self.statusbar.addWidget(self._progress_bar)

        # Processing status label (left side, shown during operations)
        self._progress_label = QtWidgets.QLabel()
        self._progress_label.hide()
        self.statusbar.addWidget(self._progress_label)

        # Create permanent hardware info label in status bar (right-aligned)
        self._hardware_status_label = QtWidgets.QLabel()
        self._hardware_status_label.setTextFormat(QtCore.Qt.RichText)
        self._hardware_status_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.statusbar.addPermanentWidget(self._hardware_status_label)

        # Create separate label for mode with tooltip
        self._mode_label = QtWidgets.QLabel()
        self._mode_label.setTextFormat(QtCore.Qt.RichText)
        self._mode_label.setToolTip(
            "<b>Backend Modes:</b><br><br>"
            "<b style='color: #28a745;'>FULL GPU</b>: Linux + NVIDIA GPU + RAPIDS (cuML)<br>"
            "All algorithms use GPU acceleration including DBSCAN and KNN.<br><br>"
            "<b style='color: #ffc107;'>PARTIAL GPU</b>: NVIDIA GPU without RAPIDS<br>"
            "PyTorch and CuPy use GPU, but DBSCAN/KNN use CPU (sklearn/scipy).<br><br>"
            "<b style='color: #dc3545;'>CPU ONLY</b>: No compatible GPU detected<br>"
            "All algorithms run on CPU."
        )
        self.statusbar.addPermanentWidget(self._mode_label)

        # Display hardware info in status bar
        self._update_status_bar_hardware_info()

        # Set up timer for dynamic hardware stats update (every 2 seconds)
        self._stats_timer = QtCore.QTimer(self)
        self._stats_timer.timeout.connect(self._update_status_bar_hardware_info)
        self._stats_timer.start(2000)  # Update every 2 seconds

    def _update_status_bar_hardware_info(self):
        """Update status bar with hardware and dynamic stats."""
        try:
            hardware = global_variables.global_hardware_info
            registry = global_variables.global_backend_registry

            if hardware is None or registry is None:
                self._hardware_status_label.setText("Hardware detection not initialized")
                return

            # Get all dynamic stats from single source (one pynvml call)
            stats = HardwareDetector.get_dynamic_stats()

            # Calculate free memory from stats
            ram_free_gb = stats['ram_total_gb'] - stats['ram_used_gb']
            gpu_free_mb = stats['vram_total_mb'] - stats['vram_used_mb']

            # RAM info with color coding based on free memory
            if ram_free_gb < 2:
                ram_color = "#dc3545"  # Red - critically low
            elif ram_free_gb < 4:
                ram_color = "#ffc107"  # Yellow - getting low
            else:
                ram_color = "#28a745"  # Green - plenty free

            ram_html = f'RAM: <span style="color: {ram_color};">{ram_free_gb:.1f} GB free</span> ({stats["ram_percent"]:.0f}% used)'

            # GPU info with dynamic stats
            if hardware.gpu_available:
                gpu_name = hardware.gpu_name

                # VRAM with color coding based on free memory
                if gpu_free_mb < 1000:
                    vram_color = "#dc3545"  # Red - critically low
                elif gpu_free_mb < 2000:
                    vram_color = "#ffc107"  # Yellow - getting low
                else:
                    vram_color = "#28a745"  # Green - plenty free

                vram_html = f'VRAM: <span style="color: {vram_color};">{gpu_free_mb:,} MB free</span> ({stats["vram_percent"]:.0f}% used)'

                # GPU utilization with color coding
                util = stats['gpu_utilization']
                if util > 80:
                    util_color = "#dc3545"  # Red for high usage
                elif util > 50:
                    util_color = "#ffc107"  # Yellow for medium
                else:
                    util_color = "#28a745"  # Green for low

                util_html = f'GPU: <span style="color: {util_color};">{util}%</span>'

                # Temperature with color coding
                temp = stats['gpu_temp_c']
                if temp is not None:
                    if temp > 80:
                        temp_color = "#dc3545"  # Red for hot
                    elif temp > 65:
                        temp_color = "#ffc107"  # Yellow for warm
                    else:
                        temp_color = "#28a745"  # Green for cool
                    temp_html = f'<span style="color: {temp_color};">{temp}°C</span>'
                else:
                    temp_html = ""

                # Build status (include temp only if available)
                if temp is not None:
                    status_html = f'{gpu_name} | {vram_html} | {util_html} | {temp_html} | {ram_html}'
                else:
                    status_html = f'{gpu_name} | {vram_html} | {util_html} | {ram_html}'
            else:
                # CPU only mode
                status_html = f'{ram_html} | GPU: None'

            self._hardware_status_label.setText(status_html)

            # Update mode label separately (with tooltip)
            scenario = registry.get_scenario()
            if scenario == "FULL GPU":
                scenario_color = "#28a745"  # Green
            elif scenario == "PARTIAL GPU":
                scenario_color = "#ffc107"  # Yellow
            else:
                scenario_color = "#dc3545"  # Red
            self._mode_label.setText(f'<span style="color: {scenario_color}; font-weight: bold;">{scenario}</span>')

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error updating status bar: {e}")
            self._hardware_status_label.setText(f"Status bar error: {e}")

    def setup_base_menus(self):
        """Set up the base menu structure for the application."""
        self.menubar = self.menuBar()
        # All menus are now dynamically created from plugin folder structure!

    def populate_menus_from_plugins(self):
        """
        Populate menus from folder-based plugin structure.

        The folder structure automatically defines the menu hierarchy:
        - plugins/Points/Clustering/dbscan.py -> Menu: Points > Clustering > DBSCAN
        - plugins/Processing/subtract.py      -> Menu: Processing > Subtract
        """
        print("Building menus from folder structure...")

        # Get the menu structure from plugin manager
        menu_structure = self.plugin_manager.get_menu_structure()

        if not menu_structure:
            print("No plugins found in folder structure.")
            return

        # Sort menu paths for consistent ordering
        sorted_menu_paths = sorted(menu_structure.keys())

        for menu_path in sorted_menu_paths:
            plugin_names = menu_structure[menu_path]

            # Create menu hierarchy from path
            # Example: "Points/Clustering" -> Create "Points" menu, then "Clustering" submenu
            self._create_menu_hierarchy(menu_path)

            # Get the target menu for this path
            target_menu = self._get_menu_by_path(menu_path)

            # Add each plugin as a menu action
            for plugin_name in plugin_names:
                self._add_plugin_menu_action(target_menu, plugin_name, menu_path)

        print(f"Created {len(self.menus)} menus with {len(self.actions)} actions")

    def _create_menu_hierarchy(self, menu_path: str):
        """
        Create menu hierarchy from a path like "Points/Clustering/Advanced".

        Automatically strips numeric prefixes (e.g., "000_File" -> "File") from display names.

        Args:
            menu_path: Menu path with forward slashes (e.g., "Points/Clustering")
        """
        parts = menu_path.split('/')
        current_path = ""
        parent_menu = None

        for i, part in enumerate(parts):
            # Build the path incrementally
            if current_path:
                current_path += f"/{part}"
            else:
                current_path = part

            # Check if menu already exists
            if current_path in self.menus:
                parent_menu = self.menus[current_path]
                continue

            # Strip numeric prefix from display name
            display_name = self.plugin_manager._strip_prefix(part)

            # Create the menu or submenu
            if i == 0:
                # Top-level menu
                menu = self.menubar.addMenu(display_name)
                self.menus[current_path] = menu
                parent_menu = menu
            else:
                # Submenu
                submenu = QtWidgets.QMenu(display_name, self)
                parent_menu.addMenu(submenu)
                self.menus[current_path] = submenu
                parent_menu = submenu

    def _get_menu_by_path(self, menu_path: str):
        """
        Get a menu by its path.

        Args:
            menu_path: Menu path (e.g., "Points/Clustering")

        Returns:
            QMenu object
        """
        return self.menus.get(menu_path)

    def _add_plugin_menu_action(self, menu, plugin_name: str, menu_path: str):
        """
        Add a plugin as a menu action.

        Args:
            menu: The QMenu to add the action to
            plugin_name: The name of the plugin
            menu_path: The menu path for this plugin
        """
        if not menu:
            print(f"Warning: Could not find menu for path '{menu_path}'")
            return

        # Get plugin class
        plugin_class = self.plugin_manager.get_plugin(plugin_name)
        if not plugin_class:
            print(f"Warning: Plugin '{plugin_name}' not found")
            return

        # Create an instance to get display name
        try:
            plugin_instance = plugin_class()
            display_name = plugin_instance.get_name()

            # Convert snake_case to Title Case for better display
            # e.g., "dbscan_clustering" -> "DBSCAN Clustering"
            display_name = self._format_plugin_name(display_name)

        except Exception as e:
            print(f"Error instantiating plugin '{plugin_name}': {e}")
            display_name = plugin_name

        # Create menu action
        action = QtWidgets.QAction(display_name, self)

        # Connect to plugin execution
        action.triggered.connect(
            lambda checked, pname=plugin_name: self.open_dialog_box(pname)
        )

        # Add to menu
        menu.addAction(action)

        # Store reference
        action_id = f"{menu_path}/{plugin_name}"
        self.actions[action_id] = action

        print(f"  Added action: {menu_path} > {display_name}")

    def _format_plugin_name(self, name: str) -> str:
        """
        Format plugin name for display in menu.

        Strips numeric prefixes and converts "dbscan_clustering" to "DBSCAN Clustering"

        Args:
            name: Plugin name (possibly with numeric prefix)

        Returns:
            Formatted display name
        """
        # Strip numeric prefix (e.g., "000_save_project" -> "save_project")
        name = self.plugin_manager._strip_prefix(name)

        # Replace underscores with spaces
        formatted = name.replace('_', ' ')

        # Title case each word
        words = formatted.split()
        formatted_words = []

        for word in words:
            # Keep acronyms uppercase (like DBSCAN, SOR, MLS)
            if word.upper() in ['DBSCAN', 'HDBSCAN', 'SOR', 'MLS', 'PCA', 'ICP']:
                formatted_words.append(word.upper())
            else:
                formatted_words.append(word.capitalize())

        return ' '.join(formatted_words)

    def open_file_dialog(self):
        """Handler for 'Open' action to open and display a point cloud file."""
        self.file_manager.open_point_cloud_file(self)

    def open_dialog_box(self, plugin_name):
        """
        Open a dialog box for parameter input or execute action directly.

        Routes to appropriate handler based on plugin type (data processing vs action).
        """
        # Check if this is an action plugin
        if self.plugin_manager.is_action_plugin(plugin_name):
            # Handle action plugin directly
            self.execute_action_plugin(plugin_name)
        else:
            # Get params via dialog (direct call, no signal)
            params = self.dialog_boxes_manager.get_analysis_params(plugin_name)
            if params is not None:
                self._start_analysis(plugin_name, params)

    def execute_action_plugin(self, plugin_name: str):
        """
        Execute an action plugin.

        Args:
            plugin_name: The name of the action plugin to execute
        """
        plugin_class = self.plugin_manager.get_plugin(plugin_name)
        if not plugin_class:
            print(f"Error: Action plugin '{plugin_name}' not found")
            return

        try:
            plugin_instance = plugin_class()
            parameter_schema = plugin_instance.get_parameters()

            # If no parameters needed, execute immediately
            if not parameter_schema or len(parameter_schema) == 0:
                plugin_instance.execute(self, {})
            else:
                # Import DynamicDialog here to avoid circular imports
                from gui.dialog_boxes.dynamic_dialog import DynamicDialog

                # Apply last-used values as defaults
                parameter_schema = self.dialog_boxes_manager._apply_last_params(
                    plugin_name, parameter_schema
                )

                # Create and open a dynamic dialog for parameter input
                dialog = DynamicDialog(f"{plugin_name.title()} Parameters", parameter_schema)
                if dialog.exec_():
                    # If the user clicked OK, get the parameters and execute
                    params = dialog.get_parameters()
                    self.dialog_boxes_manager.store_params(plugin_name, params)
                    plugin_instance.execute(self, params)

        except Exception as e:
            print(f"Error executing action plugin '{plugin_name}': {str(e)}")

    def show_progress(self, message: str = "Processing...", percent: int = None):
        """
        Show progress in status bar.

        Args:
            message: Status message to display.
            percent: Optional progress percentage (0-100). None for indeterminate.
        """
        self._progress_label.setText(message)
        self._progress_label.show()
        if percent is not None:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(percent)
        else:
            self._progress_bar.setRange(0, 0)  # Indeterminate
        self._progress_bar.show()
        QtWidgets.QApplication.processEvents()

    def clear_progress(self):
        """Clear progress display from status bar."""
        self._progress_bar.hide()
        self._progress_label.hide()
        self._progress_label.setText("")
        QtWidgets.QApplication.processEvents()

    def disable_menus(self):
        """
        Disable the entire menu bar to prevent user interaction during processing.
        """
        self.menubar.setEnabled(False)

    def enable_menus(self):
        """
        Enable the menu bar after processing is complete.
        """
        self.menubar.setEnabled(True)

    def disable_tree(self):
        """
        Disable the tree widget to prevent user interaction during processing.
        """
        self.tree_widget.setEnabled(False)

    def enable_tree(self):
        """
        Enable the tree widget after processing is complete.
        """
        self.tree_widget.setEnabled(True)

    # === Signal Handlers (forwarding to Application Layer) ===

    def _on_point_cloud_loaded(self, file_path: str, point_cloud):
        """Handle point cloud loaded signal from FileManager."""
        try:
            uid = self.controller.add_point_cloud(point_cloud, point_cloud.name)
            memory_size = self.controller.get_cache_memory_usage(uid)
            self.tree_widget.add_branch(uid, "", point_cloud.name, is_root=True)
            self.tree_widget.update_cache_tooltip(uid, memory_size)
        except Exception as e:
            logger.error(f"Failed to load point cloud: {file_path}. Error: {e}")

    def _on_branch_visibility_changed(self, visibility_status: dict):
        """Handle visibility changes from tree widget."""
        self.show_progress("Updating visibility...")
        self.disable_menus()
        self.disable_tree()
        try:
            self._render_visible_data(visibility_status, zoom_extent=False)
        finally:
            self.clear_progress()
            self.enable_menus()
            self.enable_tree()

    def _on_branch_added(self, visibility_status: dict):
        """Handle new branch additions from tree widget."""
        logger.info("_on_branch_added() triggered")
        self.show_progress("Rendering new branch...")
        self.disable_menus()
        self.disable_tree()
        try:
            self._render_visible_data(visibility_status, zoom_extent=True)
        except Exception as e:
            logger.error(f"_on_branch_added() FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.clear_progress()
            self.enable_menus()
            self.enable_tree()

    def _on_branch_selection_changed(self, uids: list):
        """Handle branch selection changes from tree widget."""
        self.controller.set_selected_branches(uids)

    # === Rendering ===

    def _render_visible_data(self, visibility_status: dict, zoom_extent: bool = False):
        """Prepare and display vertex data for all visible nodes."""
        vertices = self.controller.rendering_coordinator.prepare_vertices(
            visibility_status=visibility_status,
            sample_rate=self._current_sample_rate,
            camera_distance=self.pcd_viewer_widget.camera_distance,
            zoom_factor=self.pcd_viewer_widget.zoom_factor,
            max_extent=self.pcd_viewer_widget.max_extent or 1.0
        )

        # Sync LOD state
        self._current_sample_rate = self.controller.rendering_coordinator.current_sample_rate
        self.pcd_viewer_widget._current_sample_rate = self._current_sample_rate

        # Update tree cache UI for auto-cached nodes
        for uid, vis in visibility_status.items():
            if vis:
                node = self.controller.get_node(uid)
                if node and node.is_cached:
                    item = self.tree_widget.branches_dict.get(uid)
                    if item:
                        self.tree_widget.blockSignals(True)
                        item.setCheckState(1, Qt.Checked)
                        self.tree_widget.blockSignals(False)
                    memory_usage = self.controller.get_cache_memory_usage(uid)
                    self.tree_widget.update_cache_tooltip(uid, memory_usage)

        # Send to viewer
        if vertices is not None:
            self.pcd_viewer_widget.set_branch_offsets(
                self.controller.rendering_coordinator.branch_offsets
            )
            self.pcd_viewer_widget.set_point_vertices(vertices)
            self.pcd_viewer_widget.update()
        else:
            self.pcd_viewer_widget.set_branch_offsets({})
            self.pcd_viewer_widget.set_points(None)
            self.pcd_viewer_widget.update()

        if zoom_extent and vertices is not None:
            self.pcd_viewer_widget.zoom_to_extent(preserve_rotation=True)

    def render_visible_data(self, zoom_extent: bool = False):
        """
        Public convenience method for rendering visible data.

        Wraps _render_visible_data with the current tree visibility status.
        Used by plugins as a replacement for data_manager._render_visible_data().

        Args:
            zoom_extent: Whether to zoom to fit all visible points.
        """
        self._render_visible_data(self.tree_widget.visibility_status, zoom_extent=zoom_extent)

    def render_visible_with_lod(self, sample_rate: float = None):
        """
        Re-render visible data with specified LOD sample rate.

        Called by PCDViewerWidget when zoom changes.
        """
        if sample_rate is not None:
            self._current_sample_rate = sample_rate

        visibility_status = self.tree_widget.visibility_status
        self._render_visible_data(visibility_status, zoom_extent=False)

    # === Analysis Execution ===

    def _start_analysis(self, analysis_type: str, params: dict):
        """Start analysis with UI protection."""
        global_variables.global_progress = (None, f"Running {analysis_type}...")
        self.show_progress(f"Running {analysis_type}...")
        self.disable_menus()
        self.disable_tree()

        self.controller.run_analysis(
            plugin_name=analysis_type,
            params=params,
            on_error=self._on_analysis_error
        )

        # Start polling for completion
        self._start_completion_polling()

    def _start_completion_polling(self):
        """Start QTimer to poll for analysis thread completion."""
        if not hasattr(self, '_completion_timer'):
            self._completion_timer = QtCore.QTimer()
            self._completion_timer.timeout.connect(self._check_analysis_completion)
        self._completion_timer.start(100)  # Poll every 100ms

    def _check_analysis_completion(self):
        """Check if analysis thread completed and process results."""
        # Read progress from singleton (written by BatchProcessor or AnalysisExecutor)
        percent, message = global_variables.global_progress
        if message:
            self.show_progress(message, percent)

        if not self.controller.analysis_executor.check_and_process_completion():
            return

        self._completion_timer.stop()

        # Clear progress state
        global_variables.global_progress = (None, "")

        # Re-enable UI
        self.clear_progress()
        self.enable_menus()
        self.enable_tree()

        # Check for error
        error = self.controller.analysis_executor.get_error()
        if error:
            logger.error(f"Analysis failed: {error}")
            self.controller.analysis_executor.cleanup()
            return

        # Process result
        result_data = self.controller.analysis_executor.get_result()
        if result_data:
            self._handle_analysis_result(result_data)
        self.controller.analysis_executor.cleanup()

    def _handle_analysis_result(self, result_data: dict):
        """Handle completed analysis result."""
        result = result_data['result']
        result_type = result_data['result_type']
        dependencies = result_data['dependencies']
        parent_node = result_data['data_node']
        analysis_type = result_data['analysis_type']
        params = result_data['params']

        # Add result to data tree via ApplicationController
        uid = self.controller.add_analysis_result(
            result, result_type, dependencies, parent_node, analysis_type, params
        )

        # Update tree widget
        parent_uid_str = str(parent_node.uid)
        result_node = self.controller.get_node(uid)
        self.tree_widget.add_branch(uid, parent_uid_str, result_node.params if result_node else f"{analysis_type},{params}")

        # Show memory usage for new node
        if result_node and hasattr(result_node, 'memory_size') and result_node.memory_size:
            self.tree_widget.update_cache_tooltip(uid, result_node.memory_size)

        # Update parent cache UI if auto-cached during analysis
        if parent_node.is_cached:
            item = self.tree_widget.branches_dict.get(parent_uid_str)
            if item:
                self.tree_widget.blockSignals(True)
                item.setCheckState(1, Qt.Checked)
                self.tree_widget.blockSignals(False)
            memory_usage = self.controller.get_cache_memory_usage(parent_uid_str)
            self.tree_widget.update_cache_tooltip(parent_uid_str, memory_usage)

        # Hide parent and show only the new child result
        if parent_uid_str in self.tree_widget.visibility_status:
            self.tree_widget.visibility_status[parent_uid_str] = False
        self.tree_widget.visibility_status[uid] = True

        # Update checkboxes
        parent_item = self.tree_widget.branches_dict.get(parent_uid_str)
        if parent_item:
            parent_item.setCheckState(0, Qt.Unchecked)
        child_item = self.tree_widget.branches_dict.get(uid)
        if child_item:
            child_item.setCheckState(0, Qt.Checked)

        # Trigger visibility update to render the child
        self.tree_widget.branch_visibility_changed.emit(self.tree_widget.visibility_status)

    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error callback."""
        logger.error(f"Analysis error: {error_msg}")
        global_variables.global_progress = (None, "")
        self.clear_progress()
        self.enable_menus()
        self.enable_tree()

    def closeEvent(self, event):
        """
        Handle application close with proper memory cleanup.

        Ensures all memory resources are freed consistently to avoid
        memory leaks between sessions.
        """
        import gc
        import logging
        logger = logging.getLogger(__name__)
        logger.info("MainWindow closing - performing cleanup...")

        # Clear global references to allow garbage collection
        from config.config import global_variables
        global_variables.global_application_controller = None
        global_variables.global_data_nodes = None
        global_variables.global_file_manager = None
        global_variables.global_pcd_viewer_widget = None
        global_variables.global_tree_structure_widget = None
        global_variables.global_main_window = None

        # Clear EigenvalueUtils singleton cache (KD-tree, indices)
        try:
            from core.entities.point_cloud import _eigenvalue_utils_instance
            if _eigenvalue_utils_instance is not None:
                _eigenvalue_utils_instance.clear_cache()
                logger.info("EigenvalueUtils cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing EigenvalueUtils cache: {e}")

        # Clear CuPy memory pools if available
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            logger.info("CuPy memory pools cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error clearing CuPy memory: {e}")

        # Clear PyTorch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("PyTorch CUDA cache cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error clearing PyTorch cache: {e}")

        # Force garbage collection
        gc.collect()
        logger.info("Garbage collection completed")

        # Call parent closeEvent
        super().closeEvent(event)
