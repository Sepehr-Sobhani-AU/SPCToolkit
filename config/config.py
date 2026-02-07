# TODO: Docstrings
# TODO: Define types


# Global variables
class GlobalVariables:
    """
    Singleton class holding global references to core application components.

    These instances are initialized at application startup and provide
    centralized access to managers, widgets, and configuration.
    """

    def __init__(self):
        # Core managers
        self.global_file_manager = None
        self.global_data_nodes = None  # DataNodes collection
        self.global_data_manager = None  # DataManager instance (backward compat)
        self.global_analysis_thread_manager = None

        # Application layer
        self.global_application_controller = None

        # GUI widgets
        self.global_pcd_viewer_widget = None
        self.global_tree_structure_widget = None
        self.global_main_window = None

        # Hardware detection and backend system
        self.global_hardware_info = None  # HardwareInfo from HardwareDetector
        self.global_backend_registry = None  # BackendRegistry for algorithm backends

        # Progress state (written by background thread, read by UI polling timer)
        # (None, "msg") = indeterminate bar, (50, "msg") = 50% determinate bar
        self.global_progress = (None, "")

        # Configuration
        self.training_data_folder = "training_data"  # Default training data directory


# Singleton instance
global_variables = GlobalVariables()
