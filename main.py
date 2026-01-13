# main.py
import sys
import logging
import os

# Configure logging FIRST - write to both console and file
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spctoolkit_debug.log')


class FlushingFileHandler(logging.FileHandler):
    """File handler that flushes after every log entry (for crash debugging)."""
    def emit(self, record):
        super().emit(record)
        self.flush()


# Create file handler that logs everything including DEBUG
# Use FlushingFileHandler to ensure logs are written before crash
file_handler = FlushingFileHandler(LOG_FILE, mode='w')  # 'w' to overwrite each run
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Create console handler with INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {LOG_FILE}")

# IMPORTANT: Detect hardware BEFORE importing PyQt5/OpenGL
# This avoids conflicts with PyCharm debugger's Qt support
from services.hardware_detector import HardwareDetector
from services.backend_registry import BackendRegistry
from config.config import global_variables

# Pre-detect hardware before Qt initialization
logger.info("Pre-initializing hardware detection...")
_early_hardware_info = HardwareDetector.detect()

# Now import Qt (after CUDA libraries have initialized)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from plugins.plugin_manager import PluginManager
from gui.main_window import MainWindow
from gui.widgets.splash_screen import SplashScreen


def initialize_hardware_and_backends(splash=None):
    """Detect hardware and initialize backend registry."""
    if splash:
        splash.set_status("Detecting hardware...")
        splash.set_progress(10)
        QApplication.processEvents()

    # Get hardware info (already detected at module load to avoid Qt/CUDA conflicts)
    hardware_info = HardwareDetector.detect()  # Returns cached result
    global_variables.global_hardware_info = hardware_info

    if splash:
        splash.set_progress(30)
        QApplication.processEvents()

    # Create and register backend registry
    backend_registry = BackendRegistry(hardware_info)
    global_variables.global_backend_registry = backend_registry

    # Update splash with hardware info
    if splash:
        splash.set_hardware_info(
            os_name=hardware_info.os,
            gpu_name=hardware_info.gpu_name if hardware_info.gpu_available else None,
            mode=backend_registry.get_scenario()
        )
        splash.set_progress(50)
        QApplication.processEvents()

    # Log hardware info
    if hardware_info.gpu_available:
        gpu_info = hardware_info.gpu_name
    else:
        gpu_info = "None (CPU mode)"

    logger.info(f"Hardware: {gpu_info} | Mode: {backend_registry.get_scenario()}")


def main():
    """Main entry point for the application."""
    # Create the application
    app = QApplication(sys.argv)

    # Show splash screen
    splash = SplashScreen()
    splash.show()
    QApplication.processEvents()

    # Initialize hardware detection and backends
    initialize_hardware_and_backends(splash)

    # Load plugins
    splash.set_status("Loading plugins...")
    splash.set_progress(60)
    QApplication.processEvents()

    plugin_manager = PluginManager()

    splash.set_progress(80)
    QApplication.processEvents()

    # Create main window
    splash.set_status("Initializing main window...")
    splash.set_progress(90)
    QApplication.processEvents()

    main_window = MainWindow(plugin_manager)

    # Finish splash and show main window
    splash.finish(main_window)

    # Run the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()