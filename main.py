# main.py
import sys
from PyQt5.QtWidgets import QApplication

from plugins.plugin_manager import PluginManager
from gui.main_window import MainWindow


def main():
    """Main entry point for the application."""
    # Create the application
    app = QApplication(sys.argv)

    # Initialize the plugin manager
    plugin_manager = PluginManager()

    # Create and show the main window with the plugin manager
    main_window = MainWindow(plugin_manager)
    main_window.show()

    # Run the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()