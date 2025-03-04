# gui/dialog_boxes/dialog_boxes_manager.py
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, Any

from plugins.plugin_manager import PluginManager
from gui.dialog_boxes.dynamic_dialog import DynamicDialog


class DialogBoxesManager(QObject):
    """
    Manages dialog boxes for parameter input.

    This manager creates dynamic dialog boxes based on the parameter
    definitions provided by plugins.
    """

    analysis_params = pyqtSignal(str, dict)

    def __init__(self, plugin_manager: PluginManager, parent=None):
        """
        Initialize the dialog boxes manager.

        Args:
            plugin_manager (PluginManager): The plugin manager for accessing plugins
            parent: Parent object, if any
        """
        super().__init__(parent)
        self.plugin_manager = plugin_manager

    def open_dialog_box(self, analysis_type: str):
        """
        Open a dialog box for the specified analysis type.

        Args:
            analysis_type (str): The type of analysis to create a dialog for
        """
        # Get available analysis plugins
        analysis_plugins = self.plugin_manager.get_analysis_plugins()

        if analysis_type in analysis_plugins:
            # Create an instance of the plugin to get its parameters
            plugin_class = analysis_plugins[analysis_type]
            plugin_instance = plugin_class()
            parameter_schema = plugin_instance.get_parameters()

            # Create and open a dynamic dialog
            dialog = DynamicDialog(f"{analysis_type.title()} Parameters", parameter_schema)
            if dialog.exec_():
                # If the user clicked OK, get the parameters and emit the signal
                params = dialog.get_parameters()
                self.analysis_params.emit(analysis_type, params)
        else:
            print(f"Warning: No plugin found for analysis type '{analysis_type}'")