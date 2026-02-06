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

    def get_analysis_params(self, analysis_type: str):
        """
        Open a dialog and return analysis parameters, or None if cancelled.

        If the plugin has no parameters, returns an empty dict.

        Args:
            analysis_type: The type of analysis to create a dialog for.

        Returns:
            Dict of parameters, or None if cancelled or plugin not found.
        """
        analysis_plugins = self.plugin_manager.get_analysis_plugins()

        if analysis_type not in analysis_plugins:
            print(f"Warning: No plugin found for analysis type '{analysis_type}'")
            return None

        plugin_class = analysis_plugins[analysis_type]
        plugin_instance = plugin_class()
        parameter_schema = plugin_instance.get_parameters()

        # No parameters needed — return empty dict
        if not parameter_schema or len(parameter_schema) == 0:
            return {}

        # Open dialog for parameter input
        dialog = DynamicDialog(f"{analysis_type.title()} Parameters", parameter_schema)
        if dialog.exec_():
            return dialog.get_parameters()

        return None  # User cancelled

    def open_dialog_box(self, analysis_type: str):
        """
        Open a dialog box for the specified analysis type.

        Backward compat: emits analysis_params signal.
        New code should use get_analysis_params() instead.
        """
        params = self.get_analysis_params(analysis_type)
        if params is not None:
            self.analysis_params.emit(analysis_type, params)