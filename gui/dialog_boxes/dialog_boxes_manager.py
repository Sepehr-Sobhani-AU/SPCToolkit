# gui/dialog_boxes/dialog_boxes_manager.py
import copy

from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, Any

from plugins.plugin_manager import PluginManager
from gui.dialog_boxes.dynamic_dialog import DynamicDialog


class DialogBoxesManager(QObject):
    """
    Manages dialog boxes for parameter input.

    This manager creates dynamic dialog boxes based on the parameter
    definitions provided by plugins. Remembers last-used parameter values
    per plugin so subsequent runs start from the previous settings.
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
        self._last_params = {}  # {plugin_name: {param_name: value}}

    def _apply_last_params(self, plugin_name: str, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Override schema defaults with last-used values for this plugin.

        Returns a shallow copy of the schema with updated defaults.
        """
        if plugin_name not in self._last_params:
            return schema

        last = self._last_params[plugin_name]
        patched = {}
        for param_name, param_info in schema.items():
            if param_name in last:
                patched[param_name] = {**param_info, "default": last[param_name]}
            else:
                patched[param_name] = param_info
        return patched

    def store_params(self, plugin_name: str, params: Dict[str, Any]) -> None:
        """Store last-used parameter values for a plugin."""
        self._last_params[plugin_name] = dict(params)

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

        # Apply last-used values as defaults
        parameter_schema = self._apply_last_params(analysis_type, parameter_schema)

        # Open dialog for parameter input
        dialog = DynamicDialog(f"{analysis_type.title()} Parameters", parameter_schema)
        if dialog.exec_():
            params = dialog.get_parameters()
            self.store_params(analysis_type, params)
            return params

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