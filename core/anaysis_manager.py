"""
    This module implements a Dynamic Task Execution Framework that leverages plugins
    for analysis tasks.
"""
import uuid
from typing import Dict, Any

from PyQt5.QtCore import pyqtSignal, QObject

from core.data_node import DataNode
from plugins.plugin_manager import PluginManager


class AnalysisManager(QObject):
    """
    Manages and executes analyses on DataNodes using a plugin-based architecture.

    Instead of a hardcoded task registry, this version uses plugins discovered
    by the PluginManager to execute different types of analysis.

    Attributes:
        plugin_manager (PluginManager): Manager that discovers and provides access to plugins
        analysis_completed (pyqtSignal): Signal emitted when an analysis is completed
    """
    analysis_completed = pyqtSignal(object, str, list, object, str, dict)

    def __init__(self, plugin_manager: PluginManager):
        """
        Initializes the AnalysisManager with a plugin manager.

        Args:
            plugin_manager (PluginManager): The plugin manager that provides access to analysis plugins
        """
        super().__init__()
        self.plugin_manager = plugin_manager

        # Log available plugins for debugging
        plugins = self.plugin_manager.get_analysis_plugins()
        print(f"AnalysisManager initialized with {len(plugins)} available analysis plugins")
        for plugin_name in plugins.keys():
            print(f"  - {plugin_name}")

    def apply_analysis(self, data: DataNode, analysis_type: str, params: Dict[str, Any]) -> None:
        """
        Applies an analysis task to the given DataNode using the appropriate plugin.

        Args:
            data (DataNode): The DataNode to analyze
            analysis_type (str): The type of analysis to perform (must match a plugin name)
            params (Dict[str, Any]): Parameters for the analysis task

        Raises:
            ValueError: If no plugin is found for the specified analysis_type
        """
        # Get available analysis plugins
        analysis_plugins = self.plugin_manager.get_analysis_plugins()

        if analysis_type not in analysis_plugins:
            raise ValueError(f"Analysis type '{analysis_type}' not found in available plugins.")

        # Create an instance of the plugin class
        plugin_class = analysis_plugins[analysis_type]
        plugin_instance = plugin_class()

        print(f"Executing analysis '{analysis_type}' with plugin {plugin_instance.__class__.__name__}")

        # Execute the analysis using the plugin
        result, result_type, dependencies = plugin_instance.execute(data, params)

        # Emit signal to add the result to the DataNode
        self.analysis_completed.emit(result, result_type, dependencies, data, analysis_type, params)

    def __repr__(self) -> str:
        """
        Provides a string representation of the AnalysisManager instance.

        Returns:
            str: A string describing the AnalysisManager and available plugins
        """
        plugins = self.plugin_manager.get_analysis_plugins()
        return f"AnalysisManager({len(plugins)} available plugins)"