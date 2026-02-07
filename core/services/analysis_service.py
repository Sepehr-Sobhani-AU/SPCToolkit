"""
Analysis service for executing plugins synchronously.

Pure Core service — no GUI, no threading, no framework imports.
Threading is handled by the Application layer (AnalysisExecutor).
"""
from typing import Dict, Any, Tuple, List

from core.entities.data_node import DataNode


class AnalysisService:
    """Executes analysis plugins synchronously."""

    def execute(self, plugin, data_node: DataNode, params: Dict[str, Any],
                progress_callback=None) -> Tuple[Any, str, List[str]]:
        """
        Execute a plugin on a data node with given parameters.

        Args:
            plugin: Plugin class (must have execute() method).
            data_node: Target DataNode to analyze.
            params: Analysis parameters from dialog.
            progress_callback: Optional fn(percent, message) for progress reporting.

        Returns:
            Tuple of (result_data, result_type, depends_on).
        """
        plugin_instance = plugin()
        if progress_callback and hasattr(plugin_instance, 'progress_callback'):
            plugin_instance.progress_callback = progress_callback
        result, result_type, dependencies = plugin_instance.execute(data_node, params)
        return result, result_type, dependencies
