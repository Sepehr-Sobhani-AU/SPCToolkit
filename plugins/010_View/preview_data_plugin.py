"""
Preview Data Plugin

Opens an interactive window for browsing and previewing saved data (.npy format).
Works with any data organized in class subdirectories (training data, clusters, etc.).
Provides visualization with automatic feature-based coloring.
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from gui.dialogs.training_data_preview_window import DataPreviewWindow


class PreviewDataPlugin(ActionPlugin):
    """
    Action plugin for previewing saved data.

    This plugin opens a dedicated window that allows users to:
    - Browse data directories (any .npy format organized by class folders)
    - View metadata (if available)
    - Select classes and samples
    - Visualize samples in 3D with automatic feature-based coloring
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "preview_data"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - opens window directly.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the preview action by opening the preview window.

        Args:
            main_window: The main application window
            params: Not used for this plugin (empty dict)
        """
        # Create and show preview window with generic title
        preview_window = DataPreviewWindow(
            parent=main_window,
            window_title="Data Preview"
        )
        preview_window.exec_()