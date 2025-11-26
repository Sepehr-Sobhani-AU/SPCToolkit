"""
Preview Clusters Plugin

Opens an interactive window for browsing and previewing saved cluster data.
Provides visualization with various feature-based coloring modes.
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from gui.dialogs.training_data_preview_window import DataPreviewWindow


class PreviewClustersPlugin(ActionPlugin):
    """
    Action plugin for previewing saved cluster data.

    This plugin opens a dedicated window that allows users to:
    - Browse cluster data directories
    - View metadata (if available)
    - Select classes and samples
    - Visualize samples in 3D with feature-based coloring
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "preview_clusters"

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
        # Create and show preview window with specific title
        preview_window = DataPreviewWindow(
            parent=main_window,
            window_title="Clusters Preview"
        )
        preview_window.exec_()