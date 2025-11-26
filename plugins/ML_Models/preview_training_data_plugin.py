"""
Preview Training Data Plugin

Opens an interactive window for browsing and previewing machine learning training data.
Supports multiple model formats (PointNet, PointNet++, etc.) and provides
visualization with various feature-based coloring modes.
"""

from typing import Dict, Any

from plugins.interfaces import ActionPlugin
from gui.dialogs.training_data_preview_window import TrainingDataPreviewWindow


class PreviewTrainingDataPlugin(ActionPlugin):
    """
    Action plugin for previewing training data.

    This plugin opens a dedicated window that allows users to:
    - Browse training data directories
    - View metadata
    - Select classes and samples
    - Visualize samples in 3D with feature-based coloring
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "preview_training_data"

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
        # Create and show preview window
        preview_window = TrainingDataPreviewWindow(parent=main_window)
        preview_window.exec_()