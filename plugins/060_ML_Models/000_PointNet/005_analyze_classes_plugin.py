# plugins/060_ML_Models/000_PointNet/005_analyze_classes_plugin.py
"""
Analyze & Prepare Classes Plugin

Launches the ClassAnalysisWindow for managing imbalanced class data.
This plugin helps users:
- Visualize class distribution
- Filter out classes with too few samples
- Merge similar classes
- Configure balancing strategy
- Export configuration for training data generation
"""

from typing import Dict, Any
from plugins.interfaces import ActionPlugin


class AnalyzeClassesPlugin(ActionPlugin):
    """
    Action plugin for analyzing and preparing imbalanced class data.

    This plugin opens a dedicated window where users can:
    1. Scan a directory containing class folders
    2. View class distribution and statistics
    3. Filter classes by minimum sample count
    4. Merge semantically similar classes
    5. Configure balancing strategy
    6. Preview final distribution
    7. Export configuration for the Generate Training Data plugin
    """

    def get_name(self) -> str:
        return "analyze_classes"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - the window handles all configuration.
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Launch the Class Analysis Window.

        Args:
            main_window: The main application window
            params: Empty dict (no parameters)
        """
        # Import here to avoid circular imports and speed up plugin loading
        from gui.dialogs.class_analysis_window import ClassAnalysisWindow

        # Create and show the analysis window
        dialog = ClassAnalysisWindow(parent=main_window)

        # Show as modal dialog
        result = dialog.exec_()

        if result == ClassAnalysisWindow.Accepted:
            # User clicked "Proceed to Generate"
            config = dialog.get_config()

            # Store config for the Generate Training Data plugin
            # This uses a class variable that the generation plugin can access
            ClassAnalysisWindow.last_config = config

            print("\n" + "=" * 60)
            print("Class Analysis Complete")
            print("=" * 60)
            print(f"Source: {config['source_directory']}")
            print(f"Classes after filtering: {config['preview']['final_class_count']}")
            print(f"Samples per class: {config['preview']['samples_per_class']}")
            print(f"Total samples: {config['preview']['final_sample_count']}")
            print("=" * 60)
            print("\nConfiguration ready. Use 'Generate Training Data' to create the dataset.")
            print("You can also save the configuration for later use.")
