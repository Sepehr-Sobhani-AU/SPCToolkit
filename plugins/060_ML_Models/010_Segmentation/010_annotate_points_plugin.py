"""
Annotate Points Plugin

Opens annotation mode for manual per-point semantic labeling.
User paints labels using polygon selection and class palette.
Annotations can be exported as segmentation training data.
"""

from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class AnnotatePointsPlugin(ActionPlugin):
    """
    Action plugin to open manual annotation mode.
    """

    def get_name(self) -> str:
        return "annotate_points"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Open the annotation window for the selected point cloud."""
        controller = global_variables.global_application_controller

        # Validate branch selection
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                              "Please select a point cloud branch to annotate.")
            return

        selected_uid = selected_branches[0]

        try:
            # Reconstruct to get PointCloud
            point_cloud = controller.reconstruct(selected_uid)

            if point_cloud is None or len(point_cloud.points) == 0:
                QMessageBox.warning(main_window, "Empty Point Cloud",
                                  "The selected branch has no points.")
                return

            # Import and create annotation window
            from plugins.dialogs.annotation_window import AnnotationWindow

            annotation_window = AnnotationWindow(parent=main_window)
            annotation_window.initialize_annotations(point_cloud)
            annotation_window.show()

            print(f"Annotation mode opened for {len(point_cloud.points):,} points")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(main_window, "Error",
                               f"Failed to open annotation mode:\n\n{str(e)}")
