from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QDialog

from gui.dialog_boxes.dbscan_dialog import DbscanDialog
from gui.dialog_boxes.logical_operations_dialog import LogicalOperationsDialog
from gui.dialog_boxes.separate_selected_clusters_dialog import SeparateSelectedClustersDialog
from gui.dialog_boxes.separate_selected_points_dialog import SeparateSelectedPointsDialog
from gui.dialog_boxes.subsampling_dialog import SubsamplingDialog
from gui.dialog_boxes.filtering_dialog import FilteringDialog


class DialogBoxesManager(QObject):  # Inherit from QObject
    # Define the signal as a class attribute
    analysis_params = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)  # Initialize the QObject parent
        self.parent = parent
        self.params = {}
        self.dialog_classes = {
            "subsampling": SubsamplingDialog,
            "dbscan": DbscanDialog,
            "filtering": FilteringDialog,
            "separate_selected_points": SeparateSelectedPointsDialog,
            "separate_selected_clusters": SeparateSelectedClustersDialog,
            "logical_operations": LogicalOperationsDialog,
        }

    def open_dialog_box(self, analysis_type):
        """
        Open a dialog box for the specified analysis type.

        Args:
            analysis_type (str): The type of analysis to configure.

        Raises:
            ValueError: If the analysis type is unknown.
        """
        if analysis_type not in self.dialog_classes:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Create and execute the dialog
        dialog_class = self.dialog_classes[analysis_type]
        dialog = dialog_class(self.parent)
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_parameters()
            self.analysis_params.emit(analysis_type, params)
