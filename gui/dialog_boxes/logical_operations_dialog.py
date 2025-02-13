# TODO: convert all uids as strings to uuid.UUID objects
from PyQt5.QtWidgets import QComboBox, QVBoxLayout, QLabel
from gui.dialog_boxes.base_dialog import BaseDialog
from tasks import subtract_point_clouds
from config.config import global_variables
import uuid


class LogicalOperationsDialog(BaseDialog):
    """
    A dialog box that contains two combo boxes:
    - One for selecting a point cloud.
    - One for selecting a logical operation (Subtract, Intersect, Union).
    """

    def __init__(self, title: str, parent=None):
        """
        Initializes the dialog with combo boxes for selecting a point cloud and an operation.

        Args:
            title (str): Title of the dialog.
            point_cloud_options (list): List of point cloud names to populate the combo box.
            parent: Parent widget.
        """
        self.operation_combo_box = None
        self.operation_label = None
        self.point_cloud_combo_box = None
        self.point_cloud_label = None
        self.point_cloud_options = global_variables.global_tree_structure_widget.branches_dict.keys()
        self.operation_registry = {"Subtract": subtract_point_clouds,
 #                                  "Intersect": intersect,
 #                                  "Union": union
                                   }
        super().__init__(title, parent)

    def setup_ui(self):
        """
        Sets up the UI, including combo boxes for selecting a point cloud and an operation.
        """
        super().setup_ui()

        # Create and configure the point cloud combo box
        self.point_cloud_label = QLabel("Point Cloud:", self)
        self.point_cloud_combo_box = QComboBox(self)
        self.point_cloud_combo_box.addItems(self.point_cloud_options)

        # Create and configure the operation combo box
        self.operation_label = QLabel("Operation:", self)
        self.operation_combo_box = QComboBox(self)

        # Clear the combo box before adding items
        self.point_cloud_combo_box.clear()

        # Add point cloud names to the combo box from the global tree structure widget
        branch_uids = global_variables.global_tree_structure_widget.branches_dict.keys()
        for uid in branch_uids:
            data_node = global_variables.global_data_nodes.get_node(uuid.UUID(uid))
            self.point_cloud_combo_box.addItem(data_node.name)

        # Insert widgets into the layout before the buttons
        self.layout.insertWidget(0, self.operation_label)
        self.layout.insertWidget(1, self.operation_combo_box)
        self.layout.insertWidget(2, self.point_cloud_label)
        self.layout.insertWidget(3, self.point_cloud_combo_box)

    def get_parameters(self) -> dict:
        """
        Returns the selected options from the combo boxes.

        Returns:
            dict: Dictionary containing the selected point cloud and operation.
        """
        self.params["selected_point_cloud"] = self.point_cloud_combo_box.currentText()
        self.params["selected_operation"] = self.operation_combo_box.currentText()
        return self.params
