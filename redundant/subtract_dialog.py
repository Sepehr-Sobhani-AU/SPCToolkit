# TODO: convert all uids as strings to uuid.UUID objects
from PyQt5.QtWidgets import QComboBox, QLabel
from redundant.base_dialog import BaseDialog
from config.config import global_variables
import uuid


class subtractDialog(BaseDialog):
    """
    A dialog box that contains a combo boxes:
    - One for selecting a point cloud.
    """

    def __init__(self, title: str, parent=None):
        """
        Initializes the dialog with combo boxes for selecting a point cloud and an operation.

        Args:
            title (str): Title of the dialog.
            point_cloud_options (list): List of point cloud names to populate the combo box.
            parent: Parent widget.
        """
        self.point_cloud_combo_box = None
        self.point_cloud_label = None
        self.point_cloud_options = global_variables.global_tree_structure_widget.branches_dict.keys()

        super().__init__(title, parent)

    def setup_ui(self):
        """
        Sets up the UI, including combo boxes for selecting a point cloud and an operation.
        """
        super().setup_ui()

        # Create and configure the point cloud combo box
        self.point_cloud_label = QLabel("Point Cloud:", self)
        self.point_cloud_combo_box = QComboBox(self)
        self.point_cloud_combo_box.addItems(self.point_cloud_options,)

        # Clear the combo box before adding items
        self.point_cloud_combo_box.clear()
        # Add point cloud names to the combo box from the global tree structure widget
        branch_uids = global_variables.global_tree_structure_widget.branches_dict.keys()
        for uid in branch_uids:
            data_node = global_variables.global_data_nodes.get_node(uuid.UUID(uid))
            self.point_cloud_combo_box.addItem(data_node.params, uuid.UUID(uid))

        # Insert widgets into the layout before the buttons
        self.layout.insertWidget(2, self.point_cloud_label)
        self.layout.insertWidget(3, self.point_cloud_combo_box)

    def validate_params(self) -> bool:
        # Validate the subsampling rate
        try:
            # Step 1: Find mask using np.in1d (highly efficient for large data)
            condition = f"""
                            from config.config import global_variables
                            import uuid
                            branch_uid = global_variables.global_data_nodes.get_node(uuid.UUID("{self.point_cloud_combo_box.currentData()}")).uid
                            reconstructed_branch = global_variables.global_data_manager.reconstruct_branch(branch_uid)
                            ~np.in1d(point_cloud, reconstructed_branch)
                        """

            self.params["condition"] = condition
            return True
        except ValueError:
            return False




