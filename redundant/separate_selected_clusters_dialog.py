# TODO: docstring

from redundant.base_dialog import BaseDialog
from config.config import global_variables


class SeparateSelectedClustersDialog(BaseDialog):
    def __init__(self, parent=None):
        super().__init__(title="Separate Selected Points", parent=parent)

    def validate_params(self) -> bool:
        # Validate the subsampling rate
        try:
            global_pcd_viewer_widget = global_variables.global_pcd_viewer_widget

            points_indices = global_pcd_viewer_widget.picked_points_indices

            condition = f"np.isin(point_cloud.cluster_labels, point_cloud.cluster_labels[{points_indices}])"
            self.params["condition"] = condition

            global_pcd_viewer_widget.picked_points_indices = []

            return True
        except ValueError:
            return False
