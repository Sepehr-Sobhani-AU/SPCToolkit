
from PyQt5 import QtWidgets
import numpy as np
import open3d as o3d


class FileManager:
    @staticmethod
    def open_point_cloud_file(parent=None):
        """Opens a file dialog, loads a point cloud file, and returns the data."""

        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            "Open PLY File",
            "",
            "PLY Files (*.ply);;All Files (*)",
            options=options
        )

        if file_path:
            pcd_data = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd_data.points, dtype=np.float32)
            colors = np.asarray(pcd_data.colors, dtype=np.float32) if pcd_data.colors else None
            return points, colors
        else:
            return None, None
