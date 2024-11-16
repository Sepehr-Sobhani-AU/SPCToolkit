
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import open3d as o3d

from core.point_cloud import PointCloud


class FileManager(QObject):
    """Class to manage file operations, such as opening and saving point cloud files."""

    # Signal emitted when a point cloud file is loaded
    point_cloud_loaded = pyqtSignal(str, PointCloud)

    def open_point_cloud_file(self, parent=None):
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
            # Load the point cloud data using Open3D
            point_cloud_data = o3d.io.read_point_cloud(file_path)
            points = np.asarray(point_cloud_data.points, dtype=np.float32)
            colors = np.asarray(point_cloud_data.colors, dtype=np.float32) if point_cloud_data.colors else None
            normals = np.asarray(point_cloud_data.normals, dtype=np.float32) if point_cloud_data.normals else None

            point_cloud = PointCloud(points=points, colors=colors, normals=normals)
            point_cloud.name = file_path.split("/")[-1]

            # Emit signal with file path
            self.point_cloud_loaded.emit(file_path, point_cloud)
        else:
            return None, None, None
