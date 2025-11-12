from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import open3d as o3d
import pickle
import os

from core.point_cloud import PointCloud


class FileManager(QObject):
    """Class to manage file operations, such as opening and saving point cloud files."""

    # Signal emitted when a point cloud file is loaded
    point_cloud_loaded = pyqtSignal(str, PointCloud)

    # Signal emitted when a project is loaded
    project_loaded = pyqtSignal(object)

    # Signal emitted after a project is saved
    project_saved = pyqtSignal(str)

    def __init__(self):
        super().__init__()  # Call the parent class constructor
        self.min_bound = None
        self.current_project_path = None

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
            points = np.asarray(point_cloud_data.points, dtype=np.float64)
            colors = np.asarray(point_cloud_data.colors, dtype=np.float32) if point_cloud_data.colors else None
            normals = np.asarray(point_cloud_data.normals, dtype=np.float32) if point_cloud_data.normals else None

            # Find the min bound of the point cloud
            if self.min_bound is None:
                self.min_bound = np.min(points, axis=0)

            # Translate the point cloud to the origin to save memory by using float32
            points = (points - self.min_bound).astype(np.float32)

            point_cloud = PointCloud(points=points, colors=colors, normals=normals)

            point_cloud.translation = -self.min_bound
            point_cloud.name = file_path.split("/")[-1]

            # Emit signal with file path
            self.point_cloud_loaded.emit(file_path, point_cloud)
        else:
            return None, None, None

    def save_project(self, data_nodes, parent=None, new_file=False):
        """
        Saves the current project state to a file.

        Args:
            data_nodes: The DataNodes instance to save
            parent: The parent widget for file dialogs
            new_file (bool): If True, always prompt for a new filename

        Returns:
            tuple: (success, message) - success is a boolean, message provides details
        """
        filename = self.current_project_path

        # If no current path or new_file is True, prompt for a filename
        if filename is None or new_file:
            options = QtWidgets.QFileDialog.Options()
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                parent,
                "Save Project",
                "",
                "PCD Toolkit Project Files (*.pcdtk);;All Files (*)",
                options=options
            )

            if not filename:
                return False, "Save cancelled"

            # Ensure the file has the correct extension
            if not filename.endswith(".pcdtk"):
                filename += ".pcdtk"

        # Get tree visibility state from global variables
        from config.config import global_variables
        tree_widget = global_variables.global_tree_structure_widget

        # Extract tree visibility state
        tree_visibility = None
        if tree_widget is not None:
            tree_visibility = tree_widget.visibility_status.copy()

        # Add version information to the saved data
        project_data = {
            'version': '1.0.0',
            'data_nodes': data_nodes,
            'tree_visibility': tree_visibility
        }

        try:
            # Save the data_nodes instance using pickle
            with open(filename, 'wb') as file:
                pickle.dump(project_data, file)

            # Store the path for future saves
            self.current_project_path = filename

            # Emit the signal with the saved path
            self.project_saved.emit(filename)

            # Extract just the filename for the success message
            base_filename = os.path.basename(filename)
            return True, f"Project saved successfully to {base_filename}"

        except Exception as e:
            # Return an error message if saving fails
            return False, f"Error saving project: {str(e)}"

    def load_project(self, parent=None):
        """
        Loads a project from a file.

        Args:
            parent: The parent widget for file dialogs

        Returns:
            tuple: (data_nodes, message) - data_nodes is the loaded DataNodes instance
                   or None if loading failed, message provides details
        """
        options = QtWidgets.QFileDialog.Options()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            "Open Project",
            "",
            "PCD Toolkit Project Files (*.pcdtk);;All Files (*)",
            options=options
        )

        if not filename:
            return None, "Open cancelled"

        try:
            # Load the project data using pickle
            with open(filename, 'rb') as file:
                project_data = pickle.load(file)

            # Check version information if available
            if isinstance(project_data, dict) and 'version' in project_data:
                version = project_data['version']
                loaded_data_nodes = project_data['data_nodes']
                # Extract tree visibility state if available
                self._last_loaded_tree_visibility = project_data.get('tree_visibility', None)
            else:
                # Handle older project files without version info
                loaded_data_nodes = project_data
                self._last_loaded_tree_visibility = None

            # Store the path for future saves
            self.current_project_path = filename

            # Emit the signal with the loaded DataNodes
            self.project_loaded.emit(loaded_data_nodes)

            # Extract just the filename for the success message
            base_filename = os.path.basename(filename)
            return loaded_data_nodes, f"Project loaded successfully from {base_filename}"

        except Exception as e:
            # Return an error message if loading fails
            return None, f"Error loading project: {str(e)}"