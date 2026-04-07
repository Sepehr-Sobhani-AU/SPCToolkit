from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import open3d as o3d
import pickle
import os
import shutil
import glob

from core.entities.point_cloud import PointCloud


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
        self.flythroughs = []  # List of {name, fps, waypoints: [{name, duration, ...}]}

    def _save_version_copy(self, filepath):
        """Create an auto-versioned copy of the saved project file.

        For a file saved as 'myproject.pcdtk', creates copies named
        'myproject_001.pcdtk', 'myproject_002.pcdtk', etc.
        """
        directory = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        stem, ext = os.path.splitext(basename)

        # Find existing versioned files: stem_NNN.pcdtk
        pattern = os.path.join(directory, f"{stem}_[0-9][0-9][0-9]{ext}")
        existing = glob.glob(pattern)

        # Determine next version number
        max_version = 0
        for path in existing:
            name = os.path.splitext(os.path.basename(path))[0]
            suffix = name[len(stem) + 1:]  # skip "stem_"
            if suffix.isdigit():
                max_version = max(max_version, int(suffix))

        next_version = max_version + 1
        version_name = f"{stem}_{next_version:03d}{ext}"
        version_path = os.path.join(directory, version_name)

        shutil.copy2(filepath, version_path)

    def open_point_cloud_file(self, parent=None):
        """Opens a file dialog, loads a PLY point cloud using plyfile, and emits the loaded signal."""

        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            "Open PLY File",
            "",
            "PLY Files (*.ply);;All Files (*)",
            options=options
        )

        if not file_path:
            return

        try:
            from plyfile import PlyData
            ply = PlyData.read(file_path)
            vertex = ply['vertex']
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                None, "Import Failed",
                f"Failed to load point cloud from:\n{file_path}\n\nThe file may be empty or corrupted.\n\nDetails: {e}"
            )
            return

        # Case-insensitive property map: lowercase_name -> actual_name
        prop_names = {p.name.lower(): p.name for p in vertex.properties}

        # Coordinates (required)
        x_col = prop_names.get('x')
        y_col = prop_names.get('y')
        z_col = prop_names.get('z')
        if x_col is None or y_col is None or z_col is None:
            QtWidgets.QMessageBox.warning(
                None, "Import Failed",
                f"Failed to load point cloud from:\n{file_path}\n\nNo x/y/z properties found in PLY header."
            )
            return

        points = np.column_stack([
            np.asarray(vertex[x_col], dtype=np.float64),
            np.asarray(vertex[y_col], dtype=np.float64),
            np.asarray(vertex[z_col], dtype=np.float64),
        ])

        if len(points) == 0:
            QtWidgets.QMessageBox.warning(
                None, "Import Failed",
                f"Failed to load point cloud from:\n{file_path}\n\nThe file appears to contain no points."
            )
            return

        # Colors (optional)
        colors = None
        r_col = prop_names.get('red') or prop_names.get('r')
        g_col = prop_names.get('green') or prop_names.get('g')
        b_col = prop_names.get('blue') or prop_names.get('b')
        if r_col and g_col and b_col:
            r = np.asarray(vertex[r_col], dtype=np.float32)
            g = np.asarray(vertex[g_col], dtype=np.float32)
            b = np.asarray(vertex[b_col], dtype=np.float32)
            if r.max() > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            colors = np.column_stack([r, g, b])

        # Normals (optional)
        normals = None
        nx_col = prop_names.get('nx') or prop_names.get('normal_x')
        ny_col = prop_names.get('ny') or prop_names.get('normal_y')
        nz_col = prop_names.get('nz') or prop_names.get('normal_z')
        if nx_col and ny_col and nz_col:
            normals = np.column_stack([
                np.asarray(vertex[nx_col], dtype=np.float32),
                np.asarray(vertex[ny_col], dtype=np.float32),
                np.asarray(vertex[nz_col], dtype=np.float32),
            ])

        # Extra per-point attributes (e.g. Intensity, GPS_Time from Paris-Lille-3D)
        handled_lower = {x_col.lower(), y_col.lower(), z_col.lower()}
        if r_col:
            handled_lower.update({r_col.lower(), g_col.lower(), b_col.lower()})
        if nx_col:
            handled_lower.update({nx_col.lower(), ny_col.lower(), nz_col.lower()})

        extra_attributes = {}
        for lower_name, actual_name in prop_names.items():
            if lower_name in handled_lower:
                continue
            try:
                extra_attributes[actual_name] = np.asarray(vertex[actual_name])
            except Exception:
                pass

        # Translate to origin (preserves float32 precision)
        if self.min_bound is None:
            self.min_bound = np.min(points, axis=0)
        points = (points - self.min_bound).astype(np.float32)

        # Build PointCloud
        point_cloud = PointCloud(points=points, colors=colors, normals=normals)
        point_cloud.translation = -self.min_bound
        point_cloud.name = file_path.split("/")[-1]

        for attr_name, attr_values in extra_attributes.items():
            point_cloud.add_attribute(attr_name, attr_values)

        self.point_cloud_loaded.emit(file_path, point_cloud)

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
            'tree_visibility': tree_visibility,
            'flythroughs': self.flythroughs,
        }

        try:
            # Save the data_nodes instance using pickle
            with open(filename, 'wb') as file:
                pickle.dump(project_data, file)

            # Create an auto-versioned backup copy
            self._save_version_copy(filename)

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
                self.flythroughs = project_data.get('flythroughs', [])
                # Backward compat: migrate old camera_waypoints to a single flythrough
                if not self.flythroughs:
                    old = project_data.get('camera_waypoints', [])
                    if old:
                        for wp in old:
                            wp.setdefault('duration', 3.0)
                        self.flythroughs = [{'name': 'Default', 'fps': 30, 'waypoints': old}]
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