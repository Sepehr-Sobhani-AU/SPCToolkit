"""
Training Data Preview Window

Provides an interactive window for browsing and previewing machine learning training data.
Supports multiple model formats and displays samples with various coloring modes.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QListWidget, QSplitter, QFileDialog,
    QMessageBox, QListWidgetItem, QDoubleSpinBox
)
from PyQt5.QtCore import Qt

from gui.widgets.pcd_viewer_widget import PCDViewerWidget


class TrainingDataPreviewWindow(QDialog):
    """
    Custom dialog window for previewing training data.

    Features:
    - Directory browsing with session persistence
    - Metadata display
    - Class and sample selection
    - 3D visualization with feature-based coloring
    - Support for multiple model formats
    """

    # Class variable for session persistence
    last_training_data_dir = ""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Training Data Preview")
        self.resize(1200, 700)

        # State variables
        self.current_directory = ""
        self.metadata = None
        self.feature_order = []
        self.current_sample_data = None
        self.first_load = True  # Track first visualization to control camera reset

        # Setup UI
        self._setup_ui()

        # Initialize with last directory if available
        if TrainingDataPreviewWindow.last_training_data_dir:
            self.directory_edit.setText(TrainingDataPreviewWindow.last_training_data_dir)
            self._load_directory(TrainingDataPreviewWindow.last_training_data_dir)

    def _setup_ui(self):
        """Setup the user interface layout."""
        main_layout = QVBoxLayout()

        # === Directory Selection Section ===
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Directory:"))

        self.directory_edit = QLineEdit()
        self.directory_edit.setReadOnly(True)
        dir_layout.addWidget(self.directory_edit, stretch=1)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_directory)
        dir_layout.addWidget(self.browse_button)

        main_layout.addLayout(dir_layout)

        # === Metadata Button (Left side) ===
        metadata_layout = QHBoxLayout()

        self.metadata_button = QPushButton("Show Metadata")
        self.metadata_button.setEnabled(False)
        self.metadata_button.clicked.connect(self._show_metadata)
        metadata_layout.addWidget(self.metadata_button)

        metadata_layout.addStretch()

        main_layout.addLayout(metadata_layout)

        # === Main Content: File List + Viewer ===
        content_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Class selection, file list, and controls
        left_panel = QVBoxLayout()
        left_panel_widget = QDialog()
        left_panel_widget.setLayout(left_panel)

        # Class selection (inside left panel)
        left_panel.addWidget(QLabel("Class:"))

        self.class_combo = QComboBox()
        self.class_combo.setEditable(True)  # Allow typing to search
        self.class_combo.setEnabled(False)
        self.class_combo.currentTextChanged.connect(self._on_class_changed)
        left_panel.addWidget(self.class_combo)

        # File list
        left_panel.addWidget(QLabel("Files:"))

        self.file_list = QListWidget()
        self.file_list.setEnabled(False)
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        left_panel.addWidget(self.file_list)

        left_panel.addWidget(QLabel("Color by:"))

        self.color_combo = QComboBox()
        self.color_combo.addItem("Uniform Gray")
        self.color_combo.setEnabled(False)
        self.color_combo.currentTextChanged.connect(self._on_color_mode_changed)
        left_panel.addWidget(self.color_combo)

        # Point size control
        point_size_layout = QHBoxLayout()
        point_size_layout.addWidget(QLabel("Point Size:"))

        self.point_size_spinbox = QDoubleSpinBox()
        self.point_size_spinbox.setMinimum(0.1)
        self.point_size_spinbox.setMaximum(20.0)
        self.point_size_spinbox.setSingleStep(0.5)  # Step by 0.5
        self.point_size_spinbox.setValue(0.5)  # Default point size (start value)
        self.point_size_spinbox.setDecimals(1)
        self.point_size_spinbox.setFixedWidth(80)  # Fixed width
        self.point_size_spinbox.valueChanged.connect(self._on_point_size_changed)
        point_size_layout.addWidget(self.point_size_spinbox)
        point_size_layout.addStretch()  # Push spinbox to the left

        left_panel.addLayout(point_size_layout)

        content_splitter.addWidget(left_panel_widget)

        # Right panel: Viewer
        self.viewer = PCDViewerWidget()
        content_splitter.addWidget(self.viewer)

        # Set splitter sizes (20% left, 80% right)
        content_splitter.setSizes([240, 960])

        main_layout.addWidget(content_splitter, stretch=1)

        self.setLayout(main_layout)

    def _browse_directory(self):
        """Open file dialog to select training data directory."""
        # Default to current directory if available, otherwise use project directory
        initial_dir = self.current_directory or os.getcwd()

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Training Data Directory",
            initial_dir,
            QFileDialog.ShowDirsOnly
        )

        if directory:
            self._load_directory(directory)

    def _load_directory(self, directory: str):
        """
        Load training data directory and metadata.

        Args:
            directory: Path to training data directory
        """
        if not os.path.exists(directory):
            QMessageBox.warning(self, "Invalid Directory", f"Directory does not exist:\n{directory}")
            return

        # Check for metadata.json
        metadata_path = os.path.join(directory, "metadata.json")
        if not os.path.exists(metadata_path):
            QMessageBox.warning(
                self,
                "No Metadata Found",
                f"No metadata.json found in:\n{directory}\n\n"
                "This does not appear to be a valid training data directory."
            )
            return

        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            # Extract feature order for color mode detection
            if "data_format" in self.metadata and "feature_order" in self.metadata["data_format"]:
                self.feature_order = self.metadata["data_format"]["feature_order"]
            else:
                self.feature_order = []

            # Update state
            self.current_directory = directory
            TrainingDataPreviewWindow.last_training_data_dir = directory

            # Update UI
            self.directory_edit.setText(directory)
            self.metadata_button.setEnabled(True)

            # Scan for classes
            self._scan_classes()

            # Update color modes based on features
            self._update_color_modes()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Directory",
                f"Failed to load training data:\n{str(e)}"
            )

    def _scan_classes(self):
        """Scan directory for class subdirectories."""
        if not self.current_directory:
            return

        classes = []

        try:
            for item in os.listdir(self.current_directory):
                item_path = os.path.join(self.current_directory, item)
                if os.path.isdir(item_path):
                    # Check if directory contains .npy files
                    npy_files = [f for f in os.listdir(item_path) if f.endswith('.npy')]
                    if npy_files:
                        classes.append(item)
        except Exception as e:
            return

        # Sort alphabetically
        classes.sort()

        # Update combo box
        self.class_combo.clear()
        self.class_combo.addItems(classes)

        if classes:
            self.class_combo.setEnabled(True)
        else:
            self.class_combo.setEnabled(False)
            QMessageBox.warning(
                self,
                "No Classes Found",
                f"No class directories with .npy files found in:\n{self.current_directory}"
            )

    def _update_color_modes(self):
        """Update color mode options based on available features."""
        self.color_combo.clear()
        self.color_combo.addItem("Uniform Gray")

        if not self.feature_order:
            return

        # Check for normals (Nx, Ny, Nz)
        if "Nx" in self.feature_order and "Ny" in self.feature_order and "Nz" in self.feature_order:
            self.color_combo.addItem("Normals (RGB)")

        # Check for eigenvalues (E1, E2, E3)
        has_all_eigenvalues = "E1" in self.feature_order and "E2" in self.feature_order and "E3" in self.feature_order

        if "E1" in self.feature_order:
            self.color_combo.addItem("Eigenvalue E1")
        if "E2" in self.feature_order:
            self.color_combo.addItem("Eigenvalue E2")
        if "E3" in self.feature_order:
            self.color_combo.addItem("Eigenvalue E3")

        # Add eigenvalue-based geometric feature coloring
        if has_all_eigenvalues:
            self.color_combo.addItem("Eigenvalue Differences (RGB)")
            self.color_combo.addItem("Geometric Features (Linearity-Planarity-Sphericity)")

    def _on_class_changed(self, class_name: str):
        """
        Handle class selection change.

        Args:
            class_name: Selected class name
        """
        if not class_name or not self.current_directory:
            return

        class_dir = os.path.join(self.current_directory, class_name)

        if not os.path.exists(class_dir):
            return

        # Scan for .npy files
        try:
            files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
            files.sort()  # Sort files

            # Update file list
            self.file_list.clear()
            self.file_list.addItems(files)

            if files:
                self.file_list.setEnabled(True)
                self.color_combo.setEnabled(True)
            else:
                self.file_list.setEnabled(False)
                self.color_combo.setEnabled(False)

        except Exception as e:
            pass

    def _on_file_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """
        Handle file selection change.

        Args:
            current: Currently selected item
            previous: Previously selected item
        """
        if not current:
            return

        filename = current.text()
        class_name = self.class_combo.currentText()

        if not class_name:
            return

        filepath = os.path.join(self.current_directory, class_name, filename)

        # Load and visualize
        self._load_and_visualize(filepath)

    def _load_and_visualize(self, filepath: str):
        """
        Load sample data and visualize in viewer.

        Args:
            filepath: Path to .npy file
        """
        try:
            # Load data
            self.current_sample_data = np.load(filepath)

            # Don't reset camera when changing files - only on very first load
            # (first_load is set to True in __init__ and set to False after first visualization)

            # Visualize with current color mode
            self._update_visualization()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Sample",
                f"Failed to load sample:\n{filepath}\n\nError: {str(e)}"
            )

    def _on_color_mode_changed(self, mode: str):
        """
        Handle color mode change.

        Args:
            mode: Selected color mode
        """
        if self.current_sample_data is not None:
            self._update_visualization()

    def _on_point_size_changed(self, value: float):
        """
        Handle point size change.

        Args:
            value: New point size value
        """
        self.viewer.point_size = value
        self.viewer.update()  # Refresh the view

    def _update_visualization(self):
        """Update viewer with current sample and color mode."""
        if self.current_sample_data is None:
            return

        # Extract XYZ (first 3 columns)
        if self.current_sample_data.shape[1] < 3:
            QMessageBox.warning(self, "Invalid Data", "Sample data must have at least 3 columns (XYZ)")
            return

        points = self.current_sample_data[:, :3].copy()

        # Generate colors based on mode
        color_mode = self.color_combo.currentText()
        colors = self._generate_colors(points, color_mode)

        # Update viewer
        self.viewer.set_points(points, colors)

        # Only zoom to extent on first load, not on color changes
        if self.first_load:
            self.viewer.zoom_to_extent()
            self.first_load = False

    def _generate_colors(self, points: np.ndarray, mode: str) -> np.ndarray:
        """
        Generate colors for points based on selected mode.

        Args:
            points: Point coordinates (n, 3)
            mode: Color mode string

        Returns:
            Color array (n, 3) with values in [0, 1]
        """
        n = len(points)

        if mode == "Uniform Gray":
            return np.full((n, 3), 0.5, dtype=np.float32)

        elif mode == "Normals (RGB)":
            # Find normal columns
            try:
                nx_idx = self.feature_order.index("Nx")
                ny_idx = self.feature_order.index("Ny")
                nz_idx = self.feature_order.index("Nz")

                normals = self.current_sample_data[:, [nx_idx, ny_idx, nz_idx]]

                # Map normals from [-1, 1] to [0, 1]
                colors = (normals + 1.0) / 2.0
                return np.clip(colors, 0, 1).astype(np.float32)

            except (ValueError, IndexError):
                return np.full((n, 3), 0.5, dtype=np.float32)

        elif mode == "Eigenvalue Differences (RGB)":
            # Eigenvalue-based coloring using differences (from eigenvalue_utils.py)
            try:
                e1_idx = self.feature_order.index("E1")
                e2_idx = self.feature_order.index("E2")
                e3_idx = self.feature_order.index("E3")

                eigenvalues = self.current_sample_data[:, [e1_idx, e2_idx, e3_idx]]

                # Normalize eigenvalues row-wise
                row_sums = np.sum(eigenvalues, axis=1, keepdims=True)
                row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
                normalized_eigenvalues = eigenvalues / row_sums

                # Calculate differences
                differences = np.zeros_like(normalized_eigenvalues)
                differences[:, 0] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 1]  # E1 - E2
                differences[:, 1] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 2]  # E1 - E3
                differences[:, 2] = normalized_eigenvalues[:, 2] - normalized_eigenvalues[:, 1]  # E3 - E2

                # Normalize differences to [0, 1]
                min_vals = np.min(differences, axis=0)
                max_vals = np.max(differences, axis=0)
                range_vals = max_vals - min_vals
                range_vals = np.maximum(range_vals, 1e-10)  # Avoid division by zero

                scaled_differences = (differences - min_vals) / range_vals
                return scaled_differences.astype(np.float32)

            except (ValueError, IndexError):
                return np.full((n, 3), 0.5, dtype=np.float32)

        elif mode == "Geometric Features (Linearity-Planarity-Sphericity)":
            # Color based on geometric features: R=Linearity, G=Planarity, B=Sphericity
            try:
                e1_idx = self.feature_order.index("E1")  # Smallest
                e2_idx = self.feature_order.index("E2")  # Middle
                e3_idx = self.feature_order.index("E3")  # Largest

                lambda1 = self.current_sample_data[:, e1_idx]
                lambda2 = self.current_sample_data[:, e2_idx]
                lambda3 = self.current_sample_data[:, e3_idx]

                # Add epsilon to prevent division by zero
                epsilon = 1e-10
                lambda3_safe = np.maximum(lambda3, epsilon)

                # Calculate geometric features (from eigenvalue_utils.py)
                linearity = (lambda3 - lambda2) / lambda3_safe    # Red channel - linear structures
                planarity = (lambda2 - lambda1) / lambda3_safe    # Green channel - planar structures
                sphericity = lambda1 / lambda3_safe               # Blue channel - volumetric structures

                # Handle invalid values
                linearity = np.nan_to_num(linearity, 0.0)
                planarity = np.nan_to_num(planarity, 0.0)
                sphericity = np.nan_to_num(sphericity, 0.0)

                # Stack into RGB array
                colors = np.stack([linearity, planarity, sphericity], axis=1)

                # Clip to [0, 1] range
                colors = np.clip(colors, 0, 1).astype(np.float32)

                return colors

            except (ValueError, IndexError):
                return np.full((n, 3), 0.5, dtype=np.float32)

        elif mode.startswith("Eigenvalue"):
            # Extract eigenvalue index (E1, E2, or E3)
            if "E1" in mode:
                feature_name = "E1"
            elif "E2" in mode:
                feature_name = "E2"
            elif "E3" in mode:
                feature_name = "E3"
            else:
                return np.full((n, 3), 0.5, dtype=np.float32)

            try:
                feature_idx = self.feature_order.index(feature_name)
                values = self.current_sample_data[:, feature_idx]

                # Normalize to [0, 1]
                min_val = values.min()
                max_val = values.max()

                if max_val > min_val:
                    normalized = (values - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(values)

                # Create heatmap: blue (low) -> green -> red (high)
                colors = np.zeros((n, 3), dtype=np.float32)
                colors[:, 2] = 1.0 - normalized  # Blue decreases
                colors[:, 0] = normalized        # Red increases
                colors[:, 1] = 1.0 - np.abs(normalized - 0.5) * 2  # Green peaks at middle

                return colors

            except (ValueError, IndexError):
                return np.full((n, 3), 0.5, dtype=np.float32)

        # Default: gray
        return np.full((n, 3), 0.5, dtype=np.float32)

    def _format_key_name(self, key: str) -> str:
        """
        Convert key name to human-readable format.

        Args:
            key: Key name (e.g., "created_at", "model_target")

        Returns:
            Formatted key name (e.g., "Created At", "Model Target")
        """
        # Handle special cases
        special_cases = {
            "knn": "KNN",
            "xyz": "XYZ",
            "rgb": "RGB",
            "e1": "E1",
            "e2": "E2",
            "e3": "E3",
            "n_multiplier": "Points Multiplier (n)"
        }

        key_lower = key.lower()
        if key_lower in special_cases:
            return special_cases[key_lower]

        # Convert snake_case to Title Case
        words = key.replace('_', ' ').split()
        # Capitalize each word
        formatted = ' '.join(word.capitalize() for word in words)

        return formatted

    def _format_nested_dict(self, data: Dict, indent: int = 2) -> List[str]:
        """
        Format nested dictionary into readable lines.

        Args:
            data: Dictionary to format
            indent: Indentation level (spaces)

        Returns:
            List of formatted strings
        """
        lines = []
        prefix = " " * indent

        for key, value in data.items():
            # Format the key name to be human-readable
            formatted_key = self._format_key_name(key)

            if isinstance(value, dict):
                lines.append(f"{prefix}{formatted_key}:")
                lines.extend(self._format_nested_dict(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{prefix}{formatted_key}: {', '.join(str(v) for v in value)}")
            elif value is None:
                lines.append(f"{prefix}{formatted_key}: N/A")
            else:
                lines.append(f"{prefix}{formatted_key}: {value}")

        return lines

    def _show_metadata(self):
        """Display metadata in a message box."""
        if not self.metadata:
            return

        # Format metadata as readable text
        lines = []
        lines.append("=== TRAINING DATA METADATA ===\n")

        # Dataset Info
        if "dataset_info" in self.metadata:
            lines.append("DATASET INFO:")
            lines.extend(self._format_nested_dict(self.metadata["dataset_info"]))
            lines.append("")

        # Processing Info
        if "processing" in self.metadata:
            lines.append("PROCESSING:")
            lines.extend(self._format_nested_dict(self.metadata["processing"]))
            lines.append("")

        # Data Format
        if "data_format" in self.metadata:
            lines.append("DATA FORMAT:")
            lines.extend(self._format_nested_dict(self.metadata["data_format"]))
            lines.append("")

        # Balance Info
        if "balance_info" in self.metadata:
            lines.append("BALANCE INFO:")
            for class_name, stats in self.metadata["balance_info"].items():
                lines.append(f"  {class_name}: unique={stats['unique']}, resampled={stats['resampled']}")
            lines.append("")

        # Processing Summary
        if "processing_summary" in self.metadata:
            lines.append("PROCESSING SUMMARY:")
            lines.extend(self._format_nested_dict(self.metadata["processing_summary"]))

        metadata_text = "\n".join(lines)

        # Show in message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Training Data Metadata")
        msg_box.setText(metadata_text)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()