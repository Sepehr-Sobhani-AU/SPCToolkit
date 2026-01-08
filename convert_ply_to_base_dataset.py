# convert_ply_to_base_dataset.py
"""
Convert PLY files to base dataset format for ML training.

This script creates the same output format as 010_export_classified_clusters_plugin.py:
- XYZ + RGB data in (n, 6) numpy arrays
- Centered at origin (0,0,0)
- Subsampled to max_points if needed
- Saved as float32 .npy files

This base dataset can then be processed by model-specific plugins like:
- 000_generate_training_data_plugin.py (PointNet)
- Future plugins for other models

Usage:
    python convert_ply_to_base_dataset.py
"""

import os
import sys
import json
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSpinBox,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt


def load_ply_file(filepath):
    """
    Load a PLY file and return points and colors.

    Args:
        filepath: Path to PLY file

    Returns:
        Tuple of (points, colors) as float64 arrays, or (None, None) if error
    """
    try:
        import open3d as o3d

        # Load PLY file
        pcd = o3d.io.read_point_cloud(filepath)

        # Extract data as float64 (high precision)
        points = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64)

        if len(points) == 0:
            print(f"  WARNING: Empty point cloud in {filepath}")
            return None, None

        if len(colors) == 0:
            print(f"  WARNING: No colors in {filepath}, using default gray")
            colors = np.full((len(points), 3), 0.5, dtype=np.float64)

        return points, colors

    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None, None


def process_cluster(points, colors, max_points=20000):
    """
    Process a cluster to match export plugin format.

    Pipeline:
    1. Center at origin (0,0,0)
    2. Subsample to max_points if needed
    3. Combine XYZ + RGB
    4. Convert to float32

    Args:
        points: Point coordinates (n, 3) in float64
        colors: Point colors (n, 3) in float64, range [0, 1]
        max_points: Maximum points to keep (default: 20000)

    Returns:
        Numpy array of shape (n, 6) as float32, or None if failed
    """
    try:
        # Step 1: Center at origin (0,0,0)
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid  # Still float64

        # Step 2: Subsample if cluster is too large
        original_size = len(points_centered)
        if original_size > max_points:
            indices = np.random.choice(original_size, max_points, replace=False)
            points_centered = points_centered[indices]
            colors = colors[indices]
            print(f"    Subsampled: {original_size} → {max_points} points")

        # Step 3: Combine XYZ and RGB into single (n, 6) array
        cluster_data = np.hstack([points_centered, colors])

        # Step 4: Convert to float32 for storage efficiency
        cluster_data = cluster_data.astype(np.float32)

        print(f"    Processed: {cluster_data.shape[0]} points → (n, 6) array")

        return cluster_data

    except Exception as e:
        print(f"    ERROR processing cluster: {e}")
        return None


def get_next_file_number(class_dir, class_name):
    """
    Scan the class directory for existing files and determine the next incremental number.

    Args:
        class_dir: Path to the class-specific directory
        class_name: Name of the class (e.g., "Tree", "Car")

    Returns:
        Next available number for file naming
    """
    # Pattern to match files like "tree_1.npy", "tree_2.npy", etc.
    pattern = re.compile(rf"^{re.escape(class_name.lower())}_(\d+)\.npy$")

    max_number = 0

    # Check if directory exists and has files
    if os.path.exists(class_dir):
        for filename in os.listdir(class_dir):
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)

    # Return next number (starting at 1 if no files exist)
    return max_number + 1


def convert_dataset(input_dir, output_dir, max_points=20000, min_samples_per_class=3):
    """
    Convert entire PLY dataset to base dataset format.

    Args:
        input_dir: Directory containing class subdirectories with PLY files
        output_dir: Directory to save base dataset
        max_points: Maximum points per cluster (default: 20000)
        min_samples_per_class: Minimum samples required per class

    Returns:
        Dictionary with conversion statistics
    """
    print("=" * 80)
    print("PLY Dataset to Base Dataset Conversion")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max points per cluster: {max_points}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Scan input directory for class folders
    print("[1/3] Scanning input directory...")
    class_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and not item.startswith('_'):
            class_dirs.append(item)

    class_dirs = sorted(class_dirs)
    print(f"Found {len(class_dirs)} class directories")

    # Process each class
    print(f"\n[2/3] Processing PLY files...")
    conversion_stats = {}
    total_processed = 0
    total_skipped = 0
    total_downsampled = 0

    for class_name in class_dirs:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)

        # Find PLY files
        ply_files = [f for f in os.listdir(class_input_dir) if f.endswith('.ply')]

        if len(ply_files) == 0:
            print(f"  {class_name}: No PLY files found, skipping")
            continue

        print(f"  {class_name}: Processing {len(ply_files)} files...")

        # Create output directory
        os.makedirs(class_output_dir, exist_ok=True)

        processed = 0
        skipped = 0
        downsampled = 0

        for ply_file in ply_files:
            ply_path = os.path.join(class_input_dir, ply_file)

            # Load PLY
            points, colors = load_ply_file(ply_path)
            if points is None:
                skipped += 1
                continue

            # Check original size for downsampling statistics
            original_size = len(points)

            # Process cluster
            cluster_data = process_cluster(points, colors, max_points=max_points)

            if cluster_data is None:
                skipped += 1
                continue

            # Track downsampling
            if original_size > max_points:
                downsampled += 1

            # Determine next incremental number for this class
            next_number = get_next_file_number(class_output_dir, class_name)

            # Save as .npy
            output_filename = f"{class_name.lower()}_{next_number}.npy"
            output_path = os.path.join(class_output_dir, output_filename)
            np.save(output_path, cluster_data)

            processed += 1

        total_processed += processed
        total_skipped += skipped
        total_downsampled += downsampled

        conversion_stats[class_name] = {
            'processed': processed,
            'skipped': skipped,
            'downsampled': downsampled
        }

        print(f"    [OK] Saved {processed} samples, skipped {skipped}, downsampled {downsampled}")

    # Filter out classes with too few samples
    print(f"\n[3/3] Filtering classes (minimum {min_samples_per_class} samples)...")
    filtered_stats = {}
    removed_classes = []

    for class_name, stats in conversion_stats.items():
        if stats['processed'] >= min_samples_per_class:
            filtered_stats[class_name] = stats
        else:
            removed_classes.append(class_name)
            # Remove directory
            class_output_dir = os.path.join(output_dir, class_name)
            if os.path.exists(class_output_dir):
                import shutil
                shutil.rmtree(class_output_dir)

    if removed_classes:
        print(f"  Removed {len(removed_classes)} classes with insufficient samples:")
        for cls in removed_classes:
            print(f"    - {cls} ({conversion_stats[cls]['processed']} samples)")

    # Save metadata
    metadata = {
        "dataset_info": {
            "created_at": datetime.now().isoformat(),
            "created_by": "SPCToolkit PLY to Base Dataset Converter",
            "source_directory": input_dir,
            "output_directory": output_dir,
            "purpose": "Base dataset for model-specific training data generation",
            "compatible_with": [
                "PointNet Generate Training Data Plugin",
                "Future model-specific plugins"
            ]
        },
        "processing": {
            "centering": {
                "enabled": True,
                "method": "center_at_origin",
                "description": "Each cluster centered at (0,0,0) by subtracting centroid"
            },
            "subsampling": {
                "max_points_per_cluster": max_points,
                "method": "random_choice",
                "description": "Clusters exceeding max_points are randomly subsampled"
            }
        },
        "data_format": {
            "file_format": "numpy (.npy)",
            "array_shape": "(n, 6) where n ≤ max_points",
            "feature_order": ["X_centered", "Y_centered", "Z_centered", "R", "G", "B"],
            "data_type": "float32",
            "coordinate_space": "Centered at origin (0,0,0)",
            "color_range": "[0, 1] (normalized RGB)"
        },
        "conversion_stats": filtered_stats,
        "processing_summary": {
            "total_samples": total_processed,
            "num_classes": len(filtered_stats),
            "skipped_files": total_skipped,
            "downsampled_clusters": total_downsampled,
            "removed_classes": removed_classes
        }
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"Total samples processed: {total_processed}")
    print(f"Total classes: {len(filtered_stats)}")
    print(f"Skipped files: {total_skipped}")
    print(f"Downsampled clusters: {total_downsampled}")
    print(f"\nClass breakdown:")
    for class_name, stats in filtered_stats.items():
        print(f"  {class_name}: {stats['processed']} samples ({stats['downsampled']} downsampled)")
    print(f"\nOutput directory: {output_dir}")
    print(f"Metadata saved: {metadata_path}")
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("  1. Use 'PointNet > Generate Training Data' plugin to create training data")
    print("  2. Or use other model-specific plugins to process this base dataset")
    print("=" * 80)

    return metadata


class ConversionDialog(QDialog):
    """
    Dialog for PLY to Base Dataset conversion parameters.

    Matches the interface of 010_export_classified_clusters_plugin.py
    with additional input directory parameter.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Convert PLY to Base Dataset")
        self.setMinimumWidth(600)

        # Default values
        self.last_input_dir = r"C:\Users\Sepeh\OneDrive\AI\Point Cloud\_Data\Dataset"
        self.last_output_dir = "base_dataset"

        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout()

        # Input Directory (PLY files)
        input_layout = QHBoxLayout()
        input_label = QLabel("PLY Files Directory:")
        input_label.setMinimumWidth(180)
        self.input_dir_edit = QLineEdit(self.last_input_dir)
        self.input_dir_edit.setReadOnly(True)
        input_browse_btn = QPushButton("Browse...")
        input_browse_btn.clicked.connect(self.browse_input_directory)
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_dir_edit)
        input_layout.addWidget(input_browse_btn)
        layout.addLayout(input_layout)

        # Description for input
        input_desc = QLabel("Directory containing class subfolders with PLY files")
        input_desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(input_desc)

        layout.addSpacing(10)

        # Export Directory (Output)
        export_layout = QHBoxLayout()
        export_label = QLabel("Export Directory:")
        export_label.setMinimumWidth(180)
        self.export_dir_edit = QLineEdit(self.last_output_dir)
        self.export_dir_edit.setReadOnly(True)
        export_browse_btn = QPushButton("Browse...")
        export_browse_btn.clicked.connect(self.browse_export_directory)
        export_layout.addWidget(export_label)
        export_layout.addWidget(self.export_dir_edit)
        export_layout.addWidget(export_browse_btn)
        layout.addLayout(export_layout)

        # Description for export
        export_desc = QLabel("Main directory where class subfolders will be created")
        export_desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(export_desc)

        layout.addSpacing(10)

        # Max Points
        max_points_layout = QHBoxLayout()
        max_points_label = QLabel("Maximum Points Per Cluster:")
        max_points_label.setMinimumWidth(180)
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setMinimum(10)
        self.max_points_spin.setMaximum(100000)
        self.max_points_spin.setValue(20000)
        max_points_layout.addWidget(max_points_label)
        max_points_layout.addWidget(self.max_points_spin)
        max_points_layout.addStretch()
        layout.addLayout(max_points_layout)

        # Description for max points
        max_points_desc = QLabel("Maximum number of points to save per cluster. Clusters with more points will be randomly subsampled.")
        max_points_desc.setWordWrap(True)
        max_points_desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(max_points_desc)

        layout.addSpacing(20)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Convert")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def browse_input_directory(self):
        """Browse for input directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select PLY Files Directory",
            self.input_dir_edit.text()
        )
        if directory:
            self.input_dir_edit.setText(directory)

    def browse_export_directory(self):
        """Browse for export directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            self.export_dir_edit.text()
        )
        if directory:
            self.export_dir_edit.setText(directory)

    def get_parameters(self):
        """
        Get the parameters from the dialog.

        Returns:
            Dictionary with input_directory, export_directory, and max_points
        """
        return {
            'input_directory': self.input_dir_edit.text(),
            'export_directory': self.export_dir_edit.text(),
            'max_points': self.max_points_spin.value()
        }


def main():
    """Main conversion function with GUI dialog."""
    # Create QApplication
    app = QApplication(sys.argv)

    # Show dialog
    dialog = ConversionDialog()

    if dialog.exec_() != QDialog.Accepted:
        print("Conversion cancelled by user")
        sys.exit(0)

    # Get parameters
    params = dialog.get_parameters()
    INPUT_DIR = params['input_directory']
    OUTPUT_DIR = params['export_directory']
    MAX_POINTS = params['max_points']
    MIN_SAMPLES_PER_CLASS = 3

    # Validate input directory
    if not INPUT_DIR or not os.path.exists(INPUT_DIR):
        QMessageBox.warning(
            None,
            "Invalid Input Directory",
            f"The selected input directory does not exist:\n{INPUT_DIR}"
        )
        sys.exit(1)

    # Validate export directory
    if not OUTPUT_DIR:
        QMessageBox.warning(
            None,
            "Invalid Export Directory",
            "Please select an export directory."
        )
        sys.exit(1)

    # Run conversion
    try:
        metadata = convert_dataset(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            max_points=MAX_POINTS,
            min_samples_per_class=MIN_SAMPLES_PER_CLASS
        )

        # Show success message
        QMessageBox.information(
            None,
            "Conversion Complete",
            f"Successfully converted PLY files to base dataset!\n\n"
            f"Output directory: {OUTPUT_DIR}\n"
            f"Total samples: {metadata['processing_summary']['total_samples']}\n"
            f"Total classes: {metadata['processing_summary']['num_classes']}"
        )

    except Exception as e:
        QMessageBox.critical(
            None,
            "Conversion Error",
            f"An error occurred during conversion:\n{str(e)}"
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
