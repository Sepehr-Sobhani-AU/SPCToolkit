"""
Annotation Window for manual per-point labeling.

Provides a class palette and annotation tools for painting semantic labels
onto point cloud data. Works with the viewer's existing polygon selection.
"""

import os
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QInputDialog, QColorDialog,
    QFileDialog, QMessageBox, QGroupBox, QSpinBox, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QIcon, QPixmap

from config.config import global_variables
from core.entities.point_cloud import PointCloud


class AnnotationWindow(QDialog):
    """
    Dialog for manual point cloud annotation.

    Allows users to:
    - Define semantic classes with names and colors
    - Select a class, then use polygon selection to paint labels
    - Fill entire DBSCAN clusters with a class
    - Undo last annotation action
    - Export annotations as .npz training data
    """

    # Default class palette
    DEFAULT_CLASSES = [
        ("Ground", QColor(153, 102, 51)),
        ("Building", QColor(178, 76, 25)),
        ("Tree", QColor(0, 204, 0)),
        ("Vegetation", QColor(51, 153, 51)),
        ("Car", QColor(0, 0, 255)),
        ("Pole", QColor(128, 128, 128)),
        ("Cable", QColor(0, 0, 0)),
        ("Fence", QColor(153, 76, 0)),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point Cloud Annotation")
        self.setMinimumWidth(300)
        self.setMinimumHeight(500)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.viewer = global_variables.global_pcd_viewer_widget
        self.controller = global_variables.global_application_controller

        # Annotation state
        self.annotations = None  # Will be (N,) int array, -1 = unlabeled
        self.point_cloud = None
        self.class_list = []  # [(name, QColor), ...]
        self.current_class_idx = -1
        self.undo_stack = []  # List of (indices, old_labels) for undo

        # Polling timer to check for new selections
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._check_selection)
        self.last_selection_count = 0

        self._setup_ui()
        self._load_default_classes()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "1. Select a class below\n"
            "2. Press P in viewer for polygon select\n"
            "3. Draw polygon around points to label\n"
            "4. Click 'Apply Label' to assign class"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("QLabel { background: #f0f0f0; padding: 8px; }")
        layout.addWidget(instructions)

        # Class list
        class_group = QGroupBox("Semantic Classes")
        class_layout = QVBoxLayout()

        self.class_list_widget = QListWidget()
        self.class_list_widget.currentRowChanged.connect(self._on_class_selected)
        class_layout.addWidget(self.class_list_widget)

        # Class buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_class)
        btn_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_class)
        btn_layout.addWidget(remove_btn)

        color_btn = QPushButton("Color")
        color_btn.clicked.connect(self._change_color)
        btn_layout.addWidget(color_btn)

        class_layout.addLayout(btn_layout)
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)

        # Annotation tools
        tools_group = QGroupBox("Annotation Tools")
        tools_layout = QVBoxLayout()

        self.apply_btn = QPushButton("Apply Label to Selection")
        self.apply_btn.clicked.connect(self._apply_label)
        self.apply_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 8px; }")
        tools_layout.addWidget(self.apply_btn)

        self.fill_cluster_btn = QPushButton("Fill Cluster at Selection")
        self.fill_cluster_btn.clicked.connect(self._fill_cluster)
        self.fill_cluster_btn.setToolTip(
            "Assigns the selected class to all points in the same DBSCAN cluster "
            "as the currently selected points")
        tools_layout.addWidget(self.fill_cluster_btn)

        self.undo_btn = QPushButton("Undo Last Action")
        self.undo_btn.clicked.connect(self._undo)
        self.undo_btn.setEnabled(False)
        tools_layout.addWidget(self.undo_btn)

        self.clear_sel_btn = QPushButton("Clear Selection")
        self.clear_sel_btn.clicked.connect(self._clear_selection)
        tools_layout.addWidget(self.clear_sel_btn)

        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)

        # Status
        self.status_label = QLabel("No point cloud loaded")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()

        export_params_layout = QHBoxLayout()
        export_params_layout.addWidget(QLabel("Block size:"))
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(1, 100)
        self.block_size_spin.setValue(10)
        self.block_size_spin.setSuffix(" m")
        export_params_layout.addWidget(self.block_size_spin)

        export_params_layout.addWidget(QLabel("Pts/block:"))
        self.pts_per_block_spin = QSpinBox()
        self.pts_per_block_spin.setRange(512, 16384)
        self.pts_per_block_spin.setValue(4096)
        export_params_layout.addWidget(self.pts_per_block_spin)
        export_layout.addLayout(export_params_layout)

        export_btn = QPushButton("Export Training Data (.npz)")
        export_btn.clicked.connect(self._export_training_data)
        export_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        export_layout.addWidget(export_btn)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def _load_default_classes(self):
        """Load default class palette."""
        for name, color in self.DEFAULT_CLASSES:
            self._add_class_item(name, color)

    def _add_class_item(self, name, color):
        """Add a class to the list."""
        self.class_list.append((name, color))

        item = QListWidgetItem()
        item.setText(f"  {name}")

        # Create color icon
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        item.setIcon(QIcon(pixmap))

        self.class_list_widget.addItem(item)

    def _add_class(self):
        """Add a new class via dialog."""
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if ok and name.strip():
            color = QColorDialog.getColor(QColor(128, 128, 128), self, "Select Color")
            if color.isValid():
                self._add_class_item(name.strip(), color)

    def _remove_class(self):
        """Remove selected class."""
        row = self.class_list_widget.currentRow()
        if row >= 0:
            self.class_list.pop(row)
            self.class_list_widget.takeItem(row)

    def _change_color(self):
        """Change color of selected class."""
        row = self.class_list_widget.currentRow()
        if row < 0:
            return
        name, old_color = self.class_list[row]
        color = QColorDialog.getColor(old_color, self, f"Color for {name}")
        if color.isValid():
            self.class_list[row] = (name, color)
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            self.class_list_widget.item(row).setIcon(QIcon(pixmap))

    def _on_class_selected(self, row):
        """Handle class selection."""
        self.current_class_idx = row

    def initialize_annotations(self, point_cloud):
        """Initialize annotation array for a point cloud."""
        self.point_cloud = point_cloud
        n = len(point_cloud.points)
        self.annotations = np.full(n, -1, dtype=np.int32)
        self.undo_stack.clear()
        self.undo_btn.setEnabled(False)
        self.status_label.setText(f"Loaded: {n:,} points | 0 labeled")

        # Start polling for selections
        self.poll_timer.start(200)

    def _check_selection(self):
        """Poll viewer for new point selections."""
        if self.viewer is None:
            return
        count = len(self.viewer.picked_points_indices)
        if count != self.last_selection_count:
            self.last_selection_count = count
            self.status_label.setText(
                f"{self._count_labeled()} labeled | {count} selected")

    def _count_labeled(self):
        """Count labeled points."""
        if self.annotations is None:
            return 0
        return int(np.sum(self.annotations >= 0))

    def _apply_label(self):
        """Apply selected class to currently selected points."""
        if self.current_class_idx < 0:
            QMessageBox.warning(self, "No Class", "Please select a class first.")
            return

        if not self.viewer.picked_points_indices:
            QMessageBox.warning(self, "No Selection",
                              "No points selected. Press P in viewer to start polygon selection.")
            return

        if self.annotations is None:
            QMessageBox.warning(self, "Not Initialized",
                              "No point cloud loaded for annotation.")
            return

        indices = np.array(self.viewer.picked_points_indices, dtype=np.int64)
        # Clamp to valid range
        valid = indices < len(self.annotations)
        indices = indices[valid]

        if len(indices) == 0:
            return

        # Save undo state
        old_labels = self.annotations[indices].copy()
        self.undo_stack.append((indices.copy(), old_labels))
        self.undo_btn.setEnabled(True)

        # Apply label
        self.annotations[indices] = self.current_class_idx

        class_name = self.class_list[self.current_class_idx][0]
        print(f"Annotation: Labeled {len(indices)} points as '{class_name}'")

        self._update_visualization()
        self._clear_selection()

    def _fill_cluster(self):
        """Fill all points in the same DBSCAN cluster as selected points."""
        if self.current_class_idx < 0:
            QMessageBox.warning(self, "No Class", "Please select a class first.")
            return

        if not self.viewer.picked_points_indices:
            QMessageBox.warning(self, "No Selection",
                              "Select a point in a cluster first.")
            return

        if self.point_cloud is None or self.annotations is None:
            return

        # Get cluster labels
        cluster_labels = self.point_cloud.get_attribute("cluster_labels")
        if cluster_labels is None:
            QMessageBox.warning(self, "No Clusters",
                              "Point cloud has no cluster_labels.\n"
                              "Run DBSCAN clustering first.")
            return

        # Find clusters containing selected points
        selected_indices = np.array(self.viewer.picked_points_indices, dtype=np.int64)
        valid = selected_indices < len(cluster_labels)
        selected_indices = selected_indices[valid]

        if len(selected_indices) == 0:
            return

        target_clusters = set()
        for idx in selected_indices:
            cid = cluster_labels[idx]
            if cid >= 0:  # Ignore noise
                target_clusters.add(cid)

        if not target_clusters:
            QMessageBox.warning(self, "No Valid Clusters",
                              "Selected points are all noise (no cluster).")
            return

        # Find all points in these clusters
        fill_mask = np.isin(cluster_labels, list(target_clusters))
        fill_indices = np.where(fill_mask)[0]

        # Save undo state
        old_labels = self.annotations[fill_indices].copy()
        self.undo_stack.append((fill_indices.copy(), old_labels))
        self.undo_btn.setEnabled(True)

        # Apply label
        self.annotations[fill_indices] = self.current_class_idx

        class_name = self.class_list[self.current_class_idx][0]
        print(f"Annotation: Filled {len(target_clusters)} cluster(s) "
              f"({len(fill_indices)} points) as '{class_name}'")

        self._update_visualization()
        self._clear_selection()

    def _undo(self):
        """Undo last annotation action."""
        if not self.undo_stack:
            return

        indices, old_labels = self.undo_stack.pop()
        self.annotations[indices] = old_labels

        if not self.undo_stack:
            self.undo_btn.setEnabled(False)

        print(f"Annotation: Undone (restored {len(indices)} points)")
        self._update_visualization()

    def _clear_selection(self):
        """Clear viewer selection."""
        if self.viewer:
            self.viewer.picked_points_indices.clear()
            self.viewer.update()
            self.last_selection_count = 0

    def _update_visualization(self):
        """Update point cloud colors based on annotations."""
        if self.point_cloud is None or self.annotations is None:
            return

        # Store annotations as attribute
        self.point_cloud.attributes["annotations"] = self.annotations.copy()

        # Build colors from annotations
        colors = self.point_cloud.colors.copy() if self.point_cloud.colors is not None \
            else np.ones((len(self.point_cloud.points), 3), dtype=np.float32) * 0.5

        for class_idx, (name, qcolor) in enumerate(self.class_list):
            mask = self.annotations == class_idx
            if np.any(mask):
                colors[mask] = np.array([
                    qcolor.redF(), qcolor.greenF(), qcolor.blueF()
                ], dtype=np.float32)

        self.point_cloud.colors = colors

        # Re-render
        main_window = global_variables.global_main_window
        if main_window:
            main_window.render_visible_data(zoom_extent=False)

        # Update status
        labeled = self._count_labeled()
        total = len(self.annotations)
        self.status_label.setText(f"{labeled:,}/{total:,} labeled ({labeled/total*100:.1f}%)")

    def _export_training_data(self):
        """Export annotations as .npz training data blocks."""
        if self.annotations is None or self.point_cloud is None:
            QMessageBox.warning(self, "Nothing to Export",
                              "No annotations to export.")
            return

        labeled_mask = self.annotations >= 0
        labeled_count = np.sum(labeled_mask)
        if labeled_count == 0:
            QMessageBox.warning(self, "No Labels",
                              "No points have been labeled yet.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", "training_data_seg")
        if not output_dir:
            return

        block_size = self.block_size_spin.value()
        points_per_block = self.pts_per_block_spin.value()

        try:
            valid_points = self.point_cloud.points[labeled_mask]
            valid_labels = self.annotations[labeled_mask]

            # Build class mapping from used classes
            used_classes = sorted(set(valid_labels.tolist()))
            class_mapping = {}
            remap = {}
            for new_id, old_id in enumerate(used_classes):
                if old_id < len(self.class_list):
                    class_mapping[new_id] = self.class_list[old_id][0]
                else:
                    class_mapping[new_id] = f"Class_{old_id}"
                remap[old_id] = new_id

            # Remap labels to contiguous
            remapped = np.array([remap[l] for l in valid_labels], dtype=np.int32)

            # Create blocks
            min_xy = np.min(valid_points[:, :2], axis=0)
            max_xy = np.max(valid_points[:, :2], axis=0)
            extent = max_xy - min_xy
            nx = max(1, int(np.ceil(extent[0] / block_size)))
            ny = max(1, int(np.ceil(extent[1] / block_size)))

            saved = 0
            for ix in range(nx):
                for iy in range(ny):
                    x_min = min_xy[0] + ix * block_size
                    x_max = min_xy[0] + (ix + 1) * block_size
                    y_min = min_xy[1] + iy * block_size
                    y_max = min_xy[1] + (iy + 1) * block_size

                    mask = (
                        (valid_points[:, 0] >= x_min) & (valid_points[:, 0] < x_max) &
                        (valid_points[:, 1] >= y_min) & (valid_points[:, 1] < y_max)
                    )

                    if np.sum(mask) < points_per_block:
                        continue

                    block_pts = valid_points[mask]
                    block_lbl = remapped[mask]

                    # Normalize block
                    pc = PointCloud(points=block_pts.copy())
                    pc.normalise(apply_scaling=True, apply_centering=True,
                               rotation_axes=(False, False, False))

                    features = pc.points  # Just XYZ normalized for now

                    # Subsample
                    if len(features) > points_per_block:
                        idx = np.random.choice(len(features), points_per_block, replace=False)
                        features = features[idx]
                        block_lbl = block_lbl[idx]

                    filename = f"annotated_block_{saved:05d}.npz"
                    np.savez_compressed(
                        os.path.join(output_dir, filename),
                        features=features.astype(np.float32),
                        labels=block_lbl.astype(np.int32)
                    )
                    saved += 1

            # Save metadata
            metadata = {
                "dataset_info": {
                    "created_at": datetime.now().isoformat(),
                    "created_by": "SPCToolkit",
                    "plugin": "Manual Annotation",
                    "task_type": "Semantic Segmentation",
                    "source": "manual_annotation"
                },
                "class_mapping": class_mapping,
                "num_classes": len(class_mapping),
                "num_features": 3,
                "feature_order": ["X_norm", "Y_norm", "Z_norm"],
                "points_per_block": points_per_block,
                "block_size": block_size,
                "total_samples": saved,
                "processing": {
                    "normalization": {"enabled": True},
                    "features": {
                        "normals": {"enabled": False},
                        "eigenvalues": {"enabled": False}
                    }
                }
            }

            with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            QMessageBox.information(self, "Export Complete",
                f"Exported {saved} training blocks\n"
                f"with {len(class_mapping)} classes\n\n"
                f"Saved to: {output_dir}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Error: {str(e)}")

    def closeEvent(self, event):
        """Stop polling when dialog closes."""
        self.poll_timer.stop()
        super().closeEvent(event)
