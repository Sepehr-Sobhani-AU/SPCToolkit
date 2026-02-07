# gui/dialogs/class_analysis_window.py
"""
Class Analysis Window for managing imbalanced data.

Provides interactive UI for:
- Visualizing class distribution
- Filtering classes by sample count
- Merging similar classes
- Configuring balancing strategies
- Previewing final distribution
- Exporting configuration for training data generation
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QRadioButton, QButtonGroup,
    QFileDialog, QMessageBox, QSplitter, QWidget,
    QCheckBox, QListWidget, QListWidgetItem, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush


class ClassAnalysisWindow(QDialog):
    """
    Dialog for analyzing and preparing imbalanced class data.

    This window allows users to:
    1. Scan a directory and visualize class distribution
    2. Filter out classes with too few samples
    3. Merge semantically similar classes
    4. Configure balancing strategy
    5. Preview the final distribution
    6. Export configuration for training data generation
    """

    # Class variable for session persistence
    last_directory = ""

    # Built-in merge presets
    MERGE_PRESETS = {
        "SemanticKITTI - Merge Moving": {
            "moving-person": "person",
            "moving-car": "car",
            "moving-bicyclist": "bicycle",
            "moving-motorcyclist": "motorcycle",
            "moving-other-vehicle": "other-vehicle"
        },
        "SemanticKITTI - Merge Vehicles": {
            "car": "vehicle",
            "truck": "vehicle",
            "other-vehicle": "vehicle",
            "moving-car": "vehicle"
        },
        "Clear All": {}
    }

    def __init__(self, parent=None, input_dir: str = None):
        super().__init__(parent)
        self.setWindowTitle("Analyze & Prepare Classes")
        self.resize(1000, 800)

        # State
        self.input_directory = input_dir or ""
        self.class_distribution: Dict[str, int] = {}  # class_name -> count
        self.selected_classes: Dict[str, bool] = {}   # class_name -> is_selected
        self.merge_mappings: Dict[str, str] = {}      # source -> target
        self.min_threshold = 20
        self.balancing_strategy = "oversample_to_median"
        self.target_count = 500
        self.max_per_class = 0  # 0 = no limit

        # Setup UI
        self._setup_ui()

        # Load initial directory if provided
        if input_dir and os.path.exists(input_dir):
            self.directory_edit.setText(input_dir)
            self._scan_directory()
        elif ClassAnalysisWindow.last_directory:
            self.directory_edit.setText(ClassAnalysisWindow.last_directory)

    def _setup_ui(self):
        """Setup the user interface."""
        main_layout = QVBoxLayout()

        # === Directory Selection ===
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Input Directory:"))
        self.directory_edit = QLineEdit()
        self.directory_edit.setPlaceholderText("Select directory containing class folders...")
        dir_layout.addWidget(self.directory_edit, stretch=1)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_directory)
        dir_layout.addWidget(self.browse_button)
        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(self._scan_directory)
        dir_layout.addWidget(self.scan_button)
        main_layout.addLayout(dir_layout)

        # === Main Content Splitter ===
        content_splitter = QSplitter(Qt.Horizontal)

        # Left Panel: Distribution Table + Statistics
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Distribution Group
        dist_group = QGroupBox("Class Distribution")
        dist_layout = QVBoxLayout(dist_group)

        # Table
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(5)
        self.class_table.setHorizontalHeaderLabels(["Include", "Class", "Count", "%", "Status"])
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.class_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.class_table.setSortingEnabled(True)
        dist_layout.addWidget(self.class_table)

        # Table buttons
        table_btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        table_btn_layout.addWidget(self.select_all_btn)
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self._deselect_all)
        table_btn_layout.addWidget(self.deselect_all_btn)
        self.apply_threshold_btn = QPushButton("Apply Threshold")
        self.apply_threshold_btn.clicked.connect(self._apply_threshold)
        table_btn_layout.addWidget(self.apply_threshold_btn)
        table_btn_layout.addStretch()
        dist_layout.addLayout(table_btn_layout)

        left_layout.addWidget(dist_group)

        # Statistics Group
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        self.stats_labels = {}
        stats_items = [
            ("Total Classes:", "total_classes"),
            ("Total Samples:", "total_samples"),
            ("Imbalance Ratio:", "imbalance_ratio"),
            ("Min Count:", "min_count"),
            ("Max Count:", "max_count"),
            ("Median:", "median"),
            ("Mean:", "mean"),
            ("Classes < Threshold:", "below_threshold")
        ]
        for i, (label, key) in enumerate(stats_items):
            row, col = i // 2, (i % 2) * 2
            stats_layout.addWidget(QLabel(label), row, col)
            self.stats_labels[key] = QLabel("--")
            stats_layout.addWidget(self.stats_labels[key], row, col + 1)
        left_layout.addWidget(stats_group)

        content_splitter.addWidget(left_widget)

        # Right Panel: Filtering, Merging, Balancing, Preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Filtering Group
        filter_group = QGroupBox("Filtering")
        filter_layout = QVBoxLayout(filter_group)
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Minimum samples per class:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 10000)
        self.threshold_spin.setValue(self.min_threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()
        filter_layout.addLayout(threshold_layout)
        self.filter_info_label = QLabel("")
        self.filter_info_label.setWordWrap(True)
        filter_layout.addWidget(self.filter_info_label)
        right_layout.addWidget(filter_group)

        # Merging Group
        merge_group = QGroupBox("Class Merging")
        merge_layout = QVBoxLayout(merge_group)

        # Add merge row
        add_merge_layout = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(150)
        add_merge_layout.addWidget(self.source_combo)
        add_merge_layout.addWidget(QLabel("→"))
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("Target class name")
        self.target_edit.setMinimumWidth(150)
        add_merge_layout.addWidget(self.target_edit)
        self.add_merge_btn = QPushButton("+ Add")
        self.add_merge_btn.clicked.connect(self._add_merge)
        add_merge_layout.addWidget(self.add_merge_btn)
        add_merge_layout.addStretch()
        merge_layout.addLayout(add_merge_layout)

        # Current merges list
        merge_layout.addWidget(QLabel("Current Merges:"))
        self.merge_list = QListWidget()
        self.merge_list.setMaximumHeight(100)
        merge_layout.addWidget(self.merge_list)

        # Merge buttons
        merge_btn_layout = QHBoxLayout()
        self.remove_merge_btn = QPushButton("Remove Selected")
        self.remove_merge_btn.clicked.connect(self._remove_merge)
        merge_btn_layout.addWidget(self.remove_merge_btn)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.MERGE_PRESETS.keys()))
        merge_btn_layout.addWidget(self.preset_combo)
        self.load_preset_btn = QPushButton("Load Preset")
        self.load_preset_btn.clicked.connect(self._load_preset)
        merge_btn_layout.addWidget(self.load_preset_btn)
        merge_btn_layout.addStretch()
        merge_layout.addLayout(merge_btn_layout)

        right_layout.addWidget(merge_group)

        # Balancing Group
        balance_group = QGroupBox("Balancing Strategy")
        balance_layout = QVBoxLayout(balance_group)

        self.balance_btn_group = QButtonGroup(self)
        strategies = [
            ("oversample_to_max", "Oversample to max class count"),
            ("oversample_to_median", "Oversample to median (Recommended)"),
            ("oversample_to_target", "Oversample to target count:"),
            ("undersample_to_min", "Undersample to min class count")
        ]
        for i, (value, label) in enumerate(strategies):
            radio = QRadioButton(label)
            radio.setProperty("strategy", value)
            self.balance_btn_group.addButton(radio, i)
            if value == self.balancing_strategy:
                radio.setChecked(True)
            balance_layout.addWidget(radio)

        # Target count input
        target_layout = QHBoxLayout()
        target_layout.addSpacing(20)
        self.target_spin = QSpinBox()
        self.target_spin.setRange(10, 100000)
        self.target_spin.setValue(self.target_count)
        self.target_spin.valueChanged.connect(self._update_preview)
        target_layout.addWidget(self.target_spin)
        target_layout.addStretch()
        balance_layout.addLayout(target_layout)

        # Max per class
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max samples per class (0 = no limit):"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(0, 100000)
        self.max_spin.setValue(self.max_per_class)
        self.max_spin.valueChanged.connect(self._update_preview)
        max_layout.addWidget(self.max_spin)
        max_layout.addStretch()
        balance_layout.addLayout(max_layout)

        self.balance_btn_group.buttonClicked.connect(self._on_balance_strategy_changed)

        right_layout.addWidget(balance_group)

        # Preview Group
        preview_group = QGroupBox("Preview (After Filtering + Merging + Balancing)")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(5)
        self.preview_table.setHorizontalHeaderLabels(["Class", "Original", "After Merge", "Final", "Action"])
        self.preview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.preview_table.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_table)

        # Preview summary
        self.preview_summary = QLabel("")
        preview_layout.addWidget(self.preview_summary)

        right_layout.addWidget(preview_group)

        content_splitter.addWidget(right_widget)

        # Set splitter sizes (50/50)
        content_splitter.setSizes([500, 500])

        main_layout.addWidget(content_splitter, stretch=1)

        # === Bottom Buttons ===
        btn_layout = QHBoxLayout()
        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self._save_config)
        btn_layout.addWidget(self.save_config_btn)
        self.load_config_btn = QPushButton("Load Config")
        self.load_config_btn.clicked.connect(self._load_config)
        btn_layout.addWidget(self.load_config_btn)
        btn_layout.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        self.proceed_btn = QPushButton("Proceed to Generate →")
        self.proceed_btn.clicked.connect(self.accept)
        self.proceed_btn.setEnabled(False)
        btn_layout.addWidget(self.proceed_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def _browse_directory(self):
        """Open directory browser."""
        initial = self.directory_edit.text() or os.getcwd()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Input Directory", initial
        )
        if directory:
            self.directory_edit.setText(directory)
            self._scan_directory()

    def _scan_directory(self):
        """Scan directory for class folders and count samples."""
        directory = self.directory_edit.text().strip()
        if not directory or not os.path.exists(directory):
            QMessageBox.warning(self, "Invalid Directory", "Please select a valid directory.")
            return

        ClassAnalysisWindow.last_directory = directory
        self.input_directory = directory
        self.class_distribution = {}

        # Scan subdirectories
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    # Count .npy files
                    npy_files = [f for f in os.listdir(item_path) if f.endswith('.npy')]
                    if npy_files:
                        self.class_distribution[item] = len(npy_files)
        except Exception as e:
            QMessageBox.critical(self, "Scan Error", f"Failed to scan directory:\n{str(e)}")
            return

        if not self.class_distribution:
            QMessageBox.warning(
                self, "No Data Found",
                "No class directories with .npy files found.\n\n"
                "Expected structure:\n"
                "  input_dir/\n"
                "    ├── ClassName1/\n"
                "    │   └── *.npy\n"
                "    └── ClassName2/\n"
                "        └── *.npy"
            )
            return

        # Initialize selected classes (all selected by default)
        self.selected_classes = {name: True for name in self.class_distribution}

        # Update UI
        self._update_class_table()
        self._update_statistics()
        self._update_source_combo()
        self._update_filter_info()
        self._update_preview()

        self.proceed_btn.setEnabled(True)

    def _update_class_table(self):
        """Update the class distribution table."""
        self.class_table.setSortingEnabled(False)
        self.class_table.setRowCount(len(self.class_distribution))

        total = sum(self.class_distribution.values())

        for row, (class_name, count) in enumerate(sorted(
            self.class_distribution.items(), key=lambda x: x[1], reverse=True
        )):
            # Checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(self.selected_classes.get(class_name, True))
            checkbox.stateChanged.connect(lambda state, cn=class_name: self._on_class_toggled(cn, state))
            self.class_table.setCellWidget(row, 0, checkbox)

            # Class name
            self.class_table.setItem(row, 1, QTableWidgetItem(class_name))

            # Count
            count_item = QTableWidgetItem()
            count_item.setData(Qt.DisplayRole, count)
            self.class_table.setItem(row, 2, count_item)

            # Percentage
            pct = count / total * 100 if total > 0 else 0
            pct_item = QTableWidgetItem(f"{pct:.2f}%")
            self.class_table.setItem(row, 3, pct_item)

            # Status
            status_item = QTableWidgetItem()
            if count < self.min_threshold:
                status_item.setText("Remove")
                status_item.setForeground(QBrush(QColor(255, 0, 0)))
            elif count < self.min_threshold * 2:
                status_item.setText("Low")
                status_item.setForeground(QBrush(QColor(255, 165, 0)))
            else:
                status_item.setText("OK")
                status_item.setForeground(QBrush(QColor(0, 128, 0)))
            self.class_table.setItem(row, 4, status_item)

        self.class_table.setSortingEnabled(True)

    def _update_statistics(self):
        """Update statistics display."""
        if not self.class_distribution:
            return

        counts = list(self.class_distribution.values())
        selected_counts = [c for name, c in self.class_distribution.items()
                          if self.selected_classes.get(name, True)]

        self.stats_labels["total_classes"].setText(str(len(self.class_distribution)))
        self.stats_labels["total_samples"].setText(f"{sum(counts):,}")

        if counts:
            min_c, max_c = min(counts), max(counts)
            self.stats_labels["imbalance_ratio"].setText(f"{max_c / max(min_c, 1):.0f}:1")
            self.stats_labels["min_count"].setText(f"{min_c:,}")
            self.stats_labels["max_count"].setText(f"{max_c:,}")
            self.stats_labels["median"].setText(f"{int(np.median(counts)):,}")
            self.stats_labels["mean"].setText(f"{int(np.mean(counts)):,}")

        below = sum(1 for c in counts if c < self.min_threshold)
        self.stats_labels["below_threshold"].setText(str(below))

    def _update_source_combo(self):
        """Update the source class combo box for merging."""
        self.source_combo.clear()
        self.source_combo.addItems(sorted(self.class_distribution.keys()))

    def _update_filter_info(self):
        """Update filter info label."""
        threshold = self.threshold_spin.value()
        excluded = [
            f"{name} ({count})"
            for name, count in self.class_distribution.items()
            if count < threshold
        ]
        if excluded:
            self.filter_info_label.setText(
                f"Will exclude {len(excluded)} classes: " + ", ".join(excluded[:5]) +
                (f" and {len(excluded) - 5} more" if len(excluded) > 5 else "")
            )
            self.filter_info_label.setStyleSheet("color: #cc6600;")
        else:
            self.filter_info_label.setText("No classes will be excluded.")
            self.filter_info_label.setStyleSheet("color: green;")

    def _update_merge_list(self):
        """Update the merge list display."""
        self.merge_list.clear()
        for source, target in sorted(self.merge_mappings.items()):
            self.merge_list.addItem(f"{source} → {target}")

    def _update_preview(self):
        """Calculate and display the preview."""
        if not self.class_distribution:
            return

        # Get current settings
        threshold = self.threshold_spin.value()
        strategy = self._get_selected_strategy()
        target = self.target_spin.value()
        max_cap = self.max_spin.value()

        # Step 1: Filter by selection and threshold
        filtered = {
            name: count
            for name, count in self.class_distribution.items()
            if self.selected_classes.get(name, True) and count >= threshold
        }

        # Step 2: Apply merging
        merged = {}
        for name, count in filtered.items():
            target_name = self.merge_mappings.get(name, name)
            if target_name not in merged:
                merged[target_name] = 0
            merged[target_name] += count

        # Step 3: Calculate final counts based on strategy
        if not merged:
            self.preview_summary.setText("No classes remaining after filtering!")
            return

        counts = list(merged.values())
        if strategy == "oversample_to_max":
            balance_target = max(counts)
        elif strategy == "oversample_to_median":
            balance_target = int(np.median(counts))
        elif strategy == "oversample_to_target":
            balance_target = target
        else:  # undersample_to_min
            balance_target = min(counts)

        # Apply max cap
        if max_cap > 0:
            balance_target = min(balance_target, max_cap)

        # Calculate final distribution
        final = {}
        for name, count in merged.items():
            if count >= balance_target:
                final[name] = balance_target  # Undersample
            else:
                final[name] = balance_target  # Oversample

        # Update preview table
        self.preview_table.setRowCount(len(merged))
        for row, name in enumerate(sorted(merged.keys())):
            orig_count = sum(
                c for n, c in self.class_distribution.items()
                if self.merge_mappings.get(n, n) == name and
                self.selected_classes.get(n, True) and
                c >= threshold
            )
            merged_count = merged[name]
            final_count = final[name]

            self.preview_table.setItem(row, 0, QTableWidgetItem(name))
            self.preview_table.setItem(row, 1, QTableWidgetItem(f"{orig_count:,}"))

            merge_item = QTableWidgetItem(f"{merged_count:,}")
            if merged_count != orig_count:
                merge_item.setText(f"{merged_count:,} (+{merged_count - orig_count:,})")
            self.preview_table.setItem(row, 2, merge_item)

            self.preview_table.setItem(row, 3, QTableWidgetItem(f"{final_count:,}"))

            # Action
            if merged_count < final_count:
                ratio = final_count / merged_count
                action = f"↑ Oversample {ratio:.1f}x"
            elif merged_count > final_count:
                action = "↓ Undersample"
            else:
                action = "— Unchanged"
            self.preview_table.setItem(row, 4, QTableWidgetItem(action))

        # Update summary
        total_final = sum(final.values())
        self.preview_summary.setText(
            f"Final: {len(final)} classes, {total_final:,} samples "
            f"({balance_target:,} per class)"
        )

    def _get_selected_strategy(self) -> str:
        """Get the selected balancing strategy."""
        checked = self.balance_btn_group.checkedButton()
        if checked:
            return checked.property("strategy")
        return "oversample_to_median"

    # === Event Handlers ===

    def _on_class_toggled(self, class_name: str, state: int):
        """Handle class checkbox toggle."""
        self.selected_classes[class_name] = (state == Qt.Checked)
        self._update_preview()

    def _on_threshold_changed(self, value: int):
        """Handle threshold change."""
        self.min_threshold = value
        self._update_filter_info()
        self._update_class_table()
        self._update_statistics()
        self._update_preview()

    def _on_balance_strategy_changed(self):
        """Handle balance strategy change."""
        self._update_preview()

    def _select_all(self):
        """Select all classes."""
        for name in self.class_distribution:
            self.selected_classes[name] = True
        self._update_class_table()
        self._update_preview()

    def _deselect_all(self):
        """Deselect all classes."""
        for name in self.class_distribution:
            self.selected_classes[name] = False
        self._update_class_table()
        self._update_preview()

    def _apply_threshold(self):
        """Apply threshold to select/deselect classes."""
        threshold = self.threshold_spin.value()
        for name, count in self.class_distribution.items():
            self.selected_classes[name] = (count >= threshold)
        self._update_class_table()
        self._update_preview()

    def _add_merge(self):
        """Add a merge mapping."""
        source = self.source_combo.currentText()
        target = self.target_edit.text().strip()

        if not source or not target:
            QMessageBox.warning(self, "Invalid Merge", "Please specify both source and target class.")
            return

        self.merge_mappings[source] = target
        self._update_merge_list()
        self._update_preview()
        self.target_edit.clear()

    def _remove_merge(self):
        """Remove selected merge mapping."""
        current = self.merge_list.currentItem()
        if not current:
            return

        text = current.text()
        source = text.split(" → ")[0]
        if source in self.merge_mappings:
            del self.merge_mappings[source]
            self._update_merge_list()
            self._update_preview()

    def _load_preset(self):
        """Load a merge preset."""
        preset_name = self.preset_combo.currentText()
        if preset_name in self.MERGE_PRESETS:
            self.merge_mappings = self.MERGE_PRESETS[preset_name].copy()
            self._update_merge_list()
            self._update_preview()

    def _save_config(self):
        """Save configuration to JSON file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration",
            os.path.join(self.input_directory, "class_config.json"),
            "JSON Files (*.json)"
        )
        if not filepath:
            return

        config = self.get_config()
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Saved", f"Configuration saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{str(e)}")

    def _load_config(self):
        """Load configuration from JSON file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration",
            self.input_directory or os.getcwd(),
            "JSON Files (*.json)"
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            # Apply configuration
            if "filtering" in config:
                self.threshold_spin.setValue(config["filtering"].get("min_samples_threshold", 20))
                excluded = config["filtering"].get("excluded_classes", [])
                manually_excluded = config["filtering"].get("manually_excluded", [])
                for name in self.class_distribution:
                    self.selected_classes[name] = (
                        name not in excluded and name not in manually_excluded
                    )

            if "merging" in config and config["merging"].get("enabled", False):
                self.merge_mappings = config["merging"].get("mappings", {})

            if "balancing" in config:
                strategy = config["balancing"].get("strategy", "oversample_to_median")
                for btn in self.balance_btn_group.buttons():
                    if btn.property("strategy") == strategy:
                        btn.setChecked(True)
                        break
                if config["balancing"].get("target_count"):
                    self.target_spin.setValue(config["balancing"]["target_count"])
                if config["balancing"].get("max_per_class"):
                    self.max_spin.setValue(config["balancing"]["max_per_class"])

            # Update UI
            self._update_class_table()
            self._update_merge_list()
            self._update_preview()

            QMessageBox.information(self, "Loaded", f"Configuration loaded from:\n{filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load:\n{str(e)}")

    def get_config(self) -> Dict:
        """
        Get the current configuration as a dictionary.

        Returns:
            Configuration dictionary suitable for JSON export
        """
        threshold = self.threshold_spin.value()

        # Determine excluded classes
        excluded_by_threshold = [
            name for name, count in self.class_distribution.items()
            if count < threshold
        ]
        manually_excluded = [
            name for name, selected in self.selected_classes.items()
            if not selected and name not in excluded_by_threshold
        ]

        # Calculate preview for final distribution
        filtered = {
            name: count
            for name, count in self.class_distribution.items()
            if self.selected_classes.get(name, True) and count >= threshold
        }

        merged = {}
        for name, count in filtered.items():
            target_name = self.merge_mappings.get(name, name)
            if target_name not in merged:
                merged[target_name] = 0
            merged[target_name] += count

        strategy = self._get_selected_strategy()
        target = self.target_spin.value()
        max_cap = self.max_spin.value()

        if merged:
            counts = list(merged.values())
            if strategy == "oversample_to_max":
                balance_target = max(counts)
            elif strategy == "oversample_to_median":
                balance_target = int(np.median(counts))
            elif strategy == "oversample_to_target":
                balance_target = target
            else:
                balance_target = min(counts)

            if max_cap > 0:
                balance_target = min(balance_target, max_cap)
        else:
            balance_target = 0

        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "source_directory": self.input_directory,

            "original_distribution": self.class_distribution.copy(),

            "filtering": {
                "min_samples_threshold": threshold,
                "excluded_classes": excluded_by_threshold,
                "manually_excluded": manually_excluded
            },

            "merging": {
                "enabled": bool(self.merge_mappings),
                "mappings": self.merge_mappings.copy()
            },

            "balancing": {
                "strategy": strategy,
                "target_count": target if strategy == "oversample_to_target" else None,
                "max_per_class": max_cap if max_cap > 0 else None
            },

            "preview": {
                "final_class_count": len(merged),
                "final_sample_count": balance_target * len(merged) if merged else 0,
                "samples_per_class": balance_target
            }
        }
