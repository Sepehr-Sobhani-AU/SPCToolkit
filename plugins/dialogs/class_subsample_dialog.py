"""
Per-Class Subsample Ratio Dialog.

Displays a table of semantic classes with color swatches, point percentages,
adjustable subsample ratios, and a live preview of resulting point counts.
Columns are sortable by clicking headers. Used by the SemanticKITTI import plugin.
"""

from typing import Dict, Tuple, Optional, List

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QDoubleSpinBox, QDialogButtonBox, QPushButton, QHeaderView, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush


class ClassSubsampleDialog(QDialog):
    """Dialog with a sortable table listing each class, its point percentage, ratio, and resulting count."""

    def __init__(self, class_names: Dict[int, str], class_colors: Dict[int, Tuple[int, int, int]],
                 class_counts: Optional[Dict[int, int]] = None, parent=None):
        """
        Args:
            class_names: {label_id: "class_name"} for all classes to show
            class_colors: {label_id: (R, G, B)} with values 0-255
            class_counts: {label_id: point_count} from scanning label files.
                          If provided, shows % column, result column, and presets ratios.
            parent: parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Per-Class Subsample Ratios")
        self.class_names = class_names
        self.class_colors = class_colors
        self.class_counts = class_counts or {}
        self.ratio_spinboxes: Dict[int, QDoubleSpinBox] = {}
        self.row_label_ids: List[int] = []  # label_id for each current row
        self.total_label = None
        self.has_counts = bool(self.class_counts) and sum(self.class_counts.values()) > 0
        self._sort_column = -1
        self._sort_ascending = True
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Info label
        info = QLabel("Set subsample ratio for each class (0.0 = remove, 1.0 = keep all):")
        layout.addWidget(info)

        # Preset buttons
        btn_layout = QHBoxLayout()
        for label, value in [("All 1.0", 1.0), ("All 0.5", 0.5), ("All 0.1", 0.1), ("All 0.0", 0.0)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, v=value: self._set_all(v))
            btn_layout.addWidget(btn)
        if self.has_counts:
            balance_btn = QPushButton("Balance")
            balance_btn.setToolTip("Set ratios to balance class sizes toward the median")
            balance_btn.clicked.connect(self._set_balanced)
            btn_layout.addWidget(balance_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Determine which label_ids to show
        if self.has_counts:
            self.row_label_ids = sorted(lid for lid in self.class_counts.keys() if self.class_counts[lid] > 0)
        else:
            self.row_label_ids = sorted(self.class_names.keys())

        # Compute balanced ratios as initial values
        balanced_ratios = self._compute_balanced_ratios() if self.has_counts else {}

        # Create spinboxes first (before table, so they persist across sorts)
        for label_id in self.row_label_ids:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(balanced_ratios.get(label_id, 1.0))
            spin.valueChanged.connect(self._update_results)
            self.ratio_spinboxes[label_id] = spin

        # Table
        n_cols = 4 if self.has_counts else 2
        self.table = QTableWidget(len(self.row_label_ids), n_cols)
        if self.has_counts:
            self.table.setHorizontalHeaderLabels(["Class", "Points %", "Ratio", "Result"])
        else:
            self.table.setHorizontalHeaderLabels(["Class", "Ratio"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, n_cols):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setSortingEnabled(False)

        # Enable sort on header click
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)

        # Populate rows
        self._populate_table()

        layout.addWidget(self.table)

        # Total points label (only when counts available)
        if self.has_counts:
            self.total_label = QLabel()
            self.total_label.setAlignment(Qt.AlignRight)
            layout.addWidget(self.total_label)
            self._update_results()

        # OK / Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.resize(520, 600)

    def _populate_table(self):
        """Fill the table rows based on current self.row_label_ids order."""
        total_points = sum(self.class_counts.values()) if self.has_counts else 0

        for row, label_id in enumerate(self.row_label_ids):
            # Class name cell with color background
            name = self.class_names.get(label_id, f"class_{label_id}")
            item = QTableWidgetItem(f"  {name}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, name.lower())  # sort key
            rgb = self.class_colors.get(label_id, (128, 128, 128))
            item.setBackground(QBrush(QColor(*rgb)))
            brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            item.setForeground(QBrush(QColor(0, 0, 0) if brightness > 128 else QColor(255, 255, 255)))
            self.table.setItem(row, 0, item)

            if self.has_counts:
                # Percentage column
                count = self.class_counts.get(label_id, 0)
                pct = (count / total_points) * 100 if total_points > 0 else 0
                pct_item = QTableWidgetItem(f"{pct:.2f}%")
                pct_item.setFlags(pct_item.flags() & ~Qt.ItemIsEditable)
                pct_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                pct_item.setData(Qt.UserRole, pct)  # numeric sort key
                self.table.setItem(row, 1, pct_item)

                # Ratio spinbox (col 2)
                self.table.setCellWidget(row, 2, self.ratio_spinboxes[label_id])

                # Result column (col 3)
                ratio = self.ratio_spinboxes[label_id].value()
                result = int(count * ratio)
                result_item = QTableWidgetItem(f"{result:,}")
                result_item.setFlags(result_item.flags() & ~Qt.ItemIsEditable)
                result_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                result_item.setData(Qt.UserRole, result)  # numeric sort key
                self.table.setItem(row, 3, result_item)
            else:
                # Ratio spinbox (col 1)
                self.table.setCellWidget(row, 1, self.ratio_spinboxes[label_id])

    def _on_header_clicked(self, col: int):
        """Sort table by the clicked column."""
        # Toggle direction if same column clicked again
        if col == self._sort_column:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = col
            self._sort_ascending = True

        ratio_col = 2 if self.has_counts else 1

        # Build (sort_key, label_id) pairs
        def sort_key(label_id):
            row = self.row_label_ids.index(label_id)
            if col == ratio_col:
                return self.ratio_spinboxes[label_id].value()
            item = self.table.item(row, col)
            if item and item.data(Qt.UserRole) is not None:
                return item.data(Qt.UserRole)
            return 0

        new_order = sorted(self.row_label_ids, key=sort_key, reverse=not self._sort_ascending)
        self.row_label_ids = new_order
        self._populate_table()
        self._update_results()

    def _update_results(self):
        """Update the Result column and total label based on current ratios."""
        if not self.has_counts:
            return
        total_result = 0
        total_original = sum(self.class_counts.get(lid, 0) for lid in self.row_label_ids)
        for row, label_id in enumerate(self.row_label_ids):
            count = self.class_counts.get(label_id, 0)
            ratio = self.ratio_spinboxes[label_id].value()
            result = int(count * ratio)
            total_result += result
            item = self.table.item(row, 3)
            if item:
                item.setText(f"{result:,}")
                item.setData(Qt.UserRole, result)
        if self.total_label:
            if total_original > 0:
                self.total_label.setText(
                    f"Total: {total_result:,} / {total_original:,} points "
                    f"({total_result / total_original * 100:.1f}%)"
                )
            else:
                self.total_label.setText("Total: 0")

    def _compute_balanced_ratios(self) -> Dict[int, float]:
        """Compute ratios to balance classes toward the median size.

        Classes at or below the median get ratio 1.0.
        Classes above the median get ratio = median / count (minimum 0.01).
        """
        counts = {lid: c for lid, c in self.class_counts.items() if c > 0}
        if not counts:
            return {}
        sorted_counts = sorted(counts.values())
        median_count = sorted_counts[len(sorted_counts) // 2]
        ratios = {}
        for lid, c in counts.items():
            r = median_count / c
            if r >= 1.0:
                ratios[lid] = 1.0
            else:
                ratios[lid] = max(0.01, round(r, 2))
        return ratios

    def _set_all(self, value: float):
        for spin in self.ratio_spinboxes.values():
            spin.setValue(value)

    def _set_balanced(self):
        ratios = self._compute_balanced_ratios()
        for label_id, spin in self.ratio_spinboxes.items():
            spin.setValue(ratios.get(label_id, 1.0))

    def get_ratios(self) -> Dict[int, float]:
        """Return {label_id: ratio} for all classes."""
        return {label_id: spin.value() for label_id, spin in self.ratio_spinboxes.items()}
