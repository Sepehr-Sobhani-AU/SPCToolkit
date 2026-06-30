# gui/dialog_boxes/query_builder_dialog.py
"""
ArcGIS-style "Select By Attributes" builder for a single branch.

Unlike the schema-driven ``DynamicDialog``, the number of condition rows is
variable, so this is a hand-built ``QDialog``. It exposes the same contract the
analysis flow expects -- ``exec_()`` plus ``get_parameters()`` -- returning a
JSON-serialisable query (``new_branch_name``, ``combiner``, ``clauses``) that
``query_select`` evaluates and that a saved pipeline can replay verbatim.

Only built-in Qt widget signals (``clicked``, ``currentIndexChanged``) are used
-- the project forbids custom signals, not these.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QRadioButton, QButtonGroup, QDialogButtonBox,
    QWidget, QGroupBox, QScrollArea,
)

from services.point_fields import enumerate_fields, resolve_field
from services.attribute_query import OPERATORS, evaluate_query


class _ConditionRow(QWidget):
    """One field / operator / value(s) row, with a remove button."""

    def __init__(self, fields, operators, on_remove, on_field_changed):
        super().__init__()
        self._fields = fields
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.field_combo = QComboBox()
        for key, label in fields.items():
            self.field_combo.addItem(label, key)

        self.op_combo = QComboBox()
        for key, label in operators.items():
            self.op_combo.addItem(label, key)

        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("value")
        self.value2_edit = QLineEdit()
        self.value2_edit.setPlaceholderText("and")
        self.value2_edit.setVisible(False)

        self.remove_btn = QPushButton("−")
        self.remove_btn.setFixedWidth(28)
        self.remove_btn.setToolTip("Remove this condition")

        layout.addWidget(self.field_combo, 3)
        layout.addWidget(self.op_combo, 2)
        layout.addWidget(self.value_edit, 2)
        layout.addWidget(self.value2_edit, 2)
        layout.addWidget(self.remove_btn)

        self.op_combo.currentIndexChanged.connect(self._sync_between)
        self.field_combo.currentIndexChanged.connect(
            lambda _idx: on_field_changed(self))
        self.remove_btn.clicked.connect(lambda: on_remove(self))

    def _sync_between(self):
        self.value2_edit.setVisible(self.op_combo.currentData() == "between")

    def field_key(self):
        return self.field_combo.currentData()

    def clause(self):
        return {
            "field": self.field_combo.currentData(),
            "op": self.op_combo.currentData(),
            "value": self.value_edit.text().strip(),
            "value2": self.value2_edit.text().strip() or None,
        }

    def set_clause(self, clause):
        """Restore this row's widgets from a saved clause dict (best-effort:
        unknown fields/operators are skipped so a query saved against a
        different cloud still loads what it can)."""
        field_idx = self.field_combo.findData(clause.get("field"))
        if field_idx >= 0:
            self.field_combo.setCurrentIndex(field_idx)
        op_idx = self.op_combo.findData(clause.get("op"))
        if op_idx >= 0:
            self.op_combo.setCurrentIndex(op_idx)
        self.value_edit.setText(str(clause.get("value", "")))
        value2 = clause.get("value2")
        if value2 is not None:
            self.value2_edit.setText(str(value2))
        self._sync_between()


class QueryBuilderDialog(QDialog):
    """Build an attribute query against ``pc`` (the branch's point cloud)."""

    def __init__(self, node, pc, parent=None, initial_params=None):
        super().__init__(parent)
        self.setWindowTitle("Select By Attributes")
        self.setMinimumWidth(560)
        self._node = node
        self._pc = pc
        self._fields = enumerate_fields(node, pc)
        self._operators = OPERATORS
        self._rows = []

        outer = QVBoxLayout(self)

        if not self._fields:
            outer.addWidget(QLabel("This branch exposes no queryable fields."))

        # Combiner ---------------------------------------------------------
        combiner_box = QGroupBox("Match")
        cb_layout = QHBoxLayout(combiner_box)
        self.all_radio = QRadioButton("ALL conditions (AND)")
        self.any_radio = QRadioButton("ANY condition (OR)")
        self.all_radio.setChecked(True)
        self._combiner_group = QButtonGroup(self)
        self._combiner_group.addButton(self.all_radio)
        self._combiner_group.addButton(self.any_radio)
        cb_layout.addWidget(self.all_radio)
        cb_layout.addWidget(self.any_radio)
        cb_layout.addStretch()
        outer.addWidget(combiner_box)

        # Conditions (scrollable so many rows don't grow the dialog) --------
        self._rows_layout = QVBoxLayout()
        rows_host = QWidget()
        rows_host.setLayout(self._rows_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_host)
        scroll.setMinimumHeight(120)
        outer.addWidget(scroll)

        add_btn = QPushButton("+ Add condition")
        add_btn.clicked.connect(self._add_row)
        outer.addWidget(add_btn)

        # Field range hint + match preview ---------------------------------
        self._hint_label = QLabel("")
        self._hint_label.setStyleSheet("color: gray;")
        outer.addWidget(self._hint_label)

        preview_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview match count")
        self.preview_btn.clicked.connect(self._update_preview)
        self._preview_label = QLabel("")
        preview_row.addWidget(self.preview_btn)
        preview_row.addWidget(self._preview_label)
        preview_row.addStretch()
        outer.addLayout(preview_row)
        if self._pc is None:
            self.preview_btn.setEnabled(False)
            self.preview_btn.setToolTip(
                "Preview unavailable until the branch is reconstructed.")

        # New branch name --------------------------------------------------
        form = QFormLayout()
        self.name_edit = QLineEdit("Query Result")
        form.addRow("New branch name:", self.name_edit)
        outer.addLayout(form)

        # Buttons ----------------------------------------------------------
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # Restore the previous query if one was passed; otherwise start with a
        # single empty condition row.
        if self._fields and not self._restore(initial_params):
            self._add_row()

    def _restore(self, initial_params) -> bool:
        """Pre-populate the dialog from a previous run's params. Returns True if
        anything was restored, False otherwise (so the caller adds a blank row)."""
        if not initial_params:
            return False

        if initial_params.get("combiner") == "OR":
            self.any_radio.setChecked(True)
        else:
            self.all_radio.setChecked(True)

        name = initial_params.get("new_branch_name")
        if name:
            self.name_edit.setText(str(name))

        clauses = initial_params.get("clauses") or []
        for clause in clauses:
            self._add_row()
            self._rows[-1].set_clause(clause)

        # Always leave at least one row so the dialog isn't empty.
        if not self._rows:
            self._add_row()
        return True

    # --- rows ------------------------------------------------------------

    def _add_row(self):
        row = _ConditionRow(
            self._fields, self._operators,
            on_remove=self._remove_row,
            on_field_changed=self._show_field_range,
        )
        self._rows.append(row)
        self._rows_layout.addWidget(row)
        self._show_field_range(row)

    def _remove_row(self, row):
        if row in self._rows:
            self._rows.remove(row)
            row.setParent(None)
            row.deleteLater()

    def _show_field_range(self, row):
        """Show the selected field's value range as a hint, so the user knows
        what to type. Cheap: reads the already-loaded representative cloud."""
        if self._pc is None:
            return
        arr = resolve_field(self._pc, row.field_key())
        if arr is None:
            return
        arr = np.asarray(arr, dtype=np.float64).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        label = row.field_combo.currentText()
        self._hint_label.setText(f"{label}: range {lo:.4g} … {hi:.4g}")
        row.value_edit.setPlaceholderText(f"{lo:.4g} … {hi:.4g}")

    # --- preview & result ------------------------------------------------

    def _update_preview(self):
        if self._pc is None:
            return
        params = self.get_parameters()
        mask = evaluate_query(self._pc, params["clauses"], params["combiner"])
        n = int(mask.size)
        k = int(mask.sum())
        self._preview_label.setText(f"{k:,} of {n:,} points match")

    def get_parameters(self):
        clauses = [r.clause() for r in self._rows]
        # Keep only rows with a value entered, so an empty trailing row is ignored.
        clauses = [c for c in clauses if c["value"] != ""]
        combiner = "OR" if self.any_radio.isChecked() else "AND"
        return {
            "new_branch_name": self.name_edit.text().strip() or "Query Result",
            "combiner": combiner,
            "clauses": clauses,
        }
