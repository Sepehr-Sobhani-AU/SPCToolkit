"""
Shift Dialog

Dialog for reviewing and editing coordinate shifts on export.
"""

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit,
                               QDialogButtonBox, QGroupBox, QGridLayout)


class ShiftDialog(QDialog):
    """Dialog showing the detected coordinate shift, editable by the user."""

    def __init__(self, parent, shift: np.ndarray):
        super().__init__(parent)
        self.setWindowTitle("Coordinate Shift")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)

        info = QLabel(
            "On import, the point cloud was shifted to the origin.\n"
            "The detected shift to restore original coordinates is shown below.\n"
            "Adjust if needed, or set all to 0 for no shift."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        group = QGroupBox("Shift (added to exported points)")
        grid = QGridLayout(group)

        self._edits = {}
        for row, (axis, val) in enumerate(zip(("X", "Y", "Z"), shift)):
            grid.addWidget(QLabel(f"{axis}:"), row, 0)
            edit = QLineEdit(f"{val:.6f}")
            self._edits[axis] = edit
            grid.addWidget(edit, row, 1)

        layout.addWidget(group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_shift(self) -> np.ndarray:
        values = []
        for axis in ("X", "Y", "Z"):
            try:
                values.append(float(self._edits[axis].text()))
            except ValueError:
                values.append(0.0)
        return np.array(values, dtype=np.float64)
