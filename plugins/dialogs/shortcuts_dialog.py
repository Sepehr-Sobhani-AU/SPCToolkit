"""
Reference dialog listing input bindings (keyboard or mouse).

Used by the Help > Keyboard and Mouse Handling plugins. The dialog is purely
informational: it renders a read-only two-column table of
"input combination -> action" rows, grouped into titled sections.
"""

from typing import List, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

# A section is a title plus its (combination, action) rows.
Section = Tuple[str, List[Tuple[str, str]]]


class ShortcutsDialog(QDialog):
    """
    Read-only table of input bindings, grouped into sections.

    Args:
        parent: Parent widget (the main window).
        title: Window title.
        sections: List of (section_title, [(combination, action), ...]).
    """

    def __init__(self, parent, title: str, sections: List[Section]):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.resize(780, 620)

        layout = QVBoxLayout(self)

        self._table = QTableWidget(self)
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Input", "Action"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setFocusPolicy(Qt.NoFocus)
        self._table.setWordWrap(True)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # Row height follows the wrapped text. The Action column only reaches its
        # final width once the dialog is shown, so heights are recomputed on resize
        # as well (see resizeEvent) — otherwise every row keeps the tall height
        # computed while the column was still narrow.
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self._populate(sections)

        layout.addWidget(self._table)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def resizeEvent(self, event):
        """Re-wrap the Action column and shrink rows back to the text they hold."""
        super().resizeEvent(event)
        self._table.resizeRowsToContents()

    def _populate(self, sections: List[Section]):
        """Fill the table with section headers followed by their binding rows."""
        row_count = sum(1 + len(rows) for _, rows in sections)
        self._table.setRowCount(row_count)

        row = 0
        for section_title, rows in sections:
            header = QTableWidgetItem(section_title)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            self._table.setItem(row, 0, header)
            # Section titles span both columns
            self._table.setSpan(row, 0, 1, 2)
            row += 1

            for combination, action in rows:
                combination_item = QTableWidgetItem(combination)
                combination_font = combination_item.font()
                combination_font.setBold(True)
                combination_item.setFont(combination_font)
                combination_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)

                action_item = QTableWidgetItem(action)
                action_item.setTextAlignment(Qt.AlignLeft | Qt.AlignTop)

                self._table.setItem(row, 0, combination_item)
                self._table.setItem(row, 1, action_item)
                row += 1

        self._table.resizeRowsToContents()
