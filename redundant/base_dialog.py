# TODO: Docstrings

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class BaseDialog(QDialog):
    """
    Base class for analysis parameter dialogs.
    Provides common functionality like button setup and parameter collection.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.cancel_button = None
        self.ok_button = None
        self.button_layout = None
        self.content_layout = None
        self.layout = None
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.MSWindowsFixedSizeDialogHint)
        self.params = {}
        self.setup_ui()

    def setup_ui(self):
        """
        Common UI setup for dialogs.
        Subclasses can extend this method for specific parameter inputs.
        """

        self.resize(400, 200)  # Standard size for a dialog box
        self.layout = QVBoxLayout(self)

        # Set font size for the entire dialog
        font = QFont()
        font.setPointSize(8)  # Adjust the size as needed
        self.setFont(font)

        # Add content layout (this will be subclass-specific)
        self.content_layout = QVBoxLayout()
        self.layout.addLayout(self.content_layout)

        # Add OK/Cancel buttons in a horizontal layout
        self.button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.ok_button.clicked.connect(self.validate_and_accept)
        self.cancel_button.clicked.connect(self.reject)

        # Align buttons to the right
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

    def validate_and_accept(self):
        """
        Validate parameters and accept the dialog.
        Subclasses can override this for specific validation logic.
        """
        if not self.validate_params():
            QMessageBox.warning(self, "Validation Error", "Invalid parameters. Please check your input.")
            return
        self.accept()

    def validate_params(self) -> bool:
        """
        Default parameter validation.
        Subclasses should override this for specific validation logic.
        """
        return True

    def get_parameters(self) -> dict:
        """
        Return the collected parameters in a consistent format.
        """
        return self.params
