# gui/dialog_boxes/dynamic_dialog.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                             QDialogButtonBox, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QCheckBox, QHBoxLayout, QPushButton, QFileDialog, QWidget)
from typing import Dict, Any


class DynamicDialog(QDialog):
    """
    A dynamically generated dialog box based on parameter definitions.

    This dialog automatically creates appropriate input widgets based on
    the parameter schema provided by plugins.
    """

    def __init__(self, title: str, parameter_schema: Dict[str, Dict[str, Any]], parent=None):
        """
        Initialize the dialog with a title and parameter schema.

        Args:
            title (str): The dialog box title
            parameter_schema (Dict[str, Dict[str, Any]]): Schema defining parameters and their properties
            parent: Parent widget, if any
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.parameter_schema = parameter_schema
        self.param_widgets = {}
        self.params = {}
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog UI with widgets based on the parameter schema."""
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Create widgets based on parameter schema
        for param_name, param_info in self.parameter_schema.items():
            param_type = param_info.get("type", "string")
            default_value = param_info.get("default", "")
            label_text = param_info.get("label", param_name)
            tooltip = param_info.get("description", "")

            if param_type == "int":
                widget = QSpinBox()
                # Set min/max range BEFORE setting value to avoid clamping issues
                if "min" in param_info:
                    widget.setMinimum(param_info["min"])
                if "max" in param_info:
                    widget.setMaximum(param_info["max"])
                widget.setValue(default_value)
            elif param_type == "float":
                widget = QDoubleSpinBox()
                # Set min/max range BEFORE setting value to avoid clamping issues
                if "min" in param_info:
                    widget.setMinimum(param_info["min"])
                if "max" in param_info:
                    widget.setMaximum(param_info["max"])
                # Set decimal precision (default 4, but allow custom)
                decimals = param_info.get("decimals", 4)
                widget.setDecimals(decimals)
                widget.setValue(default_value)
            elif param_type == "dropdown":
                widget = QComboBox()
                options = param_info.get("options", {})
                for value, display_text in options.items():
                    widget.addItem(display_text, value)

                # Set the default value if provided
                if default_value and isinstance(default_value, str):
                    index = widget.findData(default_value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
            elif param_type == "choice":
                # Handle choice type (list of options)
                widget = QComboBox()
                widget.setEditable(True)  # Allow custom text input
                options = param_info.get("options", [])

                for option in options:
                    # For choice, option and display value are the same
                    widget.addItem(str(option), option)

                # Set default if provided
                if default_value:
                    index = widget.findData(default_value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                    else:
                        # If default not in options, set it as custom text
                        widget.setCurrentText(str(default_value))
            elif param_type == "bool":
                # Handle boolean type
                widget = QCheckBox()
                widget.setChecked(bool(default_value))
            elif param_type == "directory":
                # Handle directory type with browse button
                dir_widget = QWidget()
                dir_layout = QHBoxLayout()
                dir_layout.setContentsMargins(0, 0, 0, 0)

                line_edit = QLineEdit()
                line_edit.setText(str(default_value))

                browse_button = QPushButton("Browse...")
                browse_button.clicked.connect(
                    lambda checked, le=line_edit: self._browse_directory(le)
                )

                dir_layout.addWidget(line_edit)
                dir_layout.addWidget(browse_button)
                dir_widget.setLayout(dir_layout)

                widget = dir_widget
                # Store the line_edit as the actual widget for value retrieval
                self.param_widgets[param_name] = line_edit
            else:  # Default to string
                widget = QLineEdit()
                widget.setText(str(default_value))

            # Set tooltip if provided
            if tooltip:
                widget.setToolTip(tooltip)

            # For directory type, widget is already added to param_widgets above
            if param_type == "directory":
                form_layout.addRow(QLabel(label_text), widget)
            else:
                form_layout.addRow(QLabel(label_text), widget)
                self.param_widgets[param_name] = widget

        layout.addLayout(form_layout)

        # Add button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameter values from the dialog widgets.

        Returns:
            Dict[str, Any]: Dictionary of parameter names to their values
        """
        for param_name, widget in self.param_widgets.items():
            param_type = self.parameter_schema[param_name].get("type", "string")

            if param_type == "int":
                self.params[param_name] = widget.value()
            elif param_type == "float":
                self.params[param_name] = widget.value()
            elif param_type == "dropdown":
                self.params[param_name] = widget.currentData()  # Get the data value, not the display text
            elif param_type == "choice":
                # For editable combobox, get the current text (allows custom input)
                self.params[param_name] = widget.currentText()
            elif param_type == "bool":
                self.params[param_name] = widget.isChecked()
            else:  # String or directory
                self.params[param_name] = widget.text()

        return self.params

    def _browse_directory(self, line_edit: QLineEdit):
        """
        Open a directory browser dialog and update the line edit.

        Args:
            line_edit: The QLineEdit widget to update with the selected directory
        """
        current_dir = line_edit.text() if line_edit.text() else ""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            current_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if directory:
            line_edit.setText(directory)