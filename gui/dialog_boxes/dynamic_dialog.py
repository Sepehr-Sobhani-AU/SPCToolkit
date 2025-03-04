# gui/dialog_boxes/dynamic_dialog.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                             QDialogButtonBox, QLabel, QSpinBox, QDoubleSpinBox)
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
                widget.setValue(default_value)
                if "min" in param_info:
                    widget.setMinimum(param_info["min"])
                if "max" in param_info:
                    widget.setMaximum(param_info["max"])
            elif param_type == "float":
                widget = QDoubleSpinBox()
                widget.setValue(default_value)
                if "min" in param_info:
                    widget.setMinimum(param_info["min"])
                if "max" in param_info:
                    widget.setMaximum(param_info["max"])
                widget.setDecimals(4)  # Increased precision for floating-point values
            else:  # Default to string
                widget = QLineEdit()
                widget.setText(str(default_value))

            # Set tooltip if provided
            if tooltip:
                widget.setToolTip(tooltip)

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
            else:  # String
                self.params[param_name] = widget.text()

        return self.params