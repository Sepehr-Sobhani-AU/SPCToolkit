# gui/dialog_boxes/dynamic_dialog.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                             QDialogButtonBox, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QCheckBox, QHBoxLayout, QPushButton, QFileDialog, QWidget,
                             QInputDialog, QMessageBox, QFrame)
from typing import Dict, Any


class DynamicDialog(QDialog):
    """
    A dynamically generated dialog box based on parameter definitions.

    This dialog automatically creates appropriate input widgets based on
    the parameter schema provided by plugins.
    """

    def __init__(self, title: str, parameter_schema: Dict[str, Dict[str, Any]], parent=None,
                 plugin_name: str = None):
        """
        Initialize the dialog with a title and parameter schema.

        Args:
            title (str): The dialog box title
            parameter_schema (Dict[str, Dict[str, Any]]): Schema defining parameters and their properties
            parent: Parent widget, if any
            plugin_name: The plugin these are settings of. When given, the dialog
                offers named presets of them (``services/preset_service.py``).
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.parameter_schema = parameter_schema
        self.plugin_name = plugin_name
        self.preset_combo = None
        self.param_widgets = {}
        self.param_labels = {}
        self.param_containers = {}  # For directory type: the parent QWidget
        self.params = {}
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog UI with widgets based on the parameter schema."""
        layout = QVBoxLayout()
        if self.plugin_name:
            layout.addLayout(self._build_preset_row())
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)
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
                # Set decimal precision (default 3, but allow custom)
                decimals = param_info.get("decimals", 3)
                widget.setDecimals(decimals)
                widget.setSingleStep(10 ** -decimals)
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
            elif param_type == "colormap":
                # Dropdown whose items carry a gradient preview of each colormap,
                # so the user recognises it by colour, not just by name.
                widget = self._build_colormap_combo(param_info, default_value)
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
            elif param_type == "info":
                # Read-only text label (for formulas, notes, etc.)
                widget = QLabel(str(default_value))
                widget.setWordWrap(True)
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

            # Create label and add row
            label_widget = QLabel(label_text)
            self.param_labels[param_name] = label_widget

            if param_type == "directory":
                form_layout.addRow(label_widget, widget)
                self.param_containers[param_name] = widget  # The QWidget container
            else:
                form_layout.addRow(label_widget, widget)
                self.param_widgets[param_name] = widget

        # Wire up enabled_by / disabled_by dependencies between widgets
        self._setup_dependencies()

        layout.addLayout(form_layout)

        # Add button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    # ------------------------------------------------------------------ #
    # Presets                                                            #
    # ------------------------------------------------------------------ #

    def _build_preset_row(self):
        """Preset: [names v] [Load] [Save...] [Delete]"""
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(160)
        self.preset_combo.setToolTip("Saved sets of these settings. Pick one and "
                                     "press Load to fill in the settings below.")
        row.addWidget(self.preset_combo, 1)
        for text, tip, slot in (
                ("Load", "Fill in the settings from the chosen preset. You can "
                         "still change them before pressing OK.", self._load_preset),
                ("Save...", "Save the settings shown as a preset, under a name "
                            "you choose.", self._save_preset),
                ("Delete", "Delete the chosen preset.", self._delete_preset)):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        self._refresh_presets()
        return row

    def _refresh_presets(self, select=None):
        from services.preset_service import list_presets
        names = list_presets(self.plugin_name)
        self.preset_combo.clear()
        self.preset_combo.addItems(names)
        if select in names:
            self.preset_combo.setCurrentText(select)

    def _load_preset(self):
        from services.preset_service import load_preset
        name = self.preset_combo.currentText()
        if not name:
            return
        try:
            self.set_parameters(load_preset(self.plugin_name, name))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Preset Not Loaded", f"Could not read '{name}':\n{e}")

    def _save_preset(self):
        from services.preset_service import save_preset, list_presets
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:",
                                        text=self.preset_combo.currentText())
        name = name.strip()
        if not ok or not name:
            return
        if name in list_presets(self.plugin_name) and QMessageBox.question(
                self, "Replace Preset?", f"A preset '{name}' exists. Replace it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            save_preset(self.plugin_name, name, dict(self.get_parameters()))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Preset Not Saved", str(e))
            return
        self._refresh_presets(select=name)

    def _delete_preset(self):
        from services.preset_service import delete_preset
        name = self.preset_combo.currentText()
        if not name or QMessageBox.question(
                self, "Delete Preset?", f"Delete the preset '{name}'?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        delete_preset(self.plugin_name, name)
        self._refresh_presets()

    def set_parameters(self, values: Dict[str, Any]) -> None:
        """Show *values* in the widgets. Settings not in *values* keep what is
        shown; names the schema does not have are ignored."""
        for param_name, value in values.items():
            widget = self.param_widgets.get(param_name)
            if widget is None:
                continue
            param_type = self.parameter_schema[param_name].get("type", "string")
            if param_type == "info":
                continue
            if param_type == "int":
                widget.setValue(int(value))
            elif param_type == "float":
                widget.setValue(float(value))
            elif param_type in ("dropdown", "colormap"):
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif param_type == "choice":
                index = widget.findData(value)
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    widget.setCurrentText(str(value))
            elif param_type == "bool":
                widget.setChecked(bool(value))
            else:
                widget.setText(str(value))

    def _setup_dependencies(self):
        """Wire up enabled_by / disabled_by dependencies between widgets.

        Schema fields:
            "enabled_by": "bool_param"  — widget enabled when checkbox is checked
            "disabled_by": "bool_param" — widget disabled when checkbox is checked
        """
        for param_name, param_info in self.parameter_schema.items():
            for dep_key, invert in [("enabled_by", False), ("disabled_by", True)]:
                controller_name = param_info.get(dep_key)
                if not controller_name:
                    continue

                controller_widget = self.param_widgets.get(controller_name)
                if not isinstance(controller_widget, QCheckBox):
                    continue

                # Determine the target widget(s) to enable/disable
                target_widget = self.param_containers.get(param_name) or \
                                self.param_widgets.get(param_name)
                target_label = self.param_labels.get(param_name)
                if not target_widget:
                    continue

                # Connect toggled signal
                def make_callback(w, lbl, inv):
                    def callback(checked):
                        enabled = not checked if inv else checked
                        w.setEnabled(enabled)
                        if lbl:
                            lbl.setEnabled(enabled)
                    return callback

                cb = make_callback(target_widget, target_label, invert)
                controller_widget.toggled.connect(cb)

                # Set initial state
                cb(controller_widget.isChecked())

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameter values from the dialog widgets.

        Returns:
            Dict[str, Any]: Dictionary of parameter names to their values
        """
        for param_name, widget in self.param_widgets.items():
            param_type = self.parameter_schema[param_name].get("type", "string")

            if param_type == "info":
                continue  # Display-only, no value to collect
            elif param_type == "int":
                self.params[param_name] = widget.value()
            elif param_type == "float":
                self.params[param_name] = widget.value()
            elif param_type in ("dropdown", "colormap"):
                self.params[param_name] = widget.currentData()  # Get the data value, not the display text
            elif param_type == "choice":
                # For editable combobox, get the current text (allows custom input)
                self.params[param_name] = widget.currentText()
            elif param_type == "bool":
                self.params[param_name] = widget.isChecked()
            else:  # String or directory
                self.params[param_name] = widget.text()

        return self.params

    def _build_colormap_combo(self, param_info: Dict[str, Any], default_value) -> QComboBox:
        """Build a QComboBox whose items each show a gradient swatch of the
        colormap they represent. ``options`` defaults to the full colormap
        registry, so plugins need only declare ``type: "colormap"``."""
        from PyQt5.QtCore import QSize
        from PyQt5.QtGui import QIcon, QImage, QPixmap
        from services.colormap_service import COLORMAPS, colormap_swatch

        options = param_info.get("options") or COLORMAPS

        widget = QComboBox()
        widget.setIconSize(QSize(72, 16))
        for value, display_text in options.items():
            swatch = colormap_swatch(value)
            h, w, _ = swatch.shape
            buf = swatch.tobytes()  # keep alive until the QImage is copied
            image = QImage(buf, w, h, 3 * w, QImage.Format_RGB888).copy()
            widget.addItem(QIcon(QPixmap.fromImage(image)), display_text, value)

        if default_value and isinstance(default_value, str):
            index = widget.findData(default_value)
            if index >= 0:
                widget.setCurrentIndex(index)
        return widget

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