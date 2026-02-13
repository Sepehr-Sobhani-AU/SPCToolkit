from typing import Dict, Any

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class ManagePluginsDialog(QtWidgets.QDialog):
    """Dialog for managing plugins: view, reload, unload, scan for new."""

    def __init__(self, plugin_manager, parent=None):
        super().__init__(parent)
        self.plugin_manager = plugin_manager
        self.setWindowTitle("Manage Plugins")
        self.setMinimumSize(700, 450)
        self._build_ui()
        self._populate_table()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Loaded", "Name", "Type", "Menu Location"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_reload = QtWidgets.QPushButton("Reload Selected")
        self.btn_reload.setToolTip("Force-reload the selected plugin(s) from disk")
        self.btn_reload.clicked.connect(self._on_reload)
        btn_layout.addWidget(self.btn_reload)

        self.btn_scan = QtWidgets.QPushButton("Scan for New Plugins")
        self.btn_scan.setToolTip("Discover and load new plugin files from disk")
        self.btn_scan.clicked.connect(self._on_scan)
        btn_layout.addWidget(self.btn_scan)

        btn_layout.addStretch()

        self.btn_apply = QtWidgets.QPushButton("Apply && Close")
        self.btn_apply.setToolTip("Unload unchecked plugins, rebuild menus, and close")
        self.btn_apply.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.btn_apply)

        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setToolTip("Close without changes")
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)

        layout.addLayout(btn_layout)

    def _populate_table(self):
        """Fill the table with all currently loaded plugins."""
        self.table.setRowCount(0)
        infos = self.plugin_manager.get_all_plugin_info()
        # Sort by menu_path then name for consistency
        infos.sort(key=lambda d: (d['menu_path'] or '', d['name']))

        for info in infos:
            self._add_row(info, checked=True)

    def _add_row(self, info, checked=True):
        """Add a single plugin row to the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Checkbox item
        chk_item = QtWidgets.QTableWidgetItem()
        chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        chk_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        # Store plugin name for later lookup
        chk_item.setData(Qt.UserRole, info['name'])
        self.table.setItem(row, 0, chk_item)

        # Name
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(info['name']))

        # Type
        type_label = "Action" if info['type'] == 'action' else "Analysis"
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(type_label))

        # Menu location
        menu_loc = info['menu_path'] if info['menu_path'] else "(system)"
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(menu_loc))

        # Prevent unloading this plugin itself
        if info['name'] == 'manage_plugins':
            chk_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # no checkbox toggle
            chk_item.setCheckState(Qt.Checked)

    def _on_reload(self):
        """Reload selected (highlighted) plugins from disk."""
        selected_rows = set(idx.row() for idx in self.table.selectedIndexes())
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Reload", "Select one or more rows first.")
            return

        messages = []
        for row in sorted(selected_rows):
            plugin_name = self.table.item(row, 0).data(Qt.UserRole)
            success, msg = self.plugin_manager.reload_plugin(plugin_name)
            messages.append(msg)

        QtWidgets.QMessageBox.information(self, "Reload Results", "\n".join(messages))

    def _on_scan(self):
        """Scan for new plugin files on disk and add them to the table."""
        newly_loaded = self.plugin_manager.scan_and_load_new_plugins()
        if not newly_loaded:
            QtWidgets.QMessageBox.information(self, "Scan", "No new plugins found.")
            return

        # Add new plugins to table as unchecked
        for name in newly_loaded:
            info_list = self.plugin_manager.get_all_plugin_info()
            for info in info_list:
                if info['name'] == name:
                    self._add_row(info, checked=False)
                    break

        QtWidgets.QMessageBox.information(
            self, "Scan", f"Found {len(newly_loaded)} new plugin(s):\n" + "\n".join(newly_loaded)
        )

    def _on_apply(self):
        """Unload unchecked plugins, then accept (caller rebuilds menus)."""
        # Collect unchecked plugin names
        to_unload = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            plugin_name = item.data(Qt.UserRole)
            if item.checkState() != Qt.Checked and plugin_name != 'manage_plugins':
                to_unload.append(plugin_name)

        for name in to_unload:
            self.plugin_manager.unload_plugin(name)

        self.accept()  # signal caller to rebuild menus


class ManagePluginsPlugin(ActionPlugin):
    def get_name(self) -> str:
        return "manage_plugins"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        plugin_manager = main_window.plugin_manager
        dialog = ManagePluginsDialog(plugin_manager, parent=main_window)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            main_window.rebuild_plugin_menus()
