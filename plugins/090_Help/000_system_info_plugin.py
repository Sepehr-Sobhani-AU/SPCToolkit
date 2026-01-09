# plugins/090_Help/000_system_info_plugin.py
"""
Plugin for displaying system hardware and backend information.

This action plugin shows a dialog with details about:
- Operating system
- GPU hardware (if available)
- Available CUDA libraries
- Currently active backends for each algorithm
"""

from typing import Dict, Any

from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class SystemInfoPlugin(ActionPlugin):
    """
    Action plugin for displaying system and hardware information.

    Shows a dialog with details about the detected hardware configuration
    and which backends are being used for each algorithm.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "system_info"

    def get_parameters(self) -> Dict[str, Any]:
        """
        No parameters needed - directly shows info dialog.

        Returns:
            Empty dictionary (no parameters required)
        """
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the system info action - shows a dialog with hardware details.

        Args:
            main_window: The main application window
            params: Not used for this plugin (empty dict)
        """
        hardware = global_variables.global_hardware_info
        registry = global_variables.global_backend_registry

        if hardware is None or registry is None:
            QMessageBox.warning(
                main_window,
                "System Info",
                "Hardware detection not initialized."
            )
            return

        # Build info text
        lines = []

        # System info
        lines.append("=== System Information ===")
        lines.append(f"Operating System: {hardware.os}")
        lines.append("")

        # GPU info
        lines.append("=== GPU Information ===")
        if hardware.gpu_available:
            lines.append(f"GPU: {hardware.gpu_name}")
            lines.append(f"Vendor: {hardware.gpu_vendor}")
            lines.append(f"Memory: {hardware.gpu_memory_mb} MB")
        else:
            lines.append("GPU: None detected")
        lines.append("")

        # Library availability
        lines.append("=== Library Availability ===")
        lines.append(f"CUDA: {'Available' if hardware.cuda_available else 'Not available'}")
        lines.append(f"PyTorch CUDA: {'Available' if hardware.pytorch_cuda else 'Not available'}")
        lines.append(f"CuPy: {'Available' if hardware.cupy_available else 'Not available'}")
        lines.append(f"cuML (RAPIDS): {'Available' if hardware.cuml_available else 'Not available'}")
        lines.append("")

        # Backend configuration
        lines.append("=== Active Backends ===")
        lines.append(f"Scenario: {registry.get_scenario()}")
        report = registry.get_status_report()
        for algorithm, backend in report.items():
            lines.append(f"{algorithm}: {backend}")

        # Show dialog
        info_text = "\n".join(lines)
        msg_box = QMessageBox(main_window)
        msg_box.setWindowTitle("System Information")
        msg_box.setText("Hardware and Backend Configuration")
        msg_box.setDetailedText(info_text)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()
