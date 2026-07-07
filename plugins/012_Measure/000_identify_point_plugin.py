"""
Identify tool — report all stored information for a single picked point.

Workflow:
1. User Shift + Left Clicks a point in the viewer to select it.
2. User runs Measure > Identify Point.
3. The plugin locates the picked point in its source branch (full resolution),
   reads every per-point attribute the cloud carries (coordinates, colour,
   normal, intensity, distance-to-ground, classification, custom attribute
   arrays), and shows them in a read-only dialog.

The viewer's render buffer only holds xyz + rgb, so the plugin reconstructs the
branch the point belongs to and coordinate-matches it back to the full-
resolution cloud to recover the remaining attributes.
"""

from typing import Dict, Any, Optional

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox
)
from PyQt5.QtGui import QFont

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from services.coordinate_service import find_root_translation


class IdentifyPointPlugin(ActionPlugin):
    """Action plugin that reports all attributes of one selected point."""

    def get_name(self) -> str:
        return "identify_point"

    def requires_selection(self) -> Optional[str]:
        return "points"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer = global_variables.global_pcd_viewer_widget
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes

        if viewer is None or not viewer.picked_points_indices:
            QMessageBox.information(
                main_window, "Identify Point",
                "Select a point first (Shift + Left Click), then run Identify."
            )
            return

        render_pts = viewer.points
        if render_pts is None:
            return

        idx = viewer.picked_points_indices[0]
        if idx >= len(render_pts):
            return
        world_render = render_pts[idx, :3].astype(np.float64)

        # Which visible branch does this picked index belong to?
        uid = self._branch_uid_for_index(viewer, idx)
        if uid is None:
            QMessageBox.information(
                main_window, "Identify Point",
                "Could not resolve the source branch for the selected point."
            )
            return

        # Reconstruct the branch at full resolution and find the exact point.
        point_cloud = controller.reconstruct(uid)
        pts = np.asarray(point_cloud.points, dtype=np.float64)
        i = int(np.argmin(np.linalg.norm(pts[:, :3] - world_render, axis=1)))

        translation = find_root_translation(data_nodes, str(uid))
        node = data_nodes.get_node(uid) if hasattr(uid, "int") else None
        branch_name = getattr(node, "name", None) or str(uid)[:8]

        report = self._build_report(point_cloud, i, translation, branch_name, uid)
        self._show_report(main_window, report)

    @staticmethod
    def _branch_uid_for_index(viewer, idx):
        """Map a combined-buffer index back to the branch uid it came from."""
        import uuid as _uuid
        for uid_str, (start, end) in viewer._branch_offsets.items():
            if start <= idx < end:
                try:
                    return _uuid.UUID(uid_str)
                except (ValueError, AttributeError):
                    return uid_str
        return None

    @staticmethod
    def _fmt_vec(vec, decimals=4):
        return "[" + ", ".join(f"{v:.{decimals}f}" for v in vec) + "]"

    def _build_report(self, pc, i, translation, branch_name, uid) -> str:
        lines = []
        lines.append(f"Branch      : {branch_name}  ({str(uid)[:8]}…)")
        lines.append(f"Point index : {i}  of  {pc.size}")
        lines.append("")

        local = np.asarray(pc.points[i], dtype=np.float64)[:3]
        world = local + np.asarray(translation, dtype=np.float64)
        lines.append("Coordinates")
        lines.append(f"  World  X,Y,Z : {self._fmt_vec(world, 4)}")
        lines.append(f"  Local  X,Y,Z : {self._fmt_vec(local, 4)}")
        if np.any(np.asarray(translation) != 0):
            lines.append(f"  Translation  : {self._fmt_vec(translation, 4)}")
        lines.append("")

        if getattr(pc, "colors", None) is not None and len(pc.colors) > i:
            lines.append(f"Colour  R,G,B : {self._fmt_vec(pc.colors[i], 3)}")
        if getattr(pc, "normals", None) is not None and len(pc.normals) > i:
            lines.append(f"Normal  X,Y,Z : {self._fmt_vec(pc.normals[i], 4)}")

        intensity = getattr(pc, "intensity", None)
        if intensity is not None and len(intensity) > i:
            lines.append(f"Intensity     : {float(intensity[i]):.4f}")
        dtg = getattr(pc, "distToGround", None)
        if dtg is not None and len(dtg) > i:
            lines.append(f"Dist to ground: {float(dtg[i]):.4f}")

        prediction = getattr(pc, "prediction", "")
        probability = getattr(pc, "probability", 0)
        if prediction:
            lines.append(f"Prediction    : {prediction}  (p={probability})")
        feature = getattr(pc, "feature", None)
        if feature:
            lines.append(f"Feature       : {feature}")

        attributes = getattr(pc, "attributes", {}) or {}
        per_point = []
        for key, val in attributes.items():
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] == pc.size:
                row = arr[i]
                per_point.append(
                    f"  {key:14s}: "
                    f"{self._fmt_vec(np.atleast_1d(row), 4) if row.ndim else row}"
                )
        if per_point:
            lines.append("")
            lines.append("Attributes")
            lines.extend(per_point)

        return "\n".join(lines)

    @staticmethod
    def _show_report(main_window, text: str) -> None:
        dialog = QDialog(main_window)
        dialog.setWindowTitle("Identify Point")
        dialog.resize(460, 380)
        layout = QVBoxLayout(dialog)

        edit = QTextEdit(dialog)
        edit.setReadOnly(True)
        edit.setFont(QFont("Monospace"))
        edit.setPlainText(text)
        layout.addWidget(edit)

        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()
