"""
Distance measurement tool — report distances between selected points.

Workflow:
1. User Shift + Left Clicks two (or more) points in the viewer.
2. User runs Measure > Distance Measurement.
3. For two points the plugin reports the straight-line 3D distance plus the
   horizontal, vertical, and per-axis components. For more than two points it
   reports the cumulative polyline length along the selection order as well as
   the straight-line distance between the first and last point.

Distances are frame-invariant, so they are computed directly from the viewer's
render buffer; endpoint coordinates are reported in world coordinates by adding
back each point's root translation.
"""

from typing import Dict, Any, Optional

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from services.coordinate_service import find_root_translation


class DistanceMeasurementPlugin(ActionPlugin):
    """Action plugin that measures distances between selected points."""

    def get_name(self) -> str:
        return "distance_measurement"

    def requires_selection(self) -> Optional[str]:
        return "points"

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer = global_variables.global_pcd_viewer_widget
        data_nodes = global_variables.global_data_nodes

        if viewer is None or len(viewer.picked_points) < 2:
            QMessageBox.information(
                main_window, "Distance Measurement",
                "Select at least two points (Shift + Left Click), then run "
                "Distance Measurement."
            )
            return

        # Click order is the measurement: the polyline runs through the points
        # in the order they were clicked. Only individual clicks are recorded,
        # so a lasso cannot make an arbitrary polyline out of a million points.
        controller = global_variables.global_application_controller
        world = np.array(
            [self._world_coord(controller, data_nodes, uid, row)
             for uid, row in viewer.picked_points],
            dtype=np.float64,
        )
        if len(world) < 2:
            return

        if len(world) == 2:
            text = self._two_point_report(world[0], world[1])
        else:
            text = self._polyline_report(world)

        QMessageBox.information(main_window, "Distance Measurement", text)

    def _world_coord(self, controller, data_nodes, uid, row) -> np.ndarray:
        """The picked point in world coordinates.

        The pick names its branch and its row in that branch's full-resolution
        cloud, so the coordinate is read from the cloud rather than from the
        rendered buffer — which under LOD held a subsample and could only ever
        report a drawn point.
        """
        point_cloud = controller.reconstruct(uid)
        local = np.asarray(point_cloud.points[row][:3], dtype=np.float64)
        translation = find_root_translation(data_nodes, str(uid))
        return local + np.asarray(translation, dtype=np.float64)

    @staticmethod
    def _fmt(vec):
        return "(" + ", ".join(f"{v:.4f}" for v in vec) + ")"

    def _two_point_report(self, p1, p2) -> str:
        d = p2 - p1
        dist3d = float(np.linalg.norm(d))
        horizontal = float(np.hypot(d[0], d[1]))
        vertical = float(d[2])
        slope_pct = (vertical / horizontal * 100.0) if horizontal > 0 else float("inf")
        return (
            f"Point 1 : {self._fmt(p1)}\n"
            f"Point 2 : {self._fmt(p2)}\n\n"
            f"3D distance    : {dist3d:.4f}\n"
            f"Horizontal (2D): {horizontal:.4f}\n"
            f"Vertical (ΔZ)  : {vertical:.4f}\n\n"
            f"ΔX : {d[0]:.4f}\n"
            f"ΔY : {d[1]:.4f}\n"
            f"ΔZ : {d[2]:.4f}\n\n"
            f"Slope : {slope_pct:.2f}%"
        )

    def _polyline_report(self, world) -> str:
        segments = np.linalg.norm(np.diff(world, axis=0), axis=1)
        cumulative = float(np.sum(segments))
        straight = float(np.linalg.norm(world[-1] - world[0]))
        seg_lines = "\n".join(
            f"  {i + 1} → {i + 2} : {seg:.4f}" for i, seg in enumerate(segments)
        )
        return (
            f"{len(world)} points selected\n\n"
            f"Cumulative polyline length : {cumulative:.4f}\n"
            f"Straight-line (first→last) : {straight:.4f}\n\n"
            f"Segments:\n{seg_lines}"
        )
