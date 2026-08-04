"""
Reopen the guided line-extension workflow on a saved result.

Linear region growing offers this immediately after it runs, but the traces it
produces are persisted (``Clusters.line_traces``), so the work does not have to
be finished in that sitting. Select a linear-region-growing result branch and
this reopens the same stepping window on it — including in a later session, from
a reloaded project.

Nothing is recomputed: the lines' point membership comes from the cluster labels,
their centerlines and stops from the stored traces, and the grower is rebuilt
from the growth parameters recorded alongside them. See
``LINEAR_REGION_GROWING.md``.
"""

from typing import Dict, Any

import numpy as np
from scipy.spatial import cKDTree
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from plugins.dialogs.line_extension_window import LineExtensionWindow
from config.config import global_variables
from core.services.eigenvalue_utils import EigenvalueUtils
from core.services.linear_region_grower import (
    LinearRegionGrower,
    AXIS_TRACE,
    LINEARITY_CONNECTED,
    HYBRID,
    traces_to_lines,
    resolved_stop_keys,
)


_MODE_MAP = {
    "Axis Trace": AXIS_TRACE,
    "Linearity-Connected": LINEARITY_CONNECTED,
    "Hybrid": HYBRID,
}


class ExtendLinesPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "extend_traced_lines"

    def requires_selection(self) -> str:
        return "branches"

    def get_parameters(self) -> Dict[str, Any]:
        """No parameters: the growth settings are read back from the branch, so
        an extension continues under exactly the settings that produced it."""
        return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller

        prepared = self._prepare(controller, main_window)
        if prepared is None:
            return
        result_uid, pc_points, lines, grower, growth_params, resolved = prepared

        window = LineExtensionWindow(result_uid, pc_points, lines, grower,
                                     growth_params, resolved=resolved,
                                     parent=main_window)
        window.show()
        # Held on the main window so a modeless dialog is not garbage-collected
        # the moment execute() returns.
        main_window._line_extension_window = window

    # ------------------------------------------------------------------ #

    def _prepare(self, controller, main_window):
        """Validate the selection and rebuild everything the window needs.

        Returns ``(result_uid, pc_points, lines, grower, params, resolved)`` or
        ``None`` when a message has already been shown.
        """
        selected = controller.selected_branches
        if len(selected) != 1:
            QMessageBox.warning(main_window, "Select One Branch",
                                "Select exactly one linear-region-growing result "
                                "branch to continue.")
            return None

        result_uid = selected[0]
        node = controller.get_node(result_uid)
        if node is None or node.data_type != "cluster_labels":
            QMessageBox.warning(main_window, "Wrong Branch Type",
                                "Select a Clusters branch produced by Linear "
                                "Region Growing.")
            return None

        traces = getattr(node.data, "line_traces", None)
        if not traces or not traces.get("lines"):
            QMessageBox.warning(
                main_window, "No Traced Lines",
                "This branch carries no line traces.\n\n"
                "Only results from Linear Region Growing can be extended — and "
                "only those grown after line tracing began recording its stops. "
                "Re-run the growth to produce an extendable result.")
            return None

        # The cloud the labels index into is the growth input: this branch's
        # parent, reconstructed exactly as the growth saw it.
        parent_node = controller.get_node(str(node.parent_uid))
        if parent_node is None:
            QMessageBox.warning(main_window, "Missing Input Branch",
                                "The point cloud this result was grown from is "
                                "no longer in the project.")
            return None

        point_cloud = controller.reconstruct(str(node.parent_uid))
        pc_points = point_cloud.points
        labels = node.data.labels
        if len(labels) != len(pc_points):
            QMessageBox.warning(
                main_window, "Branch Changed",
                f"This result has {len(labels):,} labels but its input cloud now "
                f"reconstructs to {len(pc_points):,} points, so the two no longer "
                f"line up. Re-run the growth on the current cloud.")
            return None

        growth_params = dict(traces.get("params", {}))
        grower = self._rebuild_grower(point_cloud, pc_points, growth_params,
                                      main_window)
        if grower is None:
            return None

        return (result_uid, pc_points, traces_to_lines(traces, labels), grower,
                growth_params, resolved_stop_keys(traces))

    def _rebuild_grower(self, point_cloud, pc_points, params, main_window):
        """Rebuild the grower under the parameters that produced these lines, so
        an extension grows under the same rules as the trace it continues."""
        mode = _MODE_MAP.get(params.get("growth_mode", "Axis Trace"), AXIS_TRACE)

        linearity = None
        if mode in (LINEARITY_CONNECTED, HYBRID):
            linearity = self._linearity(point_cloud)
            if linearity is None:
                QMessageBox.warning(
                    main_window, "Linearity Not Available",
                    f"These lines were grown in {params.get('growth_mode')} mode, "
                    "which needs per-point linearity, but the input cloud no "
                    "longer carries eigenvalues.\n\n"
                    "Run Compute Eigenvalues on it first.")
                return None

        return LinearRegionGrower(
            all_points=pc_points,
            kdtree=cKDTree(pc_points),
            mode=mode,
            ransac_threshold=params.get("ransac_threshold", 0.03),
            max_iterations=params.get("ransac_iterations", 100),
            cylinder_radius=params.get("cylinder_radius", 0.03),
            cylinder_length=params.get("cylinder_length", 0.5),
            reach_factor=params.get("reach_factor", 3.0),
            overlap=params.get("cylinder_overlap", 0.0) / 100.0,
            min_points=params.get("min_points", 5),
            max_angle_deg=params.get("max_angle", 20.0),
            linearity=linearity,
            linearity_threshold=params.get("linearity_threshold", 0.4),
            neighbor_k=params.get("neighbor_k", 16),
        )

    @staticmethod
    def _linearity(point_cloud):
        """Per-point linearity consumed from upstream eigenvalues, or None.

        Consumed, never recomputed — same contract as the growth plugin.
        """
        eigenvalues = point_cloud.attributes.get("eigenvalues")
        if eigenvalues is None:
            return None
        return EigenvalueUtils().compute_geometric_features(eigenvalues)["linearity"]
