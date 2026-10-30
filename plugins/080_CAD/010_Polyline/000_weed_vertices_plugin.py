"""
Weed Vertices — remove extra vertices from an existing 3D polyline.

Select one or more polyline branches (``vector_feature`` nodes with
``geometry_type == "polyline"``, e.g. the CAD > Boundary outlines or traced
lines) and run. Each one gets a new child branch holding a weeded copy; the
source polyline is never changed.

A vertex is kept only if dropping it would move the line by more than the
tolerance across the line, or by more than the height tolerance. Both are
measured in the projection plane — by default the polyline's own best-fit
plane for a closed outline and plan view for an open line. Kept vertices are
never moved. The weeding itself lives in
``core.services.outline_cleanup.weed_polyline``.

This is an ActionPlugin, not an analysis plugin: the analysis executor
reconstructs any non-point-cloud node into its parent's point cloud, which
would hide the polyline this tool works on. Weeding is instant, so running on
the main thread is fine. Being an action, it is not captured by Save
Pipeline (pipelines replay analysis steps only).
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.data_node import DataNode
from core.entities.vector_feature import VectorFeature
from core.services.outline_cleanup import weed_polyline

logger = logging.getLogger(__name__)


_PROJECTIONS = {
    "auto": "Auto (best-fit for closed, XY for open)",
    "best_fit": "Best-fit plane (PCA)",
    "xy": "XY (plan / top-down)",
    "xz": "XZ (front)",
    "yz": "YZ (side)",
}


def weed_feature(feature: VectorFeature, tolerance: float, height_tol: float,
                 projection: str = "auto") -> VectorFeature:
    """A new polyline ``VectorFeature`` with the extra vertices of *feature*
    removed. *feature* is not modified.

    ``projection`` sets what "across" and "height" mean (see
    ``weed_polyline``). ``auto`` uses the polyline's best-fit plane for a
    closed outline and XY for an open line, whose best-fit plane can stand
    vertical (a sagging cable) and swap the two tolerances.

    Tolerances are in world units: when the feature carries a non-identity
    transform, the weeding runs on the world-space line and the kept rows
    are taken from the source's own local vertices (exact, never moved).
    """
    if feature.geometry_type != "polyline":
        raise ValueError("weed_feature needs a polyline VectorFeature.")

    local = np.asarray(feature.geometry["vertices"], dtype=np.float32)
    closed = bool(feature.geometry.get("closed", False))
    if projection == "auto":
        projection = "best_fit" if closed else "xy"
    T = np.asarray(feature.transform_matrix, dtype=np.float64)

    if np.allclose(T, np.eye(4)):
        world = local
    else:
        homo = np.hstack([local.astype(np.float64), np.ones((len(local), 1))])
        world = (T @ homo.T).T[:, :3]
    rows = weed_polyline(world, tolerance, height_tol, closed,
                         projection=projection, return_indices=True)
    kept = local[rows].copy()

    dims = (kept.max(axis=0) - kept.min(axis=0)).astype(np.float32)
    return VectorFeature(
        symbol_type=feature.symbol_type,
        geometry_type="polyline",
        geometry={"vertices": kept, "closed": closed},
        transform_matrix=T.copy(),
        dimensions=dims,
        cluster_reference=feature.cluster_reference,
        color=np.asarray(feature.color, dtype=np.float32).copy(),
    )


class WeedVerticesPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "weed_vertices"

    def requires_selection(self) -> Optional[str]:
        return "branches"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "tolerance": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "decimals": 3,
                "label": "Tolerance (m)",
                "description": "A vertex is removed if the line moves sideways "
                               "by less than this when it is gone, measured in "
                               "the projection plane.",
            },
            "height_tolerance": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "decimals": 3,
                "label": "Height Tolerance (m)",
                "description": "A vertex is removed only if the line also moves "
                               "up or down (out of the projection plane) by "
                               "less than this when it is gone. Keeps peaks "
                               "and steps.",
            },
            "projection": {
                "type": "dropdown",
                "options": _PROJECTIONS,
                "default": "auto",
                "label": "Projection Plane",
                "description": "What \"sideways\" and \"height\" mean. Auto: "
                               "the polyline's own plane for a closed outline "
                               "(a wall outline is weeded across the wall), "
                               "plan view for an open line such as a cable.",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        tolerance = float(params.get("tolerance", 0.05))
        height_tol = float(params.get("height_tolerance", 0.05))
        projection = params.get("projection", "auto")

        selected = list(controller.selected_branches or [])
        if not selected:
            QMessageBox.warning(main_window, "No Selection",
                                "Select one or more polyline branches first.")
            return

        report, skipped, new_uids = [], [], []
        for uid in selected:
            node = controller.get_node(uid)
            name = (node.alias or node.params or "polyline") if node else uid
            if (node is None or node.data_type not in ("vector_feature", "cad_object")
                    or getattr(node.data, "geometry_type", None) != "polyline"):
                skipped.append(f"{name} (not a polyline branch)")
                continue

            source: VectorFeature = node.data
            try:
                weeded = weed_feature(source, tolerance, height_tol, projection)
            except Exception as exc:
                logger.exception(f"Weeding '{name}' failed")
                skipped.append(f"{name} (failed: {exc})")
                continue

            before = len(source.geometry["vertices"])
            after = len(weeded.geometry["vertices"])
            branch_name = f"{name} (weeded)"
            new_node = DataNode(
                params=branch_name,
                data=weeded,
                data_type=node.data_type,
                parent_uid=node.uid,
                depends_on=[node.uid],
                tags=["weed_vertices", params],
            )
            new_node.memory_size = controller._estimate_derived_memory(weeded)
            new_uid = str(data_nodes.add_node(new_node))
            tree_widget.add_branch(new_uid, str(node.uid), branch_name,
                                   tooltip=f"weed_vertices,{params}")
            new_uids.append((str(node.uid), new_uid))
            report.append(f"{name}: {before:,} -> {after:,} vertices")

        # Same convention as analysis results: hide the source, show the new
        # child, so the weeded line is what is on screen.
        if new_uids:
            tree_widget.blockSignals(True)
            for parent_uid, child_uid in new_uids:
                parent_item = tree_widget.branches_dict.get(parent_uid)
                if parent_item:
                    parent_item.setCheckState(0, Qt.Unchecked)
                tree_widget.visibility_status[parent_uid] = False
                child_item = tree_widget.branches_dict.get(child_uid)
                if child_item:
                    child_item.setCheckState(0, Qt.Checked)
                tree_widget.visibility_status[child_uid] = True
            tree_widget.blockSignals(False)
            main_window.render_visible_data(zoom_extent=False)

        lines = list(report)
        if skipped:
            lines.append("")
            lines.append("Skipped:")
            lines.extend(f"  {s}" for s in skipped)

        if not report:
            QMessageBox.warning(
                main_window, "No Polylines",
                "None of the selected branches is a polyline.\n\n" + "\n".join(lines))
            return

        logger.info("Weed vertices: " + "; ".join(report))
        QMessageBox.information(main_window, "Weed Vertices Complete",
                                "\n".join(lines))
