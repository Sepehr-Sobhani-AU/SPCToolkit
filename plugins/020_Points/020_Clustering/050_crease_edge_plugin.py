"""
Crease-edge extraction plugin — the intersection line of two surfaces.

The generic "Lego brick" for edge lines: kerb top/bottom, building corners,
roof ridges, wall/floor joints are all *outcomes* of running this one brick
(see ``DECISIONS.md`` 2026-06-26). It does not know about any of those features.

Workflow:
1. Select a PointCloud branch with per-point normals (embedded PLY nx/ny/nz or
   produced by the normal_estimation plugin — normals are consumed, never
   recomputed here).
2. Shift+click **one point near the edge**. The pick only *locates* the edge;
   the code searches the whole branch and discovers the two intersecting planes.
3. Run the plugin. ``perpendicular_size`` is pre-filled from a quick estimate at
   your pick (how far to reach to capture both surfaces) — accept or override.
   The shared ``CreaseTracer`` marches one cell along the edge, fitting two
   planes per step and intersecting them; the cell re-centres on the edge and
   rotates to follow it (curves too).
4. The result is a polyline ``VectorFeature`` branch tracing the crease.

A kerb is two runs of this brick (road∩face, top∩face); a building base loop is
one run per wall, joined later by the separate junction brick.
"""

import numpy as np
from typing import Dict, Any
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.clusters import Clusters
from core.services.crease_tracer import (
    CreaseTracer,
    vertices_to_polyline_feature,
    debug_vector_features,
    suggest_perpendicular,
)

_FALLBACK_PERPENDICULAR = 0.3  # used when no seed is picked yet


class CreaseEdgePlugin(ActionPlugin):

    def get_name(self) -> str:
        return "crease_edge"

    def requires_selection(self) -> str:
        return "points"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "edge_length": {
                "type": "float",
                "default": 0.3,
                "min": 0.001,
                "max": 100.0,
                "label": "Cell Size Along Edge",
                "description": "Cell length along the edge (m) — also the step. "
                               "The only size you set; smaller follows tighter "
                               "curves and spaces vertices more finely.",
            },
            "perpendicular_size": {
                "type": "float",
                "default": self._suggest_perpendicular_default(),
                "min": 0.001,
                "max": 100.0,
                "label": "Cell Size Across Edge (auto)",
                "description": "How far each cell reaches across the edge to "
                               "capture both surfaces (m). Auto-suggested from "
                               "your picked point; adjust if it over/under-reaches.",
            },
            "min_dihedral_angle": {
                "type": "float",
                "default": 20.0,
                "min": 1.0,
                "max": 90.0,
                "label": "Min Dihedral Angle (deg)",
                "description": "Stop where a cell's two planes meet at less than "
                               "this angle — one surface, so the edge has ended.",
            },
            "ransac_threshold": {
                "type": "float",
                "default": 0.03,
                "min": 0.001,
                "max": 5.0,
                "label": "RANSAC Threshold",
                "description": "Plane RANSAC inlier distance threshold (m).",
            },
            "ransac_iterations": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
                "label": "RANSAC Iterations",
                "description": "Max RANSAC hypotheses per plane fit "
                               "(higher = more robust, slower).",
            },
            "show_cells": {
                "type": "bool",
                "default": False,
                "label": "Show Cells",
                "description": "Debug: overlay each step's cell as a wireframe box.",
            },
            "show_planes": {
                "type": "bool",
                "default": False,
                "label": "Show Planes",
                "description": "Debug: overlay the two fitted planes per step "
                               "(square patches + normal stubs).",
            },
            "show_voxel_points": {
                "type": "bool",
                "default": False,
                "label": "Show Voxel Points",
                "description": "Debug: colour the points by the march cell they "
                               "fell into (as a Clusters branch).",
            },
            "show_normals": {
                "type": "bool",
                "default": False,
                "label": "Show Normals",
                "description": "Debug: overlay the per-point normals (used to "
                               "split each cell into two planes), coloured by side.",
            },
        }

    @staticmethod
    def _suggest_perpendicular_default() -> float:
        """Estimate ``perpendicular_size`` at the current pick to pre-fill the
        dialog. Cheap (an O(N) distance pass on the viewer points, no normals or
        reconstruction); falls back to a static default when nothing is picked."""
        try:
            viewer = global_variables.global_pcd_viewer_widget
            picked = list(viewer.picked_points_indices)
            pts = np.asarray(viewer.points)[:, :3]
            coords = pts[[i for i in picked if i < len(pts)]]
            if len(coords) == 0:
                return _FALLBACK_PERPENDICULAR
            seed = coords.mean(axis=0)
            suggestion = suggest_perpendicular(pts, seed)
            if suggestion and suggestion > 0:
                return round(float(suggestion), 3)
        except Exception:
            pass
        return _FALLBACK_PERPENDICULAR

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget

        # --- Validate: exactly one branch selected ---
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a PointCloud branch first.")
            return
        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch at a time.")
            return

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)
        if node is None:
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Could not find the selected branch.")
            return

        # --- Seed: one (or more) points Shift+clicked near the edge ---
        picked = list(viewer_widget.picked_points_indices)
        if not picked:
            QMessageBox.warning(
                main_window, "No Point Picked",
                "Shift+click one point near the edge you want to trace, then run "
                "the plugin. The pick only locates the edge — the code finds the "
                "two surfaces itself.")
            return
        viewer_points = np.asarray(viewer_widget.points)
        rows = np.fromiter(picked, dtype=np.intp, count=len(picked))
        rows = rows[(rows >= 0) & (rows < len(viewer_points))]
        if rows.size == 0:
            QMessageBox.warning(main_window, "No Point Picked",
                                "Could not read the picked point coordinates.")
            return
        # Mean of the picked positions, in float32 — a seed location, not an
        # index, so this stays in the viewer's own coordinate space rather than
        # going through the cloud-index mapping the cluster plugins need.
        seed_point = viewer_points[rows, :3].astype(np.float32).mean(axis=0)

        # --- Reconstruct the whole branch (the data the march searches) ---
        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{str(e)}")
            return
        pc_points = np.asarray(point_cloud.points)

        # --- Consume upstream normals (never recomputed here) ---
        normals = point_cloud.get_attribute("normals")
        if normals is None:
            normals = getattr(point_cloud, "normals", None)
        if normals is None or len(normals) == 0:
            QMessageBox.warning(
                main_window, "Normals Required",
                "Crease-edge extraction needs per-point normals. Either import "
                "a cloud with embedded normals (e.g. PLY nx/ny/nz), or run the "
                "normal_estimation plugin on this branch first.")
            return
        normals = np.asarray(normals)

        show_cells = bool(params.get("show_cells", False))
        show_planes = bool(params.get("show_planes", False))
        show_voxel_points = bool(params.get("show_voxel_points", False))
        show_normals = bool(params.get("show_normals", False))
        any_debug = show_cells or show_planes or show_voxel_points or show_normals

        # --- Trace the crease from the seed ---
        try:
            tracer = CreaseTracer(
                points=pc_points,
                normals=normals,
                seed_point=seed_point,
                edge_length=params.get("edge_length", 0.3),
                perpendicular_size=params.get("perpendicular_size", _FALLBACK_PERPENDICULAR),
                min_dihedral_deg=params.get("min_dihedral_angle", 20.0),
                ransac_threshold=params.get("ransac_threshold", 0.03),
                ransac_iterations=params.get("ransac_iterations", 100),
                record_debug=any_debug,
            )
            vertices = tracer.trace()
        except Exception as e:
            QMessageBox.critical(main_window, "Crease Tracing Failed",
                                 f"Failed to trace crease edge:\n{str(e)}")
            return

        feature = vertices_to_polyline_feature(vertices)

        # The edge polyline (if any) becomes a branch; debug overlays attach
        # under it, or under the input branch when no edge was found — so the
        # overlays are available precisely when you need to trace *why* it failed.
        if feature is not None:
            parent_uid = controller.add_analysis_result(
                feature, "vector_feature", [node.uid], node, "crease_edge", params
            )
            tree_widget.add_branch(
                parent_uid, str(node.uid), "crease_edge",
                tooltip=f"crease_edge,{params}"
            )
            self._set_visible(tree_widget, parent_uid)
            parent_node = controller.get_node(parent_uid)
        else:
            parent_uid = str(node.uid)
            parent_node = node

        if any_debug:
            self._add_debug_branches(
                controller, tree_widget, tracer, parent_uid, parent_node, node, params,
                show_cells, show_planes, show_voxel_points, show_normals,
            )

        # --- Render and clear the pick ---
        main_window.render_visible_data(zoom_extent=False)
        viewer_widget.clear_selection()

        if feature is None:
            if any_debug:
                QMessageBox.information(
                    main_window, "No Edge Found",
                    "No crease edge was traced — debug overlays were added so you "
                    "can trace why (inspect the cells, planes, voxel points and "
                    "normals branches).")
            else:
                QMessageBox.warning(
                    main_window, "No Edge Found",
                    "No crease edge was traced. Pick a point nearer the edge, "
                    "increase the across-edge cell size, or lower the minimum "
                    "dihedral angle — or enable the debug boxes to trace it.")
            return

        QMessageBox.information(
            main_window, "Crease Edge Complete",
            f"Traced a crease edge with {len(vertices):,} vertices."
        )

    # ------------------------------------------------------------------ #
    # Debug overlays                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _set_visible(tree_widget, uid):
        tree_widget.blockSignals(True)
        item = tree_widget.branches_dict.get(uid)
        if item:
            item.setCheckState(0, Qt.Checked)
        tree_widget.visibility_status[uid] = True
        tree_widget.blockSignals(False)

    def _add_debug_branches(self, controller, tree_widget, tracer, parent_uid,
                            parent_node, input_node, params,
                            show_cells, show_planes, show_voxel_points, show_normals):
        """Add each requested processing stage as an ordinary controllable branch
        under *parent_uid*."""
        # Wireframe stages: cells (boxes), planes (patches), normals (segments).
        for name, feature in debug_vector_features(
            tracer, show_cells, show_planes, show_normals
        ):
            uid = controller.add_analysis_result(
                feature, "vector_feature", [input_node.uid], parent_node, name, params
            )
            tree_widget.add_branch(uid, parent_uid, name, tooltip=f"crease_edge,{name}")
            self._set_visible(tree_widget, uid)

        # Voxel points: colour the cloud points by the march cell they fell into
        # (a Clusters branch — points are not wireframe geometry). The tracer's
        # debug_point_cell is aligned to the whole branch it ran on.
        if show_voxel_points and tracer.debug_point_cell is not None:
            clusters = Clusters(labels=tracer.debug_point_cell.astype(np.int32))
            clusters.set_random_color()
            uid = controller.add_analysis_result(
                clusters, "cluster_labels", [input_node.uid], parent_node,
                "crease_voxel_points", params
            )
            tree_widget.add_branch(uid, parent_uid, "crease_voxel_points",
                                   tooltip="crease_edge,voxel_points")
            self._set_visible(tree_widget, uid)
