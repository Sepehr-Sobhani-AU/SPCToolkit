"""
Crease-edge extraction plugin — the intersection line of two surfaces.

The generic "Lego brick" for edge lines: kerb top/bottom, building corners,
roof ridges, wall/floor joints are all *outcomes* of running this one brick
(see ``DECISIONS.md`` 2026-06-26). It does not know about any of those features.

Workflow:
1. Select a PointCloud branch with per-point normals (embedded PLY nx/ny/nz or
   produced by the normal_estimation plugin — normals are consumed, never
   recomputed here).
2. Optionally polygon-/Shift-select a swath that straddles **both** surfaces and
   the edge between them. With no selection, the whole branch is the swath.
3. Run the plugin. The shared ``CreaseTracer`` marches one cube along the edge,
   fitting two planes per step, intersecting them, and emitting one vertex per
   step — the cube re-centres on the edge and rotates to follow it (curves too).
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
)


class CreaseEdgePlugin(ActionPlugin):

    def get_name(self) -> str:
        return "crease_edge"

    def requires_selection(self) -> bool:
        return True

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "cell_size": {
                "type": "float",
                "default": 0.3,
                "min": 0.001,
                "max": 100.0,
                "label": "Cell Size",
                "description": "Cube edge length and step along the edge (m). One "
                               "cube per step is centred on the edge and fits two "
                               "planes; smaller follows tighter curves but holds "
                               "fewer points per fit.",
            },
            "min_points_per_cell": {
                "type": "int",
                "default": 10,
                "min": 6,
                "max": 1000,
                "label": "Min Points per Cell",
                "description": "Stop the march where a cube holds fewer points "
                               "than this (too sparse, or the edge has ended).",
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
                "description": "Debug: overlay the candidate grid cells as "
                               "wireframe cubes.",
            },
            "show_planes": {
                "type": "bool",
                "default": False,
                "label": "Show Planes",
                "description": "Debug: overlay the two fitted planes per accepted "
                               "cell (square patches + normal stubs).",
            },
            "show_voxel_points": {
                "type": "bool",
                "default": False,
                "label": "Show Voxel Points",
                "description": "Debug: colour the swath points by the march cube "
                               "they fell into (as a Clusters branch).",
            },
            "show_normals": {
                "type": "bool",
                "default": False,
                "label": "Show Normals",
                "description": "Debug: overlay the per-point normals (used to "
                               "split each cell into two planes) as short segments.",
            },
        }

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

        # --- Reconstruct the selected branch ---
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

        # --- Swath: the polygon selection if one was drawn, else whole branch ---
        polygon_mask = viewer_widget.retest_polygon_selection(pc_points)
        if polygon_mask is not None and bool(polygon_mask.any()):
            swath_idx = np.where(polygon_mask)[0]
        else:
            swath_idx = np.arange(len(pc_points))

        if len(swath_idx) < params.get("min_points_per_cell", 10):
            QMessageBox.warning(
                main_window, "Swath Too Small",
                "The selected swath has too few points. Select a region that "
                "straddles both surfaces and the edge between them.")
            return

        swath_points = pc_points[swath_idx]
        swath_normals = normals[swath_idx]

        show_cells = bool(params.get("show_cells", False))
        show_planes = bool(params.get("show_planes", False))
        show_voxel_points = bool(params.get("show_voxel_points", False))
        show_normals = bool(params.get("show_normals", False))
        any_debug = show_cells or show_planes or show_voxel_points or show_normals

        # --- Trace the crease ---
        try:
            tracer = CreaseTracer(
                points=swath_points,
                normals=swath_normals,
                cell_size=params.get("cell_size", 0.3),
                min_points_per_cell=params.get("min_points_per_cell", 10),
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
                controller, tree_widget, tracer, parent_uid, parent_node, node,
                len(pc_points), swath_idx, params,
                show_cells, show_planes, show_voxel_points, show_normals,
            )

        # --- Render and clear selection ---
        main_window.render_visible_data(zoom_extent=False)
        viewer_widget.picked_points_indices.clear()
        viewer_widget._selection_polygons.clear()
        viewer_widget.update()

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
                    "No crease edge was traced. Try a larger swath, a smaller cell "
                    "size, or a lower minimum dihedral angle — or enable the "
                    "Show Cells/Planes/Voxel Points/Normals boxes to trace the "
                    "process.")
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
                            parent_node, input_node, n_points, swath_idx, params,
                            show_cells, show_planes, show_voxel_points, show_normals):
        """Add each requested processing stage as an ordinary controllable branch
        under *parent_uid*."""
        # Wireframe stages: cells (cubes), planes (patches), normals (segments).
        for name, feature in debug_vector_features(
            tracer, show_cells, show_planes, show_normals
        ):
            uid = controller.add_analysis_result(
                feature, "vector_feature", [input_node.uid], parent_node, name, params
            )
            tree_widget.add_branch(uid, parent_uid, name, tooltip=f"crease_edge,{name}")
            self._set_visible(tree_widget, uid)

        # Voxel points: colour the swath points by their grid cell (a Clusters
        # branch — points are not wireframe geometry).
        if show_voxel_points and tracer.debug_point_cell is not None:
            labels = np.full(n_points, -1, dtype=np.int32)
            labels[swath_idx] = tracer.debug_point_cell.astype(np.int32)
            clusters = Clusters(labels=labels)
            clusters.set_random_color()
            uid = controller.add_analysis_result(
                clusters, "cluster_labels", [input_node.uid], parent_node,
                "crease_voxel_points", params
            )
            tree_widget.add_branch(uid, parent_uid, "crease_voxel_points",
                                   tooltip="crease_edge,voxel_points")
            self._set_visible(tree_widget, uid)
