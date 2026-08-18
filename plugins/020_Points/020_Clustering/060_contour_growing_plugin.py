"""
Contour growing plugin — the level set of any per-point field, grown from a seed.

The generic "Lego brick" for contour lines. It does not know what the field
means: height contours, slope-break lines, distance-to-ground bands and intensity
edges are all *outcomes* of running this one brick with a different field picked
from the dropdown. Whether a given field makes a meaningful line is the user's
call, not the plugin's.

Workflow:
1. Select a branch. The field dropdown is built from that branch's actual fields
   (Z/X/Y, normal components, intensity, distance-to-ground, and any attribute),
   so a new upstream attribute shows up here with no change to this plugin.
   Fields are *consumed* from upstream, never computed here — pick Normal Z and
   the branch has no normals, and you get told to run normal_estimation first.
2. Shift+click **one point** on the contour you want. The pick sets the level
   (the field's value there) and where the flood starts.
3. Run. ``proximity`` and ``max_triangle_edge`` are pre-filled from the point
   spacing at your pick — accept or override. The shared ``ContourTracer`` floods
   the level set: per step it triangulates a ball, marches the triangles, and
   queues a ball at every open end until nothing is left to grow.
4. The result is one branch holding every contour traced, as one polyline each.

Several lines come out of one run — the flood follows the level set wherever it
reaches, not only the line through the pick. The sibling bricks make a line a
different way: ``crease_edge`` intersects two planes, ``linear_region_growing``
fits an axis.
"""

import time
import threading
import uuid
from typing import Any, Dict

import numpy as np
from scipy.spatial import cKDTree
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.services.contour_tracer import (
    ContourTracer,
    contours_to_vector_feature,
    suggest_spacing,
)
from services.point_fields import enumerate_fields, resolve_field, representative_cloud


# Pre-fill factors on the local point spacing. The ball wants enough points for a
# stable triangulation; the triangle cut wants to clear normal spacing but not
# span a real hole.
_PROXIMITY_FACTOR = 8.0
_MAX_EDGE_FACTOR = 3.0

# Used when nothing is picked yet, so the dialog still has sane numbers.
_FALLBACK_SPACING = 0.05


class ContourGrowingPlugin(ActionPlugin):

    def get_name(self) -> str:
        return "contour_growing"

    def requires_selection(self) -> str:
        return "points"

    # ------------------------------------------------------------------ #
    # Parameters                                                         #
    # ------------------------------------------------------------------ #

    def get_parameters(self) -> Dict[str, Any]:
        node = self._selected_node()
        point_cloud = representative_cloud(node)
        sources = enumerate_fields(node, point_cloud)
        default_source = "attr:values" if "attr:values" in sources else "z"
        spacing = self._suggest_spacing_default()

        return {
            "field": {
                "type": "dropdown",
                "options": dict(sources),
                "default": default_source,
                "label": "Contour Field",
                "description": "Per-point field to contour (from the selected "
                               "branch). Consumed from upstream — run the plugin "
                               "that produces it first if it isn't listed.",
            },
            "level_from_seed": {
                "type": "bool",
                "default": True,
                "label": "Level From Picked Point",
                "description": "Take the level from the field's value at your "
                               "picked point, so the contour runs through it. "
                               "Untick to use the level below instead.",
            },
            "level": {
                "type": "float",
                "default": self._seed_level_default(point_cloud, default_source),
                "min": -1e6,
                "max": 1e6,
                "decimals": 4,
                "label": "Level",
                "description": "The field value the contour follows. Only used "
                               "when 'Level From Picked Point' is off. Pre-filled "
                               "from your pick using the default field, so it goes "
                               "stale if you change the field — untick and retype.",
            },
            "proximity": {
                "type": "float",
                "default": round(spacing * _PROXIMITY_FACTOR, 3),
                "min": 0.001,
                "max": 100.0,
                "label": "Proximity (auto)",
                "description": "Ball radius per step (m) — the patch triangulated "
                               "each step, and how far the flood reaches. Auto-"
                               "suggested from the point spacing at your pick. "
                               "Bigger smooths and bridges gaps but is slower.",
            },
            "max_triangle_edge": {
                "type": "float",
                "default": round(spacing * _MAX_EDGE_FACTOR, 3),
                "min": 0.001,
                "max": 100.0,
                "label": "Max Triangle Edge (auto)",
                "description": "Drop triangles longer than this (m). Triangulation "
                               "fills its own convex hull, so this is what stops "
                               "contours running across holes and empty space. "
                               "Raise it if the line breaks up; lower it if lines "
                               "cut across gaps.",
            },
        }

    @staticmethod
    def _selected_node():
        """The selected branch's DataNode, or None — read here only to build the
        field dropdown from what that branch actually carries."""
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        if controller is None or data_nodes is None:
            return None
        selected = getattr(controller, "selected_branches", None) or []
        if not selected:
            return None
        try:
            return data_nodes.get_node(uuid.UUID(str(selected[0])))
        except Exception:
            return None

    @staticmethod
    def _picked_seed():
        """Mean of the Shift+clicked viewer points, or None when nothing is picked."""
        try:
            viewer = global_variables.global_pcd_viewer_widget
            points = np.asarray(viewer.points)
            picked = [i for i in viewer.picked_points_indices if i < len(points)]
            if not picked:
                return None
            return points[picked, :3].mean(axis=0)
        except Exception:
            return None

    def _suggest_spacing_default(self) -> float:
        """Point spacing at the pick, to pre-fill the two size boxes. Cheap (an
        O(N) pass on the viewer points, no reconstruction); falls back to a static
        default when nothing is picked yet."""
        try:
            seed = self._picked_seed()
            if seed is None:
                return _FALLBACK_SPACING
            viewer_points = np.asarray(global_variables.global_pcd_viewer_widget.points)
            spacing = suggest_spacing(viewer_points, seed)
            if spacing and spacing > 0:
                return float(spacing)
        except Exception:
            pass
        return _FALLBACK_SPACING

    def _seed_level_default(self, point_cloud, field_key) -> float:
        """The default field's value at the pick, to pre-fill the level box. Best
        effort only — 'Level From Picked Point' recomputes it properly at run time
        against whichever field was actually chosen."""
        try:
            seed = self._picked_seed()
            if seed is None or point_cloud is None:
                return 0.0
            values = resolve_field(point_cloud, field_key)
            if values is None:
                return 0.0
            points = np.asarray(point_cloud.points, dtype=np.float64)[:, :3]
            offsets = points - np.asarray(seed, dtype=np.float64)
            nearest = int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))
            return round(float(values[nearest]), 4)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # Execute                                                            #
    # ------------------------------------------------------------------ #

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        tree_widget = global_variables.global_tree_structure_widget

        prep = self._validate_and_reconstruct(controller, viewer_widget, main_window, params)
        if prep is None:
            return
        node, points, values, seed_point, level, tree_kd = prep

        try:
            tracer = ContourTracer(
                points=points,
                values=values,
                level=level,
                proximity=params.get("proximity", 0.4),
                max_triangle_edge=params.get("max_triangle_edge", 0.15),
                kdtree=tree_kd,
            )
        except ValueError as e:
            QMessageBox.warning(main_window, "Invalid Parameters", str(e))
            return

        polylines, stopped_early = self._trace_threaded(main_window, tracer, seed_point)
        if polylines is None:  # error during trace — message already shown
            return

        feature = contours_to_vector_feature(polylines)
        if feature is None:
            # Seen on real data: a hand-typed level that misses the pick's
            # surroundings leaves the very first ball with nothing crossing it, so
            # the flood ends before it starts. Name that cause first.
            QMessageBox.warning(
                main_window, "No Contour Found",
                f"No contour was traced at level {level:.4g}."
                + ("\n\nCancelled before anything was traced." if stopped_early else
                   "\n\nUsually this means the level doesn't pass near your picked "
                   "point, so there was nothing to start on — tick 'Level From "
                   "Picked Point' to take the level from the pick itself. "
                   "Otherwise raise Proximity, so each step gathers enough points "
                   "to triangulate."))
            return

        self._add_result_branch(controller, tree_widget, node, feature, params)

        main_window.render_visible_data(zoom_extent=False)
        viewer_widget.clear_selection()

        self._show_summary(main_window, tracer, polylines, level, stopped_early)

    # ------------------------------------------------------------------ #
    # execute() steps                                                    #
    # ------------------------------------------------------------------ #

    def _validate_and_reconstruct(self, controller, viewer_widget, main_window, params):
        """Validate the branch and the pick, reconstruct, consume the chosen field,
        and settle the level.

        Returns ``(node, points, values, seed_point, level, tree_kd)`` — the
        KD-tree is reused by the tracer — or ``None`` when a check fails (a
        QMessageBox has been shown).
        """
        selected_branches = controller.selected_branches
        if not selected_branches:
            QMessageBox.warning(main_window, "No Branch Selected",
                                "Please select a branch first.")
            return None
        if len(selected_branches) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch at a time.")
            return None

        selected_uid = selected_branches[0]
        node = controller.get_node(selected_uid)
        if node is None:
            QMessageBox.warning(main_window, "Invalid Branch",
                                "Could not find the selected branch.")
            return None

        seed_point = self._picked_seed()
        if seed_point is None:
            QMessageBox.warning(
                main_window, "No Point Picked",
                "Shift+click one point on the contour you want, then run the "
                "plugin. The pick sets the level and where the flood starts.")
            return None

        try:
            point_cloud = controller.reconstruct(selected_uid)
        except Exception as e:
            QMessageBox.critical(main_window, "Reconstruction Error",
                                 f"Failed to reconstruct branch:\n{str(e)}")
            return None

        # --- Consume the upstream field (never computed here) ---
        field_key = params.get("field", "z")
        values = resolve_field(point_cloud, field_key)
        if values is None:
            QMessageBox.warning(
                main_window, "Field Not Available",
                f"The branch has no '{field_key}' field to contour.\n\n"
                "Fields are consumed from upstream — run the plugin that produces "
                "it on this branch first (e.g. normal_estimation for the normal "
                "components), then re-run.")
            return None
        values = np.asarray(values, dtype=np.float64).ravel()

        points = np.asarray(point_cloud.points, dtype=np.float64)[:, :3]
        if len(values) != len(points):
            QMessageBox.warning(
                main_window, "Field Size Mismatch",
                f"'{field_key}' has {len(values):,} values but the branch has "
                f"{len(points):,} points. The field belongs to a different cloud.")
            return None

        tree_kd = cKDTree(points)
        if params.get("level_from_seed", True):
            _, seed_idx = tree_kd.query(seed_point)
            level = float(values[int(seed_idx)])
        else:
            level = float(params.get("level", 0.0))

        return node, points, values, seed_point, level, tree_kd

    def _trace_threaded(self, main_window, tracer, seed_point):
        """Run ``tracer.trace`` on a daemon thread with a status-bar progress bar
        and cancel button (matching linear_region_growing's UX).

        Returns ``(polylines, stopped_early)``. ``polylines`` is ``None`` on error
        (a QMessageBox has been shown); on cancel it holds whatever was traced
        before the user stopped, and ``stopped_early`` is True.
        """
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.show_progress("Tracing contours...")
        main_window.show_cancel_button()  # clears any stale cancel flag

        cancel_event = global_variables.global_cancel_event
        state = {"polylines": None, "error": None, "done": False}

        def _progress(done, total, message):
            percent = int(100 * done / total) if total else None
            global_variables.global_progress = (percent, message)

        def _work():
            try:
                state["polylines"] = tracer.trace(
                    seed_point, progress_cb=_progress, cancel_event=cancel_event
                )
            except Exception as e:
                state["error"] = str(e)
            finally:
                state["done"] = True

        thread = threading.Thread(target=_work, daemon=True)
        thread.start()

        while not state["done"]:
            percent, msg = global_variables.global_progress
            if msg:
                main_window.show_progress(msg, percent)
            QApplication.processEvents()
            time.sleep(0.1)

        # Read the cancel flag BEFORE hide_cancel_button clears it.
        global_variables.global_progress = (None, "")
        stopped_early = cancel_event.is_set()
        main_window.hide_cancel_button()
        main_window.clear_progress()
        main_window.enable_menus()
        main_window.enable_tree()

        if state["error"]:
            QMessageBox.critical(main_window, "Contour Growing Failed", state["error"])
            return None, stopped_early
        return state["polylines"] or [], stopped_early

    @staticmethod
    def _add_result_branch(controller, tree_widget, node, feature, params):
        """Register the contours as one child branch and show it."""
        result_uid = controller.add_analysis_result(
            feature, "vector_feature", [node.uid], node, "contour_growing", params
        )
        tree_widget.add_branch(
            result_uid, str(node.uid), "contour_growing",
            tooltip=f"contour_growing,{params}"
        )
        tree_widget.blockSignals(True)
        item = tree_widget.branches_dict.get(result_uid)
        if item:
            item.setCheckState(0, Qt.Checked)
        tree_widget.visibility_status[result_uid] = True
        tree_widget.blockSignals(False)
        return result_uid

    @staticmethod
    def _show_summary(main_window, tracer, polylines, level, stopped_early):
        closed = sum(
            1 for p in polylines
            if len(p) > 2 and np.allclose(p[0], p[-1])
        )
        vertices = sum(len(p) for p in polylines)
        cancel_note = ("\n\nCancelled early — partial result saved."
                       if stopped_early else "")
        QMessageBox.information(
            main_window,
            "Contour Growing Cancelled" if stopped_early else "Contour Growing Complete",
            f"Traced {len(polylines)} contour line(s) at level {level:.4g} "
            f"— {vertices:,} vertices, {closed} closed loop(s), "
            f"{tracer.n_balls:,} steps." + cancel_note
        )
