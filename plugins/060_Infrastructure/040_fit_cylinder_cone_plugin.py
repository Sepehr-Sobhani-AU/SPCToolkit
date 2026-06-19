"""
Fit Cylinder / Cone Plugin

Fits a cylinder or cone (rendered as a truncated frustum) to each selected
cluster of the selected branch using the RANSAC service, then emits a
render-only ``VectorFeature`` wireframe over the points so the fit can be
eyeballed. This is the geometry step of vectorising "block"-type features
(poles, bollards, pipes, tree trunks, tapered standards). Orientation is an
output of the fit, never assumed — a vertical pole and a horizontal pipe are
the same call.

Selection model: select ONE branch, make it visible, and pick points on the
clusters you want (Shift+Click / polygon select, the same mechanism as
Classify Cluster / Separate Selected Clusters). The plugin fits one primitive
per selected cluster. If the branch has no cluster labels, the whole branch is
fitted as a single primitive.

Workflow:
  1. Cluster the cloud (Points > Clustering) and/or separate a feature.
  2. Compute normals (Points > Analysis > Estimate Normals) — prerequisite.
  3. Select the (normals) branch, make it visible, pick the clusters to fit,
     and run this plugin.

Normals are a prerequisite, not computed here. The cylinder/cone RANSAC
models require per-point normals. This plugin reads them from the
reconstructed cloud (``point_cloud.normals``, populated when an Estimate
Normals step is in the branch's lineage — its attributes, including
``cluster_labels``, are carried forward) and refuses to fit a branch that has
none. Compute them once upstream and reuse.

Semantics vs geometry: this plugin only fits *geometry*. What the feature
*is* (power pole vs bollard vs pipe) comes from the cluster's class
(Clusters > Classify Cluster), which becomes the DXF layer at export. The
fitted ``VectorFeature`` is render-only; its analytic parameters (base, axis,
radius, height, cluster_id) are stashed as extra keys inside ``geometry`` for
the deferred DXF block/centerline export to read via ``cluster_reference``.

The cylinder/cone RANSAC fit is CPU-only by design — the models declare
``supports_gpu = False`` (iterative LM refit, no batched consumer). A single
cluster fits in milliseconds; this is not a silent GPU fallback.
"""

import time
import uuid
import logging
import threading
import traceback
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.entities.point_cloud import PointCloud
from core.entities.data_node import DataNode
from core.entities.vector_feature import VectorFeature
from core.services.ransac import fit

logger = logging.getLogger(__name__)

_EPS = 1e-12
_N_SEGMENTS = 24                       # wireframe resolution around the axis
_MIN_POINTS = 10                       # below this a fit is meaningless


_COLOR_CYLINDER = np.array([0.0, 1.0, 1.0], dtype=np.float32)   # cyan
_COLOR_CONE = np.array([1.0, 0.65, 0.0], dtype=np.float32)      # orange


# ── Pure geometry helpers (no Qt / no globals — unit-testable) ────────────


def _perp_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal basis (u, v) of the plane perpendicular to ``axis``."""
    axis = axis / (np.linalg.norm(axis) + _EPS)
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(axis[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u) + _EPS
    v = np.cross(axis, u)
    return u, v


def _truncate_cylinder(model, inlier_pts: np.ndarray) -> Dict[str, Any]:
    """Turn an infinite cylinder model into a finite segment over the inliers.

    The model gives an arbitrary anchor on the axis, a unit direction, and a
    radius. We project the inliers onto the axis to find the extent, and order
    the endpoints so ``base`` is the lower one in world Z (pole base at ground).
    """
    axis = np.asarray(model.direction, dtype=np.float64)
    axis /= np.linalg.norm(axis) + _EPS
    anchor = np.asarray(model.point, dtype=np.float64)
    t = (inlier_pts - anchor) @ axis
    end_a = anchor + float(t.min()) * axis
    end_b = anchor + float(t.max()) * axis
    base, top = (end_a, end_b) if end_a[2] <= end_b[2] else (end_b, end_a)
    r = float(model.radius)
    return {
        "primitive": "cylinder",
        "base": base, "top": top,
        "radius_base": r, "radius_top": r,
        "half_angle": None,
    }


def _truncate_cone(model, inlier_pts: np.ndarray) -> Dict[str, Any]:
    """Turn an infinite cone model into a finite frustum over the inliers.

    The model gives the apex, a unit axis (apex → wide end), and a half-angle.
    The radius at axial distance ``s`` from the apex is ``s * tan(half_angle)``.
    We clip to the inlier extent (never draw to the apex) and order endpoints
    by world Z.
    """
    apex = np.asarray(model.apex, dtype=np.float64)
    axis = np.asarray(model.axis, dtype=np.float64)
    axis /= np.linalg.norm(axis) + _EPS
    half_angle = float(model.half_angle)
    tan_a = float(np.tan(half_angle))

    s = (inlier_pts - apex) @ axis
    s_lo, s_hi = float(s.min()), float(s.max())
    end_lo, r_lo = apex + s_lo * axis, abs(s_lo) * tan_a
    end_hi, r_hi = apex + s_hi * axis, abs(s_hi) * tan_a
    if end_lo[2] <= end_hi[2]:
        base, r_base, top, r_top = end_lo, r_lo, end_hi, r_hi
    else:
        base, r_base, top, r_top = end_hi, r_hi, end_lo, r_lo
    return {
        "primitive": "cone",
        "base": base, "top": top,
        "radius_base": float(r_base), "radius_top": float(r_top),
        "half_angle": half_angle,
    }


def _tessellate_frustum(
    base: np.ndarray,
    top: np.ndarray,
    r_base: float,
    r_top: float,
    n_seg: int = _N_SEGMENTS,
) -> Tuple[np.ndarray, List[List[int]], np.ndarray]:
    """Tessellate a truncated cone (cylinder when ``r_base == r_top``).

    Vertices are built directly in world coordinates (so the VectorFeature
    transform is identity, mirroring the polyline path). Returns
    ``(vertices (2n,3) float32, faces list[n], edges (3n,2) int32)``.
    """
    base = np.asarray(base, dtype=np.float64)
    top = np.asarray(top, dtype=np.float64)
    axis = top - base
    height = float(np.linalg.norm(axis))
    if height < _EPS:
        raise ValueError("degenerate primitive: zero height")
    axis /= height
    u, v = _perp_basis(axis)

    ang = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    ring = np.cos(ang)[:, None] * u[None, :] + np.sin(ang)[:, None] * v[None, :]
    base_ring = base[None, :] + r_base * ring
    top_ring = top[None, :] + r_top * ring
    vertices = np.vstack([base_ring, top_ring]).astype(np.float32)

    edges: List[List[int]] = []
    faces: List[List[int]] = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        edges.append([i, j])                  # base circle
        edges.append([n_seg + i, n_seg + j])  # top circle
        edges.append([i, n_seg + i])          # vertical
        faces.append([i, j, n_seg + j, n_seg + i])
    return vertices, faces, np.asarray(edges, dtype=np.int32)


def _fit_quality(model, points: np.ndarray, mask: np.ndarray, geom: Dict[str, Any]) -> Dict[str, Any]:
    """Inspection metrics: inlier ratio, RMS over inliers, tilt from vertical."""
    d = model.distances(points)
    n_in = int(mask.sum())
    inlier_d = d[mask]
    rms = float(np.sqrt(np.mean(inlier_d ** 2))) if n_in else float("nan")
    ratio = n_in / len(points) if len(points) else 0.0
    axis = np.asarray(geom["top"]) - np.asarray(geom["base"])
    axis /= np.linalg.norm(axis) + _EPS
    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(float(axis[2]))))))
    return {"inlier_ratio": ratio, "rms": rms, "n_inliers": n_in, "tilt_deg": tilt_deg}


def _radius_label(geom: Dict[str, Any]) -> str:
    if geom["primitive"] == "cone":
        return f"r={geom['radius_base']:.3f}->{geom['radius_top']:.3f}m"
    return f"r={geom['radius_base']:.3f}m"


# ── Fit orchestration (no Qt) ─────────────────────────────────────────────


def _fit_primitive(
    points: np.ndarray,
    normals: np.ndarray,
    shape: str,
    threshold: float,
    max_iterations: int,
    min_inlier_ratio: float,
) -> Tuple[Optional[object], Optional[np.ndarray], Optional[str]]:
    """Fit the requested shape (or both for ``auto``) and keep the best.

    Best = most inliers, tie-broken by lower inlier RMS. Returns
    ``(model, mask, primitive_name)`` or ``(None, None, None)``.
    """
    shapes = ["cylinder", "cone"] if shape == "auto" else [shape]
    best = None  # ((n_inliers, -rms), name, model, mask)
    for s in shapes:
        try:
            model, mask = fit(
                points, s, threshold=threshold, normals=normals,
                max_iterations=max_iterations, min_inlier_ratio=min_inlier_ratio,
                seed=0,
            )
        except Exception as exc:
            logger.warning("%s fit raised: %s", s, exc)
            continue
        if model is None or mask is None:
            continue
        n_in = int(mask.sum())
        d = model.distances(points)
        rms = float(np.sqrt(np.mean(d[mask] ** 2))) if n_in else float("inf")
        key = (n_in, -rms)
        if best is None or key > best[0]:
            best = (key, s, model, mask)
    if best is None:
        return None, None, None
    _, name, model, mask = best
    return model, mask, name


def _build_feature(
    points: np.ndarray,
    model,
    mask: np.ndarray,
    primitive: str,
    cluster_uid: uuid.UUID,
    cluster_id: Optional[int] = None,
    cluster_class: Optional[str] = None,
) -> Tuple[VectorFeature, Dict[str, Any]]:
    """Truncate + tessellate the fitted model into a render-only VectorFeature."""
    inlier_pts = points[mask]
    geom = (_truncate_cylinder if primitive == "cylinder" else _truncate_cone)(model, inlier_pts)
    vertices, faces, edges = _tessellate_frustum(
        geom["base"], geom["top"], geom["radius_base"], geom["radius_top"]
    )
    quality = _fit_quality(model, points, mask, geom)

    axis = np.asarray(geom["top"]) - np.asarray(geom["base"])
    height = float(np.linalg.norm(axis))
    axis = axis / (np.linalg.norm(axis) + _EPS)
    max_r = max(geom["radius_base"], geom["radius_top"])

    geometry = {
        # ── render keys (read by rendering_coordinator) ──
        "vertices": vertices,
        "faces": [list(map(int, f)) for f in faces],
        "edges": edges,
        # ── analytic keys (ignored by renderer/validator; for DXF export) ──
        "primitive": primitive,
        "base": np.asarray(geom["base"], dtype=np.float32),
        "top": np.asarray(geom["top"], dtype=np.float32),
        "axis": axis.astype(np.float32),
        "radius_base": float(geom["radius_base"]),
        "radius_top": float(geom["radius_top"]),
        "half_angle": geom["half_angle"],
        "height": height,
        "cluster_id": (int(cluster_id) if cluster_id is not None else None),
        "cluster_class": cluster_class,
        "fit": quality,
    }
    color = _COLOR_CYLINDER if primitive == "cylinder" else _COLOR_CONE
    feature = VectorFeature(
        symbol_type=primitive,
        geometry_type="mesh",
        geometry=geometry,
        transform_matrix=np.eye(4),
        dimensions=np.array([2.0 * max_r, 2.0 * max_r, height], dtype=np.float32),
        cluster_reference=cluster_uid,
        color=color,
    )
    return feature, quality


def _extract_normals(pc: PointCloud) -> Optional[np.ndarray]:
    """Return precomputed per-point normals aligned to ``pc.points``, or None.

    Normals come from an upstream Estimate Normals step (the NormalsTransformer
    populates both ``pc.normals`` and the ``'normals'`` attribute). This plugin
    never computes them.
    """
    cand = pc.normals
    if cand is None or (hasattr(cand, "__len__") and len(cand) == 0):
        cand = pc.get_attribute("normals")
    if cand is None:
        return None
    cand = np.asarray(cand)
    if cand.ndim != 2 or cand.shape != (len(pc.points), 3):
        return None
    return cand.astype(np.float64)


# ── Plugin ────────────────────────────────────────────────────────────────


class FitCylinderConePlugin(ActionPlugin):
    """Fit a cylinder or cone (frustum) to each selected cluster of a branch."""

    def get_name(self) -> str:
        return "fit_cylinder_cone"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "info": {
                "type": "info",
                "default": "Select ONE branch, pick the clusters to fit "
                           "(Shift+Click), and ensure normals were computed "
                           "(Points > Analysis > Estimate Normals).",
                "label": "How to use",
            },
            "shape": {
                "type": "dropdown",
                "options": {
                    "auto": "Auto (best of both)",
                    "cylinder": "Cylinder",
                    "cone": "Cone",
                },
                "default": "auto",
                "label": "Primitive",
                "description": "Fit a cylinder, a cone/frustum, or try both and "
                               "keep the better fit (more inliers).",
            },
            "distance_threshold": {
                "type": "float",
                "default": 0.02,
                "min": 0.0,
                "max": 100.0,
                "decimals": 4,
                "label": "Inlier Threshold",
                "description": "Half-width of the inlier band, in cloud units. "
                               "Set to roughly the point noise / surface thickness.",
            },
            "min_inlier_ratio": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "decimals": 2,
                "label": "Min Inlier Ratio",
                "description": "Reject the fit if fewer than this fraction of "
                               "the cluster's points are inliers.",
            },
            "max_iterations": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 20000,
                "label": "RANSAC Iterations",
                "description": "Number of random hypotheses to try.",
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget
        viewer_widget = global_variables.global_pcd_viewer_widget

        selected = controller.selected_branches
        if not selected:
            QMessageBox.warning(main_window, "No Selection",
                                "Select the branch whose clusters you want to fit.")
            return
        if len(selected) > 1:
            QMessageBox.warning(main_window, "Multiple Branches",
                                "Please select only ONE branch at a time.")
            return
        selected_uid = selected[0]

        shape = params.get("shape", "auto")
        threshold = float(params.get("distance_threshold", 0.02))
        min_ratio = float(params.get("min_inlier_ratio", 0.3))
        max_iter = int(params.get("max_iterations", 1000))

        # ── Reconstruct the branch (cached = fast) ──
        try:
            pc: PointCloud = controller.reconstruct(selected_uid)
        except Exception as exc:
            QMessageBox.critical(main_window, "Reconstruction Error", str(exc))
            return
        if pc is None or pc.points is None or len(pc.points) == 0:
            QMessageBox.warning(main_window, "Empty Branch",
                                "The selected branch has no points.")
            return

        points = np.asarray(pc.points, dtype=np.float64)

        # ── Normals are a prerequisite (carried forward through reconstruct) ──
        normals = _extract_normals(pc)
        if normals is None:
            QMessageBox.warning(
                main_window, "Normals Required",
                "This branch has no normals.\n\n"
                "Run Points > Analysis > Estimate Normals, then select the "
                "resulting normals branch and fit again.",
            )
            return

        # ── Decide which clusters to fit ──
        # tasks: list of (cluster_id_or_None, class_name_or_None, point_mask)
        cluster_labels = pc.get_attribute("cluster_labels")
        cluster_names = pc.get_attribute("_cluster_names") or {}
        tasks: List[Tuple[Optional[int], Optional[str], np.ndarray]] = []

        if cluster_labels is not None:
            cluster_labels = np.asarray(cluster_labels)
            try:
                selection_mask = viewer_widget.get_selection_mask_for(points)
            except Exception as exc:
                logger.warning("Selection mask failed: %s", exc)
                selection_mask = None
            if selection_mask is None or not np.any(selection_mask):
                QMessageBox.warning(
                    main_window, "No Clusters Selected",
                    "Pick points on the clusters you want to fit first "
                    "(Shift+Click or polygon select), then run again.",
                )
                return
            selected_ids = [int(c) for c in np.unique(cluster_labels[selection_mask]) if int(c) != -1]
            if not selected_ids:
                QMessageBox.warning(
                    main_window, "No Valid Clusters",
                    "The selected points are all noise (cluster id -1).",
                )
                return
            for cid in selected_ids:
                cname = cluster_names.get(cid) if isinstance(cluster_names, dict) else None
                tasks.append((cid, cname, cluster_labels == cid))
        else:
            # No clustering on this branch — fit the whole branch as one primitive.
            tasks.append((None, None, np.ones(len(points), dtype=bool)))

        # ── Heavy compute in a worker thread (CPU RANSAC) ──
        main_window.disable_menus()
        main_window.disable_tree()
        main_window.show_progress("Fitting primitives...")
        global_variables.global_cancel_event.clear()

        state: Dict[str, Any] = {"results": [], "error": None, "done": False}

        def _work():
            try:
                results = []
                total = len(tasks)
                for i, (cid, cname, cmask) in enumerate(tasks, start=1):
                    if global_variables.global_cancel_event.is_set():
                        raise InterruptedError("Cancelled by user")
                    tag = cname or (f"cluster {cid}" if cid is not None else "branch")
                    global_variables.global_progress = (
                        int((i - 1) / total * 100), f"Fitting {i}/{total}: {tag}"
                    )
                    cpts = points[cmask]
                    cnorm = normals[cmask]
                    if len(cpts) < _MIN_POINTS:
                        results.append((cid, cname, None, f"too few points (<{_MIN_POINTS})"))
                        continue

                    model, inlier_mask, primitive = _fit_primitive(
                        cpts, cnorm, shape, threshold, max_iter, min_ratio
                    )
                    if model is None:
                        results.append((cid, cname, None,
                                        f"no fit met inlier ratio {min_ratio:.2f}"))
                        continue
                    try:
                        feature, quality = _build_feature(
                            cpts, model, inlier_mask, primitive,
                            uuid.UUID(selected_uid), cluster_id=cid, cluster_class=cname,
                        )
                    except Exception as exc:
                        results.append((cid, cname, None, f"geometry failed: {exc}"))
                        continue
                    results.append((cid, cname, feature, quality))
                state["results"] = results
            except Exception as exc:
                state["error"] = str(exc)
                logger.error(traceback.format_exc())
            finally:
                state["done"] = True

        thread = threading.Thread(target=_work, daemon=True)
        thread.start()
        while not state["done"]:
            percent, msg = global_variables.global_progress
            if msg:
                main_window.show_progress(msg, percent)
            QtWidgets.QApplication.processEvents()
            time.sleep(0.05)
        global_variables.global_progress = (None, "")

        if state["error"]:
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()
            QMessageBox.critical(main_window, "Fit Failed", state["error"])
            return

        # ── Add result nodes on the main thread ──
        main_window.show_progress("Adding results...", 95)
        tree_widget.blockSignals(True)
        created = 0
        ok_lines: List[str] = []
        failed_lines: List[str] = []
        try:
            for cid, cname, feature, info in state["results"]:
                disp = cname or (f"cluster {cid}" if cid is not None else "branch")
                if feature is None:
                    failed_lines.append(f"  • {disp}: {info}")
                    continue
                quality = info
                geom = feature.geometry
                primitive = feature.symbol_type
                suffix = cname or (f"c{cid}" if cid is not None else "all")
                label = f"{primitive}_fit_{suffix}"
                tooltip = (
                    f"{primitive} | {_radius_label(geom)} h={geom['height']:.2f}m "
                    f"tilt={quality['tilt_deg']:.1f}° "
                    f"inliers={quality['inlier_ratio'] * 100:.0f}% "
                    f"rms={quality['rms']:.4f}"
                )
                node = DataNode(
                    params=label,
                    data=feature,
                    data_type="vector_feature",
                    parent_uid=uuid.UUID(selected_uid),
                    depends_on=[uuid.UUID(selected_uid)],
                    tags=["cad", "fit", primitive],
                )
                new_uid = data_nodes.add_node(node)
                tree_widget.add_branch(str(new_uid), selected_uid, label, tooltip=tooltip)
                item = tree_widget.branches_dict.get(str(new_uid))
                if item:
                    item.setCheckState(0, Qt.Checked)
                tree_widget.visibility_status[str(new_uid)] = True
                created += 1
                ok_lines.append(f"  • {disp}: {tooltip}")
        except Exception:
            logger.error(traceback.format_exc())
        finally:
            tree_widget.blockSignals(False)
            main_window.clear_progress()
            main_window.enable_menus()
            main_window.enable_tree()

        main_window.render_visible_data(zoom_extent=False)

        lines: List[str] = []
        if ok_lines:
            lines.append(f"Fitted {created} primitive(s):")
            lines.extend(ok_lines)
        if failed_lines:
            if lines:
                lines.append("")
            lines.append(f"Skipped {len(failed_lines)}:")
            lines.extend(failed_lines)
        QMessageBox.information(
            main_window, "Fit Cylinder / Cone",
            "\n".join(lines) if lines else "Nothing was fitted.",
        )
