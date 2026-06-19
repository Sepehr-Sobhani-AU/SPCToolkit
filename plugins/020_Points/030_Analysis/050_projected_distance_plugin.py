"""
Vertical (Z-parallel) distance plugin (registered as "projected_distance").

For each point A in the selected branch, the distance is measured *parallel to
the Z axis* to the reference branch B:

    1. take only A's horizontal position (Ax, Ay);
    2. find B's nearest point in the XY plane (a 2-D nearest-neighbour search),
       within a limited lateral radius;
    3. the distance is the height difference  d_i = Az - Bz  to that neighbour.

The result is naturally signed: positive when A sits above B, negative when
below. This separates the two sides of a reference surface without any axis
parameter -- e.g. points above vs below ground, or floor vs ceiling inside a
room.

The lateral search is capped by ``max_radius``: an A point with no B point
within that XY radius has nothing directly above/below it to measure to (it
overhangs B's coverage), so it is marked no-measurement (NaN, rendered grey).
Default 1.0 m; set 0 to disable the cap.
"""

import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.values import Values


class ProjectedDistancePlugin(Plugin):

    def get_name(self) -> str:
        return "projected_distance"

    def get_parameters(self) -> Dict[str, Any]:
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes

        node_options = {}
        if data_nodes is not None:
            for node_uid, node in data_nodes.data_nodes.items():
                node_options[str(node_uid)] = node.alias or node.params

        default_value = next(iter(node_options)) if node_options else ""

        return {
            "reference_node": {
                "type": "dropdown",
                "options": node_options,
                "default": default_value,
                "label": "Reference Branch (B)",
                "description": "Branch to measure the vertical (Z) distance to.",
            },
            "max_radius": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 100000.0,
                "label": "Max XY search radius",
                "description": (
                    "Lateral (horizontal, XY-plane) cap on the search for B's "
                    "nearest point. An A point with no B point within this radius "
                    "has nothing directly below/above it to measure to (it "
                    "overhangs B's coverage, e.g. past the scanned ground edge): "
                    "it is marked no-measurement (NaN) and renders grey. This is "
                    "a horizontal radius only -- it does NOT limit the vertical "
                    "gap, so a tall clearance is still measured as long as a B "
                    "point lies within this radius beneath it. 0 = no cap."
                ),
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes
        controller = global_variables.global_application_controller

        target_pc: PointCloud = data_node.data
        target_points = np.asarray(target_pc.points)
        if target_points.size == 0:
            raise ValueError("Target branch has no points.")

        try:
            ref_uid = uuid.UUID(params["reference_node"])
        except Exception as e:
            raise ValueError(f"Invalid reference branch selection: {e}")

        if ref_uid == data_node.uid:
            raise ValueError("Reference branch must differ from the target branch.")

        ref_node = data_nodes.get_node(ref_uid)
        if ref_node is None:
            raise ValueError(f"Reference branch {ref_uid} not found.")

        global_variables.global_progress = (None, "Reconstructing reference branch...")
        ref_pc: PointCloud = controller.reconstruct(ref_uid)
        ref_points = np.asarray(ref_pc.points)
        if ref_points.size == 0:
            raise ValueError("Reference branch has no points.")

        # Step 1: take only the horizontal (XY) coordinates of both clouds.
        target_xy = np.ascontiguousarray(target_points[:, :2], dtype=np.float32)
        ref_xy = np.ascontiguousarray(ref_points[:, :2], dtype=np.float32)

        # Step 2: nearest B point in the XY plane (2-D nearest-neighbour). The KNN
        # backend works on any dimensionality, so a 2-column query stays on GPU.
        registry = global_variables.global_backend_registry
        knn = registry.get_knn()
        global_variables.global_progress = (
            60,
            f"XY nearest-neighbor query ({knn.name}): {len(target_xy):,} targets "
            f"vs {len(ref_xy):,} reference points..."
        )
        # Build the index on B's XY, query A's XY. Only indices are used (the XY
        # distance is recomputed below), so backend distance units never matter.
        _, idx = knn.query(target_xy, k=1, reference=ref_xy)
        idx = idx[:, 0]

        global_variables.global_progress = (85, "Computing vertical (Z) distance...")
        # Lateral (XY) distance to that neighbour -- used only for the radius cap.
        xy_dist = np.linalg.norm(target_xy - ref_xy[idx], axis=1).astype(np.float32)
        # Step 3: the distance is the signed height difference Az - Bz
        # (+ when A is above B, - when below).
        dz = (target_points[:, 2] - ref_points[idx, 2]).astype(np.float32)

        # Lateral radius cap: an A point whose nearest B is beyond `max_radius`
        # away horizontally has nothing directly under/over it -> no-measurement.
        max_radius = float(params.get("max_radius", 1.0))
        if max_radius > 0.0:
            beyond = xy_dist > max_radius
            n_beyond = int(np.count_nonzero(beyond))
            dz[beyond] = np.nan
        else:
            n_beyond = 0

        # Summary (NaN-aware: capped points are excluded from the range/mean).
        n_valid = int(np.count_nonzero(np.isfinite(dz)))
        d_min = float(np.nanmin(dz)) if n_valid else float("nan")
        d_max = float(np.nanmax(dz)) if n_valid else float("nan")
        d_mean = float(np.nanmean(dz)) if n_valid else float("nan")
        tail = (
            f"{n_beyond} beyond {max_radius:g} m laterally -> NaN"
            if max_radius > 0.0 else "no radius cap"
        )
        global_variables.global_progress = (
            100,
            f"{n_valid}/{len(dz)} measured | Z-distance [{d_min:.4f}, {d_max:.4f}]"
        )
        print(
            f"[projected_distance] {n_valid}/{len(dz)} measured ({tail}); "
            f"Z-distance min={d_min:.4f} max={d_max:.4f} mean={d_mean:.4f}"
        )

        return Values(dz), "values", [data_node.uid, ref_uid]
