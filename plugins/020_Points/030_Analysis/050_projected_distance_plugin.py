"""
Projected Distance Plugin

For each point in the selected branch A, computes the signed perpendicular
distance to a reference branch B (chosen via dropdown):

    d_i = dot(A.points[i] - B.points[j*], B.normals[j*])

where j* = index of B's nearest point to A.points[i]. Sign indicates which
side of B's local surface the A-point lies on.
"""

import uuid
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

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
        for node_uid, node in data_nodes.data_nodes.items():
            node_options[str(node_uid)] = node.alias or node.params

        default_value = next(iter(node_options)) if node_options else ""

        return {
            "reference_node": {
                "type": "dropdown",
                "options": node_options,
                "default": default_value,
                "label": "Reference Branch (B)",
                "description": "Branch to measure signed distance against. "
                               "Its normals define the projection axis."
            },
            "recompute_reference_normals": {
                "type": "bool",
                "default": False,
                "label": "Recompute B Normals",
                "description": "If off, reuse existing normals on B when present."
            },
            "knn_for_normals": {
                "type": "int",
                "default": 30,
                "min": 3,
                "max": 200,
                "label": "KNN (normal estimation)",
                "description": "Neighborhood size used if normals must be estimated on B."
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

        need_normals = (
            params.get("recompute_reference_normals", False)
            or ref_pc.normals is None
            or len(ref_pc.normals) != len(ref_points)
        )
        if need_normals:
            k = int(params.get("knn_for_normals", 30))
            global_variables.global_progress = (
                None, f"Estimating normals on reference ({len(ref_points):,} points, k={k})..."
            )
            ref_pc.estimate_normals(k=k)

        ref_normals = np.asarray(ref_pc.normals)
        if ref_normals is None or len(ref_normals) != len(ref_points):
            raise ValueError("Reference branch normals are unavailable after estimation.")

        global_variables.global_progress = (
            30, f"Building KDTree on reference ({len(ref_points):,} points)..."
        )
        tree = cKDTree(ref_points)

        global_variables.global_progress = (
            60, f"Querying nearest reference point for {len(target_points):,} targets..."
        )
        _, idx = tree.query(target_points, k=1)

        global_variables.global_progress = (85, "Projecting displacements onto B normals...")
        displacement = target_points - ref_points[idx]
        signed = np.einsum('ij,ij->i', displacement, ref_normals[idx]).astype(np.float32)

        return Values(signed), "values", [data_node.uid, ref_uid]
