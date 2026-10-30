"""
Convex hull — the tightest *convex* outline of a cluster.

Select a PointCloud branch (typically one cluster) and run. Quick and has no
method setting, but it only suits convex shapes: bays and notches are
bridged straight across. For general shapes use Grid Outline.

Every Boundary method gives the same kind of result: a clean, closed 3D
polyline (``VectorFeature`` branch) with straight segments only. It follows
the cluster's outermost points; each vertex's height comes from a small plane
fitted through the nearest cluster points; and weeding is 3D, so a vertex that
matters only for its height (e.g. a gable peak) is kept. The work lives in
``core.services.boundary_extraction`` and ``core.services.outline_cleanup``.
"""

from typing import Any, Dict, List, Optional, Tuple

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.services.boundary_extraction import outline_feature

_PROJECTIONS = {
    "best_fit": "Best-fit plane (PCA)",
    "xy": "XY (plan / top-down)",
    "xz": "XZ (front)",
    "yz": "YZ (side)",
}


class ConvexHullPlugin(Plugin):

    def get_name(self) -> str:
        return "convex_hull"

    def requires_selection(self) -> Optional[str]:
        # Pipeline replay pauses here so the user picks the cluster branch to
        # outline, instead of outlining whatever the previous step produced.
        return "branches"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "projection": {
                "type": "dropdown",
                "options": _PROJECTIONS,
                "default": "best_fit",
                "label": "Projection Plane",
                "description": "Plane the outline is traced in. Best-fit follows "
                               "the cluster's own plane (a wall is outlined on "
                               "the wall); the others are fixed views.",
            },
            "tolerance": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "decimals": 3,
                "label": "Tolerance (m)",
                "description": "How far the clean outline may stray from the "
                               "cluster's edge, measured in the projection "
                               "plane. Wiggles smaller than this are treated "
                               "as noise.",
            },
            "height_tolerance": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "decimals": 3,
                "label": "Height Tolerance (m)",
                "description": "A vertex is kept if dropping it would change the "
                               "outline's height by more than this (keeps e.g. "
                               "a gable peak).",
            },
            "ignore_strays": {
                "type": "bool",
                "default": True,
                "label": "Ignore Stray Points",
                "description": "Leave isolated stray points out of the outline.",
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables

        point_cloud: PointCloud = data_node.data
        projection = params.get("projection", "best_fit")
        tolerance = float(params.get("tolerance", 0.05))
        height_tol = float(params.get("height_tolerance", 0.05))
        drop_strays = bool(params.get("ignore_strays", True))

        global_variables.global_progress = (
            None, f"Convex hull outline ({point_cloud.size:,} points)...")
        feature, _info = outline_feature(
            point_cloud.points, "convex_hull", tolerance, height_tol=height_tol,
            projection=projection, drop_strays=drop_strays)
        if feature is None:
            raise ValueError(
                "Convex hull found no outline. The branch needs points that are "
                "not all on one line.")

        feature.cluster_reference = data_node.uid
        return feature, "vector_feature", [data_node.uid]
