"""
Convex hull — the tightest *convex* outline of a (near-planar) point cloud.

Select a PointCloud branch (typically a single footprint / cluster) and run.
The cloud is projected onto a plane, the convex hull is computed in 2D, and the
boundary is returned as a closed-polyline ``VectorFeature`` branch whose
vertices are real cloud points (coordinates stay exact). The boundary work
itself lives in ``core.services.boundary_extraction``.
"""

from typing import Dict, Any, List, Tuple

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.services.boundary_extraction import convex_hull_feature

_PROJECTIONS = {
    "best_fit": "Best-fit plane (PCA)",
    "xy": "XY (plan / top-down)",
    "xz": "XZ (front)",
    "yz": "YZ (side)",
}


class ConvexHullPlugin(Plugin):

    def get_name(self) -> str:
        return "convex_hull"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "projection": {
                "type": "dropdown",
                "options": _PROJECTIONS,
                "default": "best_fit",
                "label": "Projection Plane",
                "description": "Plane the cloud is flattened onto before the 2D "
                               "hull is computed. Best-fit follows the cloud's "
                               "own plane; XY gives a plan-view footprint.",
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables

        point_cloud: PointCloud = data_node.data
        projection = params.get("projection", "best_fit")

        global_variables.global_progress = (
            None, f"Computing convex hull ({point_cloud.size:,} points)...")
        feature = convex_hull_feature(point_cloud.points, projection)
        if feature is None:
            raise ValueError(
                "Convex hull needs at least 3 non-collinear points in the "
                "selected branch.")

        return feature, "vector_feature", [data_node.uid]
