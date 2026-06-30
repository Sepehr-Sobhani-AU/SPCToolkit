"""
Concave hull — an outline that hugs the cloud's concavities.

Like the convex hull, but the convex outline is "dug into" along Delaunay
edges longer than *Max Edge Length*, so bays and notches in the footprint are
followed (a chi-shape). Smaller *Max Edge Length* → tighter, more detailed
outline; very large → converges to the convex hull. Returns a closed-polyline
``VectorFeature`` whose vertices are real cloud points. The boundary work lives
in ``core.services.boundary_extraction``.
"""

from typing import Dict, Any, List, Tuple

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.services.boundary_extraction import concave_hull_feature

_PROJECTIONS = {
    "best_fit": "Best-fit plane (PCA)",
    "xy": "XY (plan / top-down)",
    "xz": "XZ (front)",
    "yz": "YZ (side)",
}


class ConcaveHullPlugin(Plugin):

    def get_name(self) -> str:
        return "concave_hull"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "projection": {
                "type": "dropdown",
                "options": _PROJECTIONS,
                "default": "best_fit",
                "label": "Projection Plane",
                "description": "Plane the cloud is flattened onto before the 2D "
                               "boundary is computed.",
            },
            "max_edge_length": {
                "type": "float",
                "default": 1.0,
                "min": 0.001,
                "max": 1000.0,
                "decimals": 3,
                "label": "Max Edge Length (m)",
                "description": "Boundary edges longer than this are carved inward "
                               "toward concavities. Roughly the largest gap the "
                               "outline may bridge; larger → closer to the convex "
                               "hull, smaller → tighter (set near point spacing).",
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables

        point_cloud: PointCloud = data_node.data
        projection = params.get("projection", "best_fit")
        max_edge_length = float(params.get("max_edge_length", 1.0))

        global_variables.global_progress = (
            None, f"Computing concave hull ({point_cloud.size:,} points)...")
        feature = concave_hull_feature(point_cloud.points, max_edge_length, projection)
        if feature is None:
            raise ValueError(
                "Concave hull produced no closed outline. Increase Max Edge "
                "Length, or check the branch has at least 3 non-collinear points.")

        return feature, "vector_feature", [data_node.uid]
