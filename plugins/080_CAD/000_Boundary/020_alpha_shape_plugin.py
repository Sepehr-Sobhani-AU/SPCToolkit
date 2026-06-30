"""
Alpha shape — the alpha-complex boundary of a (near-planar) point cloud.

Delaunay-triangulate the projected cloud, keep triangles whose circumradius is
≤ *Alpha*, and the resulting boundary edges are the outline. Unlike the convex
/ concave hull it can represent holes and disjoint pieces, so it is returned as
a wireframe ``VectorFeature`` (mesh edges) rather than a single polyline. As
*Alpha* grows the shape fills in toward the convex hull; as it shrinks the
outline tightens and may break into pieces. Boundary work lives in
``core.services.boundary_extraction``.
"""

from typing import Dict, Any, List, Tuple

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.services.boundary_extraction import alpha_shape_feature

_PROJECTIONS = {
    "best_fit": "Best-fit plane (PCA)",
    "xy": "XY (plan / top-down)",
    "xz": "XZ (front)",
    "yz": "YZ (side)",
}


class AlphaShapePlugin(Plugin):

    def get_name(self) -> str:
        return "alpha_shape"

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
            "alpha": {
                "type": "float",
                "default": 0.5,
                "min": 0.001,
                "max": 1000.0,
                "decimals": 3,
                "label": "Alpha (max circumradius, m)",
                "description": "Triangles whose circumradius exceeds this are "
                               "discarded, exposing the boundary. Larger → fills "
                               "toward the convex hull; smaller → tighter outline "
                               "that may split into pieces (set near point spacing).",
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables

        point_cloud: PointCloud = data_node.data
        projection = params.get("projection", "best_fit")
        alpha = float(params.get("alpha", 0.5))

        global_variables.global_progress = (
            None, f"Computing alpha shape ({point_cloud.size:,} points)...")
        feature = alpha_shape_feature(point_cloud.points, alpha, projection)
        if feature is None:
            raise ValueError(
                "Alpha shape produced no boundary. Increase Alpha, or check the "
                "branch has at least 3 non-collinear points.")

        return feature, "vector_feature", [data_node.uid]
