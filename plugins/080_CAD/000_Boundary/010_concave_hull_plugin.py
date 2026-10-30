"""
Concave hull — an outline that hugs the cluster's bays and notches.

Select a PointCloud branch (typically one cluster) and run. The convex
outline is "dug into" along Delaunay edges longer than *Max Edge Length*
(a chi-shape). Smaller → tighter, more detailed outline; larger → closer to
the convex hull; 0 picks it from the point spacing. Slightly more detailed
than Grid Outline on small, clean clusters, but the Delaunay step runs on
the CPU and is slow on very large clusters — use Grid Outline for those.

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

# Above this many points the CPU-bound method asks before running.
_SLOW_ABOVE_POINTS = 2_000_000
_METHOD_LABEL = "Concave hull"


class ConcaveHullPlugin(Plugin):

    def get_name(self) -> str:
        return "concave_hull"

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
            "max_edge_length": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1000.0,
                "decimals": 3,
                "label": "Max Edge Length (m, 0 = auto)",
                "description": "Outline edges longer than this are carved inward "
                               "toward concavities: roughly the largest gap the "
                               "outline may bridge. 0 sizes it from the point "
                               "spacing.",
            },
        }

    def confirm_before_execute(self, data_node, params) -> Optional[str]:
        """Warn before a very large cluster: this method runs on the CPU and
        can take minutes there, where Grid Outline takes seconds."""
        from config.config import global_variables
        controller = global_variables.global_application_controller
        if controller is None:
            return None
        n = controller.get_node_reconstructed_count(data_node)
        if n is None or n < _SLOW_ABOVE_POINTS:
            return None
        return (f"This cluster has {n:,} points. {_METHOD_LABEL} runs on the "
                f"CPU and may take several minutes at this size; Grid Outline "
                f"gives the same kind of outline in seconds.\n\nContinue?")

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables

        point_cloud: PointCloud = data_node.data
        projection = params.get("projection", "best_fit")
        tolerance = float(params.get("tolerance", 0.05))
        height_tol = float(params.get("height_tolerance", 0.05))
        drop_strays = bool(params.get("ignore_strays", True))
        max_edge = float(params.get("max_edge_length", 0.0)) or None

        global_variables.global_progress = (
            None, f"Concave hull outline ({point_cloud.size:,} points)...")
        feature, _info = outline_feature(
            point_cloud.points, "concave_hull", tolerance, height_tol=height_tol,
            projection=projection, drop_strays=drop_strays, max_edge=max_edge)
        if feature is None:
            raise ValueError(
                "Concave hull found no outline. Increase Max Edge Length (or set "
                "0 for auto), or check the branch has points not all on one line.")

        feature.cluster_reference = data_node.uid
        return feature, "vector_feature", [data_node.uid]
