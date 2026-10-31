"""
K-nearest concave hull — walk round the cluster's edge point by point.

Select a PointCloud branch (typically one cluster) and run. Starting from
the lowest point, each step goes to one of the k nearest unvisited points,
turning as little as possible while keeping the outline from crossing
itself (Moreira & Santos 2007). Smaller k → tighter outline. If the walk gets
stuck, or leaves points outside, it retries with a larger k automatically.
Runs on the CPU, so it is slow on very large clusters — use Grid Outline for
those.

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
_METHOD_LABEL = "K-nearest concave hull"


class KNNConcaveHullPlugin(Plugin):

    def get_name(self) -> str:
        return "knn_concave_hull"

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
            "k": {
                "type": "int",
                "default": 10,
                "min": 3,
                "max": 30,
                "label": "Neighbours (k)",
                "description": "How many nearest points each step chooses from. "
                               "Smaller is tighter; it retries with up to 30 "
                               "more automatically if the walk gets stuck.",
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
        k = int(params.get("k", 10))

        global_variables.global_progress = (
            None, f"K-nearest concave hull outline ({point_cloud.size:,} points)...")
        feature, _info = outline_feature(
            point_cloud.points, "knn_concave_hull", tolerance, height_tol=height_tol,
            projection=projection, drop_strays=drop_strays, k=k,
            should_stop=global_variables.global_cancel_event.is_set)
        if feature is None:
            raise ValueError(
                "K-nearest concave hull found no outline, even after retrying "
                "with larger k. Check the branch has points not all on one line, "
                "or use Grid Outline.")

        feature.cluster_reference = data_node.uid
        return feature, "vector_feature", [data_node.uid]
