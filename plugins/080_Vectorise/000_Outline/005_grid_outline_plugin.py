"""
Grid outline — the recommended way to outline a cluster.

Select a PointCloud branch (typically one cluster) and run. The points are
laid on a fine grid in the chosen plane; the cells holding points are filled,
small gaps closed, and the outer edge of the filled area is traced. It is
fast on millions of points and steady when the point density varies. The
cell size is worked out from the points' own spacing unless you set it.

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


class GridOutlinePlugin(Plugin):

    def get_name(self) -> str:
        return "grid_outline"

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
            "cell_size": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "decimals": 3,
                "label": "Cell Size (m, 0 = auto)",
                "description": "Grid cell size. 0 sizes it from the point "
                               "spacing; larger bridges bigger gaps but gives a "
                               "coarser outline.",
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables

        point_cloud: PointCloud = data_node.data
        projection = params.get("projection", "best_fit")
        tolerance = float(params.get("tolerance", 0.05))
        height_tol = float(params.get("height_tolerance", 0.05))
        drop_strays = bool(params.get("ignore_strays", True))
        cell = float(params.get("cell_size", 0.0)) or None

        global_variables.global_progress = (
            None, f"Grid outline ({point_cloud.size:,} points)...")
        feature, _info = outline_feature(
            point_cloud.points, "grid_outline", tolerance, height_tol=height_tol,
            projection=projection, drop_strays=drop_strays, cell=cell)
        if feature is None:
            raise ValueError(
                "Grid outline found no outline. Check the branch holds a cluster "
                "of points, or set a larger Cell Size.")

        feature.cluster_reference = data_node.uid
        return feature, "vector_feature", [data_node.uid]
