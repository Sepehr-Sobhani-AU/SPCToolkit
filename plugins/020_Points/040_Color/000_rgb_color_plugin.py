# plugins/020_Points/040_Color/000_rgb_color_plugin.py
from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.colors import Colors


class RGBColorPlugin(Plugin):
    """
    Colour a branch by the RGB colours its points already carry.

    Takes the selected branch's own per-point colours (the scan's true colour,
    preserved through subsampling/filtering/clustering subsets) and emits them as
    a ``colors`` node. Useful to restore true-colour display on a branch whose
    colours were overwritten by a downstream colouring (e.g. by Z value or
    eigenvalue visualisation).
    """

    def get_name(self) -> str:
        return "rgb_color"

    def get_parameters(self) -> Dict[str, Any]:
        # No parameters: the colours come straight from the branch's points.
        return {}

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud: PointCloud = data_node.data

        colors = point_cloud.colors
        if colors is None or len(colors) == 0:
            raise ValueError(
                "This branch has no RGB colours to apply. The point cloud was "
                "imported without colour information."
            )

        if len(colors) != point_cloud.size:
            raise ValueError(
                f"Colour count ({len(colors)}) does not match point count "
                f"({point_cloud.size})."
            )

        # Colours are stored normalised to [0, 1] on import; Colors() validates
        # and clips defensively.
        result = Colors(np.asarray(colors, dtype=np.float32))

        dependencies = [data_node.uid]
        return result, "colors", dependencies
