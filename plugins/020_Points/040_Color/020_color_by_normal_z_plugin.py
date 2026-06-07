# plugins/020_Points/040_Color/020_color_by_normal_z_plugin.py
from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.colors import Colors


class ColorByNormalZPlugin(Plugin):
    """
    Colour a branch by the Z component of each point's normal (nz).

    nz encodes surface orientation: a level upward-facing surface has nz ~ +1,
    a vertical wall has nz ~ 0, a downward-facing surface has nz ~ -1. The ramp
    runs red (nz = -1) -> green (nz = +1) over the full theoretical [-1, 1]
    range, so the colouring is comparable across branches. Emits a ``colors``
    node so the colouring travels through reconstruction like any other
    per-point colour.

    The branch must carry normals (run an estimate-normals step first).
    """

    def get_name(self) -> str:
        return "color_by_normal_z"

    def get_parameters(self) -> Dict[str, Any]:
        # No parameters: the ramp spans the full normalised nz range [-1, 1].
        return {}

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud: PointCloud = data_node.data

        if point_cloud.points is None or point_cloud.size == 0:
            raise ValueError("This branch has no points to colour.")

        normals = point_cloud.normals
        if normals is None or len(normals) == 0:
            raise ValueError(
                "This branch has no normals. Run an estimate-normals step before "
                "colouring by normal Z."
            )
        if len(normals) != point_cloud.size:
            raise ValueError(
                f"Normal count ({len(normals)}) does not match point count "
                f"({point_cloud.size})."
            )

        nz = np.asarray(normals, dtype=np.float32)[:, 2]

        # Map nz from the theoretical [-1, 1] range to [0, 1], clipping any
        # non-unit normals defensively.
        t = np.clip((nz + 1.0) * 0.5, 0.0, 1.0)

        # Red (nz = -1) -> green (nz = +1): R fades out, G fades in, B stays 0.
        colors = np.empty((len(nz), 3), dtype=np.float32)
        colors[:, 0] = 1.0 - t  # red
        colors[:, 1] = t        # green
        colors[:, 2] = 0.0      # blue

        result = Colors(colors)

        dependencies = [data_node.uid]
        return result, "colors", dependencies
