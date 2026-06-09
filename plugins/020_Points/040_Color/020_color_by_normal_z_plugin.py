# plugins/020_Points/040_Color/020_color_by_normal_z_plugin.py
from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.colors import Colors
from services.colormap_service import apply_colormap


class ColorByNormalZPlugin(Plugin):
    """
    Colour a branch by the Z component of each point's normal (nz).

    nz encodes surface orientation: a level upward-facing surface has nz ~ +1,
    a vertical wall has nz ~ 0, a downward-facing surface has nz ~ -1.

    Two ramps are available:

    - **Magnitude only** (default): colour by |nz| over [0, 1], so the direction
      the normal points is ignored. Up- and down-facing horizontal surfaces both
      read green (|nz| ~ 1) and vertical walls read red (|nz| ~ 0). This is the
      common case when normal orientation is unreliable and you only care about
      separating horizontal surfaces from vertical ones.
    - **Signed**: colour by nz over the full theoretical [-1, 1] range. Red
      (nz = -1, downward) -> green (nz = +1, upward), distinguishing the two
      facing directions.

    Emits a ``colors`` node so the colouring travels through reconstruction like
    any other per-point colour.

    The branch must carry normals (run an estimate-normals step first).
    """

    def get_name(self) -> str:
        return "color_by_normal_z"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "magnitude_only": {
                "type": "bool",
                "default": True,
                "label": "Magnitude only (ignore direction)",
                "description": (
                    "Colour by |nz| over [0, 1] so up- and down-facing surfaces "
                    "colour the same (low = vertical, high = horizontal). Uncheck "
                    "to use signed nz over [-1, 1], distinguishing facing direction."
                ),
            },
            "colormap": {
                "type": "colormap",
                "default": "turbo",
                "label": "Colormap",
                "description": "Colour ramp applied to the normalised nz values.",
            },
        }

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

        magnitude_only = params.get("magnitude_only", True)
        if magnitude_only:
            # |nz| spans [0, 1] already; direction of the normal is ignored.
            t = np.clip(np.abs(nz), 0.0, 1.0)
        else:
            # Map signed nz from the theoretical [-1, 1] range to [0, 1],
            # clipping any non-unit normals defensively.
            t = np.clip((nz + 1.0) * 0.5, 0.0, 1.0)

        colors = apply_colormap(t, params.get("colormap", "turbo"))

        result = Colors(colors)

        dependencies = [data_node.uid]
        return result, "colors", dependencies
