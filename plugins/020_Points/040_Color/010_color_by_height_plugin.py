# plugins/020_Points/040_Color/010_color_by_height_plugin.py
from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.colors import Colors


class ColorByHeightPlugin(Plugin):
    """
    Colour a branch by point height (Z) using a red -> green ramp.

    The lowest points are red, the highest are green, with a smooth linear
    interpolation in between. Emits a ``colors`` node so the colouring travels
    through reconstruction like any other per-point colour.
    """

    def get_name(self) -> str:
        return "color_by_height"

    def get_parameters(self) -> Dict[str, Any]:
        # No parameters: the ramp spans the branch's own Z range.
        return {}

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud: PointCloud = data_node.data

        if point_cloud.points is None or point_cloud.size == 0:
            raise ValueError("This branch has no points to colour.")

        z = point_cloud.points[:, 2].astype(np.float32)

        # Normalise Z to [0, 1]. Guard against a flat branch (all points at the
        # same height), which would otherwise divide by zero.
        z_min = float(np.min(z))
        z_max = float(np.max(z))
        z_range = z_max - z_min
        if z_range <= 0:
            t = np.zeros_like(z)
        else:
            t = (z - z_min) / z_range

        # Red (low) -> green (high): R fades out, G fades in, B stays 0.
        colors = np.empty((len(z), 3), dtype=np.float32)
        colors[:, 0] = 1.0 - t  # red
        colors[:, 1] = t        # green
        colors[:, 2] = 0.0      # blue

        result = Colors(colors)

        dependencies = [data_node.uid]
        return result, "colors", dependencies
