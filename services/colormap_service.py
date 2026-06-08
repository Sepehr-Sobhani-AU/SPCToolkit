"""
Colormap Service

Maps normalized scalar values (t in [0, 1]) to per-point RGB colors using a
curated set of matplotlib colormaps. Shared by the scalar-coloring plugins
(color_by_normal_z, color_by_height) so the t -> RGB mapping lives in one place.

Colors are returned as an (N, 3) float32 array in [0, 1], matching the contract
of core.entities.colors.Colors.
"""

import logging
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


# Curated {value: display_label} for the plugin dropdown. ``value`` is the
# matplotlib colormap name (or the special "red_green" legacy ramp). Order here
# is the order shown in the dialog; turbo first so it reads as the default.
COLORMAPS = OrderedDict([
    ("turbo", "Turbo (high-contrast rainbow)"),
    ("viridis", "Viridis (perceptual)"),
    ("plasma", "Plasma (perceptual)"),
    ("inferno", "Inferno (perceptual)"),
    ("cividis", "Cividis (colorblind-safe)"),
    ("jet", "Jet (classic)"),
    ("gist_rainbow", "Rainbow (red->violet)"),
    ("hsv", "HSV (cyclic)"),
    ("hot", "Hot (fire)"),
    ("cool", "Cool (cyan->magenta)"),
    ("terrain", "Terrain"),
    ("red_green", "Red->Green (legacy)"),
])

DEFAULT_COLORMAP = "turbo"


def apply_colormap(t, name=DEFAULT_COLORMAP):
    """
    Map normalized values ``t`` (clipped to [0, 1]) to an (N, 3) float32 RGB
    array using the colormap ``name``.

    ``name`` is a key of :data:`COLORMAPS`. The special "red_green" value
    reproduces the legacy red(0) -> green(1) ramp exactly. An unknown name
    falls back to the default colormap rather than raising, so a stray param
    string never aborts a long coloring run.
    """
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)

    if name == "red_green":
        # Legacy ramp: red (t=0) -> green (t=1), blue stays 0.
        colors = np.empty((len(t), 3), dtype=np.float32)
        colors[:, 0] = 1.0 - t
        colors[:, 1] = t
        colors[:, 2] = 0.0
        return colors

    from matplotlib import colormaps

    if name not in colormaps:
        logger.warning(
            "Unknown colormap '%s'; falling back to '%s'.", name, DEFAULT_COLORMAP
        )
        name = DEFAULT_COLORMAP

    # matplotlib returns RGBA in [0, 1]; drop alpha to match Colors' (N, 3).
    rgba = colormaps[name](t)
    return np.ascontiguousarray(rgba[:, :3], dtype=np.float32)
