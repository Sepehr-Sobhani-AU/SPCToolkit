# plugins/020_Points/040_Color/010_color_by_value_plugin.py
import uuid
from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.colors import Colors
from services.colormap_service import apply_colormap
from services.point_fields import (
    enumerate_fields, resolve_field, representative_cloud,
)


class ColorByValuePlugin(Plugin):
    """
    Colour a branch by any per-point scalar field, picked from a dropdown.

    Replaces the old colour-by-height / colour-by-normal-Z / colour-by-values
    plugins: those were all the same operation (map a scalar through a colormap)
    differing only in *which* scalar. Here the scalar is a parameter. The source
    dropdown is populated from the selected branch's actual fields -- geometry
    (Z/X/Y), normals (nz/nx/ny), intensity, distance-to-ground, and every key in
    the cloud's ``attributes`` dict (e.g. ``values`` from projected_distance) --
    so new attributes appear automatically with no new plugin.

    Emits a ``colors`` node so the colouring travels through reconstruction like
    any other per-point colour. (Cross-branch colouring stays in color_by_branch;
    flat colouring stays in rgb_color.)
    """

    def get_name(self) -> str:
        return "color_by_value"

    # --- field discovery -------------------------------------------------

    def get_parameters(self) -> Dict[str, Any]:
        from config.config import global_variables
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes

        node = None
        if controller is not None and data_nodes is not None:
            selected = getattr(controller, "selected_branches", None) or []
            if selected:
                try:
                    node = data_nodes.get_node(uuid.UUID(str(selected[0])))
                except Exception:
                    node = None

        pc = representative_cloud(node)
        sources = enumerate_fields(node, pc)
        # Default to the scalar field when the branch has one (e.g. a distance
        # branch); otherwise fall back to height.
        default_source = "attr:values" if "attr:values" in sources else "z"

        return {
            "source": {
                "type": "dropdown",
                "options": dict(sources),
                "default": default_source,
                "label": "Color by",
                "description": "Per-point field to colour by (from the selected branch).",
            },
            "normalize": {
                "type": "dropdown",
                "options": {
                    "robust": "Robust (2-98%, default)",
                    "minmax": "Min-max (full range)",
                    "abs": "Absolute |v|",
                    "symmetric": "Symmetric (0 centered)",
                },
                "default": "robust",
                "label": "Normalization",
                "description": (
                    "How the field maps onto the ramp. 'Robust' clamps to the "
                    "2-98th percentile so outliers don't wash out the colours "
                    "(good default for distance fields); 'min-max' uses the full "
                    "range; 'symmetric' centres 0 mid-ramp for signed fields; "
                    "'absolute' ignores sign (e.g. |nz|)."
                ),
            },
            "colormap": {
                "type": "colormap",
                "default": "turbo",
                "label": "Colormap",
                "description": "Colour ramp applied to the normalised field.",
            },
        }

    # --- execution -------------------------------------------------------

    @staticmethod
    def _normalize(values: np.ndarray, mode: str) -> np.ndarray:
        v = values.astype(np.float32)
        if mode == "abs":
            v = np.abs(v)
        elif mode == "symmetric":
            m = float(np.nanmax(np.abs(v))) if v.size else 0.0
            if m <= 0:
                return np.full_like(v, 0.5)
            return np.clip(v / m * 0.5 + 0.5, 0.0, 1.0)

        finite = v[np.isfinite(v)]
        if finite.size == 0:
            return np.zeros_like(v)
        if mode == "robust":
            # 2-98th percentile so a few outliers (e.g. sign-induced below-ground
            # points) don't compress the ramp; outliers clamp to the ends.
            lo = float(np.percentile(finite, 2.0))
            hi = float(np.percentile(finite, 98.0))
        else:  # "minmax" (also the fall-through for 'abs'): full data range
            lo = float(finite.min())
            hi = float(finite.max())
        rng = hi - lo
        return np.zeros_like(v) if rng <= 0 else (v - lo) / rng

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables
        controller = global_variables.global_application_controller

        source = params.get("source", "z")

        # Cheap path: the node's own cloud. Reconstruct only when needed (derived
        # branch, or an attribute that lives on an ancestor).
        pc = data_node.data
        if not (hasattr(pc, "points") and getattr(pc, "points", None) is not None):
            global_variables.global_progress = (None, "Reconstructing branch...")
            pc = controller.reconstruct(data_node.uid)

        values = resolve_field(pc, source)
        if values is None:
            global_variables.global_progress = (None, "Reconstructing branch for field...")
            pc = controller.reconstruct(data_node.uid)
            values = resolve_field(pc, source)
        if values is None:
            raise ValueError(f"Source '{source}' is not available on this branch.")

        values = np.asarray(values, dtype=np.float32).ravel()
        if values.size != pc.size:
            raise ValueError(
                f"Field '{source}' has {values.size} values but the branch has "
                f"{pc.size} points."
            )

        t = self._normalize(values, params.get("normalize", "robust"))
        colors = apply_colormap(t, params.get("colormap", "turbo"))
        return Colors(colors), "colors", [data_node.uid]
