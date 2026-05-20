"""
Renderable vector-feature geometry for the 3D viewport.

This is a RENDER-ONLY structure carrying the mesh / polyline geometry a
plugin produced, so the OpenGL viewport can draw it as a vector overlay
on top of the point cloud. It is intentionally not a DXF / AutoCAD export
object: DXF metadata (layer, class, attributes) lives on the source
Clusters node and is assembled at export time, not stored here.

The transform_matrix maps unit-normalized geometry onto the actual cluster's
size, orientation, and position in world coordinates.
"""

import uuid
from typing import Dict, Optional

import numpy as np


_GEOMETRY_SCHEMAS = {
    "mesh": {"vertices", "faces", "edges"},
    "polyline": {"vertices", "closed"},
}


class VectorFeature:
    """
    Renderable vector-feature geometry attached to a point-cloud cluster.

    Render-only — carries the geometry a plugin produced so the viewport can
    draw it as a wireframe / polyline overlay. Not a DXF entity: it has no
    `layer`, `class`, or `attributes` because the viewport doesn't need them.
    DXF export reads those from the source cluster (Clusters.cluster_names)
    via `cluster_reference` and assembles them at export time.

    Args:
        symbol_type: What the object represents (e.g. "pole", "sign", "tree").
        geometry_type: How the object is drawn ("mesh", "polyline").
        geometry: Geometry data dict whose keys depend on geometry_type:
            - "mesh": {'vertices': (V,3) float32, 'faces': list[list[int]],
                       'edges': (E,2) int32}
            - "polyline": {'vertices': (N,3) float32, 'closed': bool}
        transform_matrix: 4x4 matrix encoding scale, rotation, and translation
            that maps unit-normalized geometry onto the cluster's world position.
        dimensions: Cluster bounding dimensions [width, length, height] as (3,) float32.
        cluster_reference: UUID of the source cluster/class DataNode (used at
            DXF export time to look up class/layer).
        color: RGB wireframe color as (3,) float32 in [0, 1]. Render-only —
            never written to DXF (drafter's template owns layer styling).
    """

    def __init__(
        self,
        symbol_type: str,
        geometry_type: str,
        geometry: Dict,
        transform_matrix: np.ndarray,
        dimensions: np.ndarray,
        cluster_reference: Optional[uuid.UUID] = None,
        color: Optional[np.ndarray] = None,
    ):
        self.symbol_type = str(symbol_type)
        self.geometry_type = str(geometry_type)
        self.geometry = geometry
        self.transform_matrix = np.asarray(transform_matrix, dtype=np.float64)
        self.dimensions = np.asarray(dimensions, dtype=np.float32)
        self.cluster_reference = cluster_reference
        self.color = (
            np.asarray(color, dtype=np.float32)
            if color is not None
            else np.array([0.0, 1.0, 1.0], dtype=np.float32)
        )

        self._validate()

    def _validate(self):
        if not self.symbol_type:
            raise ValueError("symbol_type must be a non-empty string.")

        if self.geometry_type not in _GEOMETRY_SCHEMAS:
            raise ValueError(
                f"geometry_type must be one of {list(_GEOMETRY_SCHEMAS.keys())}, "
                f"got '{self.geometry_type}'."
            )

        required_keys = _GEOMETRY_SCHEMAS[self.geometry_type]
        missing = required_keys - set(self.geometry.keys())
        if missing:
            raise ValueError(
                f"geometry dict for '{self.geometry_type}' is missing keys: {missing}"
            )

        if self.transform_matrix.shape != (4, 4):
            raise ValueError(
                f"transform_matrix must have shape (4, 4), got {self.transform_matrix.shape}."
            )

        if self.dimensions.shape != (3,):
            raise ValueError(
                f"dimensions must have shape (3,), got {self.dimensions.shape}."
            )

        if self.color.shape != (3,):
            raise ValueError(
                f"color must have shape (3,), got {self.color.shape}."
            )

    def __repr__(self):
        w, l, h = self.dimensions
        return (
            f"VectorFeature(symbol_type='{self.symbol_type}', "
            f"geometry_type='{self.geometry_type}', "
            f"dimensions=[{w:.2f}, {l:.2f}, {h:.2f}])"
        )

    def __str__(self):
        w, l, h = self.dimensions
        return f"VectorFeature: {self.symbol_type} ({self.geometry_type}, {w:.1f}x{l:.1f}x{h:.1f})"
