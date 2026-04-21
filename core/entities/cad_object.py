"""
CAD object entity for associating 3D CAD/BIM geometry with point cloud clusters.

Stores a geometry-type-agnostic representation: meshes (from OBJ), 3D polylines,
or future feature types all use this class with different geometry_type values.
The transform_matrix maps unit-normalized geometry onto the actual cluster's
size, orientation, and position in world coordinates.
"""

import uuid
from typing import Dict, List, Optional, Union

import numpy as np


# Supported geometry types and their required dict keys
_GEOMETRY_SCHEMAS = {
    "mesh": {"vertices", "faces", "edges"},
    "polyline": {"vertices", "closed"},
}


class CADObject:
    """
    A CAD/BIM feature associated with a point cloud cluster.

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
        cluster_reference: UUID of the source cluster/class DataNode.
        color: RGB wireframe color as (3,) float32 in [0, 1].
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
            f"CADObject(symbol_type='{self.symbol_type}', "
            f"geometry_type='{self.geometry_type}', "
            f"dimensions=[{w:.2f}, {l:.2f}, {h:.2f}])"
        )

    def __str__(self):
        w, l, h = self.dimensions
        return f"CADObject: {self.symbol_type} ({self.geometry_type}, {w:.1f}x{l:.1f}x{h:.1f})"
