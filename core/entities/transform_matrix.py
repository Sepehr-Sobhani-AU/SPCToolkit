# core/entities/transform_matrix.py
"""
A class for a DataNode's 'data' attribute containing a 4x4 transformation matrix.

This is a derived DataNode type: a node holding a TransformMatrix is applied to
its parent point cloud during reconstruction (see
core/transformers/transform_matrix_transformer.py), producing a moved / rotated /
scaled cloud. A single 4x4 homogeneous matrix encodes rotation, translation, and
(possibly non-uniform, per-axis) scaling in one object.
"""

import numpy as np


class TransformMatrix:
    """
    A 4x4 homogeneous transformation matrix stored as a DataNode's `data`.

    Points are treated as row vectors in homogeneous form ``[x, y, z, 1]`` and
    transformed as ``p' = p @ matrix.T`` (i.e. ``matrix`` left-multiplies a
    column point). The top-left 3x3 block is the linear part (rotation /
    scale / shear); the first three entries of the last column are the
    translation.

    Args:
        matrix (np.ndarray): A (4, 4) array. Stored as float64.
    """

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=np.float64)

        if self.matrix.shape != (4, 4):
            raise ValueError(
                f"transform matrix must have shape (4, 4), got {self.matrix.shape}."
            )

    # --- Convenience constructors -------------------------------------------

    @classmethod
    def identity(cls) -> "TransformMatrix":
        """Return the identity transform."""
        return cls(np.eye(4, dtype=np.float64))

    @classmethod
    def from_translation(cls, tx: float, ty: float, tz: float) -> "TransformMatrix":
        """Return a pure translation transform."""
        m = np.eye(4, dtype=np.float64)
        m[:3, 3] = (tx, ty, tz)
        return cls(m)

    @classmethod
    def from_scale(cls, sx: float, sy: float, sz: float, center=None) -> "TransformMatrix":
        """
        Return a (possibly non-uniform) scale transform.

        Args:
            sx, sy, sz: Per-axis scale factors. Negative values mirror.
            center: Optional (3,) point to scale about. When given, the result
                is ``T(center) @ S @ T(-center)`` so geometry stays put while
                being scaled about that point (e.g. a cloud centroid).
        """
        s = np.diag(np.array([sx, sy, sz, 1.0], dtype=np.float64))
        if center is None:
            return cls(s)
        return cls(cls._about(s, center))

    @classmethod
    def from_euler(cls, rx: float, ry: float, rz: float,
                   degrees: bool = True, center=None) -> "TransformMatrix":
        """
        Return a rotation transform from intrinsic X→Y→Z Euler angles.

        Args:
            rx, ry, rz: Rotation angles about the X, Y, Z axes.
            degrees: If True (default), angles are in degrees, else radians.
            center: Optional (3,) point to rotate about (see `from_scale`).
        """
        if degrees:
            rx, ry, rz = np.radians([rx, ry, rz])

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

        r = rot_z @ rot_y @ rot_x
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = r
        if center is None:
            return cls(m)
        return cls(cls._about(m, center))

    @classmethod
    def compose(cls, *transforms) -> "TransformMatrix":
        """
        Compose transforms left-to-right: ``compose(A, B, C)`` applies A first,
        then B, then C (i.e. matrix product ``C @ B @ A``).

        Accepts TransformMatrix instances or raw (4, 4) arrays.
        """
        result = np.eye(4, dtype=np.float64)
        for t in transforms:
            mat = t.matrix if isinstance(t, TransformMatrix) else np.asarray(t, dtype=np.float64)
            result = mat @ result
        return cls(result)

    @staticmethod
    def _about(linear_4x4: np.ndarray, center) -> np.ndarray:
        """Wrap a 4x4 transform so it acts about ``center`` instead of the origin."""
        center = np.asarray(center, dtype=np.float64).reshape(3)
        to_origin = np.eye(4, dtype=np.float64)
        to_origin[:3, 3] = -center
        back = np.eye(4, dtype=np.float64)
        back[:3, 3] = center
        return back @ linear_4x4 @ to_origin

    # --- Introspection ------------------------------------------------------

    @property
    def translation(self) -> np.ndarray:
        """The (3,) translation component."""
        return self.matrix[:3, 3].copy()

    @property
    def linear(self) -> np.ndarray:
        """The (3, 3) linear component (rotation / scale / shear)."""
        return self.matrix[:3, :3].copy()

    def __repr__(self):
        tx, ty, tz = self.translation
        has_linear = not np.allclose(self.linear, np.eye(3))
        return (
            f"TransformMatrix(translation=[{tx:.3f}, {ty:.3f}, {tz:.3f}], "
            f"linear={'set' if has_linear else 'identity'})"
        )

    def __str__(self):
        return self.__repr__()
