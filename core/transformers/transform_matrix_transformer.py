# core/transformers/transform_matrix_transformer.py
"""
Transformer that applies a 4x4 transformation matrix to a point cloud.

Used during reconstruction: a DataNode whose `data` is a TransformMatrix maps
its parent point cloud's points (and normals) through the matrix, producing a
moved / rotated / scaled cloud. The point count is unchanged.
"""

import numpy as np

from core.entities.point_cloud import PointCloud
from core.entities.transform_matrix import TransformMatrix


class TransformMatrixTransformer:
    """
    Apply a 4x4 homogeneous transform to a PointCloud.

    Points are transformed as ``p' = p @ T.T`` (homogeneous row vectors).
    Normals are transformed by the inverse-transpose of the 3x3 linear part and
    re-normalised, which is correct for non-uniform scaling and rotation; for a
    pure rotation it reduces to ``n @ R.T``. Per-point attributes (colors,
    intensity, dist-to-ground, custom attributes) are carried over unchanged.
    """

    def __init__(self, point_cloud: PointCloud, transform: TransformMatrix):
        """
        Args:
            point_cloud (PointCloud): The point cloud to transform.
            transform (TransformMatrix): The 4x4 transform to apply.
        """
        self.point_cloud = point_cloud
        self.matrix = transform.matrix

    def execute(self) -> PointCloud:
        """
        Apply the transform and return a new PointCloud.

        Returns:
            PointCloud: A new point cloud with transformed points and normals,
            sharing all other per-point arrays with the source (shallow copy).
        """
        source = self.point_cloud
        points = source.points

        # Transform points: homogeneous row vectors p' = p @ T.T
        ones = np.ones((len(points), 1), dtype=np.float64)
        homo = np.hstack([points.astype(np.float64), ones])
        new_points = (homo @ self.matrix.T)[:, :3].astype(points.dtype)

        # Transform normals (if present) by the inverse-transpose of the linear
        # part, then re-normalise rows that have non-zero length.
        new_normals = source.normals
        if new_normals is not None and len(new_normals) > 0:
            linear = self.matrix[:3, :3]
            try:
                normal_matrix = np.linalg.inv(linear).T
            except np.linalg.LinAlgError:
                # Singular linear part (e.g. a zero scale factor): fall back to
                # the linear part itself so we still produce oriented normals.
                normal_matrix = linear
            transformed = source.normals.astype(np.float64) @ normal_matrix.T
            lengths = np.linalg.norm(transformed, axis=1, keepdims=True)
            np.divide(transformed, lengths, out=transformed, where=lengths > 0)
            new_normals = transformed.astype(source.normals.dtype)

        result = PointCloud(
            points=new_points,
            colors=source.colors,
            normals=new_normals,
            intensity=getattr(source, 'intensity', np.array([])),
            distToGround=getattr(source, 'distToGround', np.array([])),
            params=source.name,
        )
        # Carry custom attributes over unchanged (per-point, count preserved).
        result.attributes = source.attributes.copy()

        return result
