"""3D plane RANSAC model."""

from typing import Optional

import numpy as np

from ..base import RansacModel


_EPS = 1e-12


class PlaneModel(RansacModel):
    """Plane in R^3 defined by a point on the plane and a unit normal."""

    requires_normals = False
    min_samples = 3

    def __init__(self):
        self.point: Optional[np.ndarray] = None
        self.normal: Optional[np.ndarray] = None

    def fit_minimal(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bool:
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        norm = float(np.linalg.norm(normal))
        if norm < _EPS:
            return False
        self.point = points[0].astype(np.float64)
        self.normal = (normal / norm).astype(np.float64)
        return True

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs((points - self.point) @ self.normal)

    def refit(
        self,
        inlier_points: np.ndarray,
        inlier_normals: Optional[np.ndarray] = None,
    ) -> bool:
        if len(inlier_points) < 3:
            return False
        centroid = inlier_points.mean(axis=0)
        centred = inlier_points - centroid
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        normal = vh[-1]  # smallest variance = plane normal
        norm = float(np.linalg.norm(normal))
        if norm < _EPS:
            return False
        self.point = centroid
        self.normal = normal / norm
        return True
