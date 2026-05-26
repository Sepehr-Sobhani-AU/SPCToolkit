"""3D line RANSAC model."""

from typing import Optional

import numpy as np

from ..base import RansacModel


_EPS = 1e-12


class LineModel(RansacModel):
    """Line in R^3 defined by an anchor point and a unit direction."""

    requires_normals = False
    min_samples = 2

    def __init__(self):
        self.point: Optional[np.ndarray] = None
        self.direction: Optional[np.ndarray] = None

    def fit_minimal(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bool:
        diff = points[1] - points[0]
        norm = float(np.linalg.norm(diff))
        if norm < _EPS:
            return False
        self.point = points[0].astype(np.float64)
        self.direction = (diff / norm).astype(np.float64)
        return True

    def distances(self, points: np.ndarray) -> np.ndarray:
        vecs = points - self.point
        cross = np.cross(vecs, self.direction)
        return np.linalg.norm(cross, axis=1)

    def refit(
        self,
        inlier_points: np.ndarray,
        inlier_normals: Optional[np.ndarray] = None,
    ) -> bool:
        if len(inlier_points) < 2:
            return False
        centroid = inlier_points.mean(axis=0)
        centred = inlier_points - centroid
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        direction = vh[0]
        norm = float(np.linalg.norm(direction))
        if norm < _EPS:
            return False
        self.point = centroid
        self.direction = direction / norm
        return True
