"""
Reusable, model-agnostic RANSAC engine.

Provides an abstract RANSACModel interface and a concrete LineModel3D for 3D line fitting.
New geometric models (plane, cylinder, …) can be added by subclassing RANSACModel.
"""

import numpy as np
from abc import ABC, abstractmethod


class RANSACModel(ABC):
    """Interface that every geometric model must implement for use with RANSAC."""

    @abstractmethod
    def min_samples(self) -> int:
        """Minimum number of points required to fit the model."""

    @abstractmethod
    def fit(self, points: np.ndarray) -> None:
        """Fit the model to exactly ``min_samples()`` points (in-place)."""

    @abstractmethod
    def distances(self, points: np.ndarray) -> np.ndarray:
        """Return perpendicular distances from each point to the fitted model."""


class LineModel3D(RANSACModel):
    """3D line defined by a point and a unit direction vector."""

    def __init__(self):
        self.point: np.ndarray | None = None
        self.direction: np.ndarray | None = None

    def min_samples(self) -> int:
        return 2

    def fit(self, points: np.ndarray) -> None:
        self.point = points[0]
        diff = points[1] - points[0]
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            self.direction = np.array([1.0, 0.0, 0.0])
        else:
            self.direction = diff / norm

    def distances(self, points: np.ndarray) -> np.ndarray:
        vecs = points - self.point
        cross = np.cross(vecs, self.direction)
        return np.linalg.norm(cross, axis=1)


class RANSAC:
    """Model-agnostic RANSAC engine (stateless — all state lives in the model)."""

    @staticmethod
    def run(
        points: np.ndarray,
        model: RANSACModel,
        distance_threshold: float,
        max_iterations: int = 1000,
        min_inlier_ratio: float = 0.3,
    ):
        """
        Run RANSAC on *points* using the given geometric *model*.

        Returns:
            (inlier_mask, fitted_model) on success, or (None, None) on failure.
        """
        n = len(points)
        k = model.min_samples()

        if n < k:
            return None, None

        best_inlier_count = 0
        best_inlier_mask = None

        for _ in range(max_iterations):
            sample_idx = np.random.choice(n, size=k, replace=False)
            model.fit(points[sample_idx])
            dists = model.distances(points)
            inlier_mask = dists < distance_threshold
            count = int(np.sum(inlier_mask))

            if count > best_inlier_count:
                best_inlier_count = count
                best_inlier_mask = inlier_mask.copy()

        if best_inlier_mask is None or best_inlier_count < max(k, int(n * min_inlier_ratio)):
            return None, None

        # Refit on all inliers using SVD for better accuracy
        inlier_pts = points[best_inlier_mask]
        centroid = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        direction = vh[0]  # first principal component

        refit_model = type(model)()
        refit_model.point = centroid
        refit_model.direction = direction / np.linalg.norm(direction)

        # Re-evaluate inliers with the refined model
        dists = refit_model.distances(points)
        best_inlier_mask = dists < distance_threshold

        return best_inlier_mask, refit_model
