"""Scoring strategies for RANSAC."""

import numpy as np

from .base import Scorer


class MSACScorer(Scorer):
    """Truncated-quadratic loss (Torr & Zisserman, 1998).

    Each point contributes ``min(d**2, threshold**2)`` to a loss; the
    score is the negation so higher = better fit. Inliers within the
    threshold contribute their squared residual; outliers cap at the
    threshold-squared, so the score degrades smoothly across the
    inlier boundary rather than at a hard edge.
    """

    def score(self, distances: np.ndarray, threshold: float) -> float:
        t2 = threshold * threshold
        d2 = distances * distances
        return float(-np.minimum(d2, t2).sum())

    def inliers(self, distances: np.ndarray, threshold: float) -> np.ndarray:
        return distances < threshold


class InlierCountScorer(Scorer):
    """Classic RANSAC scoring: count of points inside the threshold.

    Kept for parity comparisons and debugging — MSAC is the default.
    """

    def score(self, distances: np.ndarray, threshold: float) -> float:
        return float((distances < threshold).sum())

    def inliers(self, distances: np.ndarray, threshold: float) -> np.ndarray:
        return distances < threshold
