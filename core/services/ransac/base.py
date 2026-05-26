"""Abstract interfaces for the RANSAC infrastructure.

A model defines geometry: minimal-sample fit, distance to the fitted
shape, and a refit on the inlier set. A sampler decides how to draw
minimal sets. A scorer ranks candidate models from a distance vector.
The engine in ``engine.py`` orchestrates the loop.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class RansacModel(ABC):
    """Geometric model fitted by RANSAC.

    Subclasses set ``requires_normals`` and ``min_samples`` as class
    attributes and implement the three abstract methods. Fitted state
    (e.g. ``point``, ``direction``, ``normal``) lives on the instance
    and is filled in by ``fit_minimal`` / ``refit``.
    """

    requires_normals: bool = False
    min_samples: int = 0

    @abstractmethod
    def fit_minimal(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bool:
        """Fit to exactly ``min_samples`` points (in-place).

        Returns False if the sample is degenerate (coincident points,
        collinear triplet for a plane, parallel normals for a cylinder).
        The engine skips such trials without scoring them.
        """

    @abstractmethod
    def distances(self, points: np.ndarray) -> np.ndarray:
        """Perpendicular distance from each point to the fitted model."""

    @abstractmethod
    def refit(
        self,
        inlier_points: np.ndarray,
        inlier_normals: Optional[np.ndarray] = None,
    ) -> bool:
        """Refit on all inliers (in-place). Returns False on degeneracy.

        The engine falls back to the pre-refit minimal model when this
        returns False, so the caller still gets a usable result.
        """


class Sampler(ABC):
    """Strategy for drawing minimal sample indices from a point set."""

    @abstractmethod
    def sample(self, n_points: int, k: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``k`` distinct indices in ``[0, n_points)``."""


class Scorer(ABC):
    """Strategy for ranking candidate models from a distance vector."""

    @abstractmethod
    def score(self, distances: np.ndarray, threshold: float) -> float:
        """Scalar score; higher is better."""

    @abstractmethod
    def inliers(self, distances: np.ndarray, threshold: float) -> np.ndarray:
        """Boolean inlier mask for a candidate model."""
