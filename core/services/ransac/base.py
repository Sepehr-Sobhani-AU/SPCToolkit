"""Abstract interfaces for the RANSAC infrastructure.

A model defines geometry: minimal-sample fit, distance to the fitted
shape, and a refit on the inlier set. A sampler decides how to draw
minimal sets. A scorer ranks candidate models from a distance vector.
The engine in ``engine.py`` orchestrates the loop.

Phase B extends ``RansacModel`` with optional batched GPU methods that
operate on stacks of point sets at once. Models that implement them
set ``supports_gpu = True``; the engine routes batched and large-cloud
work to those methods. Models that don't implement them still work on
the CPU path via the single-cloud abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np


class RansacModel(ABC):
    """Geometric model fitted by RANSAC.

    Subclasses set ``requires_normals``, ``min_samples``, and
    ``supports_gpu`` as class attributes. Fitted state (e.g. ``point``,
    ``direction``, ``normal``) lives on the instance and is filled in
    by ``fit_minimal`` / ``refit`` on the CPU path. The batched GPU
    methods are stateless: they take a packed state dict in and return
    a new packed state dict.
    """

    requires_normals: bool = False
    min_samples: int = 0
    supports_gpu: bool = False

    # -------- CPU, single-cloud --------

    @abstractmethod
    def fit_minimal(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bool:
        """Fit to exactly ``min_samples`` points (in-place).

        Returns False on a degenerate sample.
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
        """Refit on all inliers (in-place). Returns False on degeneracy."""

    # -------- GPU, batched (optional) --------
    #
    # Batched ops live in a ``state`` dict of (B, ...) Torch tensors.
    # The engine never inspects the keys, so subclasses are free to
    # choose their own — line uses {"point", "direction"}, plane uses
    # {"point", "normal"}, etc. The engine moves states across the
    # batched ops opaquely and finally hands them to ``unpack_to_model``
    # to recover a single-cloud model instance per row.
    #
    # The default implementations raise NotImplementedError so the
    # engine can detect missing support via ``supports_gpu`` and route
    # accordingly.

    @classmethod
    def fit_minimal_batched_gpu(
        cls,
        points_b: "Any",       # torch.Tensor (B, min_samples, 3)
        normals_b: "Any",      # torch.Tensor (B, min_samples, 3) or None
        device: "Any",
    ) -> Tuple[dict, "Any"]:
        """Return ``(state, valid_mask)`` for a batch of minimal samples.

        ``valid_mask`` is a ``(B,)`` bool tensor marking rows where the
        sample was not degenerate.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement fit_minimal_batched_gpu"
        )

    @classmethod
    def distances_batched_gpu(
        cls,
        state: dict,
        points_b: "Any",       # (B, N_max, 3)
        counts_b: "Any",       # (B,) int64
        device: "Any",
    ) -> "Any":
        """Return ``(B, N_max)`` distances, masked to +inf past per-row counts."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement distances_batched_gpu"
        )

    @classmethod
    def refit_batched_gpu(
        cls,
        state: dict,
        points_b: "Any",            # (B, N_max, 3)
        inlier_masks_b: "Any",      # (B, N_max) bool
        device: "Any",
    ) -> Tuple[dict, "Any"]:
        """Refit each row on its inliers. Returns ``(new_state, valid_mask)``."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement refit_batched_gpu"
        )

    @classmethod
    def unpack_to_model(cls, state: dict, b: int) -> "RansacModel":
        """Build a single-cloud model instance from row ``b`` of a batched state."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement unpack_to_model"
        )


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
