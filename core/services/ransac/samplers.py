"""Sampling strategies for RANSAC."""

import numpy as np

from .base import Sampler


class UniformSampler(Sampler):
    """Draw ``k`` indices uniformly without replacement."""

    def sample(self, n_points: int, k: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(n_points, size=k, replace=False)
