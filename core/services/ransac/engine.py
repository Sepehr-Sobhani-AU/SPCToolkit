"""RANSAC engine: single-cloud ``fit()`` contract.

Given a point set and a model name, draw random minimal samples, score
each candidate via a pluggable scorer, then refit the best candidate
on its inliers. Returns ``(model, inlier_mask)`` or ``(None, None)``
when no candidate meets ``min_inlier_ratio``.
"""

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np

from .base import RansacModel, Sampler, Scorer
from .primitives.line import LineModel
from .primitives.plane import PlaneModel
from .samplers import UniformSampler
from .scorers import MSACScorer


_MODEL_REGISTRY: dict[str, type[RansacModel]] = {
    "line": LineModel,
    "plane": PlaneModel,
}


def register_model(name: str, model_cls: type[RansacModel]) -> None:
    """Add a new model to the registry.

    Lets external code (e.g. tests, future cylinder/cone primitives in
    other files) plug in without modifying the engine.
    """
    _MODEL_REGISTRY[name] = model_cls


def fit(
    points: np.ndarray,
    model_type: str,
    threshold: float,
    normals: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    min_inlier_ratio: float = 0.3,
    sampler: Optional[Sampler] = None,
    scorer: Optional[Scorer] = None,
    seed: Optional[int] = None,
) -> Tuple[Optional[RansacModel], Optional[np.ndarray]]:
    """Fit a geometric model to ``points`` with RANSAC.

    Args:
        points: ``(N, 3)`` array of input points.
        model_type: Name of a registered model — ``"line"`` or ``"plane"``.
        threshold: Distance threshold for inlier classification.
        normals: Optional ``(N, 3)`` per-point normals. Required only by
            models with ``requires_normals = True``.
        max_iterations: Number of random hypotheses to try.
        min_inlier_ratio: Reject the fit if the best inlier set is
            smaller than ``max(min_samples, N * min_inlier_ratio)``.
        sampler: Sampling strategy; defaults to ``UniformSampler``.
        scorer: Scoring strategy; defaults to ``MSACScorer``.
        seed: Seed for reproducibility.

    Returns:
        ``(model, inlier_mask)`` on success.
        ``(None, None)`` if no candidate meets the inlier-ratio threshold.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}; "
            f"available: {sorted(_MODEL_REGISTRY)}"
        )

    model_cls = _MODEL_REGISTRY[model_type]
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got {points.shape}")
    n = len(points)

    if model_cls.requires_normals:
        if normals is None:
            raise ValueError(
                f"Model {model_type!r} requires normals but none were provided."
            )
        normals = np.asarray(normals)
        if normals.shape != points.shape:
            raise ValueError(
                f"normals shape {normals.shape} does not match "
                f"points shape {points.shape}"
            )

    k = model_cls.min_samples
    if n < k:
        return None, None

    sampler = sampler or UniformSampler()
    scorer = scorer or MSACScorer()
    rng = np.random.default_rng(seed)

    trial = model_cls()
    best_state: Optional[RansacModel] = None
    best_score = -np.inf

    for _ in range(max_iterations):
        idx = sampler.sample(n, k, rng)
        sample_normals = normals[idx] if normals is not None else None
        if not trial.fit_minimal(points[idx], sample_normals):
            continue
        d = trial.distances(points)
        s = scorer.score(d, threshold)
        if s > best_score:
            best_score = s
            best_state = deepcopy(trial)

    if best_state is None:
        return None, None

    distances = best_state.distances(points)
    inlier_mask = scorer.inliers(distances, threshold)
    min_inliers = max(k, int(n * min_inlier_ratio))
    if int(inlier_mask.sum()) < min_inliers:
        return None, None

    # Refit on inliers. If refit reports degeneracy, fall back to the
    # pre-refit minimal model so the caller still gets a usable result.
    inlier_points = points[inlier_mask]
    inlier_normals = normals[inlier_mask] if normals is not None else None
    refit_model = deepcopy(best_state)
    if refit_model.refit(inlier_points, inlier_normals):
        distances = refit_model.distances(points)
        inlier_mask = scorer.inliers(distances, threshold)
        return refit_model, inlier_mask

    return best_state, inlier_mask
