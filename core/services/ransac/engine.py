"""RANSAC engine.

Single-cloud ``fit()`` and batched ``fit_many()`` entry points. CPU and
GPU backends are dispatched internally based on the ``backend`` argument
and the availability of CUDA. The CPU path uses the abstract
``RansacModel`` / ``Sampler`` / ``Scorer`` interfaces; the GPU path
calls the batched ``*_batched_gpu`` classmethods on the model and
hardcodes uniform sampling + MSAC scoring (custom sampler/scorer are
CPU-only).
"""

from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from .base import RansacModel, Sampler, Scorer
from .primitives.cone import ConeModel
from .primitives.cylinder import CylinderModel
from .primitives.line import LineModel
from .primitives.plane import PlaneModel
from .samplers import UniformSampler
from .scorers import MSACScorer


_MODEL_REGISTRY: dict[str, type[RansacModel]] = {
    "line": LineModel,
    "plane": PlaneModel,
    "cylinder": CylinderModel,
    "cone": ConeModel,
}

# For backend="auto" with a single cloud, only switch to GPU when the
# cloud is large enough that per-iteration distance computation
# dominates the loop. Below this, the CPU path is faster end-to-end
# because of host↔device transfer overhead.
_GPU_FIT_SIZE_THRESHOLD = 50_000


def register_model(name: str, model_cls: type[RansacModel]) -> None:
    """Add a new model to the registry. Useful for external primitives."""
    _MODEL_REGISTRY[name] = model_cls


# ---------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _resolve_backend(backend: str, model_cls: type[RansacModel]) -> str:
    """Return ``"cpu"`` or ``"gpu"``; raise on bad input."""
    if backend == "cpu":
        return "cpu"
    if backend == "gpu":
        if not model_cls.supports_gpu:
            raise RuntimeError(
                f"Model {model_cls.__name__} does not support GPU backend."
            )
        if not _cuda_available():
            raise RuntimeError(
                "backend='gpu' requested but CUDA is not available."
            )
        return "gpu"
    if backend == "auto":
        if model_cls.supports_gpu and _cuda_available():
            return "gpu"
        return "cpu"
    raise ValueError(
        f"Unknown backend {backend!r}; expected 'auto', 'cpu', or 'gpu'."
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


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
    backend: str = "auto",
) -> Tuple[Optional[RansacModel], Optional[np.ndarray]]:
    """Fit a geometric model to a single point set with RANSAC.

    Args:
        points: ``(N, 3)`` array of input points.
        model_type: Name of a registered model ("line", "plane").
        threshold: Distance threshold for inlier classification.
        normals: Optional ``(N, 3)`` per-point normals.
        max_iterations: Number of random hypotheses to try.
        min_inlier_ratio: Reject the fit if the best inlier set is
            smaller than ``max(min_samples, N * min_inlier_ratio)``.
        sampler: Sampling strategy (CPU only). Default: ``UniformSampler``.
        scorer: Scoring strategy (CPU only). Default: ``MSACScorer``.
        seed: Seed for reproducibility.
        backend: ``"auto"`` (default), ``"cpu"``, or ``"gpu"``. ``"auto"``
            picks GPU when CUDA is available *and* ``N >= 50_000``;
            otherwise CPU.

    Returns:
        ``(model, inlier_mask)`` on success, or ``(None, None)`` if no
        candidate meets the inlier ratio.
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

    resolved = _resolve_backend(backend, model_cls)
    if backend == "auto" and resolved == "gpu" and n < _GPU_FIT_SIZE_THRESHOLD:
        resolved = "cpu"

    if resolved == "gpu":
        if sampler is not None or scorer is not None:
            raise ValueError(
                "Custom sampler/scorer are only supported by the CPU backend."
            )
        models, masks = _fit_many_gpu(
            [points],
            model_cls,
            threshold,
            [normals] if normals is not None else None,
            max_iterations,
            min_inlier_ratio,
            seed,
        )
        return models[0], masks[0]

    return _fit_cpu(
        points,
        model_cls,
        threshold,
        normals,
        max_iterations,
        min_inlier_ratio,
        seed,
        sampler=sampler or UniformSampler(),
        scorer=scorer or MSACScorer(),
    )


def fit_many(
    points_list: List[np.ndarray],
    model_type: str,
    threshold: float,
    normals_list: Optional[List[np.ndarray]] = None,
    max_iterations: int = 1000,
    min_inlier_ratio: float = 0.3,
    seed: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[List[Optional[RansacModel]], List[Optional[np.ndarray]]]:
    """Fit one model per point set in parallel.

    Args:
        points_list: List of ``B`` arrays, each shape ``(N_b, 3)``.
        model_type: Name of a registered model.
        threshold: Distance threshold for inlier classification.
        normals_list: Optional list of ``B`` arrays matching ``points_list``.
            Required only when the model requires normals.
        backend: ``"auto"`` (GPU if available), ``"cpu"``, or ``"gpu"``.

    Returns:
        ``(models, inlier_masks)`` — both lists of length ``B``. Entries
        are ``None`` for rows that failed to find a model meeting
        ``min_inlier_ratio``.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}; "
            f"available: {sorted(_MODEL_REGISTRY)}"
        )
    model_cls = _MODEL_REGISTRY[model_type]

    if model_cls.requires_normals:
        if normals_list is None:
            raise ValueError(
                f"Model {model_type!r} requires normals but normals_list is None."
            )
        if len(normals_list) != len(points_list):
            raise ValueError(
                f"normals_list length {len(normals_list)} does not match "
                f"points_list length {len(points_list)}."
            )

    if len(points_list) == 0:
        return [], []

    resolved = _resolve_backend(backend, model_cls)
    if resolved == "gpu":
        return _fit_many_gpu(
            points_list,
            model_cls,
            threshold,
            normals_list,
            max_iterations,
            min_inlier_ratio,
            seed,
        )
    return _fit_many_cpu(
        points_list,
        model_cls,
        threshold,
        normals_list,
        max_iterations,
        min_inlier_ratio,
        seed,
    )


# ---------------------------------------------------------------------
# CPU implementation
# ---------------------------------------------------------------------


def _fit_cpu(
    points: np.ndarray,
    model_cls: type[RansacModel],
    threshold: float,
    normals: Optional[np.ndarray],
    max_iterations: int,
    min_inlier_ratio: float,
    seed: Optional[int],
    sampler: Sampler,
    scorer: Scorer,
) -> Tuple[Optional[RansacModel], Optional[np.ndarray]]:
    n = len(points)
    k = model_cls.min_samples
    if n < k:
        return None, None

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

    inlier_points = points[inlier_mask]
    inlier_normals = normals[inlier_mask] if normals is not None else None
    refit_model = deepcopy(best_state)
    if refit_model.refit(inlier_points, inlier_normals):
        distances = refit_model.distances(points)
        inlier_mask = scorer.inliers(distances, threshold)
        return refit_model, inlier_mask

    return best_state, inlier_mask


def _fit_many_cpu(
    points_list: List[np.ndarray],
    model_cls: type[RansacModel],
    threshold: float,
    normals_list: Optional[List[np.ndarray]],
    max_iterations: int,
    min_inlier_ratio: float,
    seed: Optional[int],
) -> Tuple[List[Optional[RansacModel]], List[Optional[np.ndarray]]]:
    """CPU fallback for fit_many — sequential ``_fit_cpu`` calls.

    Correct but slow. Use only when CUDA is unavailable or when the
    caller explicitly requested ``backend="cpu"``.
    """
    sampler = UniformSampler()
    scorer = MSACScorer()
    models: List[Optional[RansacModel]] = []
    masks: List[Optional[np.ndarray]] = []
    for i, points in enumerate(points_list):
        normals = normals_list[i] if normals_list is not None else None
        row_seed = (seed + i) if seed is not None else None
        model, mask = _fit_cpu(
            points,
            model_cls,
            threshold,
            normals,
            max_iterations,
            min_inlier_ratio,
            row_seed,
            sampler=sampler,
            scorer=scorer,
        )
        models.append(model)
        masks.append(mask)
    return models, masks


# ---------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------


def _fit_many_gpu(
    points_list: List[np.ndarray],
    model_cls: type[RansacModel],
    threshold: float,
    normals_list: Optional[List[np.ndarray]],
    max_iterations: int,
    min_inlier_ratio: float,
    seed: Optional[int],
) -> Tuple[List[Optional[RansacModel]], List[Optional[np.ndarray]]]:
    """Batched GPU RANSAC: one model per row, processed in parallel."""
    import torch

    device = torch.device("cuda")
    dtype = torch.float64

    B = len(points_list)
    if B == 0:
        return [], []

    counts = [len(p) for p in points_list]
    N_max = max(counts)
    k = model_cls.min_samples

    # Pad inputs into (B, N_max, 3) tensors.
    points_b = torch.zeros(B, N_max, 3, device=device, dtype=dtype)
    counts_b = torch.tensor(counts, device=device, dtype=torch.int64)
    for i, p in enumerate(points_list):
        arr = np.asarray(p, dtype=np.float64)
        points_b[i, : len(p)] = torch.from_numpy(arr).to(device=device, dtype=dtype)

    normals_b = None
    if model_cls.requires_normals:
        normals_b = torch.zeros(B, N_max, 3, device=device, dtype=dtype)
        for i, normals in enumerate(normals_list):
            arr = np.asarray(normals, dtype=np.float64)
            normals_b[i, : len(normals)] = torch.from_numpy(arr).to(
                device=device, dtype=dtype
            )

    row_can_fit = counts_b >= k
    rng_seed = int(seed) if seed is not None else int(
        np.random.SeedSequence().entropy & 0xFFFFFFFF
    )
    generator = torch.Generator(device=device).manual_seed(rng_seed)

    t2 = torch.tensor(threshold * threshold, device=device, dtype=dtype)

    best_state: Optional[dict] = None
    best_score = torch.full((B,), float("-inf"), device=device, dtype=dtype)

    for _ in range(max_iterations):
        # Sample k indices per row, uniformly within each row's count.
        rand = torch.rand(B, k, device=device, generator=generator, dtype=dtype)
        idx = (rand * counts_b.unsqueeze(1).to(dtype)).to(torch.int64)
        idx = idx.clamp(max=N_max - 1)

        gather_idx = idx.unsqueeze(-1).expand(-1, -1, 3)
        sample_pts = torch.gather(points_b, 1, gather_idx)
        sample_normals = (
            torch.gather(normals_b, 1, gather_idx) if normals_b is not None else None
        )

        state, valid = model_cls.fit_minimal_batched_gpu(
            sample_pts, sample_normals, device
        )

        dists = model_cls.distances_batched_gpu(state, points_b, counts_b, device)
        d2 = dists * dists
        loss = torch.minimum(d2, t2).sum(dim=-1)
        score = -loss
        score = torch.where(
            valid & row_can_fit, score, torch.full_like(score, float("-inf"))
        )

        if best_state is None:
            best_state = {key: val.clone() for key, val in state.items()}
            best_score = score.clone()
        else:
            better = score > best_score
            for key in best_state:
                best_state[key] = torch.where(
                    better.unsqueeze(-1), state[key], best_state[key]
                )
            best_score = torch.where(better, score, best_score)

    assert best_state is not None  # loop always runs at least once

    # Final inlier evaluation on the best per-row state.
    arange = torch.arange(N_max, device=device).unsqueeze(0).expand(B, -1)
    in_bounds = arange < counts_b.unsqueeze(1)

    final_dists = model_cls.distances_batched_gpu(
        best_state, points_b, counts_b, device
    )
    inlier_masks = (final_dists < threshold) & in_bounds
    inlier_counts = inlier_masks.sum(dim=1)

    min_inliers = torch.maximum(
        torch.tensor(k, device=device, dtype=torch.int64),
        (counts_b.to(dtype) * min_inlier_ratio).to(torch.int64),
    )
    finite_score = torch.isfinite(best_score)
    row_passed = (inlier_counts >= min_inliers) & finite_score

    # Refit each row on its inliers; for failed refits, keep the
    # pre-refit minimal model so the caller still gets a usable result.
    refit_state, refit_valid = model_cls.refit_batched_gpu(
        best_state, points_b, inlier_masks, device
    )
    use_refit = refit_valid & row_passed
    for key in best_state:
        best_state[key] = torch.where(
            use_refit.unsqueeze(-1), refit_state[key], best_state[key]
        )

    final_dists = model_cls.distances_batched_gpu(
        best_state, points_b, counts_b, device
    )
    final_masks = (final_dists < threshold) & in_bounds

    row_passed_cpu = row_passed.cpu().numpy()
    counts_cpu = counts_b.cpu().numpy()
    final_masks_cpu = final_masks.cpu().numpy()

    models_out: List[Optional[RansacModel]] = []
    masks_out: List[Optional[np.ndarray]] = []
    for b in range(B):
        if not bool(row_passed_cpu[b]):
            models_out.append(None)
            masks_out.append(None)
            continue
        models_out.append(model_cls.unpack_to_model(best_state, b))
        n_b = int(counts_cpu[b])
        masks_out.append(final_masks_cpu[b, :n_b].copy())
    return models_out, masks_out
