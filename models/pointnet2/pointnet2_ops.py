"""
PointNet++ Core Operations — Tiered GPU Implementation

Pure PyTorch implementations of FPS, ball query, grouping, and interpolation
that run entirely on GPU. Optionally uses compiled CUDA kernels from
pointnet2_ops (Pointnet2_PyTorch) if installed for 5-10x speedup.

Based on: PointNet++: Deep Hierarchical Feature Learning on Point Sets
          in a Metric Space, Charles R. Qi et al., NeurIPS 2017
"""

import torch

# ---------------------------------------------------------------------------
# Tiered backend detection
# ---------------------------------------------------------------------------
_USE_CUDA_OPS = False

try:
    from pointnet2_ops.pointnet2_utils import (
        furthest_point_sample as _cuda_fps,
        ball_query as _cuda_ball_query,
        grouping_operation as _cuda_grouping,
        three_nn as _cuda_three_nn,
        three_interpolate as _cuda_three_interpolate,
    )
    _USE_CUDA_OPS = True
    print("[PointNet++] CUDA kernels available (pointnet2_ops) — using accelerated ops")
except ImportError:
    print("[PointNet++] pointnet2_ops not found — using pure PyTorch GPU ops "
          "(install pointnet2_ops for 5-10x speedup on FPS/ball_query)")


# ===================================================================
# Pure PyTorch fallbacks (still run on GPU via standard CUDA tensors)
# ===================================================================

def _fps_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling — pure PyTorch.

    Iteratively selects the point furthest from the already-selected set.

    Args:
        xyz: (B, N, 3) point coordinates
        npoint: number of points to sample

    Returns:
        (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    device = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    batch_indices = torch.arange(B, device=device)

    # Start from a random point per batch element
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        # Gather the selected point: (B, 1, 3)
        centroid_xyz = xyz[batch_indices, farthest].unsqueeze(1)
        # Squared distance from every point to selected centroid: (B, N)
        dist = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
        # Update minimum distance to the selected set (in-place)
        torch.minimum(distance, dist, out=distance)
        # Next farthest point
        farthest = torch.argmax(distance, dim=-1)

    return centroids


def _random_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Random point sampling — O(1) alternative to FPS.

    For spatially uniform blocks (from StridedSpatialHash), random sampling
    provides comparable coverage to FPS with negligible compute cost.

    Args:
        xyz: (B, N, 3) point coordinates
        npoint: number of points to sample

    Returns:
        (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # Random permutation per batch element, take first npoint
    idx = torch.argsort(torch.rand(B, N, device=device), dim=-1)[:, :npoint]
    return idx


def _ball_query_pytorch(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Ball query — pure PyTorch.

    For each centroid in new_xyz, find up to nsample points in xyz within radius.
    Uses topk instead of full sort for O(N log K) vs O(N log N).

    Args:
        radius: search radius
        nsample: max number of neighbors
        xyz: (B, N, 3) all points
        new_xyz: (B, M, 3) centroids

    Returns:
        (B, M, nsample) indices into xyz. Pads with nearest index if < nsample.
    """
    B, N, _ = xyz.shape
    M = new_xyz.shape[1]

    # Pairwise squared distances: (B, M, N)
    dists = torch.cdist(new_xyz, xyz, p=2.0).square()
    radius_sq = radius * radius

    # Mask points outside radius with inf (avoids clone — modifies dists in-place
    # which is fine since we computed it fresh above)
    dists[dists > radius_sq] = float('inf')

    # Use topk to find K nearest within radius — O(N log K) vs O(N log N) for sort
    # topk with largest=False finds the smallest K values
    _, group_idx = dists.topk(nsample, dim=-1, largest=False)  # (B, M, nsample)

    # Check for centroids with no/insufficient neighbors in radius
    # Gather distances of selected neighbors
    gathered_dists = torch.gather(dists, 2, group_idx)  # (B, M, nsample)
    invalid = gathered_dists.isinf()  # (B, M, nsample)

    if invalid.any():
        # For fully empty centroids, use nearest point from original distances
        # Recompute un-masked distances only for problematic centroids
        dists_clean = torch.cdist(new_xyz, xyz, p=2.0).square()
        nearest_idx = dists_clean.argmin(dim=-1, keepdim=True)  # (B, M, 1)
        nearest_idx_expanded = nearest_idx.expand_as(group_idx)  # (B, M, nsample)

        # Replace invalid entries with the first valid or nearest point
        first_valid = group_idx[:, :, 0:1].expand_as(group_idx)
        # Use first valid neighbor for padding, nearest for fully empty
        any_valid = ~gathered_dists[:, :, 0:1].isinf()  # (B, M, 1)
        fill = torch.where(any_valid.expand_as(group_idx), first_valid, nearest_idx_expanded)
        group_idx = torch.where(invalid, fill, group_idx)

    return group_idx


def _three_nn_pytorch(
    unknown: torch.Tensor, known: torch.Tensor
) -> tuple:
    """
    Find 3 nearest neighbors in `known` for each point in `unknown`.

    Args:
        unknown: (B, N, 3) query points
        known: (B, M, 3) reference points

    Returns:
        dist: (B, N, 3) squared distances to 3 nearest neighbors
        idx: (B, N, 3) indices of 3 nearest neighbors in known
    """
    # (B, N, M)
    dists = torch.cdist(unknown, known, p=2.0).square()
    # Top 3 nearest
    dist, idx = dists.topk(3, dim=-1, largest=False)
    return dist, idx


def _three_interpolate_pytorch(
    features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Inverse-distance weighted interpolation using 3 nearest neighbors.

    Args:
        features: (B, C, M) features at known points
        idx: (B, N, 3) indices of 3 nearest known points per unknown point
        weight: (B, N, 3) interpolation weights (inverse distance, normalized)

    Returns:
        (B, C, N) interpolated features
    """
    B, C, M = features.shape
    N = idx.shape[1]

    # Gather features for the 3 neighbors: (B, C, N, 3)
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, N, 3)
    gathered = features.unsqueeze(2).expand(-1, -1, N, -1)  # (B, C, N, M)
    # Advanced indexing for each neighbor
    neighbor_features = torch.gather(gathered, 3, idx_expanded)  # (B, C, N, 3)

    # Weighted sum: (B, C, N)
    weight_expanded = weight.unsqueeze(1)  # (B, 1, N, 3)
    interpolated = (neighbor_features * weight_expanded).sum(dim=-1)  # (B, C, N)

    return interpolated


# ===================================================================
# Public API — dispatches to CUDA or PyTorch backend
# ===================================================================

def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling.

    Args:
        xyz: (B, N, 3) input point coordinates
        npoint: number of centroids to select

    Returns:
        (B, npoint) indices of selected centroids
    """
    if _USE_CUDA_OPS and xyz.is_cuda:
        return _cuda_fps(xyz.contiguous(), npoint)
    return _fps_pytorch(xyz, npoint)


def random_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Random point sampling — fast O(1) alternative to FPS for training.

    Args:
        xyz: (B, N, 3) input point coordinates
        npoint: number of centroids to select

    Returns:
        (B, npoint) indices of selected centroids
    """
    return _random_sample(xyz, npoint)


def ball_query(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Ball Query — find neighbors within radius.

    Args:
        radius: search radius
        nsample: max neighbors per centroid
        xyz: (B, N, 3) all points
        new_xyz: (B, M, 3) centroids

    Returns:
        (B, M, nsample) neighbor indices
    """
    if _USE_CUDA_OPS and xyz.is_cuda:
        return _cuda_ball_query(radius, nsample, xyz.contiguous(), new_xyz.contiguous())
    return _ball_query_pytorch(radius, nsample, xyz, new_xyz)


def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> tuple:
    """
    Find 3 nearest neighbors for feature propagation interpolation.

    Args:
        unknown: (B, N, 3) query points
        known: (B, M, 3) reference points

    Returns:
        dist: (B, N, 3) squared distances
        idx: (B, N, 3) neighbor indices
    """
    if _USE_CUDA_OPS and unknown.is_cuda:
        dist, idx = _cuda_three_nn(unknown.contiguous(), known.contiguous())
        return dist, idx
    return _three_nn_pytorch(unknown, known)


def three_interpolate(
    features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """
    Weighted interpolation using 3 nearest neighbors.

    Args:
        features: (B, C, M) source features
        idx: (B, N, 3) neighbor indices
        weight: (B, N, 3) interpolation weights

    Returns:
        (B, C, N) interpolated features
    """
    if _USE_CUDA_OPS and features.is_cuda:
        return _cuda_three_interpolate(features.contiguous(), idx.contiguous(),
                                       weight.contiguous())
    return _three_interpolate_pytorch(features, idx, weight)


def gather_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather point features by index.

    Args:
        features: (B, C, N) point features
        idx: (B, npoint) indices to gather

    Returns:
        (B, C, npoint) gathered features
    """
    if _USE_CUDA_OPS and features.is_cuda:
        return _cuda_grouping(features.contiguous(), idx.contiguous().int())
    # Pure PyTorch
    idx_expanded = idx.unsqueeze(1).expand(-1, features.shape[1], -1)  # (B, C, npoint)
    return torch.gather(features, 2, idx_expanded)
