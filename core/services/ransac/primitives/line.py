"""3D line RANSAC model — CPU single-cloud and batched GPU ops."""

from typing import Any, Optional, Tuple

import numpy as np

from ..base import RansacModel


_EPS = 1e-12


class LineModel(RansacModel):
    """Line in R^3 defined by an anchor point and a unit direction."""

    requires_normals = False
    min_samples = 2
    supports_gpu = True

    def __init__(self):
        self.point: Optional[np.ndarray] = None
        self.direction: Optional[np.ndarray] = None

    # -------- CPU --------

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

    # -------- GPU, batched --------

    @classmethod
    def fit_minimal_batched_gpu(
        cls,
        points_b: "Any",       # (B, 2, 3)
        normals_b: "Any",
        device: "Any",
    ) -> Tuple[dict, "Any"]:
        import torch
        diff = points_b[:, 1] - points_b[:, 0]                       # (B, 3)
        norms = torch.linalg.vector_norm(diff, dim=-1)               # (B,)
        valid = norms > _EPS                                          # (B,)
        safe_norms = torch.where(valid, norms, torch.ones_like(norms))
        direction = diff / safe_norms.unsqueeze(-1)                   # (B, 3)
        direction = torch.where(
            valid.unsqueeze(-1), direction, torch.zeros_like(direction)
        )
        point = points_b[:, 0].clone()                                # (B, 3)
        return {"point": point, "direction": direction}, valid

    @classmethod
    def distances_batched_gpu(
        cls,
        state: dict,
        points_b: "Any",       # (B, N_max, 3)
        counts_b: "Any",       # (B,)
        device: "Any",
    ) -> "Any":
        import torch
        B, N_max, _ = points_b.shape
        point = state["point"]                                        # (B, 3)
        direction = state["direction"]                                # (B, 3)
        vecs = points_b - point.unsqueeze(1)                          # (B, N_max, 3)
        dir_exp = direction.unsqueeze(1).expand_as(vecs)              # (B, N_max, 3)
        cross = torch.linalg.cross(vecs, dir_exp, dim=-1)             # (B, N_max, 3)
        dists = torch.linalg.vector_norm(cross, dim=-1)               # (B, N_max)
        arange = torch.arange(N_max, device=device).unsqueeze(0).expand(B, -1)
        in_bounds = arange < counts_b.unsqueeze(1)
        return torch.where(in_bounds, dists, torch.full_like(dists, float("inf")))

    @classmethod
    def refit_batched_gpu(
        cls,
        state: dict,
        points_b: "Any",            # (B, N_max, 3)
        inlier_masks_b: "Any",      # (B, N_max)
        device: "Any",
    ) -> Tuple[dict, "Any"]:
        import torch
        weights = inlier_masks_b.to(points_b.dtype)                   # (B, N_max)
        counts = weights.sum(dim=1, keepdim=True)                     # (B, 1)
        valid = counts.squeeze(-1) >= 2                               # (B,)
        safe_counts = counts.clamp(min=1)
        centroid = (points_b * weights.unsqueeze(-1)).sum(dim=1) / safe_counts  # (B, 3)
        centred = (points_b - centroid.unsqueeze(1)) * weights.unsqueeze(-1)    # (B, N_max, 3)
        try:
            _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        except RuntimeError:
            return state, torch.zeros_like(valid)
        direction = vh[:, 0]                                          # (B, 3)
        norms = torch.linalg.vector_norm(direction, dim=-1)
        valid = valid & (norms > _EPS)
        safe_norms = torch.where(valid, norms, torch.ones_like(norms))
        direction = direction / safe_norms.unsqueeze(-1)
        new_state = {"point": centroid, "direction": direction}
        return new_state, valid

    @classmethod
    def unpack_to_model(cls, state: dict, b: int) -> "LineModel":
        model = cls()
        model.point = state["point"][b].detach().cpu().numpy().astype(np.float64)
        model.direction = state["direction"][b].detach().cpu().numpy().astype(np.float64)
        return model
