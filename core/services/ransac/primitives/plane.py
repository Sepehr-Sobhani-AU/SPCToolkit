"""3D plane RANSAC model — CPU single-cloud and batched GPU ops."""

from typing import Any, Optional, Tuple

import numpy as np

from ..base import RansacModel


_EPS = 1e-12


class PlaneModel(RansacModel):
    """Plane in R^3 defined by a point on the plane and a unit normal."""

    requires_normals = False
    min_samples = 3
    supports_gpu = True

    def __init__(self):
        self.point: Optional[np.ndarray] = None
        self.normal: Optional[np.ndarray] = None

    # -------- CPU --------

    def fit_minimal(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bool:
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        norm = float(np.linalg.norm(normal))
        if norm < _EPS:
            return False
        self.point = points[0].astype(np.float64)
        self.normal = (normal / norm).astype(np.float64)
        return True

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs((points - self.point) @ self.normal)

    def refit(
        self,
        inlier_points: np.ndarray,
        inlier_normals: Optional[np.ndarray] = None,
    ) -> bool:
        if len(inlier_points) < 3:
            return False
        centroid = inlier_points.mean(axis=0)
        centred = inlier_points - centroid
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        normal = vh[-1]  # smallest variance = plane normal
        norm = float(np.linalg.norm(normal))
        if norm < _EPS:
            return False
        self.point = centroid
        self.normal = normal / norm
        return True

    # -------- GPU, batched --------

    @classmethod
    def fit_minimal_batched_gpu(
        cls,
        points_b: "Any",       # (B, 3, 3)
        normals_b: "Any",
        device: "Any",
    ) -> Tuple[dict, "Any"]:
        import torch
        v1 = points_b[:, 1] - points_b[:, 0]                          # (B, 3)
        v2 = points_b[:, 2] - points_b[:, 0]                          # (B, 3)
        normal = torch.linalg.cross(v1, v2, dim=-1)                   # (B, 3)
        norms = torch.linalg.vector_norm(normal, dim=-1)              # (B,)
        valid = norms > _EPS                                           # (B,)
        safe_norms = torch.where(valid, norms, torch.ones_like(norms))
        normal = normal / safe_norms.unsqueeze(-1)
        normal = torch.where(
            valid.unsqueeze(-1), normal, torch.zeros_like(normal)
        )
        point = points_b[:, 0].clone()
        return {"point": point, "normal": normal}, valid

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
        normal = state["normal"]                                      # (B, 3)
        vecs = points_b - point.unsqueeze(1)                          # (B, N_max, 3)
        dots = (vecs * normal.unsqueeze(1)).sum(dim=-1)               # (B, N_max)
        dists = dots.abs()
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
        valid = counts.squeeze(-1) >= 3
        safe_counts = counts.clamp(min=1)
        centroid = (points_b * weights.unsqueeze(-1)).sum(dim=1) / safe_counts  # (B, 3)
        centred = (points_b - centroid.unsqueeze(1)) * weights.unsqueeze(-1)    # (B, N_max, 3)
        try:
            _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        except RuntimeError:
            return state, torch.zeros_like(valid)
        normal = vh[:, -1]                                            # (B, 3)
        norms = torch.linalg.vector_norm(normal, dim=-1)
        valid = valid & (norms > _EPS)
        safe_norms = torch.where(valid, norms, torch.ones_like(norms))
        normal = normal / safe_norms.unsqueeze(-1)
        new_state = {"point": centroid, "normal": normal}
        return new_state, valid

    @classmethod
    def unpack_to_model(cls, state: dict, b: int) -> "PlaneModel":
        model = cls()
        model.point = state["point"][b].detach().cpu().numpy().astype(np.float64)
        model.normal = state["normal"][b].detach().cpu().numpy().astype(np.float64)
        return model
