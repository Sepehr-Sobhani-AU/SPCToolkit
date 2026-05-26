"""3D cylinder RANSAC model — CPU only.

Cylinder is parametrised by an anchor point on the axis, a unit axis
direction, and a radius. The minimal fit uses 2 points plus their
surface normals (the standard two-normal closed-form). Refit is an
LM least-squares on perpendicular-distance-to-axis minus radius.

GPU support is intentionally omitted (``supports_gpu = False``) —
the iterative refit doesn't vectorise across rows, and no consumer
currently needs batched cylinder fitting. Revisit if a real workload
appears.
"""

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..base import RansacModel


_EPS = 1e-12


def _perp_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal basis (u, v) of the plane perpendicular to ``direction``."""
    helper = (
        np.array([1.0, 0.0, 0.0])
        if abs(direction[0]) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    return u, v


class CylinderModel(RansacModel):
    """Cylinder defined by an axis point, an axis direction, and a radius."""

    requires_normals = True
    min_samples = 2
    supports_gpu = False

    def __init__(self):
        self.point: Optional[np.ndarray] = None
        self.direction: Optional[np.ndarray] = None
        self.radius: Optional[float] = None

    def fit_minimal(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bool:
        if normals is None:
            return False
        p1, p2 = points[0], points[1]
        n1, n2 = normals[0], normals[1]

        # Axis direction from cross product of normals.
        d = np.cross(n1, n2)
        d_norm = float(np.linalg.norm(d))
        if d_norm < _EPS:
            return False  # normals parallel — sample is degenerate
        d = d / d_norm

        # Project points and normals into the plane perpendicular to the axis.
        u, v = _perp_basis(d)
        p1_2d = np.array([p1 @ u, p1 @ v])
        p2_2d = np.array([p2 @ u, p2 @ v])
        n1_2d = np.array([n1 @ u, n1 @ v])
        n2_2d = np.array([n2 @ u, n2 @ v])

        # Intersect the two normal rays in 2D: p1_2d + t1*n1_2d = p2_2d + t2*n2_2d
        # => [n1_2d, -n2_2d] @ [t1, t2] = p2_2d - p1_2d
        A = np.column_stack([n1_2d, -n2_2d])
        det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        if abs(det) < _EPS:
            return False  # projected normals parallel
        t = np.linalg.solve(A, p2_2d - p1_2d)
        centre_2d = p1_2d + t[0] * n1_2d

        # Lift back to 3D; along-axis offset = projection of (p1+p2)/2.
        centroid = 0.5 * (p1 + p2)
        along = float(centroid @ d)
        centre_3d = centre_2d[0] * u + centre_2d[1] * v + along * d

        # Radius — average of the two in-plane distances for stability.
        r1 = float(np.linalg.norm(np.array([p1 @ u, p1 @ v]) - centre_2d))
        r2 = float(np.linalg.norm(np.array([p2 @ u, p2 @ v]) - centre_2d))
        radius = 0.5 * (r1 + r2)
        if radius < _EPS:
            return False

        self.point = centre_3d
        self.direction = d
        self.radius = radius
        return True

    def distances(self, points: np.ndarray) -> np.ndarray:
        vecs = points - self.point
        along = vecs @ self.direction
        perp = vecs - along[:, None] * self.direction
        dist_to_axis = np.linalg.norm(perp, axis=1)
        return np.abs(dist_to_axis - self.radius)

    def refit(
        self,
        inlier_points: np.ndarray,
        inlier_normals: Optional[np.ndarray] = None,
    ) -> bool:
        if len(inlier_points) < 5:
            return False

        def residual(params: np.ndarray, pts: np.ndarray) -> np.ndarray:
            px, py, pz, dx, dy, dz, r = params
            d = np.array([dx, dy, dz])
            d_norm = np.linalg.norm(d)
            if d_norm < _EPS:
                return np.full(len(pts), 1e10)
            d = d / d_norm
            p0 = np.array([px, py, pz])
            vecs = pts - p0
            along = vecs @ d
            perp = vecs - along[:, None] * d
            dist_to_axis = np.linalg.norm(perp, axis=1)
            return dist_to_axis - r

        x0 = np.concatenate([self.point, self.direction, [self.radius]])
        try:
            result = least_squares(
                residual,
                x0,
                args=(inlier_points,),
                method="lm",
                max_nfev=200,
            )
        except Exception:
            return False
        if not result.success:
            return False

        px, py, pz, dx, dy, dz, r = result.x
        d = np.array([dx, dy, dz])
        d_norm = float(np.linalg.norm(d))
        if d_norm < _EPS or r < _EPS:
            return False

        self.point = np.array([px, py, pz])
        self.direction = d / d_norm
        self.radius = float(r)
        return True
