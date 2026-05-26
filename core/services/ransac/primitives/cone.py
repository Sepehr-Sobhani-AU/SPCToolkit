"""3D cone RANSAC model — CPU only.

Cone is parametrised by an apex, a unit axis direction (pointing from
apex toward the wide end), and a half-angle in radians (angle between
the axis and the lateral surface). The minimal fit uses 3 points plus
their surface normals: the apex lies on every tangent plane, giving
a 3×3 linear system, and the axis is the unique direction making
equal angles with the three apex-to-point vectors.

Both sheets of the infinite double-cone are treated as one surface.
A single-sheet restriction is a one-line filter to add later if needed.

GPU support is intentionally omitted (``supports_gpu = False``) for
the same reason as cylinder — iterative refit, no current consumer.
"""

from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from ..base import RansacModel


_EPS = 1e-12


class ConeModel(RansacModel):
    """Cone defined by apex, axis direction, and half-angle (radians)."""

    requires_normals = True
    min_samples = 3
    supports_gpu = False

    def __init__(self):
        self.apex: Optional[np.ndarray] = None
        self.axis: Optional[np.ndarray] = None
        self.half_angle: Optional[float] = None

    def fit_minimal(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
    ) -> bool:
        if normals is None:
            return False
        # Apex from the three tangent-plane equations n_i · (apex - p_i) = 0.
        N = np.stack([normals[0], normals[1], normals[2]])      # (3, 3)
        b = np.array([float(normals[i] @ points[i]) for i in range(3)])
        if abs(float(np.linalg.det(N))) < _EPS:
            return False  # linearly dependent normals
        try:
            apex = np.linalg.solve(N, b)
        except np.linalg.LinAlgError:
            return False

        # Unit vectors from apex to each sample point.
        vs = np.empty((3, 3))
        for i in range(3):
            v = points[i] - apex
            n = float(np.linalg.norm(v))
            if n < _EPS:
                return False  # apex coincides with a sample point
            vs[i] = v / n

        # Axis = vector with equal dot product against all three vs.
        # That's the unique direction perpendicular to (v2 - v1) and (v3 - v1).
        axis = np.cross(vs[1] - vs[0], vs[2] - vs[0])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < _EPS:
            return False
        axis = axis / axis_norm

        # Sign-align: axis should point from apex toward the points.
        centroid = points.mean(axis=0)
        if float(axis @ (centroid - apex)) < 0:
            axis = -axis

        # Half-angle from mean of v_i · axis (clamped for safety).
        cos_mean = float(np.mean(vs @ axis))
        cos_mean = max(-1.0, min(1.0, cos_mean))
        half_angle = float(np.arccos(cos_mean))
        if half_angle < _EPS or half_angle > (0.5 * np.pi - _EPS):
            return False  # degenerate cone (line or plane)

        self.apex = apex
        self.axis = axis
        self.half_angle = half_angle
        return True

    def distances(self, points: np.ndarray) -> np.ndarray:
        v = points - self.apex
        v_norm = np.linalg.norm(v, axis=1)
        safe = np.where(v_norm < _EPS, 1.0, v_norm)
        cos_theta = np.clip((v @ self.axis) / safe, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return v_norm * np.abs(np.sin(theta - self.half_angle))

    def refit(
        self,
        inlier_points: np.ndarray,
        inlier_normals: Optional[np.ndarray] = None,
    ) -> bool:
        if len(inlier_points) < 6:
            return False

        def residual(params: np.ndarray, pts: np.ndarray) -> np.ndarray:
            ax, ay, az, dx, dy, dz, half_angle = params
            axis = np.array([dx, dy, dz])
            axis_norm = np.linalg.norm(axis)
            if axis_norm < _EPS:
                return np.full(len(pts), 1e10)
            axis = axis / axis_norm
            apex = np.array([ax, ay, az])
            v = pts - apex
            v_norm = np.linalg.norm(v, axis=1)
            safe = np.where(v_norm < _EPS, 1.0, v_norm)
            cos_theta = np.clip((v @ axis) / safe, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            return v_norm * np.sin(theta - half_angle)

        x0 = np.concatenate([self.apex, self.axis, [self.half_angle]])
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

        ax, ay, az, dx, dy, dz, half_angle = result.x
        axis = np.array([dx, dy, dz])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < _EPS:
            return False
        if half_angle <= _EPS or half_angle >= 0.5 * np.pi:
            return False

        self.apex = np.array([ax, ay, az])
        self.axis = axis / axis_norm
        self.half_angle = float(half_angle)
        return True
