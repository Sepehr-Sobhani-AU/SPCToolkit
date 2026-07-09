"""
Shared small-vector / PCA geometry helpers.

Pure NumPy, no GUI, no torch — safe to import from ``core/services`` engines
and from plugins alike. These consolidate a handful of one-liners that were
copy-pasted across the linear-feature family (``linear_region_grower`` and
``crease_tracer``):

- ``unit`` / ``unit_rows`` — the ``vec / (norm + eps)`` zero-safe normalise idiom.
- ``perp_basis`` — an orthonormal basis of the plane perpendicular to a direction.
- ``principal_axis`` — dominant PCA axis of a set of **spatial points**
  (mean-centred covariance).
- ``dominant_direction`` — dominant axis of a set of **directions / normals**
  (raw second moment, NOT mean-centred).

``principal_axis`` and ``dominant_direction`` are deliberately kept as two
distinct functions: mean-centring is correct for spatial coordinates but wrong
for a bundle of unit normals (whose interesting structure lives in the raw
second-moment matrix about the origin, not about their mean).
"""

import numpy as np

# Shared zero-division guard for all normalisations here.
_EPS = 1e-12


def unit(vec, eps: float = _EPS) -> np.ndarray:
    """Return *vec* scaled to unit length; zero-safe (``eps`` guards the norm)."""
    vec = np.asarray(vec, dtype=np.float64)
    return vec / (np.linalg.norm(vec) + eps)


def unit_rows(vecs, eps: float = _EPS) -> np.ndarray:
    """Row-wise :func:`unit` for an ``(N, D)`` array of vectors."""
    vecs = np.asarray(vecs, dtype=np.float64)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + eps)


def perp_basis(direction):
    """Two orthonormal vectors spanning the plane perpendicular to *direction*.

    The reference axis is chosen off *direction* (z unless *direction* is itself
    near-vertical, then x) so the cross products never degenerate.
    """
    d = unit(direction)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = unit(np.cross(d, ref))
    v = np.cross(d, u)
    return u, v


def principal_axis(points) -> np.ndarray:
    """Unit dominant axis of *points* — the eigenvector of the largest
    **mean-centred** covariance eigenvalue.

    For spatial coordinates: always defined, even when the points are not
    collinear. Use :func:`dominant_direction` instead for a bundle of
    directions/normals, where mean-centring would be wrong.
    """
    pts = np.asarray(points, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    _, eigvecs = np.linalg.eigh(centered.T @ centered)
    return unit(eigvecs[:, -1])


def dominant_direction(vectors) -> np.ndarray:
    """Unit dominant axis of a bundle of *vectors* (directions / normals) — the
    eigenvector of the largest **raw second-moment** eigenvalue (``Vᵀ V``, NOT
    mean-centred).

    Use this for unit normals or heading vectors, whose meaningful spread is
    about the origin. Use :func:`principal_axis` for spatial point coordinates.
    """
    v = np.asarray(vectors, dtype=np.float64)
    _, eigvecs = np.linalg.eigh(v.T @ v)
    return unit(eigvecs[:, -1])
