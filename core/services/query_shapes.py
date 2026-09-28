"""
Query Shapes

The regions ``core.services.shape_query`` can select points by: sphere, box,
rotated box, cylinder, slab and a polygon fence drawn in plan.

**A shape answers geometry questions only.** Its bounding box, whether each
cell of the coarse spatial index is missed, fully covered or only partly covered,
and the numbers the GPU kernel needs to test a point. It never tests points
itself: the kernel in ``shape_query_kernels`` is the one definition of "inside",
so a point sitting exactly on an edge lands on the same side every time
(DECISIONS.md § 2026-09-28, "one test, one truth").

**Covered means certain.** A cell reported covered is taken whole, with no point
tested, so the claim must hold for every point the cell can hold. The service
passes cell boxes already padded against rounding in the cell numbering, and
every shape here is shrunk by a *margin* before its corners are checked, so a
corner that only just touches an edge makes the cell partial, never covered.
Partial is always safe: it only means the points get tested.

All coordinates are float32 world units, in the cloud's own (origin-shifted)
frame.
"""

import numpy as np

# Cell states, as returned by ``cell_states``.
MISS, PARTIAL, COVER = 0, 1, 2

# Kernel shape numbers. Must match ``inside()`` in shape_query_kernels.
KIND_SPHERE = 0
KIND_ORIENTED_BOX = 1
KIND_CYLINDER = 2
KIND_BOX = 3
KIND_SLAB = 4
KIND_PLAN_POLYGON = 5

# The 8 corners of a unit cube, for turning cell boxes into corner points.
_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                    dtype=np.float32)


def _vec3(value, name):
    v = np.asarray(value, dtype=np.float32).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be 3 finite numbers, got {value!r}")
    return v


def _positive(value, name):
    v = float(value)
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return v


def _unit(value, name):
    v = _vec3(value, name).astype(np.float64)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError(f"{name} must not be zero")
    return (v / norm).astype(np.float32)


def _corners(cell_lo, cell_hi):
    """(C, 8, 3) corner points of C boxes."""
    size = (cell_hi - cell_lo)[:, None, :]
    return cell_lo[:, None, :] + _CORNERS[None] * size


def _overlaps(lo, hi, cell_lo, cell_hi):
    """(C,) bool: box lo..hi touches each cell box."""
    return np.all(hi >= cell_lo, axis=1) & np.all(lo <= cell_hi, axis=1)


class QueryShape:
    """A region points can be selected by. See the module docstring."""

    kind = None

    def aabb(self):
        """``(lo, hi)`` float32 bounds. May be infinite; the service clips it."""
        raise NotImplementedError

    def kernel_params(self) -> np.ndarray:
        """float32 numbers the kernel reads for this shape, in its fixed order."""
        raise NotImplementedError

    def _corners_inside(self, corners, margin):
        """(C, 8) bool: each corner inside the shape shrunk by *margin*.

        Only used to decide COVER, so a shape whose convex hull argument does not
        hold (a concave polygon) overrides ``cell_states`` instead.
        """
        raise NotImplementedError

    def cell_states(self, cell_lo, cell_hi, margin) -> np.ndarray:
        """MISS / PARTIAL / COVER for each of the C cell boxes, as (C,) uint8.

        The default holds for every **convex** shape: all 8 corners inside means
        the whole box is inside, since a convex region contains the hull of any
        points in it.
        """
        lo, hi = self.aabb()
        states = np.zeros(len(cell_lo), dtype=np.uint8)
        hit = _overlaps(lo, hi, cell_lo, cell_hi)
        if not hit.any():
            return states
        states[hit] = PARTIAL
        inside = self._corners_inside(_corners(cell_lo[hit], cell_hi[hit]), margin)
        covered = np.flatnonzero(hit)[inside.all(axis=1)]
        states[covered] = COVER
        return states


class Sphere(QueryShape):
    """Every point within *radius* of *centre*."""

    kind = KIND_SPHERE

    def __init__(self, centre, radius):
        self.centre = _vec3(centre, "centre")
        self.radius = np.float32(_positive(radius, "radius"))

    def aabb(self):
        return self.centre - self.radius, self.centre + self.radius

    def kernel_params(self):
        return np.array([*self.centre, self.radius * self.radius], dtype=np.float32)

    def _corners_inside(self, corners, margin):
        r = max(float(self.radius) - margin, 0.0)
        d = corners.astype(np.float64) - self.centre
        return np.einsum("cki,cki->ck", d, d) <= r * r


class Box(QueryShape):
    """Axis-aligned box, *low* to *high* inclusive on every side."""

    kind = KIND_BOX

    def __init__(self, low, high):
        self.low = _vec3(low, "low")
        self.high = _vec3(high, "high")
        if np.any(self.high < self.low):
            raise ValueError(f"box high {self.high} is below low {self.low}")

    def aabb(self):
        return self.low.copy(), self.high.copy()

    def kernel_params(self):
        return np.concatenate([self.low, self.high]).astype(np.float32)

    def _corners_inside(self, corners, margin):
        return (np.all(corners >= self.low + margin, axis=2)
                & np.all(corners <= self.high - margin, axis=2))


class OrientedBox(QueryShape):
    """Box of *half_sizes* around *centre*, turned by *rotation*.

    *rotation* is 3x3; its columns are the box's own axes in world space, so a
    point's box coordinates are ``(p - centre) @ rotation``.
    """

    kind = KIND_ORIENTED_BOX

    def __init__(self, centre, rotation, half_sizes):
        self.centre = _vec3(centre, "centre")
        r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-4):
            raise ValueError("rotation must be orthonormal (a pure rotation)")
        self.rotation = r.astype(np.float32)
        self.half_sizes = _vec3(half_sizes, "half_sizes")
        if np.any(self.half_sizes < 0):
            raise ValueError("half_sizes must be >= 0")

    def aabb(self):
        extent = np.abs(self.rotation) @ self.half_sizes
        return self.centre - extent, self.centre + extent

    def kernel_params(self):
        return np.concatenate([self.centre, self.rotation.ravel(),
                               self.half_sizes]).astype(np.float32)

    def _corners_inside(self, corners, margin):
        local = (corners.astype(np.float64) - self.centre) @ self.rotation
        return np.all(np.abs(local) <= self.half_sizes - margin, axis=2)


class Cylinder(QueryShape):
    """Solid cylinder from *start* along *direction* for *length*, of *radius*."""

    kind = KIND_CYLINDER

    def __init__(self, start, direction, radius, length):
        self.start = _vec3(start, "start")
        self.direction = _unit(direction, "direction")
        self.radius = np.float32(_positive(radius, "radius"))
        self.length = np.float32(_positive(length, "length"))

    def aabb(self):
        end = self.start + self.direction * self.length
        # A disc of radius r across axis d reaches r * sqrt(1 - d_i^2) along axis i.
        reach = self.radius * np.sqrt(np.clip(1.0 - self.direction ** 2, 0.0, 1.0))
        return (np.minimum(self.start, end) - reach).astype(np.float32), \
               (np.maximum(self.start, end) + reach).astype(np.float32)

    def kernel_params(self):
        return np.array([*self.start, *self.direction,
                         self.radius * self.radius, self.length], dtype=np.float32)

    def _corners_inside(self, corners, margin):
        v = corners.astype(np.float64) - self.start
        t = v @ self.direction
        r = max(float(self.radius) - margin, 0.0)
        return ((t >= margin) & (t <= float(self.length) - margin)
                & (np.einsum("cki,cki->ck", v, v) - t * t <= r * r))


class Slab(QueryShape):
    """Everything between two parallel planes: ``low <= normal . p <= high``.

    Open in the other two directions, so it is clipped to the cloud.
    """

    kind = KIND_SLAB

    def __init__(self, normal, low, high):
        self.normal = _unit(normal, "normal")
        self.low = np.float32(low)
        self.high = np.float32(high)
        if not (np.isfinite(self.low) and np.isfinite(self.high)) or self.high < self.low:
            raise ValueError(f"slab needs finite low <= high, got {low!r}, {high!r}")

    def aabb(self):
        lo = np.full(3, -np.inf, dtype=np.float32)
        hi = np.full(3, np.inf, dtype=np.float32)
        # Only an axis-aligned slab is bounded on its axis.
        axis = np.flatnonzero(np.abs(self.normal) == 1.0)
        if len(axis) == 1:
            a = int(axis[0])
            sign = float(self.normal[a])
            ends = sorted((self.low * sign, self.high * sign))
            lo[a], hi[a] = ends
        return lo, hi

    def kernel_params(self):
        return np.array([*self.normal, self.low, self.high], dtype=np.float32)

    def _corners_inside(self, corners, margin):
        d = corners.astype(np.float64) @ self.normal
        return (d >= float(self.low) + margin) & (d <= float(self.high) - margin)


class PlanPolygon(QueryShape):
    """A fence drawn in plan: inside the XY *polygon* and between *z_min* and *z_max*.

    The polygon may be concave but must not cross itself; it is closed
    automatically. *z_min* / *z_max* may be infinite for a fence of open height.
    """

    kind = KIND_PLAN_POLYGON

    def __init__(self, polygon, z_min=-np.inf, z_max=np.inf):
        poly = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(poly) > 1 and np.array_equal(poly[0], poly[-1]):
            poly = poly[:-1]
        if len(poly) < 3 or not np.all(np.isfinite(poly)):
            raise ValueError("polygon needs at least 3 finite XY vertices")
        self.polygon = poly
        self.z_min = np.float32(z_min)
        self.z_max = np.float32(z_max)
        if np.isnan(self.z_min) or np.isnan(self.z_max) or self.z_max < self.z_min:
            raise ValueError(f"need z_min <= z_max, got {z_min!r}, {z_max!r}")

    def aabb(self):
        lo, hi = self.polygon.min(axis=0), self.polygon.max(axis=0)
        return (np.array([lo[0], lo[1], self.z_min], dtype=np.float32),
                np.array([hi[0], hi[1], self.z_max], dtype=np.float32))

    def kernel_params(self):
        return np.concatenate([[self.z_min, self.z_max, len(self.polygon)],
                               self.polygon.ravel()]).astype(np.float32)

    def _inside_xy(self, x, y):
        """Even-odd crossing test, the same rule the kernel uses."""
        px, py = self.polygon[:, 0].astype(np.float64), self.polygon[:, 1].astype(np.float64)
        qx, qy = np.roll(px, 1), np.roll(py, 1)
        x = x[..., None]; y = y[..., None]
        straddles = (py > y) != (qy > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_x = (qx - px) * (y - py) / (qy - py) + px
        return np.count_nonzero(straddles & (x < cross_x), axis=-1) % 2 == 1

    def cell_states(self, cell_lo, cell_hi, margin):
        """Concave-safe: a cell is covered only if its four plan corners are inside,
        no polygon edge comes within *margin* of it, and the height range holds it."""
        lo, hi = self.aabb()
        states = np.zeros(len(cell_lo), dtype=np.uint8)
        hit = _overlaps(lo, hi, cell_lo, cell_hi)
        if not hit.any():
            return states
        states[hit] = PARTIAL

        idx = np.flatnonzero(hit)
        clo = cell_lo[idx].astype(np.float64) - margin
        chi = cell_hi[idx].astype(np.float64) + margin
        z_ok = (clo[:, 2] >= float(self.z_min)) & (chi[:, 2] <= float(self.z_max))
        cx = np.stack([clo[:, 0], chi[:, 0], clo[:, 0], chi[:, 0]], axis=1)
        cy = np.stack([clo[:, 1], clo[:, 1], chi[:, 1], chi[:, 1]], axis=1)
        corners_in = self._inside_xy(cx, cy).all(axis=1)
        covered = z_ok & corners_in & ~self._edges_touch(clo[:, :2], chi[:, :2])
        states[idx[covered]] = COVER
        return states

    def _edges_touch(self, rect_lo, rect_hi):
        """(C,) bool: any polygon edge meets the rectangle (Liang-Barsky clip)."""
        p0 = self.polygon.astype(np.float64)
        p1 = np.roll(p0, -1, axis=0)
        d = p1 - p0                                         # (E, 2)
        t0 = np.zeros((len(rect_lo), len(p0)))
        t1 = np.ones((len(rect_lo), len(p0)))
        ok = np.ones_like(t0, dtype=bool)
        for axis in (0, 1):
            for p, q in ((-d[:, axis], p0[:, axis] - rect_lo[:, axis, None]),
                         (d[:, axis], rect_hi[:, axis, None] - p0[:, axis])):
                p = np.broadcast_to(p, t0.shape)
                parallel = p == 0
                ok &= ~(parallel & (q < 0))
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(parallel, 0.0, q / np.where(parallel, 1.0, p))
                entering = ~parallel & (p < 0)
                leaving = ~parallel & (p > 0)
                t0 = np.where(entering, np.maximum(t0, r), t0)
                t1 = np.where(leaving, np.minimum(t1, r), t1)
        return np.any(ok & (t0 <= t1), axis=1)
