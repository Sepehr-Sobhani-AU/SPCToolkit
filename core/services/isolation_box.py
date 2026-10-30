"""
The viewer's isolation box: a box in world space that hides everything
outside it.

A box is a centre, three unit axes and three half extents. Level boxes use the
world axes; a slab is the same box turned to face the camera and squashed thin
along the view direction. Nothing here touches points the user owns: the box is
view state, like the camera, and the graphics card does the hiding through
six clipping planes (see ``IsolationBoxMixin``).

The few places that read points directly — picking, the lasso, the zoom window,
and filling the box at full resolution under level of detail — use
``contains`` so they agree with what the clip planes show.

No Qt and no OpenGL imports, so it can be tested on its own.
"""

from dataclasses import dataclass, field

import numpy as np

# Rows per pass in ``contains``: keeps the temporaries bounded at 170M points.
_CHUNK = 4_000_000

# A box face is never dragged closer than this to the opposite one (metres).
MIN_HALF_EXTENT = 0.01

# The seven grip handles, in a fixed order: +X, -X, +Y, -Y, +Z, -Z faces, then
# the centre. A face handle is (axis, sign).
FACE_HANDLES = ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0))
CENTRE_HANDLE = len(FACE_HANDLES)

# The twelve edges as corner-index pairs, corners numbered by the sign bits of
# (x, y, z) — see ``corners``.
_EDGES = np.array([
    (0, 1), (2, 3), (4, 5), (6, 7),      # along x
    (0, 2), (1, 3), (4, 6), (5, 7),      # along y
    (0, 4), (1, 5), (2, 6), (3, 7),      # along z
], dtype=np.int64)


@dataclass
class IsolationBox:
    """A box in world space.

    Attributes:
        centre: (3,) world position of the middle of the box.
        half_extents: (3,) half size along each of ``axes``.
        axes: (3, 3) rows are the box's unit axes in world space. Identity for
            a level box.
    """

    centre: np.ndarray
    half_extents: np.ndarray
    axes: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.centre = np.asarray(self.centre, dtype=np.float64).reshape(3)
        self.half_extents = np.maximum(
            np.asarray(self.half_extents, dtype=np.float64).reshape(3),
            MIN_HALF_EXTENT)
        self.axes = np.asarray(self.axes, dtype=np.float64).reshape(3, 3)

    @classmethod
    def cube(cls, centre, size):
        """A level cube *size* across, centred on *centre*."""
        return cls(centre, np.full(3, float(size) / 2.0))

    def copy(self) -> "IsolationBox":
        return IsolationBox(self.centre.copy(), self.half_extents.copy(),
                       self.axes.copy())

    # ------------------------------------------------------------------
    # What it hides
    # ------------------------------------------------------------------

    def planes(self) -> np.ndarray:
        """(6, 4) plane equations ``a*x + b*y + c*z + d >= 0`` for inside.

        In world coordinates, in the form ``glClipPlane`` takes when it is
        called with the scene's model-view matrix loaded. One pair per axis:
        the +face keeps points below it, the -face keeps points above it.
        """
        eqs = np.empty((6, 4), dtype=np.float64)
        for i in range(3):
            n = self.axes[i]
            c = float(n @ self.centre)
            h = float(self.half_extents[i])
            eqs[2 * i] = (-n[0], -n[1], -n[2], c + h)        # n.x <= c + h
            eqs[2 * i + 1] = (n[0], n[1], n[2], h - c)       # n.x >= c - h
        return eqs

    def contains(self, points) -> np.ndarray:
        """Boolean mask of the rows of *points* (N, >=3) inside the box.

        The same test the clip planes apply, so a point is reported inside
        exactly when it is drawn. Chunked and float32 so a 170M-point cloud
        needs no full-size float64 copy.

        Runs on the CPU on purpose. The job is one read of every point and
        almost no arithmetic, so it is bound by memory, and copying the cloud
        to the graphics card costs more than the test: measured at 50M points,
        0.12 s here against 0.31 s with CuPy. The trick is a cheap first pass
        on one coordinate against the box's world-space bounds, so the full
        rotated test only runs on the few points that pass it.
        """
        pts = np.asarray(points)
        n = len(pts)
        out = np.zeros(n, dtype=bool)
        if n == 0:
            return out
        axes = self.axes.astype(np.float32)
        centre = self.centre.astype(np.float32)
        half = self.half_extents.astype(np.float32)

        # World-space bounds of the box along x: a point outside them is
        # outside the box, whichever way the box is turned. Padded by a hair
        # so float32 rounding can never drop a point the full test would keep.
        reach = float(np.abs(self.axes[:, 0]) @ self.half_extents)
        pad = 1e-5 * (abs(float(self.centre[0])) + reach + 1.0)
        lo = np.float32(self.centre[0] - reach - pad)
        hi = np.float32(self.centre[0] + reach + pad)

        for start in range(0, n, _CHUNK):
            block = pts[start:start + _CHUNK, :3]
            near = block[:, 0] >= lo
            near &= block[:, 0] <= hi
            rows = np.flatnonzero(near)
            if rows.size == 0:
                continue
            local = (block[rows] - centre) @ axes.T    # (k, 3) box coordinates
            np.abs(local, out=local)
            out[start + rows[np.all(local <= half, axis=1)]] = True
        return out

    def contains_point(self, point) -> bool:
        local = self.axes @ (np.asarray(point, dtype=np.float64) - self.centre)
        return bool(np.all(np.abs(local) <= self.half_extents))

    # ------------------------------------------------------------------
    # Drawing and handles
    # ------------------------------------------------------------------

    def corners(self) -> np.ndarray:
        """(8, 3) corners; corner k has sign bit i set when it is on the -axis i
        side."""
        out = np.empty((8, 3), dtype=np.float64)
        for k in range(8):
            signs = np.array([-1.0 if k & (1 << i) else 1.0 for i in range(3)])
            out[k] = self.centre + (signs * self.half_extents) @ self.axes
        return out

    def edges(self) -> np.ndarray:
        """(12, 2, 3) start and end of each outline edge."""
        c = self.corners()
        return np.stack([c[_EDGES[:, 0]], c[_EDGES[:, 1]]], axis=1)

    def handle_positions(self) -> np.ndarray:
        """(7, 3) world positions of the grip handles: six faces, then the
        centre (see ``FACE_HANDLES``)."""
        out = np.empty((7, 3), dtype=np.float64)
        for k, (axis, sign) in enumerate(FACE_HANDLES):
            out[k] = self.centre + sign * self.half_extents[axis] * self.axes[axis]
        out[CENTRE_HANDLE] = self.centre
        return out

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def drag_face(self, handle: int, distance: float, one_sided: bool,
                  scale_axes=(0, 1, 2)) -> None:
        """Move face *handle* outward by *distance* metres (negative = inward).

        Normally the whole box grows or shrinks about its centre, keeping its
        shape: the dragged face moves *distance* and every side along
        *scale_axes* scales by the same factor. A slab passes ``(0, 1)`` so its
        thickness stays as set.

        With *one_sided*, only the dragged face moves: the opposite face stays
        put and the centre follows halfway.
        """
        axis, sign = FACE_HANDLES[handle]
        old = self.half_extents[axis]
        if one_sided:
            new = max(old + distance / 2.0, MIN_HALF_EXTENT)
            # The opposite face stays put: shift the centre by however much the
            # half extent actually changed, towards the dragged face.
            self.centre = self.centre + sign * (new - old) * self.axes[axis]
            self.half_extents[axis] = new
            return

        axes = list(scale_axes)
        # Never let the smallest scaled side go below the minimum.
        factor = max((old + distance) / old,
                     MIN_HALF_EXTENT / float(np.min(self.half_extents[axes])))
        self.half_extents[axes] = self.half_extents[axes] * factor

    def move(self, offset) -> None:
        self.centre = self.centre + np.asarray(offset, dtype=np.float64)

    def slab(self, view_axes, thickness: float) -> "IsolationBox":
        """This box as a slab through its centre, facing the camera.

        *view_axes* rows are the camera's right, up and viewing direction in
        world space. The slab keeps the box's centre and covers the box
        sideways — its in-plane size is the box's largest side — so turning a
        box into a slab never loses the part of the scene you were looking at.
        """
        side = float(np.max(self.half_extents))
        return IsolationBox(self.centre.copy(),
                       np.array([side, side, float(thickness) / 2.0]),
                       np.asarray(view_axes, dtype=np.float64))

    def step(self, axis: int, distance: float) -> None:
        """Move the box *distance* metres along its own *axis*."""
        self.centre = self.centre + distance * self.axes[axis]

    def key(self) -> tuple:
        """A hashable summary, for caching anything computed from the box."""
        return tuple(np.round(np.concatenate(
            [self.centre, self.half_extents, self.axes.ravel()]), 6))
