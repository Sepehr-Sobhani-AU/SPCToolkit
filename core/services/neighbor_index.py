"""
Neighbour queries over a point cloud.

A ``cKDTree``-shaped facade over ``core.services.spatial_grid``, so that an
algorithm asking "which points are near here?" goes through the same indexing
machinery the viewer's picking uses instead of building a KD-tree of its own.

Why not just keep the KD-tree. Building one is O(N log N) in Python-free scipy
code, but it is still the single most expensive thing a growing algorithm does:
measured on the linear region grower, the ``cKDTree`` construction was 96-98% of
a run, dwarfing every query it then served. A grid is one pass to number the
points and one counting sort, which is measured in seconds where the tree is
measured in minutes, and it holds 4 bytes a point where the tree holds far more.

What that costs. A single query is a little slower than the tree's — the grid
returns whole cells and this class measures them, where the tree walks straight
to the answer. That trade is worth taking whenever the number of queries is
bounded by the *feature* being traced rather than by the cloud, which is the
case for every grower here: a cable is thousands of queries over a cloud of
millions.

The two methods below are the two the growers use, with ``cKDTree``'s signatures
and return shapes, so a caller swaps one for the other and nothing else moves.

**Why it keeps a second copy of the coordinates.** In cell order, so that the
points of one cell sit next to each other in memory and reading a cell is a
slice rather than a gather from all over the cloud. That is the single largest
factor in a query's cost — measured on a real 12M cloud, a 0.5 m ball query went
from 156 us to 33 us — and it is the same trick a KD-tree plays when it builds
its own reordered copy of the data. It costs 12 bytes a point, and the whole
index still comes to a third of what the tree wanted.
"""

import numpy as np

from core.services.spatial_grid import SpatialGrid

# Points per cell the index aims for when it is not given a cell size. This is
# the knob that matters: a k-NN query looks at the home cell and its 26
# neighbours first, so this number times 27 is roughly how many points get
# measured to answer one query. Too small and a query walks many cells to find
# anything; too large and it measures a crowd to return sixteen.
DEFAULT_POINTS_PER_CELL = 16

# Ceiling on how finely the bounding box is divided, whatever the cloud size.
# Cells cost 8 bytes each in the grid's offset table and nothing per point, so
# this is a 64 MB cap on the part of the index that does not scale with points.
MAX_CELLS = 8_000_000

# Query-candidate pairs held at once while measuring distances. Bounds the
# (queries x candidates x 3) scratch to about 24 MB regardless of how many
# query points share a cell.
QUERY_PAIR_BLOCK = 1_000_000

# Give up widening after this many doublings. Only reachable if the grid is
# somehow degenerate; a box that has grown past the grid on every side reports
# an infinite guaranteed reach and finishes on its own.
_MAX_RINGS = 40


class NeighborIndex:
    """Radius and k-nearest queries over a fixed set of points.

    Immutable once built, and safe to share between threads: every query reads
    the index and writes only its own answer.

    Attributes:
        points: the (N, >=3) array the index was built over. Held, not copied —
            the same array must be passed to the queries' caller, which is
            exactly how ``cKDTree`` behaves.
        n: number of points. Also the "no neighbour" index, matching
            ``cKDTree``.
        grid: the underlying ``SpatialGrid``, sorted. None for an empty cloud.
    """

    __slots__ = ("points", "n", "grid", "_xyz", "_order")

    def __init__(self, points, points_per_cell=DEFAULT_POINTS_PER_CELL,
                 cell_size=None, max_cells=MAX_CELLS, backend=None):
        """Index *points*.

        Args:
            points: (N, >=3) array. Only xyz is read.
            points_per_cell: how many points a cell should hold on average. The
                cell *size* then follows the cloud, which is what makes one
                default work for a 3 m room and a 400 m street.
            cell_size: set the cell edge length directly instead, in world
                units, when the caller knows its query radius. Overrides
                *points_per_cell*.
            max_cells: ceiling on the cell count, so a small *cell_size* on a
                large site cannot allocate an unbounded offset table.
            backend: override the registry's CPU/GPU choice. Tests pass one.
        """
        self.points = np.asarray(points)
        self.n = len(self.points)

        self._xyz = None
        self._order = None

        if self.n == 0:
            self.grid = None
            return

        if cell_size is not None:
            # The ceiling is enforced inside build, before anything is
            # allocated: a cell_size small against the site can ask for more
            # cells than a cell number can even hold.
            self.grid = SpatialGrid.build(self.points, cell_size=cell_size,
                                          sort=True, backend=backend,
                                          max_cells=int(max_cells))
        else:
            per_cell = max(int(points_per_cell), 1)
            target = min(max(self.n // per_cell, 1), int(max_cells))
            self.grid = SpatialGrid.build(self.points, target_cells=target,
                                          sort=True, backend=backend)

        if self.grid is not None:
            self._order = self.grid.order
            # xyz only, and float32 as the cloud is: this is the copy every
            # query reads, so it stays as small as the data allows.
            self._xyz = np.ascontiguousarray(
                self.points[self._order, :3], dtype=np.float32)

    def index_nbytes(self) -> int:
        """Bytes the index occupies, including the cell-ordered coordinates."""
        if self.grid is None:
            return 0
        return self.grid.index_nbytes() + self._xyz.nbytes

    # ------------------------------------------------------------------
    # Radius query
    # ------------------------------------------------------------------

    def query_ball_point(self, x, r) -> np.ndarray:
        """Indices of every point within *r* of *x*.

        One centre only — that is what the growers ask, and a list-of-lists
        return for a batch would defeat the point of not allocating per query.

        Returns an ``np.intp`` array rather than ``cKDTree``'s Python list, so
        callers must test it with ``.size`` and not with ``if not ...``. The
        order is by grid cell, not by distance; ``cKDTree`` makes no ordering
        promise either.
        """
        if self.grid is None:
            return np.empty(0, dtype=np.intp)

        centre = np.asarray(x, dtype=np.float64).reshape(3)
        radius = float(r)
        rows, near = self._candidates(*self.grid.runs_near(centre, radius))
        if rows.size == 0:
            return np.empty(0, dtype=np.intp)

        # The grid hands back whole cells, so this is the measuring step that
        # turns a box into a ball. float64, matching cKDTree, so that a point
        # sitting on the radius falls the same side of it for both.
        near = near.astype(np.float64)
        near -= centre
        sq = np.einsum('ij,ij->i', near, near)
        return np.asarray(rows[sq <= radius * radius], dtype=np.intp)

    def _candidates(self, begins, ends):
        """``(rows, xyz)`` for every point in the given slices of the index.

        Both come out of contiguous reads: the row numbers from ``order`` and
        the coordinates from the cell-ordered copy, at the very same offsets.
        That pairing is the reason the copy exists.
        """
        row_parts, xyz_parts = [], []
        for begin, end in zip(begins, ends):
            if end > begin:
                row_parts.append(self._order[begin:end])
                xyz_parts.append(self._xyz[begin:end])

        if not row_parts:
            return np.empty(0, dtype=np.intp), np.empty((0, 3), dtype=np.float32)
        if len(row_parts) == 1:
            return row_parts[0], xyz_parts[0]
        return np.concatenate(row_parts), np.concatenate(xyz_parts)

    # ------------------------------------------------------------------
    # k-nearest query
    # ------------------------------------------------------------------

    def query(self, x, k=1, distance_upper_bound=np.inf):
        """The *k* nearest points to *x*, nearest first.

        Follows ``cKDTree.query``: one point in gives scalars for ``k=1`` and a
        ``(k,)`` pair of arrays otherwise; a ``(m, 3)`` batch gives ``(m,)`` or
        ``(m, k)``. A missing neighbour comes back as distance ``inf`` and index
        ``self.n``, which is what lets the growers reject it with ``j >= n``.

        Args:
            x: (3,) or (m, 3) query coordinates.
            k: how many neighbours per query.
            distance_upper_bound: ignore anything further away than this.
        """
        query_points = np.asarray(x, dtype=np.float64)
        single = query_points.ndim == 1
        query_points = np.atleast_2d(query_points)[:, :3]
        m = len(query_points)
        k = int(k)
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

        dist = np.full((m, k), np.inf, dtype=np.float64)
        idx = np.full((m, k), self.n, dtype=np.intp)
        if self.grid is not None and m:
            self._nearest_k(query_points, k, float(distance_upper_bound),
                            dist, idx)

        if k == 1:
            dist, idx = dist[:, 0], idx[:, 0]
        if single:
            return (float(dist[0]), int(idx[0])) if k == 1 else (dist[0], idx[0])
        return dist, idx

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _nearest_k(self, query_points, k, upper, dist, idx):
        """Fill *dist* / *idx* by widening a cell box around each query.

        Queries sharing a home cell share a search box, so they are grouped and
        answered together against one gathered candidate set — which is what
        makes a batch worth batching. A query is finished only once its k-th
        distance is no further than the *guaranteed* reach of the box searched,
        so nothing outside can beat it. The rest go round again with the box
        twice as wide.
        """
        grid = self.grid
        shape = np.asarray(grid.shape, dtype=np.int64)
        home = np.asarray(grid.cell_of(query_points),
                          dtype=np.int64).reshape(-1, 3)

        pending = np.arange(len(query_points), dtype=np.intp)
        radius_cells = 1
        for _ in range(_MAX_RINGS):
            if pending.size == 0:
                return

            cells = home[pending]
            flat = ((cells[:, 0] * shape[1] + cells[:, 1]) * shape[2]
                    + cells[:, 2])
            order = np.argsort(flat, kind="stable")
            splits = np.flatnonzero(np.diff(flat[order])) + 1

            unfinished = []
            for group in np.split(order, splits):
                members = pending[group]
                cell = home[members[0]]
                rows, candidates = self._candidates(
                    *grid.runs_in_cell_box(cell - radius_cells,
                                           cell + radius_cells))
                if rows.size:
                    self._measure(members, rows, candidates, query_points, k,
                                  upper, dist, idx)

                reach = self._guaranteed_reach(query_points[members], cell,
                                               radius_cells, shape)
                done = (dist[members, k - 1] <= reach) | (reach >= upper)
                if not np.all(done):
                    unfinished.append(members[~done])

            pending = (np.concatenate(unfinished) if unfinished
                       else np.empty(0, dtype=np.intp))
            radius_cells *= 2

    def _measure(self, members, rows, candidates, query_points, k, upper,
                 dist, idx):
        """Distances from the queries in *members* to every one of *candidates*.

        *rows* and *candidates* are the two halves of one read of the index —
        the row numbers and the matching coordinates — so this never touches the
        original cloud at all.

        Chunked by query so the scratch is bounded by ``QUERY_PAIR_BLOCK``
        pairs and not by how many queries happened to land in one cell.

        Writes the k nearest, ascending. A later ring always searches a superset
        of this one, so overwriting an earlier answer is always an improvement
        and never a loss.
        """
        candidates = candidates.astype(np.float64)
        keep = min(k, len(rows))
        columns = np.arange(keep)
        per_chunk = max(1, QUERY_PAIR_BLOCK // len(rows))

        for start in range(0, len(members), per_chunk):
            chunk = members[start:start + per_chunk]
            offset = candidates[None, :, :] - query_points[chunk][:, None, :]
            sq = np.einsum('ijk,ijk->ij', offset, offset)
            # A NaN coordinate gives a NaN distance, and argpartition would rank
            # it as small rather than skipping it. +inf loses every comparison.
            np.nan_to_num(sq, copy=False, nan=np.inf, posinf=np.inf,
                          neginf=np.inf)

            picked = np.argpartition(sq, keep - 1, axis=1)[:, :keep]
            picked_sq = np.take_along_axis(sq, picked, axis=1)
            ranked = np.argsort(picked_sq, axis=1)
            picked = np.take_along_axis(picked, ranked, axis=1)
            picked_sq = np.take_along_axis(picked_sq, ranked, axis=1)

            found = np.sqrt(picked_sq)
            within = found <= upper
            dist[chunk[:, None], columns] = np.where(within, found, np.inf)
            idx[chunk[:, None], columns] = np.where(within, rows[picked], self.n)

    def _guaranteed_reach(self, query_points, cell, radius_cells, shape):
        """How far from each query the searched box is known to be complete.

        A neighbour nearer than this cannot be hiding outside the box, so a k-th
        distance within it is final. An axis whose box already runs off the edge
        of the grid reaches infinitely far that way — there is nothing beyond it
        to find.
        """
        low_cell = cell - radius_cells
        high_cell = cell + radius_cells
        low = self.grid.lo.astype(np.float64) + low_cell * self.grid.step
        high = self.grid.lo.astype(np.float64) + (high_cell + 1) * self.grid.step
        low = np.where(low_cell <= 0, -np.inf, low)
        high = np.where(high_cell >= shape - 1, np.inf, high)

        return np.minimum(query_points - low, high - query_points).min(axis=1)
