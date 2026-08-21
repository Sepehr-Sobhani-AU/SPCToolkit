"""
Spatial Grid

A uniform grid over a set of points, so that "what is near here" does not have
to measure the distance to every point in the cloud. A shared service: the
viewer uses it to make clicking fast, and any algorithm that needs a cheap
spatial index uses the same class.

**How it works.** Each point gets one number saying which cell of the grid it
falls in. A query works out which cells it can reach, collects the points in
them, and measures only those.

**Two shapes of use, one class.**

*Coarse and unsorted* — the viewer's pick grid. 11 x 11 x 2 = 242 cells so the
cell number fits in one byte (0.17 GB at 170M points rather than 0.68 GB), no
sorting, and ``nearest()`` answers one click. Finding the rows in a cell means
reading the whole index array, which is ~2 ms at 20M — invisible on a mouse
click, and it buys back the 8 s a full scan used to cost.

*Fine and sorted* — what an algorithm wants. ``target_cells=`` for the
resolution and ``sort=True`` so the rows of a cell are a slice instead of a
scan. That matters when there are thousands of queries rather than one: the
whole-array read the viewer shrugs off would dominate.

Sorting is opt-in for exactly that reason. The viewer would pay for a sort it
cannot perceive; an algorithm cannot afford to skip it. Sorting also *replaces*
``cell_ids`` rather than adding to it — once the rows are bucketed nothing reads
the per-point cell number again, so it is dropped and a sorted index costs
4 bytes a point.

**Sizing the cells.** The divisions apply to the bounding box, so cells stretch
with the data — on a 2 m thick facade the eleven Y divisions are 0.18 m each. A
fixed split is not a fixed cell size, which is why the viewer's 11 x 11 x 2 holds
up across very different clouds.

**An algorithm should pass ``cell_size`` equal to its query radius.** That is
the physically meaningful number and it is what the measurements land on. On the
real 168M cloud, querying at a 1.0 m radius:

    cells        cell size   points returned   rows_near
    58,144         4.84 m            349,462       124 us
    482,664        2.28 m            137,048        63 us
    3,974,880      1.11 m             70,113        58 us
    19,923,540     0.65 m             50,955        53 us

Cell size roughly equal to the radius is the knee. Beyond it the lookup barely
improves while ``starts`` — 8 bytes per cell — starts to show.

Note what those numbers do *not* say. An earlier round on synthetic uniform
points suggested 65,536 cells was both the cheapest and the fastest choice. Real
survey data is clustered, not uniform: only **12%** of cells hold any points at
all, so a query hits cells far denser than the average implies and comes back
with 350,000 points where the ball holds 12,000. The 5x difference lands on the
caller, which then measures every one of them.

The width of the cell number is **not** a reason to choose a cell count. Sorting
drops ``cell_ids``, so the finished index is about 4 bytes a point whether the
cells numbered in one byte or four; the width only changes the transient peak
while building (6 bytes a point against 8). ``target_cells`` remains for a
caller with no natural radius.
"""

import logging
import numpy as np

from core.services.compute_backend import DEFAULT_BLOCK, resolve_backend

logger = logging.getLogger(__name__)

# The viewer's pick-grid shape. Cells per axis; the product is <= 256, so a cell
# number fits in one byte. Two rows in Z because a survey bounding box is usually
# much flatter than it is wide.
PICK_GRID_SHAPE = (11, 11, 2)

# How many cells each integer width can number.
UINT8_CELL_LIMIT = 256
UINT16_CELL_LIMIT = 65_536

# A reasonable fallback for a caller with no natural query radius. NOT the
# fastest choice on real data — see the module docstring. Prefer ``cell_size``.
DEFAULT_TARGET_CELLS = 4_000_000


class SpatialGrid:
    """Cell number per point, plus what is needed to find a cell from a position.

    Immutable once built.

    Attributes:
        cell_ids: (N,) uint8/uint16/int32 — the cell each point falls in, or
            **None on a sorted grid**. Sorting turns it into ``order`` and
            ``starts``, after which nothing reads it again, so it is dropped
            rather than carried: it is 2-4 bytes per point, which is half the
            index on a large cloud.
        lo: (3,) float32 low corner of the bounding box.
        step: (3,) float32 cell size per axis.
        shape: (nx, ny, nz) cells per axis.
        order: (N,) int32 row indices sorted by cell, or None when unsorted.
        starts: (n_cells + 1,) int64 offsets into *order*, or None when unsorted.
    """

    __slots__ = ("cell_ids", "lo", "step", "inv_step", "shape", "n_points",
                 "n_cells", "order", "starts")

    def __init__(self, cell_ids, lo, step, shape, n_points,
                 order=None, starts=None):
        self.cell_ids = cell_ids
        self.lo = lo
        self.step = step
        self.inv_step = (np.float32(1.0) / step).astype(np.float32)
        self.shape = tuple(int(v) for v in shape)
        # Held explicitly rather than as len(cell_ids), because a sorted grid
        # does not keep cell_ids.
        self.n_points = int(n_points)
        self.n_cells = int(np.prod(self.shape))
        self.order = order
        self.starts = starts

    @property
    def sorted(self) -> bool:
        """Whether the rows were bucketed, making ``rows_in_cell`` a slice."""
        return self.order is not None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, points, shape=None, cell_size=None, target_cells=None,
              sort=False, block=DEFAULT_BLOCK, backend=None):
        """Build a grid over *points*.

        Args:
            points: (N, >=3) float32 array. Only xyz is read, so the viewer's
                interleaved xyz+rgb render buffer can be passed unchanged.
            shape: cells per axis. Give one of *shape*, *cell_size* or
                *target_cells*; the default is the viewer's ``PICK_GRID_SHAPE``.
            cell_size: target cell edge length in world units, as a scalar or
                per axis. The shape is derived from the bounding box, so the
                cells come out at most this big.
            target_cells: an upper bound on how many cells to divide the
                bounding box into, letting the cell *size* follow the site. For
                a caller with no natural query radius; one that has a radius
                should pass *cell_size* instead, since that is what the timings
                key off. Never exceeded, so it also pins the width of the cell
                number.
            sort: bucket the rows by cell, so ``rows_in_cell`` is a slice rather
                than a scan over the whole index. Costs one pass at build time
                and an extra 4 bytes per point; worth it above a handful of
                queries, not worth it for a mouse click.
            block: points numbered at once. Bounds the transfer to the graphics
                card and the CPU scratch; does not affect the result.
            backend: override the registry's choice (used by the tests to
                compare the CPU and GPU paths against each other).

        Returns:
            SpatialGrid, or None when there is nothing to index.
        """
        points = np.asarray(points)
        n = len(points)
        if n == 0:
            return None
        given = [name for name, value in (("shape", shape),
                                          ("cell_size", cell_size),
                                          ("target_cells", target_cells))
                 if value is not None]
        if len(given) > 1:
            raise ValueError(f"give only one of shape, cell_size or "
                             f"target_cells; got {', '.join(given)}")

        impl = resolve_backend("grid", backend)
        lo, hi = _bounds(points, block, impl)
        span = hi - lo
        # A flat or single-valued axis would divide by zero and, worse, put every
        # point in cell 0 of that axis. Give it a nominal span so the arithmetic
        # stays finite; all its points then land in the first cell, which is
        # correct — there is nothing to separate along it.
        span[span <= 0] = np.float32(1.0)

        if cell_size is not None:
            shape = _shape_for_cell_size(span, cell_size)
        elif target_cells is not None:
            shape = _shape_for_target_cells(span, target_cells)
        elif shape is None:
            shape = PICK_GRID_SHAPE
        shape = tuple(int(v) for v in shape)
        if any(v < 1 for v in shape):
            raise ValueError(f"grid {shape} has an axis with no cells")

        step = (span / np.asarray(shape, dtype=np.float32)).astype(np.float32)
        inv_step = (np.float32(1.0) / step).astype(np.float32)

        n_cells = int(np.prod(shape))
        # The narrowest width that can number this many cells. At 168M points
        # each byte saved is 168 MB, and the jump straight from uint8 to int32
        # used to cost 2 bytes a point on every grid between 257 and 65,536
        # cells — which is exactly the range a useful algorithm grid lands in.
        dtype = _cell_dtype(n_cells)

        cell_ids = np.empty(n, dtype=dtype)
        for start in range(0, n, block):
            stop = min(start + block, n)
            impl.cell_ids(points[start:stop], lo, inv_step, shape,
                          cell_ids[start:stop])

        order = starts = None
        if sort:
            order, starts = _bucket(cell_ids, n_cells, impl)
            # Dropped, not kept: every lookup on a sorted grid goes through
            # order/starts, so this would be 2-4 bytes a point of pure weight.
            cell_ids = None

        return cls(cell_ids, lo, step, shape, n, order, starts)

    # ------------------------------------------------------------------
    # Cell arithmetic
    # ------------------------------------------------------------------

    def cell_of(self, position) -> np.ndarray:
        """(3,) int32 cell coordinates holding *position*, clamped to the grid.

        Clamped because a query can land just outside the bounding box, and the
        cells that can answer it are then the edge ones.
        """
        position = np.asarray(position, dtype=np.float32)
        return np.clip(
            ((position - self.lo) * self.inv_step).astype(np.int32),
            0, np.asarray(self.shape, dtype=np.int32) - 1,
        )

    def cell_number(self, cell) -> int:
        """Flatten (ix, iy, iz) into the number stored in ``cell_ids``."""
        nx, ny, nz = self.shape
        ix, iy, iz = (int(v) for v in cell)
        return (ix * ny + iy) * nz + iz

    def rows_in_cell(self, cell) -> np.ndarray:
        """Rows of the indexed points that fall in one cell.

        A slice when the grid was sorted, a scan over the whole index when it
        was not. The scan is the reason ``sort=True`` exists: it is ~2 ms per
        call at 20M points regardless of how few points the cell holds.
        """
        number = cell if np.isscalar(cell) else self.cell_number(cell)
        number = int(number)
        if self.order is not None:
            return self.order[self.starts[number]:self.starts[number + 1]]
        return np.flatnonzero(self.cell_ids == number)

    def index_nbytes(self) -> int:
        """Bytes the index occupies, for reporting and budgeting."""
        total = 0 if self.cell_ids is None else self.cell_ids.nbytes
        if self.order is not None:
            total += self.order.nbytes + self.starts.nbytes
        return total

    def rows_in_box(self, low, high) -> np.ndarray:
        """Rows in every cell touched by the world-space box *low* to *high*.

        The answer is a superset of the points actually inside the box — whole
        cells come back — so callers still test what they collected. That is the
        point: the grid narrows, the caller decides.
        """
        a = self.cell_of(low)
        b = self.cell_of(high)
        if self.order is not None:
            return self._rows_in_runs(*self._z_runs(a, b))
        # Unsorted: one pass over the index answers any set of cells at once,
        # which beats one pass per cell.
        return np.flatnonzero(self._cell_table(self._cells_in(a, b))[self.cell_ids])

    def rows_near(self, centre, radius) -> np.ndarray:
        """Rows in every cell a ball of *radius* around *centre* can reach.

        A superset, as ``rows_in_box`` — the caller measures.
        """
        centre = np.asarray(centre, dtype=np.float64)
        radius = float(radius)
        return self.rows_in_box(centre - radius, centre + radius)

    # ------------------------------------------------------------------
    # Nearest point (the viewer's click path)
    # ------------------------------------------------------------------

    def nearest(self, points, target, max_dist=None):
        """Row of *points* closest to *target*, searching outwards by cell ring.

        Args:
            points: the same array the grid was built over.
            target: (3,) world coordinates, normally from the depth buffer.
            max_dist: give up beyond this distance, matching the caller's snap
                radius. ``None`` means no limit.

        Returns:
            (row, squared_distance), or (None, inf) when nothing qualifies.
        """
        target = np.asarray(target, dtype=np.float64)
        tx, ty, tz = (float(target[0]), float(target[1]), float(target[2]))
        limit_sq = np.inf if max_dist is None else float(max_dist) ** 2

        home = self.cell_of(target)

        # Nothing in the cloud can be within reach: answer without touching it.
        box_sq = self._box_distance_sq(target)
        if box_sq > limit_sq:
            return None, np.inf

        best_sq, best_row = limit_sq, None

        # Two widening steps, then give up on cells. On an unsorted grid each
        # extra ring costs a pass over the whole index, and past the first ring
        # the box holds so much of the cloud that a plain scan is cheaper than
        # narrowing it — bounding the worst case at one scan instead of eleven
        # ring passes.
        for ring in (0, 1):
            rows, covers_all = self._ring_rows(home, ring)
            if rows.size:
                sq = squared_distances(points, rows, tx, ty, tz)
                j = int(np.argmin(sq))
                if sq[j] < best_sq:
                    best_sq, best_row = float(sq[j]), int(rows[j])

            if covers_all:
                return best_row, best_sq
            # Nothing outside the searched box can beat what we have.
            reach = self._distance_to_ring_edge(home, ring, target)
            if best_row is not None and best_sq <= reach * reach:
                return best_row, best_sq
            if reach * reach >= limit_sq:
                return best_row, best_sq

        row, sq = scan_nearest(points, tx, ty, tz, limit_sq)
        return (row, sq) if row is not None else (best_row, best_sq)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _z_runs(self, low, high):
        """(first, last) cell numbers of each Z column in the cell box.

        The cell number is ``(ix * ny + iy) * nz + iz``, so **iz is the fastest
        axis** and a column of cells in Z is a contiguous range of cell numbers.
        One run per (ix, iy) pair therefore covers the whole box, and a sorted
        grid can answer each run with a single slice of ``order``.

        This is what stops cell size being a trap. Fetching cell by cell, a
        query cost 22 us at 6,000 cells but 317 us at 8M — so a grid fine enough
        to keep the candidate list short was slower than the coarse one it
        replaced. By run it is 28 us to 48 us over the same range, which is what
        makes ``target_cells`` a free choice.
        """
        nx, ny, nz = self.shape
        ix = np.arange(low[0], high[0] + 1, dtype=np.int64)[:, None]
        iy = np.arange(low[1], high[1] + 1, dtype=np.int64)[None, :]
        base = ((ix * ny + iy) * nz).ravel()
        return base + int(low[2]), base + int(high[2])

    def _rows_in_runs(self, first, last):
        """Rows of every cell in the given runs, sorted grids only.

        Each run is one slice of ``order``, so the Python loop is over columns
        rather than over cells — smaller than the cell count by a factor of the
        Z span, and independent of how fine the grid is in Z.
        """
        begins = self.starts[first]
        ends = self.starts[last + 1]
        parts = [self.order[b:e] for b, e in zip(begins, ends) if e > b]
        if not parts:
            return np.empty(0, dtype=self.order.dtype)
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)

    def _cells_in(self, low, high):
        """Every cell number in the cell box, for the unsorted table lookup."""
        nx, ny, nz = self.shape
        ix = np.arange(low[0], high[0] + 1, dtype=np.int64)[:, None, None]
        iy = np.arange(low[1], high[1] + 1, dtype=np.int64)[None, :, None]
        iz = np.arange(low[2], high[2] + 1, dtype=np.int64)[None, None, :]
        return ((ix * ny + iy) * nz + iz).ravel()

    def _box_distance_sq(self, target):
        """Squared distance from *target* to the cloud's bounding box, 0 inside.

        An O(1) reject. A click that lands on line geometry far from any points
        — the depth buffer resolves whatever was drawn there, not only clouds —
        would otherwise widen ring by ring across the whole index before finding
        nothing within the snap radius.
        """
        hi = self.lo + self.step * np.asarray(self.shape, dtype=np.float32)
        outside = np.maximum(np.maximum(self.lo - target, target - hi), 0.0)
        return float(np.dot(outside, outside))

    def _cell_table(self, cells):
        """Boolean lookup over cell numbers, for resolving many cells in one pass.

        A lookup table rather than a comparison per cell: one pass over the index
        then answers any set of cells, whatever shape that set is.
        """
        table = np.zeros(self.n_cells, dtype=bool)
        table[np.asarray(cells, dtype=np.int64)] = True
        return table

    def _ring_rows(self, home, ring):
        """(rows, covers_all) for every cell within *ring* cells of *home*."""
        nx, ny, nz = self.shape
        x0, x1 = max(0, home[0] - ring), min(nx - 1, home[0] + ring)
        y0, y1 = max(0, home[1] - ring), min(ny - 1, home[1] + ring)
        z0, z1 = max(0, home[2] - ring), min(nz - 1, home[2] + ring)

        covers_all = (x0 == 0 and y0 == 0 and z0 == 0
                      and x1 == nx - 1 and y1 == ny - 1 and z1 == nz - 1)

        if ring == 0 and self.order is None:
            # The viewer's common path. A single cell answers virtually every
            # click, because the cursor position comes from the depth buffer and
            # so already sits on a point. On an unsorted grid a plain
            # ``cell_ids == c`` comparison is the fastest way to resolve one
            # cell: measured 2.3 ms against 31.6 ms for a table gather at 20M,
            # because a comparison vectorises and a gather does not.
            return self.rows_in_cell(self.cell_number(home)), covers_all

        low = np.array([x0, y0, z0], dtype=np.int64)
        high = np.array([x1, y1, z1], dtype=np.int64)
        if self.order is not None:
            rows = self._rows_in_runs(*self._z_runs(low, high))
        else:
            rows = np.flatnonzero(
                self._cell_table(self._cells_in(low, high))[self.cell_ids])
        return rows, covers_all

    def _distance_to_ring_edge(self, home, ring, target):
        """How far *target* is from the nearest unsearched cell.

        Any point not yet measured lies outside the box of cells within *ring*
        of home, so it is at least this far away. Once the best point found is
        closer than this, widening cannot improve on it. Faces that sit on the
        edge of the grid are ignored — there is nothing beyond them.
        """
        reach = np.inf
        for axis in range(3):
            low_cell = home[axis] - ring
            high_cell = home[axis] + ring
            if low_cell > 0:
                face = self.lo[axis] + low_cell * self.step[axis]
                reach = min(reach, float(target[axis]) - float(face))
            if high_cell < self.shape[axis] - 1:
                face = self.lo[axis] + (high_cell + 1) * self.step[axis]
                reach = min(reach, float(face) - float(target[axis]))
        # A target outside the box (it can sit outside the cloud entirely) would
        # give a negative reach, which guarantees nothing. Treat it as zero.
        return max(reach, 0.0)


def _cell_dtype(n_cells):
    """The narrowest integer that can number *n_cells* cells."""
    if n_cells <= UINT8_CELL_LIMIT:
        return np.uint8
    if n_cells <= UINT16_CELL_LIMIT:
        return np.uint16
    return np.int32


def _shape_for_target_cells(span, target_cells):
    """Cells per axis dividing the box into roughly *target_cells* cells.

    Cells come out near-cubic — the axis divisions are proportional to the axis
    lengths — so the *size* follows the site rather than being fixed.

    *target_cells* is a **ceiling, never exceeded**, because the caller passes it
    to control the width of the cell number: one cell over 65,536 and the index
    doubles from two bytes a point to four. Rounding each axis to nearest would
    overshoot (measured: a 200 x 30 x 8 m box asking for 65,536 came out at
    65,934, and silently cost an extra byte a point), so each axis rounds down
    and a flat axis clamped up to one cell is paid for by trimming elsewhere.
    """
    target = int(target_cells)
    if target < 1:
        raise ValueError(f"target_cells must be at least 1, got {target_cells!r}")
    span = np.asarray(span, dtype=np.float64)
    # Cube root of the volume per cell gives the edge length that would divide
    # the box into `target` near-cubic cells.
    edge = float((np.prod(span) / target) ** (1.0 / 3.0))
    if not np.isfinite(edge) or edge <= 0:
        return np.ones(3, dtype=np.int64)

    counts = np.maximum(np.floor(span / edge), 1).astype(np.int64)
    # Flooring alone keeps the product under target, but an axis too thin for a
    # whole cell gets clamped up to 1, which can push it back over. Give the
    # cells back from whichever axis has the most.
    while int(np.prod(counts)) > target and counts.max() > 1:
        counts[int(np.argmax(counts))] -= 1
    return counts


def _shape_for_cell_size(span, cell_size):
    """Cells per axis giving cells no larger than *cell_size*."""
    size = np.broadcast_to(np.asarray(cell_size, dtype=np.float32), (3,))
    if np.any(size <= 0):
        raise ValueError(f"cell_size must be positive, got {cell_size!r}")
    return np.maximum(np.ceil(span / size), 1).astype(np.int64)


def _bucket(cell_ids, n_cells, backend=None):
    """(order, starts) — row indices sorted by cell, and where each cell begins.

    A counting sort in all but name: ``argsort`` on a small integer type is a
    radix pass, and ``bincount`` gives the offsets. This is what turns fetching
    a cell from a scan over the whole index into a slice.

    The sort goes through the backend — it was the largest phase of a sorted
    build by a wide margin (5.97 s of an 11 s build at 50M) and is 27x faster on
    the graphics card. ``bincount`` stays on the CPU: it is a fraction of the
    cost and the result is per cell, not per point.

    ``order`` is int32, not the default int64, which halves it: 0.63 GB at 168M
    points rather than 1.26 GB.
    """
    impl = backend if backend is not None else resolve_backend("grid")
    order = impl.argsort(cell_ids)
    counts = np.bincount(cell_ids.astype(np.int64, copy=False), minlength=n_cells)
    starts = np.empty(n_cells + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])
    return order, starts


def _bounds(points, block, backend=None):
    """(low, high) corners of the xyz bounding box, in float32, read in blocks.

    Non-finite coordinates are ignored rather than propagated. ``np.minimum``
    spreads a NaN to the whole result, so one bad point — which a scanner or a
    lossy export can produce — would make the box NaN on that axis, every cell
    number on it garbage, and the grid collapse to a handful of cells. The plain
    scan this replaced shrugged NaN off, because ``abs(x - tx) < radius`` is
    False for it, so ignoring them here keeps the old behaviour: they clamp into
    an edge cell and are never the nearest point to anything.

    Goes through the backend, so it runs on the graphics card when there is one.
    This is not a detail: measured over 50M points it was the single largest
    phase of a build at 3.3 s, larger than numbering every cell, and it drops to
    0.31 s on the card.
    """
    impl = backend if backend is not None else resolve_backend("grid")
    lo = np.full(3, np.inf, dtype=np.float32)
    hi = np.full(3, -np.inf, dtype=np.float32)
    for start in range(0, len(points), block):
        block_lo, block_hi = impl.block_bounds(points[start:start + block])
        np.minimum(lo, block_lo, out=lo)
        np.maximum(hi, block_hi, out=hi)

    # Every point on an axis was non-finite, leaving +/-inf, which would give an
    # infinite span. Fall back to a unit box so the arithmetic stays finite and
    # everything lands in cell 0, as it does for a flat axis.
    degenerate = ~(np.isfinite(lo) & np.isfinite(hi))
    lo[degenerate] = np.float32(0.0)
    hi[degenerate] = np.float32(0.0)
    return lo, hi


def scan_nearest(points, tx, ty, tz, limit_sq, block=DEFAULT_BLOCK):
    """Nearest row over every point, in blocks, float32.

    The fallback for the rare query the grid cannot narrow down — a target
    outside the cloud, or right on a cell wall with the nearest point beyond it.
    Blocked so its scratch stays bounded, and float32 so it does not build the
    4 GB float64 copy the old picking code made at 170M points.
    """
    best_sq, best_row = float(limit_sq), None
    for start in range(0, len(points), block):
        chunk = np.asarray(points[start:start + block, :3], dtype=np.float32)
        offset = chunk - np.array([tx, ty, tz], dtype=np.float32)
        sq = np.einsum('ij,ij->i', offset, offset)
        reject_non_finite(sq)
        j = int(np.argmin(sq))
        if sq[j] < best_sq:
            best_sq, best_row = float(sq[j]), start + j
    return best_row, best_sq


def reject_non_finite(sq):
    """Turn non-finite squared distances into +inf, in place.

    A NaN coordinate gives a NaN distance, and ``np.argmin`` returns the index
    of a NaN rather than skipping it — so a single bad point in a cell would
    hide the real nearest one and the query would find nothing. +inf loses every
    comparison instead, which is how the plain scan behaved: ``abs(x - tx) <
    radius`` is False for NaN, so those points were simply never candidates.
    """
    np.nan_to_num(sq, copy=False, nan=np.inf, posinf=np.inf, neginf=np.inf)


def squared_distances(points, rows, tx, ty, tz):
    """Squared distances from the given rows of *points* to (tx, ty, tz).

    float32 throughout, with the offset applied before squaring, so what gets
    squared is a distance within the cloud rather than a coordinate.
    """
    near = np.asarray(points[rows, :3], dtype=np.float32)
    near[:, 0] -= np.float32(tx)
    near[:, 1] -= np.float32(ty)
    near[:, 2] -= np.float32(tz)
    sq = np.einsum('ij,ij->i', near, near)
    reject_non_finite(sq)
    return sq
