"""
Pick Grid

A coarse cell grid over a branch's drawn points, so that clicking on a point
does not have to measure the distance to every point in the cloud.

**What it replaces.** Clicking reads the depth buffer to get the world position
under the cursor, then finds the nearest drawn point to it. That search walked
the whole cloud. It looks as though a snap radius bounds the work, but the
radius is ``max_extent * picking_point_threshold_factor`` with the factor
defaulting to 1.0 — the whole size of the cloud — so it rejects nothing.
Measured at 20M points and scaled: **8.0 s per click at 170M**, and that is paid
by Shift+Left select, Ctrl+Shift+Right cluster deselect, *and* double-click to
re-centre the view, since all three go through the same function.

**How it works.** Each point gets one number saying which cell of an 11 x 11 x 2
grid it falls in. A click works out the cursor's cell, collects the points in it,
and measures only those. If the nearest point found is further away than the
edge of the searched region, the search widens by a ring of cells and repeats —
which almost never happens, because the cursor position comes from the depth
buffer and therefore sits on a real point.

**Why 242 cells.** They fit in one byte, so the index is 0.17 GB at 170M points
rather than 0.68 GB for a 32-bit one. The cell count barely affects the time,
which is not obvious: most of a click goes on reading the whole index array to
find the matching entries, and that read is the same size whatever the cell
count. Cell count only changes how many points are left to measure at the end.
Measured at 170M: 90-200 ms per click depending on the cloud, against 8,000 ms
without the grid.

**Why the shape holds up.** The divisions are applied to the bounding box, so
cells stretch with the data — on a 2 m thick facade the eleven Y divisions are
0.18 m each and all of them are used. A fixed split is not a fixed cell size.
Measured against splits derived from each cloud's own proportions, 11 x 11 x 2
was within 15% on a 10-storey building and ahead on a facade.

**No sorting.** Sorting the points by cell would remove the read and get a click
to 0.19 ms, but nobody can tell that from 150 ms on a mouse click, and the sort
costs 31 s on the CPU, forces a graphics card, and doubles the memory.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Cells per axis. The product must stay <= 256 so a cell number fits in a byte.
# Two rows in Z because a survey bounding box is usually much flatter than wide.
GRID_SHAPE = (11, 11, 2)

# Points numbered at once when building. Bounds the transfer to the graphics
# card and the CPU scratch arrays; does not affect the result.
BUILD_BLOCK = 8_000_000


class PointGrid:
    """Cell number per point, plus what is needed to find a cell from a position.

    Immutable once built. Held per branch by the viewer and thrown away when
    that branch's drawn points are replaced.
    """

    __slots__ = ("cell_ids", "lo", "step", "inv_step", "shape", "n_points")

    def __init__(self, cell_ids, lo, step, shape):
        self.cell_ids = cell_ids          # (N,) uint8
        self.lo = lo                      # (3,) float32, bounding box low corner
        self.step = step                  # (3,) float32, cell size per axis
        self.inv_step = (np.float32(1.0) / step).astype(np.float32)
        self.shape = tuple(int(v) for v in shape)
        self.n_points = len(cell_ids)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, points, shape=GRID_SHAPE, block=BUILD_BLOCK, backend=None):
        """Build a grid over *points*.

        Args:
            points: (N, >=3) float32 array. Only xyz is read, so the viewer's
                interleaved xyz+rgb render buffer can be passed unchanged.
            shape: cells per axis; the product must be <= 256.
            block: points numbered at once.
            backend: override the registry's choice (used by the tests to
                compare the CPU and GPU paths against each other).

        Returns:
            PointGrid, or None when there is nothing to index.
        """
        points = np.asarray(points)
        n = len(points)
        if n == 0:
            return None
        if int(np.prod(shape)) > 256:
            raise ValueError(f"grid {shape} has more than 256 cells, "
                             "which does not fit in one byte per point")

        lo, hi = _bounds(points, block)
        span = hi - lo
        # A flat or single-valued axis would divide by zero and, worse, put every
        # point in cell 0 of that axis. Give it a nominal span so the arithmetic
        # stays finite; all its points then land in the first cell, which is
        # correct — there is nothing to separate along it.
        span[span <= 0] = np.float32(1.0)
        step = (span / np.asarray(shape, dtype=np.float32)).astype(np.float32)
        inv_step = (np.float32(1.0) / step).astype(np.float32)

        impl = _resolve_backend(backend)
        cell_ids = np.empty(n, dtype=np.uint8)
        for start in range(0, n, block):
            stop = min(start + block, n)
            impl.cell_ids(points[start:stop], lo, inv_step, shape,
                          cell_ids[start:stop])

        return cls(cell_ids, lo, step, shape)

    # ------------------------------------------------------------------
    # Querying
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

        # The cursor's cell, clamped: a click can land just outside the cloud's
        # bounding box, and the nearest point is then in the edge cell.
        home = np.clip(
            ((target.astype(np.float32) - self.lo) * self.inv_step).astype(np.int32),
            0, np.asarray(self.shape, dtype=np.int32) - 1,
        )

        # Nothing in the cloud can be within reach: answer without touching it.
        box_sq = self._box_distance_sq(target)
        if box_sq > limit_sq:
            return None, np.inf

        best_sq, best_row = limit_sq, None

        # Two widening steps, then give up on cells. Each extra ring costs a
        # pass over the whole index, and past the first ring the box holds so
        # much of the cloud that a plain scan is cheaper than narrowing it —
        # bounding the worst case at one scan instead of eleven ring passes.
        for ring in (0, 1):
            table, covers_all = self._ring_table(home, ring)
            rows = self._rows_in_cells(table)
            if rows.size:
                sq = _squared_distances(points, rows, tx, ty, tz)
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

        row, sq = _scan_all(points, tx, ty, tz, limit_sq)
        return (row, sq) if row is not None else (best_row, best_sq)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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

    def _rows_in_cells(self, table):
        """Rows whose cell is marked in *table*, a (256,) boolean lookup.

        Split on purpose. Indexing the lookup with the whole index array —
        ``table[cell_ids]`` — is the general answer but measured 31.6 ms against
        20M points, while a plain ``cell_ids == c`` is 2.3 ms: a comparison
        vectorises, a gather does not. The first ring is a single cell and
        answers virtually every click, because the cursor position comes from
        the depth buffer and so already sits on a point. So the common path
        takes the comparison and only a widened search pays for the gather.
        """
        cells = np.flatnonzero(table)
        if cells.size == 0:
            return np.empty(0, dtype=np.intp)
        if cells.size == 1:
            return np.flatnonzero(self.cell_ids == np.uint8(cells[0]))
        return np.flatnonzero(table[self.cell_ids])

    def _ring_table(self, home, ring):
        """(256,) bool lookup marking every cell within *ring* cells of *home*.

        A lookup table rather than a comparison per cell: the cell number is a
        byte, so ``table[cell_ids]`` resolves any set of cells in one pass over
        the index, whatever shape that set is.
        """
        nx, ny, nz = self.shape
        table = np.zeros(256, dtype=bool)

        x0, x1 = max(0, home[0] - ring), min(nx - 1, home[0] + ring)
        y0, y1 = max(0, home[1] - ring), min(ny - 1, home[1] + ring)
        z0, z1 = max(0, home[2] - ring), min(nz - 1, home[2] + ring)

        ix = np.arange(x0, x1 + 1, dtype=np.int32)[:, None, None]
        iy = np.arange(y0, y1 + 1, dtype=np.int32)[None, :, None]
        iz = np.arange(z0, z1 + 1, dtype=np.int32)[None, None, :]
        table[((ix * ny + iy) * nz + iz).ravel()] = True

        covers_all = (x0 == 0 and y0 == 0 and z0 == 0
                      and x1 == nx - 1 and y1 == ny - 1 and z1 == nz - 1)
        return table, covers_all

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


def _bounds(points, block):
    """(low, high) corners of the xyz bounding box, in float32, read in blocks."""
    lo = np.full(3, np.inf, dtype=np.float32)
    hi = np.full(3, -np.inf, dtype=np.float32)
    for start in range(0, len(points), block):
        chunk = points[start:start + block, :3]
        np.minimum(lo, chunk.min(axis=0), out=lo)
        np.maximum(hi, chunk.max(axis=0), out=hi)
    return lo, hi


def _scan_all(points, tx, ty, tz, limit_sq, block=BUILD_BLOCK):
    """Nearest row over every point, in blocks, float32.

    The fallback for the rare click the grid cannot narrow down — a cursor
    outside the cloud, or right on a cell wall with the nearest point beyond it.
    Blocked so its scratch stays bounded, and float32 so it does not build the
    4 GB float64 copy the old picking code made at 170M points.
    """
    best_sq, best_row = float(limit_sq), None
    for start in range(0, len(points), block):
        chunk = np.asarray(points[start:start + block, :3], dtype=np.float32)
        offset = chunk - np.array([tx, ty, tz], dtype=np.float32)
        sq = np.einsum('ij,ij->i', offset, offset)
        j = int(np.argmin(sq))
        if sq[j] < best_sq:
            best_sq, best_row = float(sq[j]), start + j
    return best_row, best_sq


def _squared_distances(points, rows, tx, ty, tz):
    """Squared distances from the given rows of *points* to (tx, ty, tz).

    float32 throughout, with the offset applied before squaring, so what gets
    squared is a distance within the cloud rather than a coordinate.
    """
    near = np.asarray(points[rows, :3], dtype=np.float32)
    near[:, 0] -= np.float32(tx)
    near[:, 1] -= np.float32(ty)
    near[:, 2] -= np.float32(tz)
    return np.einsum('ij,ij->i', near, near)


def _resolve_backend(backend):
    """The backend to number cells with: the caller's, else the registry's."""
    if backend is not None:
        return backend
    try:
        from config.config import global_variables
        registry = global_variables.global_backend_registry
        if registry is not None:
            return registry.get_selection()
    except Exception as exc:                        # registry not up yet (tests)
        logger.debug(f"Selection backend registry unavailable ({exc}); using NumPy")
    from plugins.backends.selection_backends import NumpySelection
    return NumpySelection()
