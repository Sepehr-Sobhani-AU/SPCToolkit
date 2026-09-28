"""
Shape Query Service

Answers one question for every plugin: *which points of this cloud are inside
this shape?* The shapes are in ``core.services.query_shapes``; the design and the
measurements that chose it are in DECISIONS.md § 2026-09-28.

    rows = global_variables.global_shape_query.points_in(cloud.points, shape)

That is the whole surface. There is no build call and no release call: the first
query on a cloud prepares it, and everything is freed when the plugin run ends
(``AnalysisExecutor`` calls ``release_all``) or as soon as nothing references
the cloud's array any more.

**How a query is answered.**

1. *Cells.* The full cloud is numbered into the coarse spatial index, 11 x 11 x 2
   cells. The shape says which cells it misses, fully covers or partly covers;
   empty cells are dropped. This is not the viewer's coarse index, which is built
   over the drawn (LOD) rows rather than the cloud.
2. *Per-cell index, built on first touch.* The first query to reach a cell sorts
   that cell's rows by a sub-grid of about 32 points a cell (Z fastest, so a
   column of sub-cells is one contiguous run). 4 bytes a point on the GPU and no
   second copy of the coordinates. Real clouds are so clustered that one coarse
   cell can hold 21M points; without this a 0.5 m sphere tested ~10M of them.
3. *One kernel.* The host works out which runs of the sub-grid the shape's box
   reaches, and one CUDA kernel tests every candidate and collects the matches.
   Covered cells are taken whole, untested.
4. *Big shapes skip the index.* When the candidate cells hold more than half the
   cloud, a single kernel tests every point instead — cheaper than gathering.
5. *Fixed order.* The kernel collects matches in whatever order threads finish,
   so the answer is sorted: the same query always returns the same array, which
   pipeline replay relies on.

**GPU only.** The coordinates are held on the GPU for the run: 12 bytes a point
plus 1 for the cell number (2.2 GB at 168M points), and at most 4 more once every
cell is indexed. There is deliberately no CPU path; a missing GPU, too little
memory or a failed allocation is raised as an error saying so.

**Uploads are blocked.** ``cp.asarray`` on a whole array leaves a same-size
staging copy in host RAM — 2 GB for the 168M cloud. Copying in blocks with
``.set()`` leaves none.

**Its own memory pool.** In the app CuPy allocates through RAPIDS' RMM, which
makes a real GPU allocation for every array; a small query makes about seven, so
that alone was over half its time. The service allocates from a CuPy pool of its
own instead, and empties it on release.

**Rounding at borders.** A point's cell is computed on the GPU, so right at a
cell border it can land a hair outside the box the host computes for that cell.
Cell boxes and query boxes are therefore padded by a small margin, and a shape
must hold a cell's corners with that margin to spare before the cell counts as
covered. Both only ever turn "skip" into "test", never the reverse.

Thread-safe: queries may come from any thread. A cell is built once even when
two queries reach it together.
"""

import logging
import threading
import time
import weakref

import numpy as np

from core.services.query_shapes import MISS, PARTIAL, COVER
from core.services.spatial_grid import SpatialGrid

logger = logging.getLogger(__name__)

# Points copied to or from the GPU at once, and points tested per every-point
# launch. Bounds the scratch memory; does not change any answer.
BLOCK = 8_000_000

# Sub-cell size of the per-cell index. Measured on the 168M cloud: the fixed
# cost of a query dominates well before this matters, so it is not a tuning knob.
POINTS_PER_SUBCELL = 32

# Candidate cells holding more than this share of the cloud -> test every point.
EVERY_POINT_SHARE = 0.5

# Cell and query boxes are padded by this fraction of the largest cell edge.
# Cell numbering rounds at about 1e-7 of a cell; this is a thousand times that.
PAD_FRACTION = 1e-4

# Unused GPU memory the service's own pool may keep cached between queries
# before handing it back. See ``ShapeQueryService._pool``.
POOL_SLACK_BYTES = 256 * 1024 * 1024

# GPU bytes a point needs while one cell's index is built (coordinates gathered,
# sub-cell numbers, sort order), used to check memory before building.
_CELL_BUILD_BYTES_PER_POINT = 40


class ShapeQueryError(RuntimeError):
    """A shape query could not run: no GPU, not enough GPU memory, and so on."""


class ShapeQueryService:
    """Which points of a cloud are inside a shape. One instance, in
    ``global_variables.global_shape_query``. See the module docstring."""

    def __init__(self):
        self._lock = threading.Lock()
        # (id of owning array, data pointer, shape, strides) -> (weakref, _CloudState)
        self._clouds = {}
        # The service's own CuPy memory pool, created on first use. In the app
        # CuPy allocates through RAPIDS' RMM, which makes a real cudaMalloc for
        # every array: ~35 us each, ~7 per query, over half of a small query's
        # time. A pool reuses the buffers. Emptied by release_all.
        self._pool = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def points_in(self, points, shape, as_mask=False) -> np.ndarray:
        """Rows of *points* inside *shape*.

        Args:
            points: (N, >=3) array, the cloud's own points (``PointCloud.points``).
                Pass the same array each time: its identity is how the service
                finds what it already built for it.
            shape: a ``core.services.query_shapes.QueryShape``.
            as_mask: return an (N,) bool mask instead of row numbers.

        Returns:
            int32 row numbers, sorted ascending; or the bool mask.

        Raises:
            ShapeQueryError: no usable GPU, or not enough GPU memory.
        """
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"points must be (N, >=3), got {points.shape}")
        n = len(points)
        if n == 0:
            return np.zeros(0, dtype=bool) if as_mask else np.empty(0, dtype=np.int32)

        state = self._state_for(points)
        try:
            with self._allocator():
                rows = state.query(shape)
        except Exception as exc:
            error = _as_query_error(exc, "running a shape query")
            if error is exc:
                raise
            raise error from exc
        finally:
            pool = self._pool
            if pool is not None and pool.total_bytes() - pool.used_bytes() > POOL_SLACK_BYTES:
                pool.free_all_blocks()             # a big query's scratch; do not hoard it

        if as_mask:
            mask = np.zeros(n, dtype=bool)
            mask[rows] = True
            return mask
        return rows

    def release_all(self) -> None:
        """Free everything built for every cloud. Called when a plugin run ends."""
        with self._lock:
            had = len(self._clouds)
            self._clouds.clear()
        if self._pool is not None:
            self._pool.free_all_blocks()
        _free_gpu_pools()
        if had:
            logger.debug(f"Shape query: released {had} cloud(s)")

    def cloud_count(self) -> int:
        """Clouds currently prepared. For tests and diagnostics."""
        with self._lock:
            return len(self._clouds)

    def index_nbytes(self) -> int:
        """GPU bytes held by per-cell indexes across all clouds."""
        with self._lock:
            states = [state for _, state in self._clouds.values()]
        return sum(state.index_nbytes() for state in states)

    # ------------------------------------------------------------------
    # Per-cloud state
    # ------------------------------------------------------------------

    def _state_for(self, points):
        owner = _owner(points)
        key = (id(owner), points.__array_interface__["data"][0],
               points.shape, points.strides)
        with self._lock:
            entry = self._clouds.get(key)
            if entry is not None and entry[0]() is owner:
                return entry[1]
            # Built under the lock: two queries on a new cloud must not both
            # upload it. Only one plugin runs at a time, so nobody waits long.
            try:
                with self._allocator():
                    state = _CloudState(points)
            except Exception as exc:
                if self._pool is not None:
                    self._pool.free_all_blocks()
                _free_gpu_pools()
                error = _as_query_error(exc, "preparing the cloud")
                if error is exc:
                    raise
                raise error from exc
            ref = weakref.ref(owner, self._forget_owner(id(owner)))
            self._clouds[key] = (ref, state)
            return state

    def _allocator(self):
        """Context in which CuPy allocates from the service's own pool."""
        cp = _cupy()
        if self._pool is None:
            # The memory checks inside the context run hardware detection the
            # first time, which imports cuML, which installs RMM as CuPy's global
            # allocator — refused inside ``using_allocator``, and cuML is then
            # broken for the rest of the process. Detect once, outside. (In the
            # app detection already ran at startup; this is for everything else.)
            from infrastructure.memory_manager import MemoryManager
            MemoryManager.get_available_gpu_mb()
            self._pool = cp.cuda.MemoryPool()
        return cp.cuda.using_allocator(self._pool.malloc)

    def _forget_owner(self, owner_id):
        """Weakref callback: the cloud's array is gone, so drop what was built for it."""
        service = weakref.ref(self)

        def forget(_ref):
            self_ = service()
            if self_ is None:
                return
            with self_._lock:
                for key in [k for k in self_._clouds if k[0] == owner_id]:
                    del self_._clouds[key]
            try:
                if self_._pool is not None:
                    self_._pool.free_all_blocks()
                logger.debug("Shape query: cloud array gone, released its state")
            except Exception:               # interpreter shutting down: nothing to free
                pass

        return forget


class _CloudState:
    """Everything built for one cloud: GPU coordinates, cell numbers, cell indexes."""

    def __init__(self, points):
        cp = _cupy()
        from infrastructure.memory_manager import MemoryManager
        from plugins.backends.grid_backends import CuPyGrid

        started = time.perf_counter()
        n = len(points)
        need_mb = (n * 13) // (1024 * 1024) + 64
        if not MemoryManager.can_use_gpu(need_mb):
            raise ShapeQueryError(
                f"Not enough GPU memory for shape queries on {n:,} points: "
                f"need about {need_mb:,} MB, have "
                f"{MemoryManager.get_available_gpu_mb():,} MB free.")

        grid = SpatialGrid.build_coarse_spatial_index(points, backend=CuPyGrid())
        self.n = n
        self.xyz = cp.empty((n, 3), dtype=cp.float32)
        self.ids = cp.empty(n, dtype=grid.cell_ids.dtype)
        for s in range(0, n, BLOCK):
            e = min(s + BLOCK, n)
            self.xyz[s:e].set(np.ascontiguousarray(points[s:e, :3], dtype=np.float32))
            self.ids[s:e].set(grid.cell_ids[s:e])

        n_cells = grid.n_cells
        self.counts = np.bincount(grid.cell_ids, minlength=n_cells)[:n_cells]
        grid.cell_ids = None                     # on the GPU now; nothing reads it here

        # Cell boxes, indexed by cell number, padded against rounding.
        nx, ny, nz = grid.shape
        ijk = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                                   indexing="ij"), axis=-1).reshape(-1, 3)
        # meshgrid "ij" with Z last enumerates (ix, iy, iz) in cell-number order.
        self.pad = float(PAD_FRACTION * float(np.max(grid.step)))
        low = grid.lo + grid.step * ijk.astype(np.float32)
        self.cell_lo = (low - self.pad).astype(np.float32)
        self.cell_hi = (low + grid.step + self.pad).astype(np.float32)
        self.lo = grid.lo.astype(np.float32)
        self.hi = (grid.lo + grid.step * np.asarray(grid.shape, dtype=np.float32)).astype(np.float32)

        self._cells = {}
        self._cell_locks = [threading.Lock() for _ in range(n_cells)]
        logger.info(f"Shape query: prepared {n:,} points in "
                    f"{time.perf_counter() - started:.2f} s "
                    f"({self.xyz.nbytes / 1e9:.2f} GB on the GPU)")

    def index_nbytes(self):
        return sum(cell.nbytes for cell in list(self._cells.values()))

    def cell(self, number):
        """The index of one cell, built the first time it is asked for."""
        cell = self._cells.get(number)
        if cell is not None:
            return cell
        with self._cell_locks[number]:
            cell = self._cells.get(number)
            if cell is None:
                cell = _CellIndex(self, number)
                self._cells[number] = cell
            return cell

    def query(self, shape):
        cp = _cupy()
        from core.services.shape_query_kernels import (
            kernels, blocks_for, MAX_GATHER_BLOCKS, MAX_MASK_BLOCKS, THREADS)
        gather_test, mask_all = kernels()

        # The shape's box, clipped to the cloud (open shapes become finite).
        lo_s, hi_s = (np.asarray(v, dtype=np.float32) for v in shape.aabb())
        lo_q = np.maximum(lo_s, self.lo) - self.pad
        hi_q = np.minimum(hi_s, self.hi) + self.pad
        if np.any(lo_q > hi_q):
            return np.empty(0, dtype=np.int32)

        states = shape.cell_states(self.cell_lo, self.cell_hi, self.pad)
        states[self.counts == 0] = MISS
        partial = np.flatnonzero(states == PARTIAL)
        cover = np.flatnonzero(states == COVER)
        if len(partial) == 0 and len(cover) == 0:
            return np.empty(0, dtype=np.int32)

        params = cp.asarray(shape.kernel_params())
        kind = np.int32(shape.kind)

        if self.counts[partial].sum() + self.counts[cover].sum() > EVERY_POINT_SHARE * self.n:
            # Every point, one kernel per block. Blocks come out in row order,
            # so the answer is already sorted.
            parts, mask = [], cp.empty(min(BLOCK, self.n), dtype=cp.bool_)
            for s in range(0, self.n, BLOCK):
                block = self.xyz[s:s + BLOCK]
                m = len(block)
                mask_all((blocks_for(m, MAX_MASK_BLOCKS),), (THREADS,),
                         (block, np.int64(m), kind, params, mask))
                parts.append(cp.flatnonzero(mask[:m]).astype(cp.int32) + np.int32(s))
            return _to_host(cp.concatenate(parts))

        parts = [self.cell(int(c)).rows for c in cover]
        for c in partial:
            cell = self.cell(int(c))
            runs = cell.runs(lo_q, hi_q)
            if runs is None:
                continue
            begins, offs = runs
            total = int(offs[-1])
            out = cp.empty(total, dtype=cp.int32)
            count = cp.zeros(1, dtype=cp.uint32)
            gather_test((blocks_for(total, MAX_GATHER_BLOCKS),), (THREADS,),
                        (self.xyz, cell.rows, cp.asarray(np.concatenate([begins, offs])),
                         np.int32(len(begins)), np.int64(total), kind, params, out, count))
            parts.append(out[:int(count.get()[0])])

        parts = [p for p in parts if len(p)]
        if not parts:
            return np.empty(0, dtype=np.int32)
        joined = parts[0] if len(parts) == 1 else cp.concatenate(parts)
        return _to_host(cp.sort(joined))


class _CellIndex:
    """One coarse cell's finite rows, sorted by a sub-grid over its own points.

    Rows with a non-finite coordinate are left out: they are inside no shape, and
    a covered cell is taken whole without testing, so they must not be in it.
    """

    __slots__ = ("rows", "lo", "step", "shape", "starts", "nbytes")

    def __init__(self, state, number):
        cp = _cupy()
        from infrastructure.memory_manager import MemoryManager

        count = int(state.counts[number])
        need_mb = (count * _CELL_BUILD_BYTES_PER_POINT) // (1024 * 1024) + 16
        if not MemoryManager.can_use_gpu(need_mb):
            raise ShapeQueryError(
                f"Not enough GPU memory to index a cell of {count:,} points: "
                f"need about {need_mb:,} MB, have "
                f"{MemoryManager.get_available_gpu_mb():,} MB free.")

        rows = cp.flatnonzero(state.ids == number).astype(cp.int32)
        pts = state.xyz[rows]
        finite = cp.isfinite(pts).all(axis=1)
        if not bool(finite.all()):
            rows, pts = rows[finite], pts[finite]
        del finite

        if len(rows) == 0:
            self.rows = rows
            self.lo = np.zeros(3, dtype=np.float32)
            self.step = np.ones(3, dtype=np.float32)
            self.shape = np.ones(3, dtype=np.int64)
            self.starts = np.zeros(2, dtype=np.int64)
            self.nbytes = 0
            return

        self.lo = cp.asnumpy(pts.min(axis=0)).astype(np.float32)
        hi = cp.asnumpy(pts.max(axis=0)).astype(np.float32)
        span = np.maximum(hi.astype(np.float64) - self.lo, 1e-3)
        n_sub = max(1, len(rows) // POINTS_PER_SUBCELL)
        side = (np.prod(span) / n_sub) ** (1.0 / 3.0)
        self.shape = np.maximum(1, np.ceil(span / side)).astype(np.int64)
        # A flat cell would otherwise ask for far more sub-cells than points.
        while np.prod(self.shape) > 4 * n_sub + 64:
            side *= 1.25
            self.shape = np.maximum(1, np.ceil(span / side)).astype(np.int64)
        self.step = (span / self.shape).astype(np.float32)

        nx, ny, nz = (int(v) for v in self.shape)
        ijk = cp.floor((pts - cp.asarray(self.lo)) / cp.asarray(self.step)).astype(cp.int32)
        del pts
        for axis, size in enumerate((nx, ny, nz)):
            cp.clip(ijk[:, axis], 0, size - 1, out=ijk[:, axis])
        sub = (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2]
        del ijk
        self.rows = rows[cp.argsort(sub)]
        counts = cp.bincount(sub, minlength=nx * ny * nz)
        del sub
        starts = cp.zeros(nx * ny * nz + 1, dtype=cp.int64)
        cp.cumsum(counts, out=starts[1:])
        self.starts = cp.asnumpy(starts)          # small; the run arithmetic is on the host
        self.nbytes = int(self.rows.nbytes)

    def runs(self, lo_q, hi_q):
        """``(begins, offs)`` of the sub-cell runs the box reaches, or None.

        One run per (ix, iy) column: Z is the fastest axis of the sub-cell number,
        so a column of sub-cells is one contiguous slice of ``rows``. ``offs`` is
        the prefix sum of the run lengths, which is what the kernel searches.
        """
        if len(self.rows) == 0:
            return None
        a = np.floor((lo_q - self.lo) / self.step).astype(np.int64)
        b = np.floor((hi_q - self.lo) / self.step).astype(np.int64)
        if np.any(b < 0) or np.any(a >= self.shape):
            return None
        a = np.clip(a, 0, self.shape - 1)
        b = np.clip(b, 0, self.shape - 1)
        ny, nz = int(self.shape[1]), int(self.shape[2])
        ix = np.arange(a[0], b[0] + 1)[:, None]
        iy = np.arange(a[1], b[1] + 1)[None, :]
        column = ((ix * ny + iy) * nz).ravel()
        begins = self.starts[column + a[2]]
        ends = self.starts[column + b[2] + 1]
        keep = ends > begins
        if not keep.any():
            return None
        begins, ends = begins[keep], ends[keep]
        offs = np.zeros(len(begins) + 1, dtype=np.int64)
        np.cumsum(ends - begins, out=offs[1:])
        return begins, offs


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise ShapeQueryError("Shape queries need CuPy and an NVIDIA GPU; "
                              "CuPy is not installed.") from exc
    return cp


def _owner(points):
    """The array that owns *points*' memory, so views of one cloud share a state."""
    owner = points
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    return owner


def _to_host(rows_gpu):
    """Copy to host in blocks: no pinned staging buffer the size of the answer."""
    out = np.empty(len(rows_gpu), dtype=np.int32)
    for s in range(0, len(rows_gpu), BLOCK):
        out[s:s + BLOCK] = rows_gpu[s:s + BLOCK].get()
    return out


def _free_gpu_pools():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:                               # no CuPy: nothing was held
        pass


def _as_query_error(exc, doing):
    """Turn a CuPy / CUDA failure into a ShapeQueryError a user can act on."""
    if isinstance(exc, ShapeQueryError):
        return exc
    name = type(exc).__name__
    if "OutOfMemory" in name or isinstance(exc, MemoryError):
        _free_gpu_pools()
        return ShapeQueryError(f"Ran out of GPU memory while {doing}: {exc}")
    if "CUDA" in name or "cuda" in type(exc).__module__:
        return ShapeQueryError(f"GPU error while {doing}: {name}: {exc}")
    return exc if isinstance(exc, (ValueError, TypeError)) else \
        ShapeQueryError(f"Shape query failed while {doing}: {name}: {exc}")
