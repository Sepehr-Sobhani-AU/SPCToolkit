"""
Grid KNN — exact k-nearest neighbours for 2D points on the GPU.

Why not the KNN backend (cuML). cuML's search is built for high-dimensional
data; for 2D points and a handful of neighbours it measured 6 s on a 2M-point
cluster, slower than a CPU KD-tree (2.2 s). A grid suits low dimensions: the
points are sorted by grid cell, and one CUDA thread per query scans the cells
around it ring by ring, keeping the best k. It stops once the k-th distance is
no farther than the nearest point outside the scanned square could be, so the
answer is exact — sparse areas just scan more rings. 2M points: 0.14 s.

Used by ``core.services.outline_cleanup``, where every query is in a cluster's
own plane (2D). GPU only, like ``core.services.shape_query``: without a CUDA
device ``GridKNN`` raises, it does not fall back to the CPU.
"""

import threading

import numpy as np

# Most neighbours one query can ask for (the per-thread best-k lists live in
# registers / local memory).
MAX_K = 32

# Target points per cell. About 2 keeps the first 3x3 scan small.
POINTS_PER_CELL = 2.0

# Ceiling on cells, whatever the cluster size: 16M cells is 128 MB of
# start/end tables.
MAX_CELLS = 16_000_000

# Queries sent to the GPU at once; bounds the (queries x k) output scratch.
QUERY_BLOCK = 4_000_000

THREADS = 128

_SOURCE = r'''
extern "C" __global__
void grid_knn(const float* ref, const int* order, const int* cell_start,
              const int* cell_end, const float* qry, const int nq,
              const int nx, const int ny, const float lox, const float loy,
              const float cell, const int k, const int max_ring,
              float* out_d, int* out_i)
{
    int q = blockDim.x * blockIdx.x + threadIdx.x;
    if (q >= nq) return;
    float qx = qry[2 * q], qy = qry[2 * q + 1];
    float bd[32]; int bi[32];
    for (int j = 0; j < k; ++j) { bd[j] = 3.4e38f; bi[j] = -1; }
    int cx = (int)floorf((qx - lox) / cell);
    int cy = (int)floorf((qy - loy) / cell);
    cx = min(max(cx, 0), nx - 1);
    cy = min(max(cy, 0), ny - 1);
    for (int r = 0; r <= max_ring; ++r) {
        // Ring r: full top and bottom rows, only the two end cells between.
        // Rows and columns outside the grid are skipped, not looped over, so
        // a thin cluster (one row of cells) costs one row per ring.
        int gy0 = max(cy - r, 0), gy1 = min(cy + r, ny - 1);
        for (int gy = gy0; gy <= gy1; ++gy) {
            bool edge_row = (gy == cy - r) || (gy == cy + r);
            int gx0 = max(cx - r, 0), gx1 = min(cx + r, nx - 1);
            for (int gx = gx0; gx <= gx1; ++gx) {
                if (!edge_row && gx != cx - r && gx != cx + r) {
                    if (cx + r > gx1) break;      // right end cell is off-grid
                    gx = cx + r;                  // jump to the right end cell
                }
                int c = gy * nx + gx;
                for (int s = cell_start[c]; s < cell_end[c]; ++s) {
                    int p = order[s];
                    float dx = ref[2 * p] - qx, dy = ref[2 * p + 1] - qy;
                    float d2 = dx * dx + dy * dy;
                    if (d2 < bd[k - 1]) {
                        int j = k - 1;
                        while (j > 0 && bd[j - 1] > d2) { bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; --j; }
                        bd[j] = d2; bi[j] = p;
                    }
                }
            }
        }
        // Nearest a point outside the scanned square can be.
        float x0 = lox + (cx - r) * cell, x1 = lox + (cx + r + 1) * cell;
        float y0 = loy + (cy - r) * cell, y1 = loy + (cy + r + 1) * cell;
        float reach = fminf(fminf(qx - x0, x1 - qx), fminf(qy - y0, y1 - qy));
        bool covers_all = (cx - r <= 0) && (cy - r <= 0) && (cx + r >= nx - 1) && (cy + r >= ny - 1);
        if (covers_all || (bi[k - 1] >= 0 && bd[k - 1] <= reach * reach)) break;
    }
    for (int j = 0; j < k; ++j) {
        out_d[q * k + j] = sqrtf(bd[j]);
        out_i[q * k + j] = bi[j];
    }
}
'''

_lock = threading.Lock()
_kernel = None


def _get_kernel():
    """The CUDA kernel, compiled on first use and kept for the process.

    Raises whatever CuPy raises when there is no CUDA device.
    """
    global _kernel
    with _lock:
        if _kernel is None:
            import cupy as cp
            _kernel = cp.RawKernel(_SOURCE, "grid_knn")
        return _kernel


class GridKNN:
    """A GPU index over 2D reference points: build once, query many times.

    Args:
        ref: (N, 2+) reference points; only the first two columns are used.

    Raises:
        RuntimeError: no usable CUDA device (the message says so).
    """

    def __init__(self, ref):
        try:
            import cupy as cp
            kernel = _get_kernel()
        except Exception as exc:
            raise RuntimeError(
                f"Grid KNN needs a CUDA GPU with CuPy ({type(exc).__name__}: {exc})"
            ) from exc
        self._kernel = kernel

        ref = cp.ascontiguousarray(cp.asarray(np.asarray(ref)[:, :2], dtype=cp.float32))
        self.n = len(ref)
        if self.n == 0:
            raise ValueError("Grid KNN needs at least one reference point.")
        lo = ref.min(axis=0)
        span = cp.maximum(ref.max(axis=0) - lo, 1e-6)
        sx, sy = float(span[0]), float(span[1])
        # Area-based size, but never finer than the cluster's length shared
        # out among its points: a near-straight cluster would otherwise get
        # cells far thinner than its point spacing.
        cell = max(float(np.sqrt(sx * sy * POINTS_PER_CELL / self.n)),
                   max(sx, sy) * POINTS_PER_CELL / self.n)
        nx, ny = int(sx / cell) + 1, int(sy / cell) + 1
        while nx * ny > MAX_CELLS:
            cell *= 1.25
            nx, ny = int(sx / cell) + 1, int(sy / cell) + 1

        ix = cp.clip(((ref[:, 0] - lo[0]) / cell).astype(cp.int32), 0, nx - 1)
        iy = cp.clip(((ref[:, 1] - lo[1]) / cell).astype(cp.int32), 0, ny - 1)
        cell_id = iy * nx + ix
        order = cp.argsort(cell_id).astype(cp.int32)
        sorted_id = cell_id[order]
        cells = cp.arange(nx * ny, dtype=cp.int32)
        self._start = cp.searchsorted(sorted_id, cells, side="left").astype(cp.int32)
        self._end = cp.searchsorted(sorted_id, cells, side="right").astype(cp.int32)
        self._ref, self._order = ref, order
        self._lo = (float(lo[0]), float(lo[1]))
        self._cell, self._nx, self._ny = cell, nx, ny

    def query(self, query, k):
        """(distances, indices) of the k nearest reference points, as numpy
        arrays of shape (M, k), nearest first. A reference point queried
        against itself comes back as its own nearest neighbour (distance 0),
        as with ``scipy.spatial.cKDTree.query``."""
        import cupy as cp

        if k > MAX_K:
            raise ValueError(f"Grid KNN supports k <= {MAX_K}, got {k}.")
        k = int(min(k, self.n))
        query = np.asarray(query, dtype=np.float32)[:, :2]
        m = len(query)
        out_d = np.empty((m, k), np.float32)
        out_i = np.empty((m, k), np.int64)
        max_ring = max(self._nx, self._ny)
        for a in range(0, m, QUERY_BLOCK):
            b = min(m, a + QUERY_BLOCK)
            q = cp.ascontiguousarray(cp.asarray(query[a:b]))
            nq = b - a
            d = cp.empty((nq, k), cp.float32)
            i = cp.empty((nq, k), cp.int32)
            self._kernel(((nq + THREADS - 1) // THREADS,), (THREADS,),
                         (self._ref, self._order, self._start, self._end, q,
                          np.int32(nq), np.int32(self._nx), np.int32(self._ny),
                          np.float32(self._lo[0]), np.float32(self._lo[1]),
                          np.float32(self._cell), np.int32(k), np.int32(max_ring),
                          d, i))
            out_d[a:b] = cp.asnumpy(d)
            out_i[a:b] = cp.asnumpy(i)
        return out_d, out_i

    def release(self):
        """Drop this index's GPU arrays. Their memory goes back to CuPy's pool
        (reused by the next index); ``free_gpu_memory`` returns the pool to
        the driver once a whole job is done."""
        self._ref = self._order = self._start = self._end = None


def free_gpu_memory():
    """Hand CuPy's cached blocks back to the driver — call once at the end of
    a job, not per query (other GPU work shares the pool)."""
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
