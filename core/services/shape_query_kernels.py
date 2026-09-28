"""
Shape Query Kernels

The two CUDA kernels behind ``core.services.shape_query``, and the one
definition of whether a point is inside a shape (``inside()`` below).

Why one kernel per query rather than CuPy array steps. A small query touches a
few hundred candidate points, so its cost is almost all fixed overhead. As ~15
separate CuPy steps (upload the runs, search, gather, subtract, square, sum,
compare, compact ...) a 0.5 m sphere on the 168M cloud took 1.2 ms; as this one
kernel it takes 0.35-0.5 ms, as fast as ``NeighborIndex`` on the CPU
(DECISIONS.md § 2026-09-28).

**Every test is written as "inside when all conditions hold"**, never as "outside
as soon as one fails". A NaN coordinate makes every comparison false, so written
this way it is outside every shape; written the other way round it would slip
past every early exit and count as inside.

Compiled with ``--fmad=false``: without it the compiler may fuse a multiply and
an add into one rounding, and a point sitting exactly on an edge could change
sides between two builds of the same code.

Shape numbers and parameter layouts must match ``core.services.query_shapes``.
"""

import threading

_SOURCE = r'''
__device__ __forceinline__ bool inside(int kind, const float* __restrict__ p,
                                       float x, float y, float z) {
    if (kind == 0) {                        // sphere: centre, r^2
        float dx = x - p[0], dy = y - p[1], dz = z - p[2];
        return dx * dx + dy * dy + dz * dz <= p[3];
    }
    if (kind == 1) {                        // oriented box: centre, R (row major), half sizes
        float dx = x - p[0], dy = y - p[1], dz = z - p[2];
        bool in = true;
        for (int j = 0; j < 3; ++j) {
            float l = dx * p[3 + j] + dy * p[6 + j] + dz * p[9 + j];
            in = in && (fabsf(l) <= p[12 + j]);
        }
        return in;
    }
    if (kind == 2) {                        // cylinder: start, unit direction, r^2, length
        float vx = x - p[0], vy = y - p[1], vz = z - p[2];
        float t = vx * p[3] + vy * p[4] + vz * p[5];
        return (t >= 0.f) && (t <= p[7]) && (vx * vx + vy * vy + vz * vz - t * t <= p[6]);
    }
    if (kind == 3) {                        // axis-aligned box: low, high
        return (x >= p[0]) && (x <= p[3]) && (y >= p[1]) && (y <= p[4])
            && (z >= p[2]) && (z <= p[5]);
    }
    if (kind == 4) {                        // slab: unit normal, low, high
        float d = x * p[0] + y * p[1] + z * p[2];
        return (d >= p[3]) && (d <= p[4]);
    }
    if (kind == 5) {                        // plan polygon: z_min, z_max, n, then x0 y0 x1 y1 ...
        if (!((z >= p[0]) && (z <= p[1]))) return false;
        int n = (int)p[2];
        const float* v = p + 3;
        bool in = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            float xi = v[2 * i], yi = v[2 * i + 1], xj = v[2 * j], yj = v[2 * j + 1];
            if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi))
                in = !in;
        }
        return in;
    }
    return false;
}

// Every candidate in the given runs of `rows` is tested; survivors are appended
// to `out` through an atomic counter, so their order is not fixed (the caller
// sorts). runs = [begin_0 .. begin_{n-1}, off_0 .. off_n], off = prefix sum of
// the run lengths, so candidate i lives in the run r with off_r <= i < off_{r+1}.
extern "C" __global__ void gather_test(
        const float* __restrict__ xyz, const int* __restrict__ rows,
        const long long* __restrict__ runs, int n_runs, long long total,
        int kind, const float* __restrict__ params,
        int* __restrict__ out, unsigned int* __restrict__ count) {
    const long long* begins = runs;
    const long long* offs = runs + n_runs;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        int lo = 0, hi = n_runs - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (offs[mid] <= i) lo = mid; else hi = mid - 1;
        }
        int row = rows[begins[lo] + (i - offs[lo])];
        const float* q = xyz + 3LL * row;
        if (inside(kind, params, q[0], q[1], q[2])) out[atomicAdd(count, 1u)] = row;
    }
}

// The every-point path: one bool per point of a block of the cloud.
extern "C" __global__ void mask_all(
        const float* __restrict__ xyz, long long n, int kind,
        const float* __restrict__ params, bool* __restrict__ mask) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x)
        mask[i] = inside(kind, params, xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
}
'''

THREADS = 256
# Grid-stride loops, so the grid is capped and each thread takes several items.
MAX_GATHER_BLOCKS = 4096
MAX_MASK_BLOCKS = 65535

_lock = threading.Lock()
_kernels = None


def kernels():
    """``(gather_test, mask_all)``, compiled on first use and kept for the process.

    Raises whatever CuPy raises when there is no CUDA device; the service turns
    that into a message a user can act on.
    """
    global _kernels
    with _lock:
        if _kernels is None:
            import cupy as cp
            module = cp.RawModule(code=_SOURCE, options=("--fmad=false",))
            _kernels = (module.get_function("gather_test"),
                        module.get_function("mask_all"))
        return _kernels


def blocks_for(n, cap):
    """Thread blocks for *n* items under a grid-stride cap."""
    return int(min(max((int(n) + THREADS - 1) // THREADS, 1), cap))
