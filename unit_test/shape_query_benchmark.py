"""
Benchmark for core.services.shape_query on a real cloud.

Reproduces the numbers in DECISIONS.md § 2026-09-28 against the real service:
small shapes (the growers' case), medium and large shapes, and two workloads of
1,000 small queries, with RAM and VRAM after every step. Each answer is checked
against a CuPy test of every point written as separate array steps, independent
of the service's kernel; the two can disagree only on a point sitting exactly on
an edge, which is reported, not failed.

Everything is appended to logs/shape_query_benchmark.log, flushed line by line,
so a run that is killed can still be read. The 168M cloud is close to the limits
of a 15 GB / 8 GB machine, so run it detached and memory-capped:

    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \\
        python unit_test/shape_query_benchmark.py "Middle Head-Part 2 - 168M.ply"
    python unit_test/shape_query_benchmark.py <file.ply> --limit 5000000
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cupy as cp

from spatial_grid_benchmark import read_ply_xyz
from core.services.query_shapes import Sphere, Cylinder, OrientedBox, Box
from core.services.shape_query import ShapeQueryService

N_QUERIES = 30
N_WORKLOAD = 1000
BLOCK = 8_000_000

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG = open(os.path.join(_LOG_DIR, "shape_query_benchmark.log"), "a", buffering=1)
_T0 = time.perf_counter()


def log(msg=""):
    rss = peak = -1.0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                rss = int(line.split()[1]) / 1e6
            elif line.startswith("VmHWM"):
                peak = int(line.split()[1]) / 1e6
    free, total = cp.cuda.runtime.memGetInfo()
    line = (f"[{time.perf_counter() - _T0:7.1f}s  RAM {rss:5.2f} GB (peak {peak:5.2f})  "
            f"VRAM {(total - free) / 1e9:5.2f} GB] {msg}")
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()
    os.fsync(_LOG.fileno())


def make_cases(xyz):
    """Deterministic shapes; same seed and draw order as the design benchmark."""
    rng = np.random.default_rng(0)
    ext = xyz.max(0)
    pool = xyz[rng.integers(0, len(xyz), 10 * N_QUERIES)].copy()

    def pick(n):
        return pool[rng.integers(0, len(pool), n)]

    def rot():
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        return q

    cases = [
        ("sphere r=0.5 m", [Sphere(c, 0.5) for c in pick(N_QUERIES)]),
        ("cylinder r=0.05 L=1 m", [Cylinder(c, rng.normal(size=3), 0.05, 1.0)
                                   for c in pick(N_QUERIES)]),
        ("rotated box 20x20x10 m", [OrientedBox(c, rot(), [10, 10, 5])
                                    for c in pick(N_QUERIES)]),
        ("sphere r=50 m", [Sphere(c, 50.0) for c in pick(N_QUERIES)]),
        ("axis box 1/4 of site", [Box(c - ext / 4, c + ext / 4) for c in pick(N_QUERIES)]),
    ]
    big_pool = xyz[rng.integers(0, len(xyz), N_WORKLOAD)].copy()
    workloads = [
        ("1000 small spheres", [Sphere(c, 0.5) for c in big_pool]),
        ("1000 small cylinders", [Cylinder(c, rng.normal(size=3), 0.05, 1.0)
                                  for c in big_pool]),
    ]
    return cases, workloads


def reference(shape, xyz_gpu):
    """Every point, as separate CuPy steps — independent of the service's kernel."""
    out = []
    for s in range(0, len(xyz_gpu), BLOCK):
        p = xyz_gpu[s:s + BLOCK]
        if isinstance(shape, Sphere):
            d = p - cp.asarray(shape.centre)
            m = (d * d).sum(1) <= shape.radius * shape.radius
        elif isinstance(shape, Cylinder):
            v = p - cp.asarray(shape.start)
            t = v @ cp.asarray(shape.direction)
            m = (t >= 0) & (t <= shape.length) & ((v * v).sum(1) - t * t <= shape.radius ** 2)
        elif isinstance(shape, OrientedBox):
            local = (p - cp.asarray(shape.centre)) @ cp.asarray(shape.rotation)
            m = cp.all(cp.abs(local) <= cp.asarray(shape.half_sizes), 1)
        else:
            m = cp.all((p >= cp.asarray(shape.low)) & (p <= cp.asarray(shape.high)), 1)
        out.append(cp.flatnonzero(m).astype(cp.int32) + s)
    return cp.concatenate(out)


def mismatches(rows, ref, n):
    got = cp.zeros(n, cp.bool_)
    for s in range(0, len(rows), BLOCK):
        g = cp.empty(min(BLOCK, len(rows) - s), cp.int32)
        g.set(rows[s:s + BLOCK])
        got[g] = True
    want = cp.zeros(n, cp.bool_)
    want[ref] = True
    return int(cp.count_nonzero(got != want))


def timed(f, *args):
    t = time.perf_counter()
    result = f(*args)
    return result, (time.perf_counter() - t) * 1e3


def main(path, limit):
    log(f"=== start, file {path}, limit {limit}")
    xyz = read_ply_xyz(path, limit=limit)
    n = len(xyz)
    log(f"loaded {n:,} points ({xyz.nbytes / 1e9:.2f} GB; kept in RAM, as the app does)")
    cases, workloads = make_cases(xyz)

    service = ShapeQueryService()
    _, ms = timed(service.points_in, xyz, Sphere(xyz[0], 0.5))
    log(f"SETUP: first query prepared the cloud in {ms / 1e3:.2f} s")

    # Reference copy, uploaded in blocks (a whole-array upload leaves a
    # same-size staging copy in RAM). For checking only.
    xyz_gpu = cp.empty((n, 3), cp.float32)
    for s in range(0, n, BLOCK):
        xyz_gpu[s:s + BLOCK].set(xyz[s:s + BLOCK])
    log("reference copy on the GPU (for checking only, not part of the service)")

    for name, shapes in cases:
        log(f"-- {name}")
        first, again, res, bad = [], [], [], 0
        for qi, shape in enumerate(shapes):
            before = service.index_nbytes()
            rows, ms = timed(service.points_in, xyz, shape)
            m = mismatches(rows, reference(shape, xyz_gpu), n)
            n_rows = len(rows)
            del rows
            _, ms2 = timed(service.points_in, xyz, shape)
            bad += m
            first.append(ms); again.append(ms2); res.append(n_rows)
            if qi % 10 == 0 or m:
                log(f"   q{qi}: first {ms:.3f} ms (index +{(service.index_nbytes() - before) / 1e6:.0f} MB), "
                    f"again {ms2:.3f} ms, result {n_rows:,}, edge disagreements {m}")
        log(f"RESULT {name:24s} first {np.median(first):9.3f} ms  again {np.median(again):9.3f} / "
            f"p90 {np.percentile(again, 90):9.3f} ms  result {int(np.median(res)):,}  "
            f"edge disagreements {bad}")

    for name, shapes in workloads:
        _, ms = timed(lambda: [service.points_in(xyz, s) for s in shapes])
        _, ms2 = timed(lambda: [service.points_in(xyz, s) for s in shapes])
        log(f"RESULT {name}: first pass {ms:.0f} ms, second pass {ms2:.0f} ms = "
            f"{ms2 / len(shapes):.3f} ms each")

    log(f"RESULT per-cell index held: {service.index_nbytes() / 1e9:.2f} GB GPU "
        f"({service.index_nbytes() / n:.2f} B/pt)")
    del xyz_gpu
    service.release_all()
    log("released")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
        args = [a for a in args if a != str(limit)]
    if not args:
        print(__doc__)
        sys.exit(0)
    try:
        main(args[0], limit)
        log("=== finished")
    except BaseException:
        import traceback
        log("=== FAILED\n" + traceback.format_exc())
        raise
