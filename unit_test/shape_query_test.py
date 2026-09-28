"""
Tests for core.services.shape_query and core.services.query_shapes.

The oracle is a float64 numpy test of every point. Rounding can put a point that
sits exactly on a shape's edge on either side, and the kernel is the only
definition of "inside" for those (DECISIONS.md § 2026-09-28), so the comparison
leaves out points within a hair of an edge. Everything else must match exactly.

Every run appends to logs/shape_query_test.log, flushed line by line with RAM and
VRAM, so a run that dies can still be read afterwards.

Needs CuPy and an NVIDIA GPU: the service has no CPU path by design.

Run:
    python unit_test/shape_query_test.py
"""

import gc
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.services import shape_query as sq
from core.services.query_shapes import (
    Sphere, Box, OrientedBox, Cylinder, Slab, PlanPolygon, COVER, PARTIAL, MISS)
from core.services.shape_query import ShapeQueryService

_RNG = np.random.default_rng(7)
# Points closer than this to an edge are not compared (see module docstring).
_EDGE = 1e-3

# ----------------------------------------------------------------------
# Log
# ----------------------------------------------------------------------

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG = open(os.path.join(_LOG_DIR, "shape_query_test.log"), "a", buffering=1)
_T0 = time.perf_counter()


def _rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    return -1.0


def _vram_gb():
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return (total - free) / 1e9
    except Exception:
        return -1.0


def log(msg):
    line = (f"[{time.perf_counter() - _T0:7.1f}s  RAM {_rss_gb():5.2f} GB  "
            f"VRAM {_vram_gb():5.2f} GB] {msg}")
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()
    os.fsync(_LOG.fileno())


# ----------------------------------------------------------------------
# Clouds
# ----------------------------------------------------------------------

def _clustered(n=400_000):
    """Like real survey data: a few dense blobs, one very dense, much empty space.

    One blob holds half the points, so one coarse cell is far denser than the
    rest — the case the per-cell index exists for.
    """
    heavy = _RNG.normal([20, 20, 2], [1.5, 1.5, 0.5], (n // 2, 3))
    blobs = _RNG.integers(0, 6, (n - n // 2, 3)) * [8.0, 8.0, 1.0]
    light = blobs + _RNG.normal(0, 0.8, (n - n // 2, 3))
    return np.concatenate([heavy, light]).astype(np.float32)


def _uniform(n=300_000, extent=(44.0, 44.0, 8.0)):
    return (_RNG.random((n, 3)) * extent).astype(np.float32)


# ----------------------------------------------------------------------
# float64 oracle, with an "edge band" left out of comparison
# ----------------------------------------------------------------------

def _inside(shape, p, grow):
    """float64 test of *shape* grown by *grow* (negative shrinks)."""
    p = p.astype(np.float64)
    if isinstance(shape, Sphere):
        return np.sum((p - shape.centre) ** 2, axis=1) <= (float(shape.radius) + grow) ** 2
    if isinstance(shape, Box):
        return np.all((p >= shape.low - grow) & (p <= shape.high + grow), axis=1)
    if isinstance(shape, OrientedBox):
        local = (p - shape.centre) @ shape.rotation.astype(np.float64)
        return np.all(np.abs(local) <= shape.half_sizes + grow, axis=1)
    if isinstance(shape, Cylinder):
        v = p - shape.start
        t = v @ shape.direction.astype(np.float64)
        perp2 = np.maximum(np.sum(v * v, axis=1) - t * t, 0)
        return ((t >= -grow) & (t <= float(shape.length) + grow)
                & (np.sqrt(perp2) <= float(shape.radius) + grow))
    if isinstance(shape, Slab):
        d = p @ shape.normal.astype(np.float64)
        return (d >= float(shape.low) - grow) & (d <= float(shape.high) + grow)
    if isinstance(shape, PlanPolygon):
        z_ok = (p[:, 2] >= float(shape.z_min) - grow) & (p[:, 2] <= float(shape.z_max) + grow)
        inside_xy = shape._inside_xy(p[:, 0], p[:, 1])
        near_edge = _distance_to_edges(shape.polygon.astype(np.float64), p[:, :2]) <= abs(grow)
        xy = (inside_xy | near_edge) if grow > 0 else (inside_xy & ~near_edge)
        return z_ok & xy
    raise TypeError(shape)


def _distance_to_edges(poly, xy):
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(((xy[:, None, :] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    closest = a + t[..., None] * ab
    return np.sqrt(((xy[:, None, :] - closest) ** 2).sum(-1)).min(axis=1)


def _check(service, points, shape, label):
    """The service's answer matches the oracle on every point clear of an edge."""
    rows = service.points_in(points, shape)
    assert rows.dtype == np.int32
    assert np.all(np.diff(rows) > 0), f"{label}: rows not sorted and unique"
    got = np.zeros(len(points), dtype=bool)
    got[rows] = True
    sure_in = _inside(shape, points, -_EDGE)
    maybe_in = _inside(shape, points, +_EDGE)
    clear = sure_in == maybe_in
    wrong = np.count_nonzero(got[clear] != sure_in[clear])
    assert wrong == 0, f"{label}: {wrong} points wrong (of {clear.sum():,} clear of an edge)"
    extra = np.count_nonzero(got & ~maybe_in)
    assert extra == 0, f"{label}: {extra} points returned that are well outside"
    return rows


def _rotation():
    q, _ = np.linalg.qr(_RNG.normal(size=(3, 3)))
    return q


def _state(service, points):
    return service._state_for(points)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_every_shape_matches_the_oracle():
    """Each shape, at several sizes, on a clustered cloud."""
    service = ShapeQueryService()
    points = _clustered()
    centres = points[_RNG.integers(0, len(points), 6)]
    shapes = []
    for c in centres:
        shapes += [
            ("sphere 0.5", Sphere(c, 0.5)),
            ("sphere 6", Sphere(c, 6.0)),
            ("box", Box(c - [2, 3, 0.5], c + [3, 2, 0.7])),
            ("oriented box", OrientedBox(c, _rotation(), [4, 1, 0.5])),
            ("cylinder thin", Cylinder(c, _RNG.normal(size=3), 0.1, 3.0)),
            ("cylinder fat", Cylinder(c, [0, 0, 1], 5.0, 2.0)),
            ("slab", Slab(_RNG.normal(size=3), float(c.sum()) - 1, float(c.sum()) + 1)),
            ("plan polygon", PlanPolygon(c[:2] + [[-4, -4], [5, -3], [1, 0], [4, 5], [-3, 4]],
                                         c[2] - 1, c[2] + 1)),
        ]
    total = 0
    for label, shape in shapes:
        total += len(_check(service, points, shape, label))
    log(f"  every shape: {len(shapes)} queries match the oracle ({total:,} points returned)")
    service.release_all()


def test_the_three_search_paths():
    """Covered cells taken whole, partly covered cells searched, and the
    every-point path for shapes over half the cloud — each checked, and each
    confirmed to have actually been taken."""
    service = ShapeQueryService()
    points = _uniform()
    state = _state(service, points)

    # Two full columns of coarse cells, well inside: some cells must be COVER.
    lo, step = state.lo, (state.hi - state.lo) / np.array([11, 11, 2], np.float32)
    box = Box(lo + step * [2.5, 2.5, -1], lo + step * [6.5, 6.5, 3])
    states = box.cell_states(state.cell_lo, state.cell_hi, state.pad)
    assert np.count_nonzero(states == COVER) >= 16, np.bincount(states)
    _check(service, points, box, "covered cells")

    small = Sphere(points[123], 0.7)
    before = len(state._cells)
    _check(service, points, small, "small sphere")
    assert len(state._cells) - before <= 8, "a small query built too many cells"

    big = Box(state.lo - 1, state.hi + 1)
    assert big.cell_states(state.cell_lo, state.cell_hi, state.pad).sum() > 0
    rows = _check(service, points, big, "every point")
    assert len(rows) == len(points)
    log("  covered, partial and every-point paths all match the oracle")
    service.release_all()


def test_points_on_cell_borders_are_counted_once():
    """A lattice whose points sit exactly on coarse cell borders."""
    service = ShapeQueryService()
    g = np.arange(0, 45, 0.5, dtype=np.float32)
    x, y, z = np.meshgrid(g, g, np.arange(0, 8.5, 0.5, dtype=np.float32), indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    for shape in (Box([3.9, 3.9, 0.9], [16.1, 20.1, 5.1]),
                  Sphere([22, 22, 4], 9.3), Slab([1, 0, 0], 3.9, 12.1)):
        rows = _check(service, points, shape, f"lattice {type(shape).__name__}")
        assert len(np.unique(rows)) == len(rows)
    log("  points on cell borders: each counted once, answers match")
    service.release_all()


def test_edges_are_decided_the_same_way_every_time():
    """Same query twice gives the identical array, points on edges included."""
    service = ShapeQueryService()
    g = np.arange(0, 20, 0.25, dtype=np.float32)
    x, y, z = np.meshgrid(g, g, g[:12], indexing="ij")
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    shapes = [Box([2, 2, 0.5], [9, 11, 2]), Sphere([10, 10, 1.5], 4.0),
              Cylinder([1, 1, 0.5], [1, 1, 0], 2.0, 10.0),
              PlanPolygon([[1, 1], [12, 1], [12, 6], [6, 6], [6, 12], [1, 12]], 0.5, 2.0)]
    for shape in shapes:
        first = service.points_in(points, shape)
        for _ in range(3):
            assert np.array_equal(first, service.points_in(points, shape))
    log("  repeated queries return identical arrays")
    service.release_all()


def test_a_concave_fence_does_not_cover_its_notch():
    """An L-shaped fence. The cell in the notch lies inside the fence's bounding
    box but outside the fence; the cell holding the L's inner corner has an edge
    running through it. Neither may be taken whole."""
    service = ShapeQueryService()
    points = _uniform()
    state = _state(service, points)
    step = (state.hi - state.lo) / np.array([11, 11, 2], np.float32)
    o = state.lo
    # L covering cells x 1..6, y 1..6 minus the notch x 4..6, y 4..6 (in cell units),
    # with the inner corner in the middle of a cell.
    cell = lambda cx, cy: (o[:2] + step[:2] * [cx, cy]).tolist()
    fence = PlanPolygon([cell(1, 1), cell(7, 1), cell(7, 4.5), cell(4.5, 4.5),
                         cell(4.5, 7), cell(1, 7)], -100, 100)
    states = fence.cell_states(state.cell_lo, state.cell_hi, state.pad)
    notch_cell = SpatialGridCell(state, 5.5, 5.5)
    inner_corner_cell = SpatialGridCell(state, 4.5 - 0.01, 4.5 - 0.01)
    for number in notch_cell + inner_corner_cell:
        assert states[number] != COVER, f"cell {number} wrongly covered"
    assert np.count_nonzero(states == COVER) > 0, "nothing covered at all"
    _check(service, points, fence, "L fence")
    log("  concave fence: notch and corner cells not covered, answer matches")
    service.release_all()


def SpatialGridCell(state, cx, cy):
    """Cell numbers (both Z layers) of the coarse cell at cell coordinates cx, cy."""
    ix, iy = int(cx), int(cy)
    return [(ix * 11 + iy) * 2 + iz for iz in (0, 1)]


def test_empty_outside_and_open_shapes():
    service = ShapeQueryService()
    points = _clustered(100_000)
    assert len(service.points_in(np.empty((0, 3), np.float32), Sphere([0, 0, 0], 1))) == 0
    assert service.cloud_count() == 0, "an empty cloud must not touch the GPU"
    assert len(service.points_in(points, Sphere([1e4, 1e4, 1e4], 5))) == 0
    assert len(service.points_in(points, Box([500, 500, 500], [600, 600, 600]))) == 0
    _check(service, points, Slab([0, 0, 1], 1.0, 2.5), "open slab")
    _check(service, points, PlanPolygon([[0, 0], [30, 0], [30, 30]]), "open-height fence")
    mask = service.points_in(points, Sphere(points[0], 3.0), as_mask=True)
    rows = service.points_in(points, Sphere(points[0], 3.0))
    assert mask.dtype == bool and len(mask) == len(points)
    assert np.array_equal(np.flatnonzero(mask), rows)
    log("  empty cloud, far-away shapes, open shapes and the mask form all behave")
    service.release_all()


def test_non_finite_points_are_never_returned():
    """Not by the kernel test, and not when their cell is taken whole."""
    service = ShapeQueryService()
    points = _uniform(200_000)
    bad = _RNG.choice(len(points), 500, replace=False)
    points[bad[:250], 0] = np.nan
    points[bad[250:], 2] = np.inf
    everything = Box([-1e6, -1e6, -1e6], [1e6, 1e6, 1e6])
    rows = service.points_in(points, everything)
    assert not np.isin(bad, rows).any()
    assert len(rows) == len(points) - len(bad)
    state = _state(service, points)
    middle = Box(state.lo + [5, 5, -1], state.lo + [30, 30, 10])
    rows = service.points_in(points, middle)
    assert not np.isin(bad, rows).any()
    log("  non-finite points never returned (every-point and covered-cell paths)")
    service.release_all()


def test_views_of_one_cloud_share_one_preparation():
    service = ShapeQueryService()
    cloud = np.zeros((200_000, 6), dtype=np.float32)
    cloud[:, :3] = _uniform(200_000)
    service.points_in(cloud[:, :3], Sphere(cloud[0, :3], 2))
    service.points_in(cloud[:, :3], Sphere(cloud[1, :3], 2))       # a new view object
    assert service.cloud_count() == 1
    _check(service, cloud[:, :3], Sphere(cloud[5, :3], 3.0), "xyz view of a wider array")
    log("  views of the same array share one preparation")
    service.release_all()


def test_release_frees_the_gpu():
    """Measured on the device, not CuPy's pool: in the app CuPy allocates through
    RAPIDS' RMM (installed when the hardware detector loads), whose memory never
    shows in ``get_default_memory_pool()``."""
    service = ShapeQueryService()
    points = _uniform(10_000_000)                 # ~130 MB on the GPU once prepared
    sq._free_gpu_pools()
    base = _vram_gb()
    service.points_in(points, Sphere(points[0], 5.0))
    held = _vram_gb() - base
    assert service.cloud_count() == 1 and held > 0.1, f"only {held:.3f} GB held"
    service.release_all()
    freed = _vram_gb() - base
    assert service.cloud_count() == 0
    assert freed < 0.02, f"{freed:.3f} GB still held after release_all"

    service.points_in(points, Sphere(points[0], 5.0))
    del points
    gc.collect()
    assert service.cloud_count() == 0, "state kept after the cloud's array was freed"
    sq._free_gpu_pools()
    left = _vram_gb() - base
    assert left < 0.02, f"{left:.3f} GB still held after the cloud was dropped"
    log(f"  release_all and dropping the cloud both free the GPU ({held:.2f} GB held, then freed)")


def test_preparing_a_cloud_does_not_copy_it_into_host_ram():
    """Uploads are blocked: host RAM must not grow by anything like the cloud's size."""
    service = ShapeQueryService()
    points = _uniform(40_000_000)
    size_gb = points.nbytes / 1e9
    gc.collect()
    before = _rss_gb()
    service.points_in(points, Sphere(points[0], 1.0))
    grown = _rss_gb() - before
    log(f"  preparing {size_gb:.2f} GB of points grew host RAM by {grown:.2f} GB")
    assert grown < 0.5 * size_gb, f"host RAM grew {grown:.2f} GB for a {size_gb:.2f} GB cloud"
    service.release_all()


def test_a_cell_is_built_once_when_two_threads_reach_it():
    service = ShapeQueryService()
    points = _clustered(300_000)
    builds = []
    original = sq._CellIndex

    class Counting(original):
        __slots__ = ()

        def __init__(self, state, number):
            builds.append(number)
            time.sleep(0.05)                     # widen the race window
            super().__init__(state, number)

    sq._CellIndex = Counting
    try:
        shape = Sphere([20, 20, 2], 1.0)          # inside the one dense cell
        results = [None, None]

        def run(i):
            results[i] = service.points_in(points, shape)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sq._CellIndex = original
    assert len(builds) == len(set(builds)), f"a cell was built twice: {builds}"
    assert np.array_equal(results[0], results[1])
    log(f"  two threads, same new cells: each built once ({len(builds)} cells)")
    service.release_all()


def test_bad_shapes_are_refused():
    for make in (lambda: Sphere([0, 0], 1), lambda: Sphere([0, 0, 0], -1),
                 lambda: Box([1, 1, 1], [0, 0, 0]), lambda: Cylinder([0, 0, 0], [0, 0, 0], 1, 1),
                 lambda: OrientedBox([0, 0, 0], np.ones((3, 3)), [1, 1, 1]),
                 lambda: Slab([0, 0, 1], 2, 1), lambda: PlanPolygon([[0, 0], [1, 1]]),
                 lambda: PlanPolygon([[0, 0], [1, 0], [0, 1]], 5, 1)):
        try:
            make()
        except ValueError:
            continue
        raise AssertionError("a bad shape was accepted")
    log("  malformed shapes are refused")


if __name__ == "__main__":
    try:
        import cupy
        cupy.zeros(1)
    except Exception as exc:
        print(f"Shape query tests need CuPy and a GPU ({type(exc).__name__}); skipped.")
        sys.exit(0)

    log("=== shape_query_test start")
    try:
        test_bad_shapes_are_refused()
        test_every_shape_matches_the_oracle()
        test_the_three_search_paths()
        test_points_on_cell_borders_are_counted_once()
        test_edges_are_decided_the_same_way_every_time()
        test_a_concave_fence_does_not_cover_its_notch()
        test_empty_outside_and_open_shapes()
        test_non_finite_points_are_never_returned()
        test_views_of_one_cloud_share_one_preparation()
        test_release_frees_the_gpu()
        test_a_cell_is_built_once_when_two_threads_reach_it()
        test_preparing_a_cloud_does_not_copy_it_into_host_ram()
    except BaseException:
        import traceback
        log("=== FAILED\n" + traceback.format_exc())
        raise
    log("=== All shape query tests passed.")
