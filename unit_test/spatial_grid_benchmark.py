# spatial_grid_benchmark.py
#
# Measures core.services.spatial_grid on a real point cloud, because every
# design decision in it was made from measurements and none of them should be
# taken on trust at a size nobody tested.
#
# Reports, for each grid configuration and backend:
#   build time, split into bounding box / cell numbering / sort
#   index size, peak host RAM, peak VRAM
#   rows_near  — the query an algorithm makes thousands of times
#   nearest    — the query the viewer makes once per mouse click
#
# Usage:
#     python unit_test/spatial_grid_benchmark.py "Middle Head-Part 2 - 168M.ply"
#     python unit_test/spatial_grid_benchmark.py <file.ply> --limit 20000000
#
# Reads only xyz out of the file. The 168M cloud is 5.2 GB on disk because it
# also carries colour, normals and intensity; xyz alone is 1.88 GB, and pulling
# just that is the difference between fitting in memory and not.
import os
import sys
import gc
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.services.compute_backend import DEFAULT_BLOCK
from core.services.spatial_grid import (
    DEFAULT_TARGET_CELLS, SpatialGrid, _bounds, _bucket, _cell_dtype,
    _shape_for_cell_size, _shape_for_target_cells, PICK_GRID_SHAPE,
)
from plugins.backends.grid_backends import NumpyGrid

MB = 1024 ** 2
GB = 1024 ** 3


def _psutil():
    try:
        import psutil
        return psutil.Process(os.getpid())
    except Exception:
        return None


def _cupy():
    try:
        import cupy
        cupy.zeros(1)
        return cupy
    except Exception:
        return None


PROC = _psutil()
CP = _cupy()


class Watch:
    """Peak host RAM and device VRAM over the lifetime of a `with` block."""

    def __init__(self, period=0.002):
        self.period = period
        self.rss = self.vram = 0

    def __enter__(self):
        gc.collect()
        if CP is not None:
            CP.get_default_memory_pool().free_all_blocks()
        self._rss0 = PROC.memory_info().rss if PROC else 0
        self._vram0 = self._vram_now()
        self._stop = False
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self.seconds = time.perf_counter()
        return self

    def _vram_now(self):
        if CP is None:
            return 0
        free, total = CP.cuda.Device().mem_info
        return total - free

    def _sample(self):
        while not self._stop:
            if PROC:
                self.rss = max(self.rss, PROC.memory_info().rss - self._rss0)
            self.vram = max(self.vram, self._vram_now() - self._vram0)
            time.sleep(self.period)

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.seconds
        self._stop = True
        self._thread.join()


def read_ply_xyz(path, limit=None, block=8_000_000):
    """xyz only, as (N, 3) float32 shifted to the origin.

    Parses the header rather than assuming it, then memory-maps the body with
    the exact per-point dtype and copies out the three fields it wants. Using
    plyfile here would materialise every property — 5.2 GB against 1.88 GB on
    the 168M cloud — and the extra 3.3 GB is colour and normals nothing in this
    benchmark looks at.

    Shifted to the origin because that is what the importers do
    (``PointCloud.translation``), and it is what makes float32 safe.
    """
    ply_to_numpy = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
        "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
    }

    fields, n_points = [], None
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header has no end_header")
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == b"format":
                fmt = parts[1].decode()
            elif key == b"element" and parts[1] == b"vertex":
                n_points = int(parts[2])
            elif key == b"property" and n_points is not None:
                if parts[1] == b"list":
                    raise ValueError("list properties are not supported here")
                fields.append((parts[2].decode(), ply_to_numpy[parts[1].decode()]))
            elif key == b"end_header":
                offset = handle.tell()
                break
    if fmt != "binary_little_endian":
        raise ValueError(f"need binary_little_endian, got {fmt}")
    for axis in ("x", "y", "z"):
        if axis not in [name for name, _ in fields]:
            raise ValueError(f"PLY has no '{axis}' property")

    dtype = np.dtype(fields)
    total = n_points if limit is None else min(n_points, int(limit))
    print(f"  {os.path.basename(path)}: {n_points:,} points, "
          f"{dtype.itemsize} bytes each, data at byte {offset}")
    if total != n_points:
        print(f"  reading the first {total:,}")

    mapped = np.memmap(path, dtype=dtype, mode="r", offset=offset,
                       shape=(n_points,))
    xyz = np.empty((total, 3), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, total, block):
        stop = min(start + block, total)
        chunk = mapped[start:stop]
        for axis, name in enumerate(("x", "y", "z")):
            xyz[start:stop, axis] = chunk[name]
    del mapped

    low = _bounds(xyz, DEFAULT_BLOCK, NumpyGrid())[0]
    xyz -= low
    print(f"  read {xyz.nbytes / GB:.2f} GB of xyz in "
          f"{time.perf_counter() - started:.1f} s, shifted to the origin; "
          f"extent {np.round(xyz.max(axis=0), 1)}")
    return xyz


def build_by_phase(points, backend, shape=None, cell_size=None,
                   target_cells=None, sort=False, block=DEFAULT_BLOCK):
    """SpatialGrid.build, with each phase timed separately.

    Deliberately mirrors ``build`` rather than calling it — the whole point is
    to see which phase costs what, and build only returns the finished grid.
    """
    n = len(points)
    started = time.perf_counter()
    lo, hi = _bounds(points, block, backend)
    t_bounds = time.perf_counter() - started

    span = hi - lo
    span[span <= 0] = np.float32(1.0)
    if cell_size is not None:
        shape = _shape_for_cell_size(span, cell_size)
    elif target_cells is not None:
        shape = _shape_for_target_cells(span, target_cells)
    elif shape is None:
        shape = PICK_GRID_SHAPE
    shape = tuple(int(v) for v in shape)
    step = (span / np.asarray(shape, dtype=np.float32)).astype(np.float32)
    inv_step = (np.float32(1.0) / step).astype(np.float32)
    n_cells = int(np.prod(shape))

    started = time.perf_counter()
    cell_ids = np.empty(n, dtype=_cell_dtype(n_cells))
    for start in range(0, n, block):
        stop = min(start + block, n)
        backend.cell_ids(points[start:stop], lo, inv_step, shape,
                         cell_ids[start:stop])
    t_cells = time.perf_counter() - started

    order = starts = None
    t_sort = 0.0
    if sort:
        started = time.perf_counter()
        order, starts = _bucket(cell_ids, n_cells, backend)
        t_sort = time.perf_counter() - started
        cell_ids = None

    grid = SpatialGrid(cell_ids, lo, step, shape, n, order, starts)
    return grid, t_bounds, t_cells, t_sort


def free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / GB
    except Exception:
        return float("nan")


def run(points, radius):
    n = len(points)
    rng = np.random.default_rng(0)
    targets = points[rng.integers(0, n, 200)].astype(np.float64)

    configs = [
        ("viewer  11x11x2 unsorted", dict()),
        (f"algo    {DEFAULT_TARGET_CELLS:,} sorted",
         dict(target_cells=DEFAULT_TARGET_CELLS, sort=True)),
    ]
    backends = [("CPU", NumpyGrid())]
    if CP is not None:
        from plugins.backends.grid_backends import CuPyGrid
        backends.append(("GPU", CuPyGrid()))

    header = (f"{'grid':<26} {'be':<4} {'build':>7} {'bounds':>7} {'cells':>7} "
              f"{'sort':>7} {'cells#':>9} {'B/pt':>6} {'index':>9} "
              f"{'pkRAM':>8} {'pkVRAM':>8} {'near':>9} {'click':>9}")
    print("\n" + header)
    print("-" * len(header))

    for label, kwargs in configs:
        for name, backend in backends:
            if free_ram_gb() < 1.5:
                print(f"  stopping: only {free_ram_gb():.1f} GB RAM free")
                return
            with Watch() as watch:
                grid, t_bounds, t_cells, t_sort = build_by_phase(
                    points, backend, **kwargs)

            per_point = grid.index_nbytes() / n
            n_click = 200 if grid.sorted else 20

            started = time.perf_counter()
            returned = 0
            for centre in targets:
                returned += grid.rows_near(centre, radius).size
            t_near = (time.perf_counter() - started) / len(targets) * 1e6

            started = time.perf_counter()
            for centre in targets[:n_click]:
                grid.nearest(points, centre)
            t_click = (time.perf_counter() - started) / n_click * 1e3

            print(f"{label:<26} {name:<4} {watch.seconds:>7.2f} {t_bounds:>7.2f} "
                  f"{t_cells:>7.2f} {t_sort:>7.2f} {grid.n_cells:>9,} "
                  f"{per_point:>6.2f} {grid.index_nbytes() / MB:>8.0f}M "
                  f"{watch.rss / MB:>7.0f}M {watch.vram / MB:>7.0f}M "
                  f"{t_near:>8.0f}u {t_click:>8.1f}m")
            print(f"{'':<31} -> {returned / len(targets):,.0f} points per "
                  f"rows_near, cell {np.round(grid.step, 2)} m, "
                  f"{n / grid.n_cells:,.0f} points per cell")
            del grid
            gc.collect()
            if CP is not None:
                CP.get_default_memory_pool().free_all_blocks()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit"):
            limit = int(arg.split("=", 1)[1] if "=" in arg
                        else sys.argv[sys.argv.index(arg) + 1])
    if limit is not None:
        args = [a for a in args if not a.isdigit()]

    if not args:
        print(__doc__ or "")
        print("No PLY given — nothing to measure. Pass a file path.")
        return 0
    path = args[0]
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return 1

    print(f"RAM free at start: {free_ram_gb():.1f} GB")
    print(f"GPU: {'yes' if CP is not None else 'no'}")
    points = read_ply_xyz(path, limit=limit)
    print(f"RAM free after load: {free_ram_gb():.1f} GB")

    extent = float(np.max(points.max(axis=0)))
    radius = max(round(extent / 400.0, 2), 0.05)
    print(f"\nrows_near radius: {radius} m "
          f"(a realistic tube for a {extent:,.0f} m site)")
    run(points, radius)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
