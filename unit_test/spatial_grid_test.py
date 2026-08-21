# spatial_grid_test.py
#
# Tests for core.services.spatial_grid, the shared point index.
#
# It grew out of the viewer's pick grid, which was one coarse unsorted grid
# tuned for a single mouse click. It is now a service two very different callers
# share: the viewer still wants 242 cells and no sorting, while an algorithm
# wants cells sized to its query radius and the rows bucketed so a lookup is a
# slice instead of a scan.
#
# So most of what is checked here is that those two configurations are the same
# index underneath — same cell contents, same nearest point, same answer from
# the CPU and the GPU — and that the viewer's preset did not move while the
# service was generalised.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.services.spatial_grid import (
    DEFAULT_TARGET_CELLS, PICK_GRID_SHAPE, UINT8_CELL_LIMIT, UINT16_CELL_LIMIT,
    SpatialGrid, _bounds,
)
from plugins.backends.grid_backends import NumpyGrid

_RNG = np.random.default_rng(20260821)


def _cloud(n=200_000):
    """A street-shaped slab: much wider than it is tall, like a survey box."""
    return np.column_stack([
        _RNG.random(n) * 200.0,
        _RNG.random(n) * 30.0,
        _RNG.random(n) * 8.0,
    ]).astype(np.float32)


def _gpu_backend():
    """The CuPy grid backend, or None when this machine has no usable GPU."""
    try:
        import cupy  # noqa: F401
        from plugins.backends.grid_backends import CuPyGrid
        cupy.zeros(1)
        return CuPyGrid()
    except Exception:
        return None


def test_viewer_preset_is_unchanged():
    """The default is still the pick grid: 242 cells, one byte, no sorting.

    Generalising the service must not quietly change what the viewer builds —
    the one-byte cell number is what keeps the index at 0.17 GB rather than
    0.68 GB on a 170M point cloud, and sorting is deliberately skipped because
    nobody can perceive it on a mouse click.
    """
    grid = SpatialGrid.build(_cloud())
    assert grid.shape == PICK_GRID_SHAPE, grid.shape
    assert grid.n_cells <= UINT8_CELL_LIMIT
    assert grid.cell_ids.dtype == np.uint8, grid.cell_ids.dtype
    assert not grid.sorted
    assert grid.order is None and grid.starts is None
    print(f"  viewer preset: {grid.shape}, {grid.n_cells} cells, "
          f"{grid.cell_ids.dtype}, sorted={grid.sorted}")


def test_cell_size_gives_a_finer_grid():
    """An algorithm can still ask by cell size when it knows its query radius."""
    grid = SpatialGrid.build(_cloud(), cell_size=2.0, sort=True)

    assert grid.n_cells > UINT8_CELL_LIMIT
    assert grid.sorted
    # int32, not the default int64 — it halves the index on a large cloud.
    assert grid.order.dtype == np.int32, grid.order.dtype
    assert len(grid.starts) == grid.n_cells + 1
    assert grid.starts[-1] == grid.n_points

    # Cells come out no larger than asked.
    assert np.all(grid.step <= 2.0 + 1e-6), grid.step
    print(f"  cell_size=2.0 -> {grid.shape}, {grid.n_cells:,} cells, "
          f"order {grid.order.dtype}")


def test_target_cells_never_exceeds_its_ceiling():
    """The count is a cap, because it decides the width of the cell number.

    One cell over 65,536 and the index goes from two bytes a point to four —
    on a 168M cloud that is 337 MB against 674 MB. Rounding each axis to
    nearest used to overshoot: a 200 x 30 x 8 m box asking for 65,536 came back
    with 65,934 and quietly cost the extra byte.
    """
    for box in ([200., 30., 8.], [1000., 1000., 30.], [50., 50., 50.],
                [300., 2., 10.], [1000., 1., 1.]):
        points = (_RNG.random((40_000, 3)) * np.asarray(box)).astype(np.float32)
        grid = SpatialGrid.build(points, target_cells=UINT16_CELL_LIMIT)
        assert grid.n_cells <= UINT16_CELL_LIMIT, (box, grid.n_cells)
        assert grid.cell_ids.dtype == np.uint16, (box, grid.cell_ids.dtype)
        print(f"  {str(box):<22} -> {str(grid.shape):>16} "
              f"{grid.n_cells:>7,} cells, uint16")


def test_every_cell_width_is_reachable():
    """uint8, uint16 and int32 in turn — the jump from 1 to 4 was the bug."""
    points = _cloud(60_000)
    widths = {}
    for target in (200, 60_000, 400_000):
        grid = SpatialGrid.build(points, target_cells=target)
        widths[str(grid.cell_ids.dtype)] = grid.n_cells
    assert set(widths) == {"uint8", "uint16", "int32"}, widths
    print(f"  reachable widths: "
          f"{', '.join(f'{k} at {v:,} cells' for k, v in widths.items())}")


def test_a_sorted_grid_drops_its_cell_ids():
    """Sorting replaces cell_ids rather than adding to it.

    Nothing reads the per-point cell number once the rows are bucketed, so
    keeping it would be 2-4 bytes a point of pure weight — half the index.
    """
    points = _cloud()
    grid = SpatialGrid.build(points, target_cells=DEFAULT_TARGET_CELLS, sort=True)

    assert grid.cell_ids is None
    assert grid.n_points == len(points)          # not len(cell_ids) any more

    # And it still answers everything.
    centre = points[123].astype(np.float64)
    assert grid.rows_near(centre, 1.0).size > 0
    assert grid.rows_in_cell(grid.cell_number(grid.cell_of(centre))).size > 0
    row, _sq = grid.nearest(points, centre)
    assert row == 123

    per_point = grid.index_nbytes() / grid.n_points
    print(f"  sorted grid: cell_ids is None, {per_point:.2f} bytes/point "
          f"(order only; starts is per cell)")


def test_the_three_sizing_arguments_are_mutually_exclusive():
    pairs = [("shape", "cell_size"), ("shape", "target_cells"),
             ("cell_size", "target_cells")]
    values = {"shape": (4, 4, 4), "cell_size": 2.0, "target_cells": 1000}
    for a, b in pairs:
        try:
            SpatialGrid.build(_cloud(1000), **{a: values[a], b: values[b]})
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {a} + {b}")
    print("  shape / cell_size / target_cells all refuse each other")


def test_sorting_does_not_change_what_is_in_a_cell():
    """Bucketing is an index layout, not a different answer."""
    points = _cloud()
    sorted_grid = SpatialGrid.build(points, cell_size=2.0, sort=True)
    plain_grid = SpatialGrid.build(points, cell_size=2.0, sort=False)

    checked = 0
    for cell in _RNG.integers(0, sorted_grid.n_cells, 60):
        a = set(sorted_grid.rows_in_cell(int(cell)).tolist())
        b = set(plain_grid.rows_in_cell(int(cell)).tolist())
        assert a == b, f"cell {cell} differs"
        checked += len(a)
    print(f"  60 cells hold the same rows either way ({checked:,} points)")


def test_rows_near_never_misses_a_point_in_the_ball():
    """The grid narrows; the caller measures.

    ``rows_near`` returns whole cells, so it is allowed to hand back points
    outside the ball — but it must never drop one inside it, or a query built
    on top of it silently loses points near a cell wall.
    """
    points = _cloud()
    grids = {
        "sorted": SpatialGrid.build(points, cell_size=2.0, sort=True),
        "unsorted": SpatialGrid.build(points, cell_size=2.0, sort=False),
        "viewer": SpatialGrid.build(points),
    }

    for radius in (0.15, 1.5, 6.0):
        for _ in range(15):
            centre = points[_RNG.integers(0, len(points))].astype(np.float64)
            inside = set(np.flatnonzero(
                np.linalg.norm(points - centre, axis=1) <= radius).tolist())
            for name, grid in grids.items():
                got = set(grid.rows_near(centre, radius).tolist())
                missing = inside - got
                assert not missing, f"{name} r={radius} lost {len(missing)} points"
    print("  135 ball queries across 3 radii and 3 grids: nothing lost")


def test_a_query_outside_the_cloud_is_answered_not_crashed():
    points = _cloud(5_000)
    for grid in (SpatialGrid.build(points),
                 SpatialGrid.build(points, cell_size=2.0, sort=True)):
        rows = grid.rows_near([1e4, -1e4, 500.0], 1.0)
        assert isinstance(rows, np.ndarray)
        # Cells are clamped, so an edge cell can come back — but nothing in it
        # is actually within reach, which is the caller's job to notice.
        row, sq = grid.nearest(points, [1e4, -1e4, 500.0], max_dist=1.0)
        assert row is None and sq == np.inf, (row, sq)
    print("  a target far outside the box: no crash, and nothing within reach")


def test_nearest_agrees_with_brute_force():
    """The viewer's click path, on every configuration of the service."""
    points = _cloud(120_000)
    grids = {
        "viewer": SpatialGrid.build(points),
        "fine-sorted": SpatialGrid.build(points, cell_size=2.0, sort=True),
        "fine-unsorted": SpatialGrid.build(points, cell_size=2.0, sort=False),
    }
    targets = np.column_stack([
        _RNG.random(150) * 200.0,
        _RNG.random(150) * 30.0,
        _RNG.random(150) * 8.0,
    ])

    for name, grid in grids.items():
        for target in targets:
            row, _sq = grid.nearest(points, target)
            truth = int(np.argmin(np.linalg.norm(points - target, axis=1)))
            assert row == truth, f"{name}: got {row}, brute force says {truth}"
    print(f"  {len(targets)} targets x {len(grids)} grids all match brute force")


def test_z_run_gather_matches_a_per_cell_gather():
    """The Z-run shortcut must be a layout trick, not a different answer.

    Cell numbers are ``(ix * ny + iy) * nz + iz``, so a column of cells in Z is
    a contiguous range and one slice of ``order`` covers it. That is what made
    cell size a free choice — fetching cell by cell cost 22 us at 6,000 cells
    but 317 us at 8M, so a grid fine enough to keep the candidate list short was
    slower than the coarse one it replaced.
    """
    points = _cloud()
    grid = SpatialGrid.build(points, target_cells=DEFAULT_TARGET_CELLS, sort=True)

    def per_cell(low, high):
        """What the old code did: one slice per cell, in cell-number order."""
        rows = []
        for ix in range(low[0], high[0] + 1):
            for iy in range(low[1], high[1] + 1):
                for iz in range(low[2], high[2] + 1):
                    rows.append(grid.rows_in_cell(grid.cell_number((ix, iy, iz))))
        rows = [r for r in rows if r.size]
        return np.concatenate(rows) if rows else np.empty(0, np.int32)

    nx, ny, nz = grid.shape
    boxes = [
        ((0, 0, 0), (0, 0, 0)),                      # a single cell
        ((5, 5, 0), (5, 5, nz - 1)),                 # a whole Z column
        ((0, 0, 0), (nx - 1, ny - 1, nz - 1)),       # the entire grid
    ]
    for _ in range(25):                              # and random boxes
        low = [int(_RNG.integers(0, n)) for n in (nx, ny, nz)]
        high = [int(_RNG.integers(lo, n)) for lo, n in zip(low, (nx, ny, nz))]
        boxes.append((tuple(low), tuple(high)))

    biggest = 0
    for low, high in boxes:
        want = per_cell(low, high)
        got = grid._rows_in_runs(*grid._z_runs(np.array(low), np.array(high)))
        assert np.array_equal(np.sort(want), np.sort(got)), (low, high)
        biggest = max(biggest, want.size)
    print(f"  {len(boxes)} cell boxes gather identically by run and by cell "
          f"(largest {biggest:,} rows)")


def test_block_bounds_matches_the_masked_reduction():
    """The fast path takes a plain min/max and only redoes a bad block.

    Plain reductions are ~1.7x faster than masked ones (3.30 s against 1.89 s
    over 50M points), and real clouds almost never hold a NaN — so the masked
    form should be the exception, not the rule. It still has to give the same
    answer when one turns up.
    """
    backends = [("CPU", NumpyGrid())]
    gpu = _gpu_backend()
    if gpu is not None:
        backends.append(("GPU", gpu))

    clean = _cloud(50_000)
    dirty = clean.copy()
    dirty[3, 1] = np.nan
    dirty[99, 0] = np.inf
    dirty[100, 2] = -np.inf
    all_bad = np.full((1_000, 3), np.nan, dtype=np.float32)

    def masked(block):
        finite = np.isfinite(block[:, :3])
        return (np.min(block[:, :3], axis=0, where=finite, initial=np.inf),
                np.max(block[:, :3], axis=0, where=finite, initial=-np.inf))

    for name, backend in backends:
        for label, block in (("clean", clean), ("nan+inf", dirty),
                             ("all non-finite", all_bad)):
            lo, hi = backend.block_bounds(block)
            want_lo, want_hi = masked(block)
            assert np.array_equal(lo, want_lo.astype(np.float32)), (name, label)
            assert np.array_equal(hi, want_hi.astype(np.float32)), (name, label)
        # and the whole-cloud helper agrees across backends
        a = _bounds(dirty, 8_000, backend)
        b = _bounds(dirty, 8_000, NumpyGrid())
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), name
        print(f"  {name}: clean, nan+inf and all-non-finite blocks all match")


def test_argsort_buckets_identically_on_cpu_and_gpu():
    """Different sorts may break ties differently; the buckets must not differ."""
    gpu = _gpu_backend()
    ids = _RNG.integers(0, 5_000, 400_000).astype(np.uint16)

    cpu_order = NumpyGrid().argsort(ids)
    assert cpu_order.dtype == np.int32, cpu_order.dtype
    assert np.array_equal(np.sort(cpu_order), np.arange(len(ids)))
    # Sorted by cell is the whole contract.
    assert np.all(np.diff(ids[cpu_order].astype(np.int64)) >= 0)

    if gpu is None:
        print("  CPU order is a stable permutation sorted by cell "
              "(GPU skipped: none available)")
        return

    gpu_order = gpu.argsort(ids)
    assert gpu_order.dtype == np.int32, gpu_order.dtype
    assert np.array_equal(ids[cpu_order], ids[gpu_order]), \
        "the two orders put different cells in different places"
    assert np.array_equal(cpu_order, gpu_order), \
        "both sorts claim to be stable, so they should agree exactly"
    print(f"  CPU and GPU orders agree exactly over {len(ids):,} points")


def test_cpu_and_gpu_number_cells_identically():
    """Both cell widths, because the GPU kernel is compiled per output type.

    The narrow kernel is what the viewer uses and was the only one that existed;
    the wide one is new, so this is the check that the second compilation says
    the same thing as the first.
    """
    gpu = _gpu_backend()
    if gpu is None:
        print("  skipped: no usable GPU on this machine")
        return

    points = _cloud()
    for label, kwargs in (("uint8", {"target_cells": 200}),
                          ("uint16", {"target_cells": UINT16_CELL_LIMIT}),
                          ("int32", {"target_cells": 400_000})):
        cpu_grid = SpatialGrid.build(points, backend=NumpyGrid(), **kwargs)
        gpu_grid = SpatialGrid.build(points, backend=gpu, **kwargs)
        assert np.array_equal(cpu_grid.cell_ids, gpu_grid.cell_ids), label
        print(f"  CPU == GPU for the {label} grid "
              f"({cpu_grid.n_cells:,} cells)")


def test_an_unsupported_output_width_is_refused():
    """Better a loud TypeError than a silently truncated cell number."""
    out = np.empty(4, dtype=np.int64)
    block = np.zeros((4, 3), dtype=np.float32)
    try:
        NumpyGrid().cell_ids(block, np.zeros(3, np.float32),
                             np.ones(3, np.float32), (2, 2, 2), out)
    except TypeError:
        print("  an int64 out array is refused rather than truncated")
        return
    raise AssertionError("expected TypeError for an int64 out array")


def test_degenerate_clouds():
    """Empty, single point, and a flat axis — all of which real data produces."""
    assert SpatialGrid.build(np.empty((0, 3), dtype=np.float32)) is None

    one = np.zeros((1, 3), dtype=np.float32)
    grid = SpatialGrid.build(one, cell_size=1.0, sort=True)
    row, sq = grid.nearest(one, [5.0, 5.0, 5.0])
    assert row == 0, row
    assert abs(sq - 75.0) < 1e-3, sq

    flat = _cloud(10_000)
    flat[:, 2] = 3.0                       # no extent in Z at all
    flat_grid = SpatialGrid.build(flat, cell_size=2.0, sort=True)
    assert flat_grid.shape[2] == 1, flat_grid.shape
    assert np.all(np.isfinite(flat_grid.step))
    print("  empty -> None; one point; a flat axis collapses to one cell")


def test_one_bad_coordinate_does_not_collapse_the_grid():
    """A single NaN used to make the bounding box NaN and every cell useless."""
    points = _cloud(50_000)
    clean = SpatialGrid.build(points, cell_size=2.0, sort=True)

    dirty = points.copy()
    dirty[17, 1] = np.nan
    dirty[999, 0] = np.inf
    grid = SpatialGrid.build(dirty, cell_size=2.0, sort=True)

    assert grid.n_cells == clean.n_cells, (grid.n_cells, clean.n_cells)
    assert np.all(np.isfinite(grid.lo)) and np.all(np.isfinite(grid.step))
    # The bad points land somewhere valid rather than corrupting the index.
    # Read through an unsorted twin, since a sorted grid drops cell_ids.
    unsorted = SpatialGrid.build(dirty, cell_size=2.0, sort=False)
    assert unsorted.cell_ids[17] < unsorted.n_cells
    assert unsorted.cell_ids[999] < unsorted.n_cells
    # Every row is still reachable through the buckets.
    assert grid.starts[-1] == len(dirty)
    # And they are never the nearest point to anything.
    row, _sq = grid.nearest(dirty, dirty[500, :3].astype(np.float64))
    assert row not in (17, 999)
    print(f"  one NaN and one inf: still {grid.n_cells:,} cells, "
          f"and neither is ever picked")


if __name__ == "__main__":
    test_viewer_preset_is_unchanged()
    test_cell_size_gives_a_finer_grid()
    test_target_cells_never_exceeds_its_ceiling()
    test_every_cell_width_is_reachable()
    test_a_sorted_grid_drops_its_cell_ids()
    test_the_three_sizing_arguments_are_mutually_exclusive()
    test_sorting_does_not_change_what_is_in_a_cell()
    test_z_run_gather_matches_a_per_cell_gather()
    test_block_bounds_matches_the_masked_reduction()
    test_argsort_buckets_identically_on_cpu_and_gpu()
    test_rows_near_never_misses_a_point_in_the_ball()
    test_a_query_outside_the_cloud_is_answered_not_crashed()
    test_nearest_agrees_with_brute_force()
    test_cpu_and_gpu_number_cells_identically()
    test_an_unsupported_output_width_is_refused()
    test_degenerate_clouds()
    test_one_bad_coordinate_does_not_collapse_the_grid()
    print("\nAll spatial grid tests passed.")
