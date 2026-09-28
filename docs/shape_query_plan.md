# Shape query service — implementation plan

Design and the reasons for it: `DECISIONS.md` § 2026-09-28. This file is the
*how*: files, steps, tests. The benchmark scripts behind the numbers are ported
into `unit_test/` in step 6.

## What a plugin sees

```python
from config.config import global_variables
from core.services.query_shapes import Cylinder

shapes = global_variables.global_shape_query
rows = shapes.points_in(point_cloud.points, Cylinder(a, d, radius=0.05, length=1.0))
# rows: np.ndarray int32, sorted ascending — rows of point_cloud.points inside
mask = shapes.points_in(point_cloud.points, shape, as_mask=True)   # (N,) bool
```

That is the whole surface. No build call, no release call: the first query on a
cloud prepares it, and everything is released when the plugin run ends or the
cloud is no longer referenced.

## Files

| File | New / changed | What |
|---|---|---|
| `core/services/query_shapes.py` | new | The shapes. Geometry only. |
| `core/services/shape_query_kernels.py` | new | CUDA source + the two kernels, loaded once. |
| `core/services/shape_query.py` | new | `ShapeQueryService`: per-cloud state, lazy per-cell index, query, release. |
| `config/config.py` | changed | `global_shape_query` attribute. |
| `main.py` | changed | Create the service next to the backend registry. |
| `application/analysis_executor.py` | changed | Release in a `finally` of `_run_in_thread`. |
| `unit_test/shape_query_test.py` | new | Correctness tests, logged. |
| `unit_test/shape_query_benchmark.py` | new | The 168M benchmark, logged. |
| `ARCHITECTURE.md` | changed | Short section: what it is, when to use it. |

## Step 1 — Shapes (`query_shapes.py`)

A small base class. Every shape gives:

- `aabb()` → `(lo, hi)` float32 on the host. May be infinite (slab); the service
  clips it to the cloud's bounds.
- `cell_states(cell_lo, cell_hi)` → per-cell MISS / PARTIAL / COVER for all 242
  cells at once (vectorised numpy). **COVER only when it is certain**: the cell's
  8 corners must be inside by a small margin, so rounding at an edge can never
  make a cell "covered" that holds an outside point.
- `kernel_kind` (int) and `kernel_params()` → float32 array the kernel reads.

Shapes in the first version:

| Shape | Parameters | COVER rule |
|---|---|---|
| `Sphere` | centre, radius | 8 corners inside (convex) |
| `Box` | min, max (axis-aligned) | cell box inside the box |
| `OrientedBox` | centre, rotation (3x3), half sizes | 8 corners inside (convex) |
| `Cylinder` | start point, direction, radius, length | 8 corners inside (convex) |
| `Slab` | normal, two offsets | 8 corners inside (convex) |
| `PlanPolygon` | XY polygon (any, not self-crossing), z min, z max | 4 XY corners inside **and** no polygon edge crosses the cell — it may be concave |

Shapes do **not** test points themselves. The kernel is the only definition of
"inside" (DECISIONS: one test, one truth).

## Step 2 — Kernels (`shape_query_kernels.py`)

Port from `compare_fused.py`, with two changes:

- Parameters come in as a device array of any length, so `PlanPolygon` can pass
  its vertices. The kernel copies up to a fixed size into shared memory; longer
  polygons read from global memory.
- Add `PlanPolygon` (even-odd crossing test over the vertices, then the z range)
  and `Slab` to the `inside()` device function.

Two kernels, as benchmarked:
- `gather_test`: candidates given as sub-cell runs → matching rows, appended with
  an atomic counter.
- `mask_all`: every point of a block → bool mask. Used when the candidate cells
  hold more than half the cloud.

Compiled with `--fmad=false`. Loaded lazily on first use and cached per process.

## Step 3 — The service (`shape_query.py`)

`ShapeQueryService`, one instance in `global_variables.global_shape_query`.

**Per-cloud state**, created on the first query for that cloud:
1. Check VRAM with `MemoryManager.can_use_gpu` for xyz + cell numbers; if it
   does not fit, raise a clear error (no CPU fallback).
2. Build the full-cloud coarse spatial index with
   `SpatialGrid.build_coarse_spatial_index(points, backend=CuPyGrid())`.
3. Upload xyz and the cell numbers **in 8M-point blocks** with `.set()`, then
   drop the host copy of the cell numbers.
4. Keep per-cell point counts and the 242 cell boxes (and corners) on the host.

State is keyed by the identity of the points array, with a `weakref` callback
that drops the state when the array is garbage-collected — this is the release
path for action plugins with their own threads. (`ndarray` supports weakrefs
but is not hashable, so it is a plain dict `id → (weakref, state)`.)

**Per-cell index** (`_CellIndex`, from `compare_percell.CellIndex`): built the
first time a query touches the cell. Rows sorted by sub-cell on the GPU, sub-cell
offsets on the host. One lock per cell, so two threads never build the same cell.

**Query** (`points_in`):
1. Clip the shape's box to the cloud's bounds; classify the 242 cells; drop empty cells.
2. If the candidate cells hold > 50% of the cloud → `mask_all` over the whole cloud.
3. Otherwise: covered cells' rows are taken whole; each partly covered cell →
   host works out the runs → one `gather_test` launch.
4. Join the pieces, **sort**, return as numpy int32 (or a mask when `as_mask=True`).

**Release** (`release_all()`): drops every cloud's state and frees CuPy's device
and pinned pools. Called from the `finally` of `AnalysisExecutor._run_in_thread`
(covers success, error and cancel, and pipeline replay, which runs through the
executor). Plugins never call it.

**Errors**: CuPy missing, no GPU, not enough VRAM, or OOM during a build → free
the pools, then raise `RuntimeError` saying which and how much was needed.

## Step 4 — Wiring

- `config/config.py`: `self.global_shape_query = None` with a one-line comment.
  Add it to the table in `CLAUDE.md` (Global Variables).
- `main.py`: create `ShapeQueryService()` after the backend registry. Creating it
  allocates nothing.
- `analysis_executor.py`: `finally: global_variables.global_shape_query.release_all()`
  (guarded for `None` in tests).

## Step 5 — Tests (`unit_test/shape_query_test.py`)

Every run writes `logs/shape_query_test.log`, flushed per line, with RAM/VRAM.

- Each shape against a numpy reference on synthetic clouds — a clustered cloud
  with one very dense cell and many empty ones, like the real data. Test points
  are kept away from shape edges so the numpy and GPU answers must be identical.
- Edge cases on exactly representable coordinates: a point on a cell border
  (counted once), on a shape's edge (the same answer every run).
- Paths: covered-cell path, partly-covered path, over-half → every-point path,
  shape fully outside the cloud, empty cloud, open slab.
- Concave `PlanPolygon` (an L shape) whose notch sits over a cell that must
  **not** be marked covered.
- Same query twice → identical array (order fixed).
- Lazy build: a small query builds only the cells it touches.
- Release: VRAM back to baseline after `release_all()`; state dropped when the
  points array is deleted (weakref).
- Blocked upload: host RAM does not grow by the cloud's size on upload.
- Two threads querying the same new cell → built once.

## Step 6 — Benchmark (`unit_test/shape_query_benchmark.py`)

Port `compare_fused.py` to call the real service. Writes
`logs/shape_query_benchmark.log`; run detached with a memory cap on the 168M
cloud. It should reproduce the DECISIONS numbers within noise; if it does not,
stop and look before going on.

## Step 7 — Docs

`ARCHITECTURE.md`: a short section — what the service answers, the API, its
lifetime, when to use it rather than `NeighborIndex` (shapes, any size) and when
not (k-nearest, which it does not do).

## Commits

One per step: shapes → kernels → service → wiring → tests → benchmark → docs.

## Not in this plan

- Combining shapes (union / difference) and many shapes in one call — deferred.
- Moving existing plugins or `NeighborIndex` users onto the service — each moves
  when a real scenario needs it.
- k-nearest queries.
