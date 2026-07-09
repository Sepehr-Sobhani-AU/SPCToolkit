# Linear Region Growing

The linear counterpart to [Surface Region Growing](SURFACE_REGION_GROWING.md). Picked points on a 1-D linear feature (cable, pipe, rail, kerb, edge) become seeds; the plugin grows each feature outward from its seeds and returns one cluster per feature plus a remaining-points mask.

- **Plugin (UI / orchestration):** `040_linear_region_growing_plugin.py` → `"linear_region_growing"` (Action plugin, consumes the live viewer selection).
- **Engine (the growth itself):** `core/services/linear_region_grower.py` → `LinearRegionGrower`. Line fitting is delegated to the shared RANSAC infrastructure (`core/services/ransac.fit`, `model_type="line"`); everything else is orchestration on top of it, per `DECISIONS.md` 2026-05-26.

The same `LinearRegionGrower` (in `AXIS_TRACE` mode) backs the `power_line_detection` plugin — power-line tracing is just one preset of this generic grower.

---

## Inputs and output

**Inputs (from the application):**
- Exactly one `PointCloud` branch selected.
- At least two seed points picked in the viewer (polygon select with `P`, or Shift+Click) lying along one linear feature.
- For the linearity-based modes only: per-point **eigenvalues** on the selected branch (run **Compute Eigenvalues** first, select the eigenvalues node). Linearity is *consumed* from them, never recomputed here.

**Output:**
- A single `Clusters` branch (`linear_region_growing`) over the input cloud: label `0` = the grown feature, `-1` = everything else — the same output shape as [Surface Region Growing](SURFACE_REGION_GROWING.md), ready to classify (cluster → class → DXF layer).
- If **Show Search Cylinders** / **Show Centerlines** are ticked, render-only `vector_feature` branches (`search_cylinders`, `centerlines`) are added under the result and shown — wireframe geometry, fully controllable in the tree (toggle, delete) like any other branch. Everything drawn lives in a branch; nothing is an ad-hoc viewer overlay.

One selection grows one feature. To trace several features, run the plugin once per feature. After running, the input branch is hidden and the result is shown.

**Progress & cancellation:** growing runs on a background thread (like [Surface Region Growing](SURFACE_REGION_GROWING.md)). While it runs the menus and tree are disabled, a status-bar progress bar reports how many lines have been traced, and a **Cancel** button stops the trace early — the lines grown so far are still saved as a result branch, and the completion message notes that it was cancelled. The 3D viewer stays interactive for camera manipulation throughout.

---

## Growth modes

Selected with the `growth_mode` parameter. All three start from the picked seed points and call the line-RANSAC engine; they differ only in how the region expands.

### Axis Trace (`axis_trace`)
Raw points only — needs no upstream features. Best for **isolated thin features** (cables, pipes, rails).

1. Take the seeds' dominant (**PCA**) axis — used **only** to order the seeds along the feature and locate the span centre, never as a march heading. (A RANSAC line is the wrong tool here: seeds spanning a *curved* feature aren't collinear, so the fit fails outright and nothing grows. PCA always returns a usable sort axis.)
2. Anchor at the seed nearest the span centre; take the initial heading from a line fit to the dense **cloud** within one `cylinder_length` of it — a genuine local tangent, independent of how sparsely the seeds were picked.
3. **March a search cylinder outward from the anchor in both directions.** Each pass traverses half the seed body, reaches the far end, and continues past it, so the picked points obey the *same* cylinder rule as the growth (no single straight fit across the seeds); the two passes share the anchor and tile into one continuous chain. Per step, along the current direction:
   - Ball-query the KD-tree at a point half a cylinder-length ahead, then keep candidates inside the forward half-cylinder (`0 < along < cylinder_length`, perpendicular distance `< cylinder_radius`).
   - Re-fit a line to the cylinder contents; flip it to keep pointing forward.
   - **Stop** if the direction change exceeds `max_angle` (a bend — e.g. a pole or feature end), fewer than `min_points` fall in the cylinder, or the cylinder is empty.
   - Collect the inliers, then advance the tip by `(1 − cylinder_overlap%)` of a cylinder length, **re-projecting it onto the freshly-fit local axis** so the march stays on the feature through curves. **Show Search Cylinders** draws each step's *actual* selection cylinder (full `cylinder_length` × `cylinder_radius`, along the search direction), faithfully showing where points were selected — so consecutive cylinders overlap when `cylinder_overlap > 0` and step across bends. Repeat (capped at `max_steps = 500` per direction).

### Linearity-Connected (`linearity_connected`)
Requires precomputed linearity. Best for **edges/kerbs embedded in a surface**, where an axis cylinder would leak into the surrounding plane.

Breadth-first expansion over the KD-tree from the seeds: pop a point, query its `neighbor_k` nearest neighbours, and admit any unvisited neighbour whose `linearity ≥ linearity_threshold` into the region (and onto the queue). Traversal only walks through linear points, so it stays on the feature and the queue stays bounded by the feature's size.

### Hybrid (`hybrid`)
The Axis-Trace march with the linearity gate additionally applied to candidate points before the per-step line fit. Combines directional ordering with surface-leak resistance — also requires precomputed linearity.

---

## Parameters

| Parameter             | Default | Role                                                                 |
|-----------------------|---------|----------------------------------------------------------------------|
| `growth_mode`         | Axis Trace | Axis Trace / Linearity-Connected / Hybrid (see above).            |
| `ransac_threshold`    | `0.03`  | Line-RANSAC inlier distance threshold (m).                          |
| `ransac_iterations`   | `100`   | Max RANSAC hypotheses per line fit (higher = more robust, slower).  |
| `cylinder_radius`     | `0.03`  | Axis-trace search cylinder radius per step (m).                     |
| `cylinder_length`     | `0.5`   | Axis-trace search cylinder length per step (m).                     |
| `cylinder_overlap`    | `0.0`   | Percent each step's cylinder overlaps the previous (0 = end-to-end … 90). Higher follows curves better, slower. |
| `min_points`          | `5`     | Stop the axis march below this many points in a cylinder.            |
| `max_angle`           | `20°`   | Max per-step direction change before the axis march stops.           |
| `linearity_threshold` | `0.4`   | Linearity / Hybrid: accept a point only above this linearity.        |
| `neighbor_k`          | `16`    | Linearity-Connected: k-NN used to expand the region.                |
| `show_cylinders`      | `off`   | Add the axis-trace search cylinders as a controllable wireframe branch (debug). |
| `show_lines`          | `off`   | Add the traced centerline as a controllable wireframe branch (debug). |

All picked points seed a single feature; RANSAC robustly ignores the occasional stray pick when fitting the initial line direction. To trace multiple features, run the plugin once per feature.

---

## Design lineage

- **`DECISIONS.md` 2026-05-26** — RANSAC is a single-cloud contract; region growing is orchestration on top. `LinearRegionGrower` is one of the orchestrators that decision predicted (the line analog of the plane one).
- Generalised from the original `power_line_tracer.py`, which was hardwired to cables and predated — and broke against — the 2026-05-26 RANSAC refactor. `power_line_detection` now calls `LinearRegionGrower(mode=AXIS_TRACE)`.
- Addresses the line side of `PROJECT.md`'s #1 open problem (generic line extraction), previously "solved only for power lines."
