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

Several features can be traced from one selection — the picked points are grouped with DBSCAN (`seed_eps`) and each group grows its own line. After running, the input branch is hidden and the result is shown.

### One line per physical feature

DBSCAN routinely splits the picks on *one* feature into several groups (sparse picks, a small `seed_eps`), and growing each of those would draw duplicate lines on top of each other. The greedy loop in `grow_lines` prevents that without any merge heuristic:

1. Grow a line from the **largest** remaining seed group.
2. **Discard** that line if most of its points lie in the region already **swept** by the lines kept so far — it is retracing a feature that is already traced.
3. **Drop** every remaining seed group whose points mostly fall in the swept region — those groups were pieces of a feature already grown.
4. Repeat until no groups remain.

Both tests use the **swept region** (everything that fell inside a step's fit window) rather than the lines' member points. Membership is gated at `ransac_threshold` while the search tube is `cylinder_radius` wide, so a march can drive straight *through* a seed group and claim none of its points — testing membership would leave that group in the pool to regrow the same feature. Step 2 is the backstop for the case step 3 cannot see: a group that legitimately survives (it sits past the point where the first line stopped) and then marches *back* over that line, since every march runs both ways from its anchor.

Because a group is only ever dropped when growth genuinely reached it, parallel neighbouring features are never fused (growth never crosses the gap to them), collinear features separated by more than the search reach stay separate, and a feature split across many groups comes out whole.

### Extending traces that stopped short

Growth rarely reaches the end of every feature — a march gives up at an occlusion, a density drop, or a spurious bend. Rather than re-picking seeds and re-running (which produces a *competing* branch), the stops are handed back to you to walk through.

Every march end records **where** it stopped, **which way** it was heading, and **why**. After growing, the plugin offers to open **Extend Traced Lines**; the same window reopens later on a saved result via the `extend_traced_lines` plugin, since the stops and centerlines are persisted on the result branch (`Clusters.line_traces`).

For each stop the window brings it into view, marks it with a green "you are here" wireframe branch, and reports how many unclaimed points lie ahead:

- **Extend from picks** — pick a few points beyond the marker; growth re-runs seeded with **the line's own points around the stop ∪ your picks**, and the result is spliced into the existing line.
- **Undo** — put everything back as it was before the last change.
- **Real end** — the feature genuinely ends here; never ask again (survives save/reload).
- **Previous / Next** — move along the queue, leaving the stop undecided.
- **Discard last N cylinder(s)** — throw away the last N steps of march *before* growing again (default 0).

Stops are **ranked** by unclaimed points ahead, so the ones worth extending come first and genuine ends sink to the bottom. With up to two ends per line, thirty features produce sixty stops and most are real ends — ranking is the difference between reviewing eight and sixty.

The queue is ranked **once, on open**, and you keep your place as you work: settling a stop removes it and lands you on the next one. Re-ranking after every extension would reshuffle the list and bounce you back to the top each time you fixed something. The points-ahead figure for the current stop is recomputed live, so it stays honest as earlier extensions claim points.

**Extending leaves you on the same line, and leaves the camera alone.** The end the extension reached becomes the new marker, in the same place in the queue — so you can look at what your picks did, pick further ahead, and extend again, as many times as the feature needs. Nothing moves you off that line until you press **Real end**, **Previous** or **Next**. Only those three move the camera: you framed the view yourself to pick into, so extending and undoing leave it exactly where it was.

**Undo** reverses the last change — an extension or a "real end". Stepping past a stop is not a change, so **Previous** is how you go back to one.

All the debug wireframes you asked for at growth time — search cylinders, centerlines and the per-reason stop markers — are rebuilt after every change, so they follow the lines instead of stopping where the original run did. Stop markers are redrawn from the lines' **current** stops: a branch appears when a reason first occurs and is removed when its last stop is gone, so you never see a red "too few points" marker sitting in the middle of a line that now runs past it. The green marker still says which stop is under review.

Search cylinders are stored with the trace for the same reason, so reopening a saved result and extending it redraws the whole wireframe rather than just the new piece. A result saved before that was stored comes back without them, and its cylinder branch is left alone rather than being replaced by the extension's few.

### Only the points worth picking can be picked

In a real cloud there is far too much in the way to pick accurately — vegetation, poles, ground, the traced cable itself. So while you are working a stop, the points you could sensibly pick are put into **a cluster of their own**: bright yellow, and the only points in the cloud that will accept a click. Everything else stays dark grey and ignores the mouse entirely, so you can drag a polygon straight across the clutter and pick up only what you meant.

That falls out of what a cluster label already means, rather than being a display mode bolted on top. A point labelled `-1` is drawn grey **and** refused by the viewer's picking filters. Unclaimed points are `-1` — which is why, before this, they could not be picked at all and the workflow only functioned with the input cloud shown alongside the result, drawing every point twice.

The candidates are given a **name** as well as a label (`Pick candidates`). A growth result names its clusters (`Line 1`, `Line 2`), and for a named `Clusters` the renderer colours by name (`ClustersTransformer` → `get_named_colors`) and never reads the per-point colour array — so on exactly the branches this window opens on, painting the colour array yellow shows nothing at all and the candidates come out the unnamed 0.7-grey default. The name and its colour are removed again the moment the points stop being candidates, so no phantom class reaches classification or export.

**The window hides every other point cloud while it is open**, and shows them again when it closes. This is not tidiness: a second cloud on screen draws an unlabelled copy of the same points in the same place, and those copies take the clicks. Measured on the real controller — with the input cloud visible and selected in the tree, *every* one of its points is selectable and *none* of the result's are, so the candidates look right and clicking does nothing. If another point cloud does come back on screen, the window says so under the pick count.

- **Offer only points ahead, within [N] m** — the candidates are the unclaimed points inside a cone reaching that far along the stop's heading, plus a small ball at the marker. Raise the range to pick across a long occlusion; anything beyond it stays grey.
- Untick it to offer *every* unclaimed point, wherever it is — for the rare feature that turns hard at the stop.

The panel reports **two** numbers, and they are not the same thing. "*N* unclaimed points sit in the line's own search corridor ahead" counts the narrow tube the march itself would have searched — 9 m long and 0.6 m wide at the defaults, about 10 m³. "*N* points offered" counts the pick cone, which at the same 9 m is roughly 15× that volume, and grows as the cube of the range: at 24 m it is 157×. In open ground the two are nearly equal; in a tree canopy the cone can hold tens of thousands of points while the corridor holds a handful. The default range is set to the corridor's length so the two at least describe the same distance, and every point the first number counts is always among those offered. If the offer becomes a wall of yellow, shorten the range.

The cone is deliberately narrow (15°). Measured on a cable 8 m above ground with a tree beside the hole: a 45° cone leaves 36% of the cloud clickable, a 15° one leaves 2.3%, and both offer every point of the cable beyond the hole. Erring tight is also the cheaper mistake — you can see a point is not offered and widen the range, whereas too wide quietly gives the clutter back.

Points already on a traced line keep their cluster colour and stay clickable; they are already seeds, so picking one changes nothing. The candidate cluster is temporary — it is rebuilt as you step from stop to stop and erased when the window closes, so it never reaches classification or a saved project.

### Your picks are the decision

A point you picked is a point you looked at and judged to be on this line, so **the picks always join the line** — whether or not growth could get there by itself. The march is the bonus: it runs, and whatever it reaches is spliced in, but if it stops dead your picks are still adopted and the marker still moves out to them. The window says which happened ("grew +240 points" versus "growth could not carry on, so your 6 picked point(s) were added — pick further ahead and extend again"), so you always know whether the trace advanced on its own.

That is what makes the workflow always able to make progress. Across a long occlusion, or a cable strung through canopy, worst case you walk the feature yourself a few points at a time — which is slower than automatic growth but never dead-ends, and dead-ending is exactly what a short trace used to do.

For the same reason, `min_points` does not overrule you when bridging a gap. That gate exists to stop the march hopping blindly onto a couple of stray returns — a handful of points beyond a hole is not evidence of a feature. A person who looked at the cloud and pointed at them *is* that evidence, so an extension may bridge onto **your picks** however few of them there are. One lone return inside a 5 m occlusion is enough to carry the march across and let it pick the cable back up on the far side; blind bridging refuses that same return, and should.

### Discarding the last cylinders

A march often stops because the last step or two went *wrong*, not because the feature ended — the fit window caught a neighbouring object, the axis drifted off centre, and the heading that came out points somewhere the feature never went. Re-seeding from that tip inherits the bad heading and repeats the mistake.

Setting **Discard last N cylinder(s)** trims that much march off the end first: the centerline is cut back by `N × cylinder_length × (1 − overlap)` of arc length, the points the trimmed stretch collected are dropped, and the re-seed starts from the clean tip with the heading the line had there.

Trimming works on **arc length along the centerline**, not by slicing the recorded cylinder list — cylinders accumulate across both march directions and any earlier extensions, so their order no longer identifies "the last few of this end", and arc length follows curves correctly. Asking to roll back further than the line is long is refused rather than deleting the feature.

Note what this does and does not do: it backs the end *out* of a trouble spot. Whether the following march then gets *past* the obstacle depends on the growth parameters (`cylinder_radius`, `max_angle`), not on the rollback.

**Why re-seeding rather than automatic extension.** An automatic pass would have to *guess* whether a feature continues, and a wrong guess drives a line through a pole top into empty sky — worse than a short trace, because you have to notice it to fix it. Your picks *are* the answer. Growth is never loosened on its own initiative.

**What the picks authorise.** The re-seed runs the ordinary growth — same fit window, same angle gate, same membership threshold — with exactly three relaxations, all bounded by where you pointed:

1. The march runs **only outward**, since the opposite direction would re-walk line already traced.
2. The search is opened up just far and wide enough to *arrive* at your furthest pick.
3. It may bridge a gap onto your picks below `min_points` (see above).

The width matters as much as the distance, and less obviously: the search tube is aimed along the heading the march **drifted** to, and angular error scales with reach. A heading 2° off — routine after a few re-fits — misses by 0.2 m at 6 m and 0.85 m at 25 m. Granting reach without width sails the tube straight past the points you picked.

With no picks, nothing is relaxed at all — which is what stops the workflow from quietly guessing.

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

Picked points are grouped into one seed group per feature by `seed_eps` / `seed_min_samples`, and each group grows a line; RANSAC robustly ignores the occasional stray pick when fitting the initial line direction. Groups that turn out to be the same physical feature collapse into one line (see [One line per feature](#one-line-per-physical-feature)) — so an over-small `seed_eps` costs a little time, not duplicate lines.

---

## Design lineage

- **`DECISIONS.md` 2026-05-26** — RANSAC is a single-cloud contract; region growing is orchestration on top. `LinearRegionGrower` is one of the orchestrators that decision predicted (the line analog of the plane one).
- Generalised from the original `power_line_tracer.py`, which was hardwired to cables and predated — and broke against — the 2026-05-26 RANSAC refactor. `power_line_detection` now calls `LinearRegionGrower(mode=AXIS_TRACE)`.
- Addresses the line side of `PROJECT.md`'s #1 open problem (generic line extraction), previously "solved only for power lines."
