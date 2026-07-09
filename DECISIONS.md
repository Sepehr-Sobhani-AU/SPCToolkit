# Decisions log

Append-only log of non-obvious Definition-level decisions. One entry per call.
Keep entries to 1–3 sentences. The point is the *why*, not the *what* —
the *what* is already captured in `PROJECT.md` or in code. Newest at the top.

---

## 2026-07-09 — Linear march: fit the per-step axis from the full band, not a threshold-wide stripe
The per-step axis (direction + centre) is fit from ALL near-tube points (within
`cylinder_radius`), not from the `ransac_threshold`-wide "on-axis" subset. Gating
the fit by the tight threshold silently assumes the feature is thinner than the
threshold; on a real line whose lateral scatter EXCEEDS it, the threshold picks a
diagonal *stripe* through the band and PCA follows that stripe's tilt, so the axis
walks off to one side, the drawn cylinder hangs off the points, and the march
stops early with a false "too few points" (both symptoms the user hit). Fitting
the whole band gives its true long axis (along-extent dwarfs lateral scatter) and
a centroid that re-centres the tip each step; `ransac_threshold` is kept only to
pick members and clutter beyond `cylinder_radius` is excluded by the tube. This
does not reintroduce the old free-RANSAC chord-locking (that was minimal-sample
RANSAC; global PCA of a symmetric band/annulus still returns the long axis). Gap
bridging likewise looks for the continuation within the tube radius, not the
stripe, so wide bands are not falsely stopped at minor sparsities. Search
cylinders are now drawn on the fitted (centred) axis so the debug overlay sits on
the points.

## 2026-07-09 — Linear march: decouple fit-length from reach; keep plain PCA (reject density normalization)
Two follow-ups to the prior-gated march. (1) **Fit window vs search reach are
decoupled** via a `reach_factor` knob: `cylinder_length` is now only the fit
window / step (kept short so a straight per-step fit does not chord-cut a curve
and drift outward), while the tube is searched out to `reach = cylinder_length ×
reach_factor`. When the near window is empty the march **bridges the gap** —
jumps the tip forward to the nearest collinear cluster within reach, keeping the
heading (no data in the gap) — so fragmented features trace through. The lateral
on-axis gate means a bridge can only follow the same line longitudinally, never
hop to a parallel neighbour; `reach_factor = 1` disables bridging. (2) A
**density-normalized per-step fit** (equal weight per along-axis section) was
implemented and then **rejected on measurement**: the step estimator is
variance-weighted PCA, which already maximizes spread (so it inherently prefers
the longest-extent structure — the user's "fit-length" term) and down-weights a
compact dense off-axis cluster (a stub/bracket/junction near the seed) by its low
leverage. Forcing equal-per-section weight *re-inflated* exactly such clusters
and made a controlled case worse (1.3°→2.4° off-axis). The user's two goals are
already met by the current design: PCA's max-spread objective is the fit-length
term, and the on-axis gate + max-angle check is the prior-alignment term — so
plain PCA is kept.

## 2026-07-08 — Linear march step: prior-gated PCA, not a free per-step RANSAC
The axis-trace march no longer runs a fresh RANSAC line fit each step. A free
fit locks onto whichever chord has the most inliers, so wherever the search tube
overlaps other geometry (a crossing feature, a pole, the surface the feature
sits on) a denser off-axis clutter chord wins — tilting the march off the feature
or, via the `min_inlier_ratio` gate, wrongly reporting "fit failed" on a clearly
visible line (the user's `stop_fit_failed` complaint). During a march the current
heading is a strong prior (RANSAC is only justified at the seed, where there is
none): each step now keeps the tube points within the fit threshold of the
CURRENT axis, takes the next direction from a PCA of just those points (rejecting
turns > max_angle as a bend), then collects tube points within the threshold of
the UPDATED axis so curves are still hugged fully. This matches the existing
surface-region-growing pattern (calls RANSAC with `min_inlier_ratio=0.0` and
"applies its own filters"). The `fit_failed` stop reason is retired (no per-step
fit can fail); stop reasons are now too-few-points, sharp-bend, empty-space and
step-cap, each drawn as its own coloured end-cylinder branch.

## 2026-07-07 — Adopt existing standards for deliverable structure, don't invent a taxonomy
The deliverable/cartography structure is *adopted* from open, internationally-portable standards rather than authored from scratch (no client mandate → portability is the tie-breaker over US/UK vendor schemes): **ISO 13567** for CAD layer structure, **Uniclass 2015** as a per-feature classification *code attribute* (not the layer), **ISO 19650-2** for file/container naming, and the **ISO 128 family** (ISO 128 lines/widths, 128-50 hatching, ISO 3098 text, ISO 129 dims, ISO 5455 scales, ISO 5457 sheets, ISO 7200 title block) for drafting cartography. All four presentation concerns collapse into one **class-keyed style table** (`feature.class → layer, class_code, color/linetype/lineweight, hatch, text style`), which is the OGC-SLD / ISO 19117 *portrayal* idea resolved into DXF entity properties instead of an SLD XML — style stays decoupled from geometry and lives in config, so no new taxonomy and no code taxonomy.

This **qualifies, does not overturn, the 2026-05-17 "layer == class, no styling" decision**: the drafter's AutoCAD template still resolves visual properties by layer name when the client supplies one; the ISO 128 style table is the *fallback portrayal* for the generic/no-client case, applied only on explicit opt-in. `VectorFeature` stays render-only (2026-05-17) — the class field it needs is the single hook the whole stack hangs off, and it's an attribute assembled at export time, not a persistent style on the entity.

The **only actionable item now** is ensuring each extracted feature carries a **class field**; the style table, linetypes, scales and sheets all belong to the deferred export milestone (2026-05-20, extraction before export) and are not built yet. This is a *locked target-standard choice*, not a scope or priority change.

## 2026-06-26 — Crease edges via a generic two-plane intersection brick, not per-feature plugins
Edge lines (kerb top/bottom, building corners, etc.) are built by composing one generic "crease tracer" Lego brick, never by a kerb- or building-specific plugin — the named features are *outcomes* of putting bricks together. Brick contract (basic version): user selects a swath covering exactly two planes including the edge; the selection is partitioned into a user-sized hash grid; each cell 2-clusters its points by upstream normal into two planes (ignored if too few points or the two cluster-normals are near-parallel, i.e. one plane); the two planes are intersected and the cell emits **one vertex** (point on the line nearest the cell centroid); vertices are PCA-ordered along the crease into a `VectorFeature` polyline. Vertex-per-cell was locked over clipped-segment-per-cell because independent per-cell fits make segment ends disagree (broken/kinked line needing stitching), whereas connecting vertices gives a continuous line for free; clipped segments stay the upgrade path for tighter curvature. No seed point (the selection defines extent); the grid *is* the piecewise-linearization that handles curved creases. Corner resolution (extend-to-intersection of two edge polylines) is deliberately a **separate, deferred brick** — the kerb scenario validates the tracer alone since its two lines are parallel and never meet.

## 2026-06-29 — Crease tracer: seed-point input + auto perpendicular cell size
The selection now only *locates* the edge — the user Shift+clicks **one point near it** and the tracer searches the **whole branch**, discovering the two intersecting planes itself (local 2-means at the seed); the swath-as-fitting-data model is dropped. The cube also stops being a cube: the user sets only the **along-edge** size (`edge_length`, = the step), while the **across-edge** size (`perpendicular_size`) is auto-suggested by `suggest_perpendicular` (an O(N), normals-free k-th-nearest reach at the pick) and pre-filled as the dialog default, which the user can override — the human-in-the-loop preset is the leak guard, so no automatic third-surface detection. Side-colour consistency no longer uses a global normal split (gone with the swath); each march step's split is chained to the previous step's plane normals. A live preview cell was considered and deliberately deferred (out of scope for now).

## 2026-06-28 — Crease tracer is a local march, superseding the fixed grid
A fixed lattice (even crease-aligned) cannot guarantee one cell-row on the edge, keep the edge centred in each cell, or follow a curve — so the grid is replaced by a local march: one `cell_size` cube per step, centred on the local plane-intersection point and rotated to the local edge tangent `nA x nB`, stepping one cell along the edge and refitting; it runs both directions from a seed and stops when a cube no longer straddles two distinctly-oriented surfaces. This realises the previously-deferred march and handles curved creases (validated: a 90° arc kerb traced end-to-end at ~5 mm). The seed must be where the two surfaces meet *spatially* — the point whose neighbourhood is the most balanced mix of the two global normal-clusters — NOT the intersection of the two global flat planes, whose straight chord runs nowhere near a curved edge (that bug placed the seed in empty space and the march never started). Debug overlays now record per-step oriented cubes (drawn rotated to the edge), the two planes, and per-point cube/side ids. Corner junctions and clipped-segment output remain deferred.

## 2026-06-26 — Crease-tracer grid is crease-aligned, not world-aligned
A world-axis cell grid cuts an obliquely-oriented crease unevenly: cells that clip the edge at a corner catch too few points of one surface, so that plane is dropped (gap) or fit from a sparse/degenerate patch (outlier vertex). Fix: orient the grid so axis 0 is the edge direction `nA x nB`, derived once from a *global* 2-means split of all the swath's normals before bucketing; cells then tile evenly along the edge and each holds a balanced cross-section of both surfaces. Bucketing is done in frame coords; plane fitting and intersection stay in world coords (debug cubes are rotated back to world to match). This is exact for a straight crease and good for gentle curves; a sharply curving crease drifts from the single global frame — following it with always-plane-aligned cells is the deferred local march's job. A cube cannot be face-parallel to *both* planes at once (they meet at a dihedral angle ≠ 90°), so aligning the primary axis to the edge is the achievable and sufficient fix.

## 2026-05-26 — RANSAC is a single-cloud contract; iteration is orchestration on top
The current code has two disconnected RANSAC paths (CPU line engine and a GPU plane-RANSAC fused inside the surface-region-growing plugin), and adding cylinder/cone primitives would compound the duplication. The locked design: a `core/services/ransac/` layer with a single canonical contract `fit(points, model_type, threshold, normals=None) -> (model, inlier_mask)`, refit logic owned by each model (line, plane, cylinder, cone), pluggable sampler/scorer, and an optional batched `fit_many` fast-path for performance-sensitive orchestrators. Region growing, multi-model extraction, and curve-as-line-segments are orchestrators that *call* RANSAC, not variants of it — splines/arcs/catenaries are deliberately out of scope as primitives. Full design captured in `core/services/RANSAC.md`.

## 2026-05-21 — Pipeline capture & replay is scoped-in infrastructure, not a milestone
Every plugin step (plugin name + parameters + parent branch) is deliberately
recorded on each branch so a sequence from root cloud to outcome can later be
saved as a reusable tool and replayed on a new cloud — this was the original
design intent behind the per-step metadata. It is in scope but unranked, built
opportunistically once the basic plugin set is solid, not now. Replay is
semi-automated: it pauses at interactive steps (e.g. manual cluster selection),
so it is a guided pipeline, not an unattended batch file.

## 2026-05-21 — Scenario-driven plugin development
Plugins (the "Lego blocks") are not built speculatively or in bulk. The method:
define a concrete scenario (e.g. ground/non-ground segmentation), attempt it
with existing plugins, and only create or update a plugin when that scenario
demonstrates a real gap. Keeps every plugin justified by a real workflow and
avoids a sprawl of unused plugins.

## 2026-05-20 — Extraction before DXF export
DXF export (formerly M1) is deferred until vector-feature extraction from
clusters is solid on real data. Designing an export wrapper before
extraction is mature risks locking in a metadata contract against imagined
producer output, and a polished export over weak extraction hides the real
problem. Once extractors are real, the required export metadata becomes
obvious from the patterns instead of guessed.

## 2026-05-20 — Definition and Construction live in separate sessions
Stage-2 (Definition) work and Stage-3 (Construction) work happen in
different chat sessions. The handoff is via `PROJECT.md` (scope + current
priority), `DECISIONS.md` (this file, the *why*), and the Stage-gate
section in `CLAUDE.md` (the rule that forces Construction sessions to read
them first). Keeps each session honest about what kind of work it is doing.

## 2026-05-17 — `CADObject` renamed to `VectorFeature`; stays render-only
The class in `core/entities/vector_feature.py` is viewport drawing geometry,
not a DXF entity. It deliberately carries no `layer`, `class`, or
`attributes`. Export metadata will be assembled at export time from
`VectorFeature.geometry` + `Clusters.cluster_names` + the
`cluster_reference` UUID — no new persistent type. Old name kept as a shim
for pickled projects.

## 2026-05-17 — Point-cloud-source-agnostic
SPCToolkit accepts any point cloud regardless of acquisition method
(terrestrial, mobile-mapping, airborne LiDAR, photogrammetric). The source
does not change the pipeline. Validation stays concrete (one outdoor + one
indoor reference dataset for M2/M3) but architecture must not assume a
specific source.

## 2026-05-17 — BIM / IFC export is a committed year-2+ direction, not excluded
Walls/floors as point-cloud primitives are in scope for v1.0; turning them
into Revit families / IFC entities is deferred. M1's internal
geometry-with-metadata model must stay rich enough that IFC export can be
added later as another target without rewriting the pipeline.

## 2026-05-17 — DXF schema: layer == class, no styling
DXF layer name equals the cluster's class string verbatim. SPCToolkit sets
no color / linetype / lineweight — the drafter's AutoCAD template resolves
visual properties from the layer name. Cluster UUID is round-tripped on
each entity so AutoCAD → SPCToolkit re-import is possible later.

## 2026-07-06 — Seed-DBSCAN reintroduced in linear region growing (on demonstrated need)
`linear_region_growing` now clusters the picked seed points with DBSCAN and
traces several linear features from one selection. This deliberately reverses
the earlier call to strip seed-DBSCAN as scenario-specific machinery: the
reversal is justified by a concrete demonstrated scenario (selecting multiple
overhead lines in one go), which is how features earn their way into the
toolkit.

A single physical line is often split by clustering into several clusters. The
first implementation recovered it by growing each cluster separately then
MERGING grown lines with an endpoint/collinearity heuristic — this proved
fragile (fused parallel side-by-side lines into one zig-zag centerline). It was
replaced (user's proposal, simpler and more robust) with a GREEDY consume loop:
grow from the largest remaining cluster; drop every cluster that growth
consumed (>= `_CONSUME_FRAC` of its points ended up in the line); repeat. No
merge heuristic — a cluster is joined only when the growth actually reached it,
so parallel lines never fuse and a line split across many clusters comes out
whole.

Output is one Clusters branch (label per line). Centerlines and cylinders are
both optional and each collapses to a SINGLE branch holding all lines: one
"centerlines" branch (a mesh with one connected edge chain per line, no
bridging between lines) and one "cylinders" branch. The plugin stays generic
(any linear feature; cables were only an example).
