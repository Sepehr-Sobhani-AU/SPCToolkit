# Decisions log

Append-only log of non-obvious Definition-level decisions. One entry per call.
Keep entries to 1–3 sentences. The point is the *why*, not the *what* —
the *what* is already captured in `PROJECT.md` or in code. Newest at the top.

---

## 2026-07-17 — Contours via a generic field-agnostic brick, keyed on cloud edges
Contour lines are one generic `contour_growing` brick over *any* per-point field
(the `services/point_fields` dropdown), never a per-field plugin — height
contours, slope-break lines and intensity edges are outcomes of picking a
different field, and whether a given field makes a meaningful line is the user's
call, not the plugin's. Locked design: the seed pick sets the level (its own field
value) and the start; per step a ball is PCA-flattened, Delaunay-triangulated and
marched; **every crossing is keyed by the pair of cloud point indices whose edge
it lies on**, so overlapping balls dedupe exactly, meeting ends join, and loops
close with no distance tolerance anywhere — and since a cloud point is a ball
centre at most once, the flood is finite and stops by itself. Two triangle cuts
earn their keep: longer than `max_triangle_edge` (Delaunay fills its convex hull,
so it invents surface across holes) and circumcircle leaving the ball (rim
triangles are built from points whose real neighbours the ball never saw).
Crossings are capped at two segments because neighbouring balls fit slightly
different local planes and Delaunay flips near-tie diagonals between them — the
vertices agree exactly, only the pairing differs, so a third segment is a second
opinion on a settled pairing, and allowing it forked a curved-surface test into 11
fragments instead of 1 closed loop. ALL lines in a ball are kept and every open
end grown, not only the seeded line (user's explicit call): the seed picks the
level and where to start, not which line survives. Output is ONE branch holding
every contour, as `geometry_type="mesh"` with vertices+edges (as
`centerlines_to_vector_feature` already does) — the `"polyline"` schema holds only
one line per node, and a closed contour needs no `closed` flag since it is simply
a chain returning to its first vertex.

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

## 2026-07-30 — "Growth reached this cluster" is a SWEPT-region test, not a membership test
Fixes duplicate lines (two centerlines drawn on top of one cable) from
`linear_region_growing`. The greedy consume loop of 2026-07-06 asked "did the
grown line CLAIM this cluster's points?" — the wrong question, for two reasons:

1. Membership is gated at `ransac_threshold`, while the search tube is
   `cylinder_radius` wide. Whenever radius > threshold (the normal setting for a
   thick or noisy cable — `power_line_detection` ships 0.5 vs 0.3), a march can
   drive straight through a cluster and claim none of its points. The cluster
   stays in the pool and regrows the same feature.
2. The test only ran BEFORE a cluster grew, so it could not catch a cluster that
   legitimately survived — e.g. one sitting past the point where the first line
   stopped — and then marched BACK over that line, since every march runs both
   ways from its anchor.

Locked: the grower now records the region it SWEPT (every point inside a step's
fit window, member or not; `swept_indices()`), and `grow_lines` tests against
the sweep accumulated over the kept lines — both to drop clusters from the pool
(`_CONSUME_FRAC`) and, after growing, to discard a line that retraces a kept one
(`_DUPLICATE_FRAC`). The sweep, not the membership, is what "growth reached
here" means. No merge heuristic was reintroduced, so the 2026-07-06 guarantees
hold: parallel features never fuse, and collinear features separated by more
than the search reach stay separate.

`power_line_detection` still runs its own pre-growth `already_grown` check
against member points and has no post-growth check, so it retains bug (1) and
(2). Left alone deliberately — not in this fix's scope.

## 2026-08-04 — Short traces are extended by USER-CONFIRMED re-seeding, not by automatic relaxation

Linear region growing does not reach the end of every feature, and the manual
cleanup that follows is almost entirely *extending traces that gave up early*.

The obvious automation is an automatic post-growth pass: at each stop, retry with
the search relaxed according to the stop reason, and accept the extension if
points re-appear. Rejected as the first step. Such a pass has to **guess**
whether a feature continues, and a wrong guess drives a line through a pole top
into empty sky — worse than a short trace, because it has to be *noticed* before
it can be fixed. It also has to be tuned against synthetic cases before anyone
has seen how real stops distribute between "obviously continues" and "genuine
end".

Locked: extension is **guided re-seeding**. Every march end now reports where it
stopped, its heading, and why (`MarchStop`, `march_stops()`). A window steps
through those stops — ranked by how many unclaimed points lie ahead, so genuine
ends sink to the bottom — and for a stop where the feature really does continue,
the user picks a few points beyond it. Growth then re-runs seeded with the line's
own points around the stop ∪ those picks, and the result is spliced into the
existing line (`extend_from_stop`). The picks are the evidence; growth never
loosens on its own initiative.

Exactly two things are relaxed for a re-seed, both bounded by the user's gesture
(`_pick_bounded_search`):

1. The march runs **only outward** (`grow(only_direction=...)`) — the opposite
   direction would only re-walk traced line. Measured: 26 centerline vertices
   instead of 37 for the same feature. Cost, not correctness; the union and the
   centerline join absorb a retrace either way.
2. The search reach **and tube width** open just enough to arrive at the furthest
   pick. Both are needed. The tube is aimed along the heading the march *drifted*
   to, and angular error scales with reach: a 2° drift (routine after a few
   re-fits) misses by 0.2 m at 6 m and 0.85 m at 25 m. Granting reach alone was
   measured to sail straight past the picked points — the extension collected the
   picks and nothing else.

With no picks, nothing is relaxed, which is what prevents silent guessing.

Also fixed in the same path: stopping on a sharp bend discarded the entire fit
window, while the too-few-points branch keeps its near on-axis points. A bend
usually means something else entered the window, so the points still hugging the
current heading belong to this line — up to a full `cylinder_length` was being
thrown away at every bend. Those points are now kept, but the window is
deliberately **not** marked swept: a bend often means a second feature crosses
there, and marking its neighbourhood swept would consume that feature's seed
group (see 2026-07-30).

Traces are persisted on the result (`Clusters.line_traces`, plain data — it
outlives the session in the project file, so it must not depend on a service
class staying importable), so extension can continue in a later session via the
`extend_traced_lines` plugin. The automatic pass is deferred, to be reconsidered
once real clouds show how many stops are trivially recoverable.

## 2026-08-04 (b) — Stop queue is ranked once; a bad last cylinder is rolled back, not grown from

Two corrections to the guided-extension workflow above, both from using it.

**The queue is ranked once, on open.** The first version re-ranked after every
extension and reset the walk to the top, so fixing one stop threw away the user's
place and re-offered stops they had already passed. Ranking is a starting
suggestion, not a live scoreboard. Settling a stop now removes it in place and
lands on the next; stops produced by a fresh extension go to the BACK of the
queue. Only the points-ahead figure for the *current* stop is recomputed live, so
it stays honest as earlier extensions claim points.

**A march that stopped badly is rolled back before re-seeding.** A stop is often
caused by the last step or two going wrong — the fit window caught a neighbouring
object and the heading drifted — so the tip is already off the feature and
re-seeding from it inherits the bad direction. `rollback_stop(line, stop, n)`
trims N steps off that end, drops the points the trimmed stretch collected, and
returns a stop on what remains. Exposed as "Discard last N cylinder(s)".

Trimming is by **arc length along the centerline**, not by slicing the recorded
cylinder list: cylinders accumulate across both march directions and any earlier
extensions, so their order no longer identifies "the last few of this end", and
arc length follows curves correctly. Rolling back further than the line is long
is refused rather than deleting the feature. It backs the end *out* of a trouble
spot; whether the following march gets *past* the obstacle is down to
`cylinder_radius` / `max_angle`, not to the rollback.

Also fixed: `extend_from_stop` reported success by point count, but `grow()`
always returns the seeds it was given — so an extension whose march stopped dead
still came back holding the user's picks, and the user was told the trace had
advanced when it had not. It now requires the march to contribute something
beyond the seeds.

## 2026-08-05 — A pick is a membership decision; the extension window stays put and can be undone

Three corrections from the first real use of guided extension, all the same
mistake in different places: the engine treating its own judgement as better
than the user's, and the window moving on before the user could see what
happened.

**The picks are always adopted.** `extend_from_stop` used to return failure when
the march contributed nothing beyond the seeds, and the picks were discarded
with it — so the workflow dead-ended precisely where it was needed most (a long
occlusion, a cable through canopy). But a point the user picked is a point they
looked at and judged to be on this line, and the engine's opinion about whether
it could have got there by itself does not override that. The march is now the
*bonus*: it runs, and whatever it reaches is spliced in, but when it stops dead
the picks still join the line and the frontier still advances to their far end
(`STOP_PICKED_END`). Worst case the user walks the feature themselves, a few
points at a time — which is a workflow that always terminates, unlike one that
refuses. The distinction is kept honestly: `Extension.marched` says whether the
march advanced, so "the trace grew" and "your points were added, pick further"
are reported differently. This reverses the last paragraph of 2026-08-04 (b),
which was right about the diagnosis (point count cannot tell the two apart) and
wrong about the remedy (throwing the picks away).

**`min_points` no longer overrules a human when bridging.** The gate exists to
stop the march bridging blindly onto a couple of stray returns — a handful of
points beyond a gap is not evidence of a feature. A person who looked at the
cloud and pointed at them *is* that evidence. During an extension the march may
now bridge onto the user's picks however few of them there are
(`_bridge_targets`); everything else about the gate is unchanged, and ordinary
growth is exactly as strict as before. Measured on a cable with one lone return
inside a 5 m occlusion: blind bridging refuses it and the trace ends at the
hole; one pick on that return carries the march across and it recovers the
remaining 7 m by itself.

**The window stays on the line after an extension, and can be undone.** It used
to settle the stop and jump to the next one, so the user never saw the result of
their own picks. The frontier now REPLACES the stop in place: pick, extend, look,
pick further, extend again — the same line, the same position in the queue,
until the user says "real end" or skips. An Undo stack restores the state before
any change (extend, real end, or skip — so Undo doubles as the only way back to
a stop already stepped past). Growth is a judgement made from a picture on
screen; the user has to be able to look at a result and say "no, not that".

## 2026-08-05 (b) — Only navigation moves the camera; stepping is not a change

Follow-ups from using the extension window (supersedes the last paragraph of
2026-08-05, which had Undo doubling as "go back").

**Extend and Undo leave the view alone.** The user frames a view themselves in
order to pick into it; re-centring the camera the moment they press a button
hides the very thing they pressed it to see. Only landing on a stop they have
not seen yet moves the camera — Previous, Next, and Real end (which settles the
current entry, so the next stop takes its place).

**Skip became Previous / Next.** Stepping past a stop is not a change to
anything, so it is not snapshotted and Undo no longer covers it; Previous is how
you go back. That leaves Undo meaning exactly one thing: reverse a change to the
lines.

**The debug wireframes are rebuilt after every change.** The window only
refreshed the centerlines branch, so search cylinders stopped at the original
stop while the line ran on past it — which reads as the extension not having
happened. Both the `centerlines` and `cylinders` branches are now rebuilt from
the current lines (only if the user asked for them at growth time — the window
never creates branches they did not want). The per-reason `stop_*` branches are
deliberately NOT repainted: they record where the original run ended, and the
green "you are here" marker is the live answer. Two live answers on screen is
worse than one stale record clearly labelled as history.

## 2026-08-05 (c) — Stop markers are drawn from the stops, not from march leftovers

Supersedes the last paragraph of 2026-08-05 (b), which left the per-reason
`stop_*` branches as a record of the original run. In practice that reads as a
bug: after an extension the line visibly runs straight past a red "too few
points" marker, and the end it actually has now is unmarked.

The window now repaints those branches from the lines' CURRENT `stops` on every
change — a branch is created when a reason first appears and removed when its
last stop is gone. `stops` is the machine-readable truth; `end_cylinders` is
display geometry that only ever accumulates (an extension appends to it), so it
cannot answer "where does this line end now". Only branches the growth run drew
are managed: which debug geometry is on screen stays the user's choice.

One visible consequence: markers the window repaints sit at the stop tip along
its heading, where the run drew the last search cylinder of the march. Same
meaning, and it now matches the green "under review" marker, which is drawn the
same way.

**Search cylinders are now persisted with the trace** (`line_traces`, eight
floats per step). Without them, a session that reopened a saved result and
extended a line would rebuild the cylinder branch from the extension's handful
of cylinders alone, throwing away the whole original run's geometry. Traces
written before this come back with none, and the window leaves that branch
alone rather than gutting it.

## 2026-08-07 — Pick focus is a cluster label, not a viewer mode

Picking the few points beyond a stop is the slow, error-prone part of extending a
line: a real cloud has vegetation, poles, ground and the cable itself competing
for every click. Two facts about the viewer decided how to fix it.

A point labelled `-1` is drawn dark grey (`Clusters.set_random_color`'s noise
colour) **and** refused by the picking filters (`_filter_noise_points`). So the
viewer already fades and locks exactly what should be faded and locked — nothing
needed adding for that. The real problem was the opposite one: unclaimed points
are `-1` too, so the points the user must pick were unpickable, and the workflow
only functioned because the input cloud was shown alongside the result, drawing
every point twice.

**So the candidates are promoted to a real cluster** (`pick_candidates` → the
window's `_mark_candidates`): they stop being noise, become clickable, and get a
bright colour of their own. The first design was a viewer-level focus mask with
a new mixin, a three-pass GL draw and a callback re-evaluated on every LOD
re-slice. Same picture, an order of magnitude more code, and one more thing to
keep in step with the renderer. Credit where due: this was the user's design.

Three things it turns on:

- **The label must be `max + 1`.** Colours are handed out in sorted label order,
  so a label that sorts last leaves every existing cluster's colour untouched.
  One sorting first (`-2`) shifts every line's colour each time candidates appear
  and vanish — the cables would change colour on every step.
- **The window claims the tree selection.** Picking is filtered to the branches
  selected in the tree, and after growth that is still the input cloud (which is
  hidden by then), so nothing is pickable whatever the labels say.
- **Label lookups had to be made LOD-aware.** Under LOD the viewer holds
  `points[indices]`, but `_branch_helpers` looked up `labels[index - start]` — an
  unrelated point's label. Pre-existing, but this feature stands on that lookup:
  the candidate the user sees glowing would refuse the click. The coordinator now
  keeps the subsample indices and the viewer maps through them (`cloud_index`).

**The cone is 15°, and that number is the whole filter.** Measured on a cable 8 m
above ground with a 6 m hole and a tree beside it, offering 24 m ahead: at 45°
the cone swallows the ground and 36% of the cloud stays clickable; at 15° it is
2.3%. Both offer every point of the cable beyond the hole, so recovery alone
cannot tell the useful setting from the useless one — `test_pick_candidates_
leave_the_clutter_out` asserts both halves. Erring tight is the cheaper mistake:
the user can see a point is not offered and widen the range, while too wide
silently returns the problem.

The cluster is transient. `_commit` rebuilds labels from the lines every time, so
it is erased by construction rather than needing to be unpicked, and stepping
between stops only relabels the points involved instead of the whole cloud.

## 2026-08-08 — The extension window owns the viewport while it is open

The candidate cluster (2026-08-07) shipped looking correct and behaved as if it
did nothing: candidates drew grey, and every unclaimed point stayed clickable.
Neither symptom was in the labelling — measured on the real controller, the
window labels 560 points, paints them yellow, and leaves 20,000 as grey noise.

The cause was a second point branch on screen. The cloud a result was grown from
draws an unlabelled copy of the same points in the same place; those copies have
no cluster labels, so the noise filter has nothing to refuse and they take every
click. Selecting that branch in the tree — which is what clicking its row to
show it does — switches the pick filter over to it. Measured: with both visible
and the input cloud selected, every one of ITS points is selectable and none of
the result's are.

So the window now clears the viewport on open: every other point-drawing branch
is hidden (vector features are left alone — they draw as lines and are the
context worth keeping), the result branch is forced visible, and the tree
selection is claimed. All of it is restored on close. If another point cloud
reappears while the window is open, the pick count says so rather than letting
the clicks fail silently.

The general lesson, since this cost a release: a rule about which points may be
picked means nothing while a second copy of those points is on screen. Anything
that decides pickability from a branch's own data has to own what is drawn.

Also fixed on the way: `ApplicationController.remove_node` called
`DataNodes.remove_data`, which has never existed. Every call failed into its own
except clause and only logged, so removed branches lost their tree row and
stayed in the project forever — including this window's "you are here" marker,
one per close.

## 2026-08-08 (b) — Candidate points must be NAMED, not just coloured

The candidate cluster still drew grey on real data. The labels were right and
the per-point colour array was yellow; neither reached the screen.

`ClustersTransformer` colours a **named** `Clusters` with `get_named_colors()`
and never looks at `clusters.colors`. A linear-region-growing result is always
named — `_build_result_branch` sets `{0: "Line 1", 1: "Line 2", ...}` — so on
exactly the branches this window is opened on, the colour array is dead weight,
and every label without a name is painted the 0.7 grey default. The candidates
were that: unnamed, and therefore indistinguishable from the rest of the cloud.

So `_mark_candidates` now names the cluster (`Pick candidates`) and registers its
colour in `cluster_colors` when the branch is named, and falls back to the colour
array when it is not. The naming is removed in `_forget_candidate_naming`, called
both when the offer is withdrawn and from `_commit` before `set_random_color`,
so no phantom class survives into classification or a DXF layer.

The test now asserts on what the RENDERER produces
(`ClustersTransformer(...).execute().colors`), not on `clusters.colors`. The old
assertion passed the whole time this was broken — the array really was yellow;
it just was not what anyone drew. Where two paths can produce a colour, a test
that reads the wrong one is worse than no test.

## 2026-08-08 (c) — Show what is offered, and default the range to the corridor

"77 unclaimed points lie ahead" beside a canopy solid with yellow reads as a bug.
It was not: the two describe different volumes. The count is the line's own
search corridor — 9 m long, 0.6 m wide, about 10 m3. The yellow was the pick cone
at its 8x-reach default: 24 m long, 7.4 m wide at the end, 1600 m3. 157x the
volume it was being compared with, and in dense cover that is tens of thousands
of points.

Two changes. The panel now reports the offered count next to the ahead count, so
the two numbers can be compared instead of one being read as a claim about the
other. And the default range drops from 8x the search reach to 3x — the same
length as the corridor the ahead count uses — so the two figures describe the
same distance and differ only in width (15x, not 157x). Reaching further is one
spin away, and the offered count makes the cost of doing so visible.

Worth stating plainly: the cone's volume grows as the CUBE of the range. Every
metre of extra reach is bought at compounding cost in points to aim through, so
the default should be the smallest one that usually works, not the largest one
that might.

## 2026-08-08 (d) — The offer IS the count: the pick cone is gone

Tried on real data and rejected. Reverses the cone in 2026-08-07 and the
range/offered-count tuning in (c).

The window offered a cone while the panel counted a corridor, so the two
disagreed by construction — 1600 m3 against 10 m3 at the original default. On a
cable ending inside a tree canopy that meant thousands of yellow points beside
"77 unclaimed points lie ahead", which reads as a bug however carefully the two
are explained. Reporting both numbers (c) made it comprehensible, not good.

``_candidate_indices`` now returns ``unclaimed_ahead`` — the same set the count
reports. One number on screen, one set of yellow points, and no way for them to
disagree. ``pick_candidates`` and its cone constants are deleted rather than
left unused; the range spin and the offered-count line go with them.

The corridor is narrow (9 m x 0.6 m at the defaults) and that is now a
user-confirmed choice, not a compromise: measured on the cluttered test scene it
offers 0.46% of the cloud and 124 of those 125 points are on the cable, against
2.3% for the cone. The escape hatch stays — unticking offers every unclaimed
point in the cloud — for a continuation that genuinely lies outside it.

What this cost was worth learning: the cone was designed against a synthetic
scene with a thin ground plane and one small tree, where it looked fine (2.3%).
Real vegetation is orders of magnitude denser, and no amount of geometry
distinguishes a cable from the leaves around it. The corridor works because it
is the same question the march asks, not because it is cleverer.

## 2026-08-08 (e) — A polygon may only pick what was offered

Extending from a polygon selection swallowed a whole bush: the viewer reported
32 points picked and the line gained thousands.

``picked_cloud_indices`` re-tests the stored polygon against the FULL cloud, so
a polygon covers everything it encloses rather than only the points LOD drew.
That re-test bypasses the viewer's selection filters — they run when the polygon
is closed, in rendered-index space, while the re-test works in cloud-index
space and simply unions in every enclosed point. Combined with "the picks are
always adopted" (2026-08-05), every point inside the polygon joined the line.

``picked_cloud_indices`` now takes an ``allowed`` set of cloud indices and
intersects with it; the extension window passes its candidate set. The gate is
in the shared helper rather than the window on purpose — the trap lives there,
so the warning and the way out should too. Only the caller can supply the
answer, since only it knows what is admissible in cloud-index space.

Note the same leak still applies to seed picking in the growth plugin, which
calls the helper without ``allowed``. Left alone deliberately: growth seeds are
a starting body that RANSAC and the seed DBSCAN are built to tolerate strays in,
and nobody has reported it biting. If it does, the fix is one argument.
