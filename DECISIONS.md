# Decisions log

Append-only log of non-obvious Definition-level decisions. One entry per call.
Keep entries to 1–3 sentences. The point is the *why*, not the *what* —
the *what* is already captured in `PROJECT.md` or in code. Newest at the top.

---

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
