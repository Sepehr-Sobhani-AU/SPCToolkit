# SPCToolkit

## Purpose
SPCToolkit helps surveying firms and CAD/BIM drafters turn point clouds
into as-built CAD deliverables faster, on ordinary hardware, by automating
the geometry-tracing step. It is point-cloud-source-agnostic — terrestrial,
mobile-mapping, airborne LiDAR, and photogrammetric scans all enter the
same pipeline. The tool's job ends at clean DXF output with metadata
(layers, attributes); finalization (symbolization, dimensioning, sheet
layout) happens in AutoCAD.

## Users
- **Primary**: CAD/BIM drafters (daily users — the people whose hands do the tracing)
- **Secondary**: Surveying professionals at small/mid-firms who can't justify $10k workstations
- **Tertiary**: Sepehr (researcher / maintainer) — the project doubles as an R&D platform

## Workflow this tool fits into
Point cloud (PLY/LAS/E57) → SPCToolkit (extract + classify + export) →
DXF with metadata → AutoCAD (symbols, dimensions, sheet, title block) → client

## In scope
- Extraction of three CAD primitive types from point clouds:
  - **Blocks**: discrete features (trees, poles, signs, traffic lights, wheel stops)
    → DXF block insertions at correct locations with attribute data
  - **Lines**: linear features (kerb tops/bottoms, power cables, pipes, fences)
    → 3D polylines on the correct layer
  - **Surfaces**: ground, terrain
    → TIN exported as POLYFACEMESH / MESH (drafter converts on their side if needed)
- DXF export with metadata (layer == class, attributes) as the primary handoff format
- Plugin-based architecture for adding new feature extractors
- Both **outdoor** scans (street surveys, infrastructure) and **indoor** scans
  (rooms, corridors, heritage interiors) — same primitive types apply (walls
  as lines, floors as surfaces, fixtures as blocks)
- **Input-source-agnostic**: any point cloud regardless of acquisition method
  (terrestrial laser scanning, mobile mapping, airborne LiDAR, drone /
  photogrammetric) — the source does not change the pipeline
- Single-user desktop application
- Ordinary hardware target (8–32 GB RAM, single consumer GPU)
- AI/ML-assisted classification and segmentation

## Out of scope (next 12 months)
- Replacing AutoCAD/Revit (no symbol library, no dimensioning, no sheet layout, no title blocks)
- **BIM authoring and IFC export (for v1.0 only)** — extracting walls/floors as
  point-cloud primitives is in scope; turning them into Revit families,
  parametric BIM elements, or IFC-class entities is deferred to year 2+. This
  is a committed long-term direction, not an exclusion. Today's architectural
  choices (especially the geometry-with-metadata model in M1) should keep
  the IFC door open.
- Multi-user / cloud / web UI — single-user desktop only
- Real-time scanning / SLAM — offline batch processing only
- 3D mesh authoring (Blender-style modeling) — we extract CAD primitives, not author meshes
- Photogrammetry / SfM / SLAM **reconstruction** — SPCToolkit *consumes* a
  point cloud, it does not *generate* one. A point cloud produced by
  photogrammetry is perfectly valid input; producing it from images is not
  SPCToolkit's job.
- Photorealistic rendering — the viewer is a working viewport, not a presentation tool

## Open problems (known unknowns)
- **Generic line extraction from point clusters** — turning a cluster of points
  into a clean, ordered, fitted polyline is genuinely hard (skeletonization,
  ordering, intersection handling). Currently solved only for power lines via
  RANSAC. This is the highest research risk in the project.
- ~~**Cluster → block correspondence**~~ — RESOLVED: `Clusters → Classify Cluster`
  plugin already does this. User selects clusters, picks class from dropdown
  (or types a new one), stored in the project file.
- ~~**DXF schema**~~ — RESOLVED in the M1 data model (see roadmap). Layer = class
  (company standard: `Category-Subtype`, e.g. `Kerb-Top`, `Tree-Palm`).
  No visual styling set by SPCToolkit; the drafter's AutoCAD template handles it.

## Success criteria (for the v1.0 milestone)
A drafter can:
1. Open a real point-cloud scan in SPCToolkit.
2. Run a sequence of plugins to detect blocks, lines, and surfaces.
3. Export a single DXF file.
4. Open it in AutoCAD and find clean geometry on the right layers, ready for
   symbolization — without manual cleanup of the geometry itself.

If steps 1–4 work for one full real-world dataset, the architecture is proven.

## Non-goals for code style
This file does not describe code conventions. See `CLAUDE.md` for those
(singleton over signals, GPU-first, batching, plugin templates).

---
*Roadmap (M1–M4), data model, and re-drift practices: see `plans/` strategy
draft until folded here. PROJECT.md stays one page — strategy detail lives
elsewhere.*