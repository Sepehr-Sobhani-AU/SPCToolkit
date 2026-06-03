# Octree / tile out-of-core point clouds — reference

**Status:** Deferred. Not a current priority. This document captures the
direction and tradeoffs so the decision can be picked up later without
re-deriving everything.

**Revisit when:** a real user routinely opens scans where the raw file is
bigger than their RAM (in practice, ~500M+ points / ~12–15 GB LAS), or
SPCToolkit needs to host clouds over HTTP for remote drafters.

## Why this matters for SPCToolkit

`PROJECT.md` targets "ordinary hardware: 8–32 GB RAM, single consumer GPU."
Today the architecture is built on the assumption that a branch is an
in-memory `PointCloud` with flat NumPy `points` / `colors` arrays. That
assumption is reasonable up to ~100–200M points but breaks past it. The
per-branch VBO refactor (commit `25dafaa`) and `LODManager`'s point budget
extend the ceiling but don't change the model — the whole cloud still has
to fit in RAM, and the LOD subsample is the only knob.

Out-of-core point cloud rendering removes that ceiling. The dataset lives
on disk in a spatially organised multi-resolution format; the viewer only
ever loads what the current viewport needs. Per-frame VRAM and RAM usage
become bounded by viewport size, not dataset size.

## Core idea

The cloud is reorganised into an **octree**: each node owns a fixed-size
sample (say 50k points) of the points inside its bounding box, plus
pointers to up to eight children that further subdivide.

- Root node: full bounding box, 50k uniformly-sampled preview points.
  Renders when the whole scene fits on screen.
- Depth N node: 1/8ᴺ of the volume, still 50k points → 8ᴺ× the density of
  the root. Renders when the camera zooms in.
- Leaves: small regions at full resolution.

Two properties fall out for free:

1. **Spatial culling.** Only nodes whose bounding box intersects the view
   frustum need to be loaded. Everything else stays on disk.
2. **Distance LOD baked into the data.** Walking the tree top-down and
   stopping at the first level where each node hits roughly one point per
   screen pixel automatically produces the correct LOD for the view —
   denser near the camera, sparser far away. No per-frame downsampling
   decision.

## How rendering works

Per frame:

1. Compute viewport frustum + per-pixel density target (from
   `camera_distance`, `fov`, screen size).
2. Walk the octree top-down.
   - If the current node's sample is dense enough for its screen
     footprint → render this node, stop descending.
   - Otherwise → recurse into children whose bbox overlaps the frustum.
3. Any node we want to render that isn't resident → schedule async load
   from disk (worker thread → numpy array → VBO upload).
4. Maintain LRU eviction over resident nodes. When the resident VBO
   budget exceeds, say, 70% of detected VRAM, drop least-recently-visible
   nodes' VBOs.

The per-branch VBO refactor we already shipped is the right scaffolding
for this: the toggle/visibility code is per-key. Going from
`key = branch_uid` to `key = (branch_uid, tile_id)` is a natural
extension.

## Format options

Listed roughly in the order to consider today.

### COPC — Cloud Optimized Point Cloud (recommended target)

- LAS 1.4 with points reorganised into an octree layout inside the same
  file, plus a small VLR header telling readers where each node lives.
- Single file. HTTP-range-friendly — you can read just one tile over the
  wire without fetching the rest.
- Drops straight into existing LAS toolchains.
- Pure-Python read support via `laspy >= 2.0`.
- Maintained by the PDAL / Hobu team.
- Spec: <https://copc.io/>

This is what to target if going down this path.

### Entwine / EPT (Entwine Point Tile)

- Older, also Hobu-maintained. AWS Open Data hosts most public LiDAR in
  this format.
- JSON metadata + a directory of binary tile files (many small files).
- Production-proven, but COPC is the same idea in a tidier single-file
  package — no reason to pick EPT for a new project.

### Potree (v1/v2)

- The classic browser-first viewer; the format is a directory tree of
  binary files (v2 is more compact than v1).
- Drafter clients have likely seen Potree pages — it's the de-facto
  surveying-deliverable web viewer.
- As a format, eclipsed by COPC. As a reference implementation of
  out-of-core rendering, still useful to study.

### 3D Tiles (OGC)

- Cesium ecosystem. Supports point clouds via the `pnts` tile type.
- Heavier spec. Worth picking only if rendering alongside Cesium's
  terrain / 3D building / web context.

## What it would change in SPCToolkit

This is **not** a localised optimisation. It changes the data model. Be
honest about that before starting.

### Concrete impact

- **`PointCloud` interface gains query methods.** Today plugins do
  `pc.points[mask]`. Tile-aware needs `pc.query(bbox, max_points)`
  returning a generator/iterator of tile-array fragments. Plugins that
  truly need all points call `pc.iter_all()` and pay the cost; plugins
  that work on a region (most extraction plugins) stream only the
  relevant tiles.

- **Viewer becomes `(branch, tile_id) -> VBO`.** Frustum culling +
  per-tile LRU eviction. Most of the per-branch VBO scaffolding ports
  cleanly.

- **Picking / selection get harder.** Today picking returns a global
  index into a flat `Nx6` array. Tile-aware picking returns
  `(branch, tile_id, local_index)` triples. Every selection-aware plugin
  and every reader of `viewer_widget.points[idx, :3]` (see `_branch_helpers.py`
  and the picking-coordinate paths in `040_Clusters/*` plugins, etc.)
  needs touching.

- **Derived branches are the awkward case.** A root cloud loaded from
  COPC is naturally tiled. The output of "run DBSCAN on the visible
  points" is just a flat array of labels — not tiled. Re-tiling derived
  outputs every time is expensive.
  → Likely answer: **hybrid model.** Roots are tile-streamed; derived
  branches stay as in-memory flat `PointCloud` objects. That isolates
  tile complexity to the loader + the viewer, leaves the plugin surface
  mostly intact.

### What stays the same

- Plugin contract for analysis (RANSAC, region growing, classification)
  — they get full point data either way; whether it came from a flat
  array or a tile iterator is a loader concern.
- Branch / tree structure model.
- Reconstruction service (still rebuilds derived branches from parent +
  operation history).
- Cache service (still tracks `is_cached` per branch).
- DXF export pipeline (operates on derived extraction results, not the
  raw cloud).

### What likely gets thrown away

- Nothing in the current per-branch VBO path. The work shipped in
  `25dafaa` extends naturally.
- The "load entire cloud into one `PointCloud.points` array" code path in
  `services/file_manager.py` for COPC inputs. (Stays for PLY / non-COPC
  LAS.)

## Python ecosystem

- **`laspy >= 2.0`** — pure-Python LAS / LAZ / COPC reader. Has native
  COPC support, including range queries by bbox / depth.
- **PDAL** — C++ point cloud processing library with Python bindings.
  Can read/write COPC, EPT, LAS, many formats. The reference for
  "industrial" point cloud I/O.
- **`copclib`** — C++ with Python bindings, COPC-focused. Lighter than
  PDAL if all you need is COPC reads.
- **py3dtiles** — for the 3D Tiles ecosystem.

For SPCToolkit, the natural starting point is `laspy` + (optionally)
`copclib` for read performance. Avoid pulling in PDAL unless its broader
pipeline features are needed — it's heavy.

## Risks / tradeoffs

- **Indexing cost.** Building a COPC for a 100M-point cloud takes
  minutes (often several). The model is "index once on import, query
  forever." Need a clear import workflow.
- **Plugin compatibility.** Any plugin that does
  `pc.points[mask]` on a tile-streamed cloud has to be reviewed. The
  hybrid model (tiled roots, flat derived) limits this to the very first
  step of every workflow.
- **Disk space.** Octree formats may be 1.2–1.5× raw size due to
  overlapping samples + metadata.
- **Loss of "everything in memory" simplicity.** Debugging,
  introspecting, and writing one-off scripts against a flat numpy array
  is genuinely easier than against a tile iterator. Real cost.

## The cheap intermediate step (if needed before going full out-of-core)

Backing cold derived branches with `numpy.memmap`. Extends the RAM
ceiling without changing the model:

- A `PointCloud.points` array lives as an `np.memmap` over a file on
  disk. Pages fault in on access; the OS evicts under memory pressure.
- Cheap to add (numpy supports it natively).
- Random access is slower than RAM, but sequential reads (building a
  VBO slice, iterating for analysis) are fine.
- Doesn't help with viewports larger than VRAM — that still needs LOD
  subsampling or tile streaming.

This buys time and handles the "many derived branches piling up" failure
mode without committing to the full out-of-core architecture.

## When to invest

**Worth it when:**
- Drafters routinely open scans bigger than their RAM (~500M+ points).
- Hosting clouds over HTTP for remote drafters / clients becomes a real
  need (COPC was designed for this).
- "I can't even pan smoothly because the dataset doesn't fit" becomes
  the dominant friction.

**Not worth it when:**
- The bottleneck is extraction quality, not viewport scale. The current
  M1 priority (`PROJECT.md`: extraction before export) is a smarter
  place to invest effort.
- Datasets comfortably fit in 32 GB RAM. The LOD cap + per-branch VBO
  model handles 100–200M points fine on the project's target hardware.
- You'd have to maintain two implementations of every plugin
  (tile-aware and flat). That's a tax on every future plugin.

## Honest read for v1.0

Defer past v1.0. The current architecture handles "ordinary hardware,
ordinary scans" adequately. When user demand actually becomes "I need to
open scans that don't fit in RAM," COPC is the target — clean
integration with `laspy` and the rest of the LAS toolchain SPCToolkit
already lives in.

The cheap intermediate step (`np.memmap` for cold derived branches) is
the move if RAM pressure shows up before viewport pressure. Costs a
weekend instead of a quarter.
