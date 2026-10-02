## ISSUES:

- I expect the length of data nodes in all the branches under the root point cloud be exactly the same as number of points in the root. This way applying logical operations on the branches will be very fast.
  - **Report (Open / design):** Invariant does **not** hold today (masks are boolean over their *immediate parent*; nested masks / point_cloud-subsets are shorter).
  - ***Do you have any suggestion?***
  - **↳ Suggestion:** Move to **root-anchored, bit-packed boolean masks** — every selection branch stores a boolean mask of length = **root** point count; nested filters AND into the parent's root-space mask. Wins all at once: logical ops (AND/OR/NOT/subtract) become native O(n) array ops; mask memory drops to 1 bit/point (≈9 MB for 73M — the issue you raised last round); the length invariant holds, so the color cue only flags genuine new-root branches (merge / duplicate-to-root). Cost: a data-model change (mask producers must emit root-space masks, or a transformer lifts immediate-parent masks once). This is Definition-level — wants a `DECISIONS.md` entry before building. Recommend doing the color cue + points column first, then schedule this as its own change.

- The menus and their contents should be rearranged, as many of the plugins are not in a proper location.
  - **Report (Open):** Organizational; needs a deliberate menu taxonomy pass (Definition-level).
  - ***OK, leave it for later***

- ~~Review all the plugins using plugin reviewer agent in parallel and generate a report here for not complying items~~
  - **Done:** ran the `plugin-reviewer` agent across 4 parallel partitions. Conventions checked: singleton-over-signal/slot, no custom `pyqtSignal`/`.connect`/`.emit`, interface inheritance, GPU usage (no silent CPU fallback), `threading.Thread`+`QTimer` polling, backend-registry abstraction, data immutability, batching/chunking, progress reporting. **No violations** of singleton/no-custom-signals/interface/data-immutability anywhere. Non-compliant items found:
    - **Runtime-breaking:** `070_ML_Models/010_PointNet2/010_Segmentation/000_train_model_plugin.py:39` imports `plugins.060_ML_Models...` (dir is `070_ML_Models`) → `ModuleNotFoundError` at load (already visible in test output). Fix the path or extract the shared code out of the numbered package.
    - **Backend abstraction bypassed (hardcoded libs + silent CPU fallback on ImportError):** `040_Clusters/030_cluster_by_class_plugin.py` & `040_Clusters/040_cluster_by_value_plugin.py` (`run_clustering_direct()` hardcodes cuML/hdbscan instead of `backend_registry.get_hdbscan()`); `020_Points/030_Analysis/010_knn_analysis_plugin.py` uses CPU `point_cloud.KNN()` (scipy KDTree, unbatched) instead of `backend_registry.get_knn()`.
    - **Heavy work on the main thread (no `threading.Thread`+QTimer):** `070_ML_Models/000_PointNet/000_Classification/010_train_model_plugin.py` (epoch loop driven by `processEvents()`); same pattern in `cluster_by_class`/`cluster_by_value`.
    - **Silent GPU→CPU fallback (logged only, not surfaced):** `backends/dbscan_backends.py`, `backends/hdbscan_backends.py`, `backends/masking_backends.py`, and `000_File/.../060_semantickitti_plugin.py`. Recurs across 3 backends → wants a `DECISIONS.md` entry (accept & log, or raise/surface).
    - **Silent wrong result on error:** `020_Points/020_Clustering/020_cluster_size_filter_plugin.py` returns an all-True "keep everything" fallback mask on any exception instead of raising.
    - **Arbitrary code execution:** `020_Points/010_Filtering/000_filtering_plugin.py:64` runs `exec(f"filter_mask = {filter_condition}")` on user input → restrict the namespace / use a safe evaluator.
    - **Batching/OOM:** `030_Selection/010_separate_selected_clusters_plugin.py` had an O(n) per-point Python loop (**fixed as part of issue below**); `backends/eigenvalue_backends.py` per-point eigvec loop; `050_Processing/010_average_distance_plugin.py` unbatched KNN.
    - **Missing cancel/progress:** `070_ML_Models/000_PointNet/000_Classification/000_generate_training_data_plugin.py` (no `global_progress`/`global_cancel_event`); `semantickitti` per-file loops don't check cancel.
    - **Minor/hygiene:** unused/duplicate imports (`las_laz`, `semantickitti`, PointNet trainers); `"dropdown"`/`"info"` param types work but aren't in CLAUDE.md's documented type list (doc gap).
  - *These are reported only (not fixed in this pass) except where noted. Recommend prioritising the runtime-breaking import and the backend-abstraction bypasses.*

- The algorithm of coordinate matching 2 branches looks very basic. we have to discuss it. For example, we have to use hash grid to speed up the process
  - **Note (discussion — not changed):** the current exact-match (now chunked, see Cancel fix) reduces each row to a void-byte key (CPU) / uses `cp.unique` (GPU) and tests membership — it's an **exact** O((n+m)·d) hash/sort match, *not* spatial. A **hash grid only helps approximate / tolerance matching** (snap-to-voxel within ε), which changes semantics (float-exact vs. nearest-within-radius). Recommended direction: keep the exact path as the safe fallback, and the *real* fix is **root-anchored bit-packed masks** (the open data-model item above) so most subtracts never reach coordinate matching at all. If approximate matching is desired, that's a Definition-level decision (tolerance, voxel size) → `DECISIONS.md`. **Left for discussion as requested.**

- Surface region growing outcome is not %100 correct. We have to investigate it.
  - **Note (investigation — not changed):** read `020_Points/020_Clustering/030_surface_region_growing_plugin.py`. Likely correctness contributors, in order of suspicion: (1) **`is_processed_voxel_t[chunk] = True` permanently retires a boundary voxel after its first fit** — if that voxel later gains surface points from a neighbour, it is never re-evaluated, so growth can stop short of full coverage (under-segmentation at concave joins). (2) Candidate acceptance uses the **boundary-voxel plane** to pre-filter (`distance_threshold`) and then a **separate candidate-voxel RANSAC** + angle gate; a candidate that the boundary plane barely misses is dropped before its own plane is ever fit. (3) `_CANDIDATE_RANSAC_POINTS_CAP = 256` caps the RANSAC sample set, so dense voxels fit on a non-representative subset (stochastic, non-deterministic — `seed=None`). (4) the 26-neighbour expansion + plane/AABB-cross test can miss surfaces thinner than `voxel_size`. Suggested next step: make boundary voxels **re-eligible** when they acquire new surface points (clear their processed flag on update), and set a fixed RANSAC `seed` for reproducibility while debugging. **Needs a dedicated session + test cloud to verify.**
- Filter plugin needs a dialog box to create a filter. We need to discuss it later. please remind me.
  - **Reminder (open, Definition-level):** the filtering plugin currently takes a raw Python expression string; you want a guided builder dialog instead (cf. the query_select "Select By Attributes" builder, which is the obvious model). Left for a later discussion as requested.

- The shape query service (`core/services/shape_query.py`) has no CPU fallback: on a machine with no NVIDIA GPU, or without enough GPU memory, any plugin using it fails.
  - **Report (Open / by design, Definition-level):** GPU only is deliberate (`DECISIONS.md` § 2026-09-28): it follows the CLAUDE.md rule "report GPU failure, never silently fall back to CPU", and the kernel is the one definition of "inside", so a point on a shape's edge is decided the same way on every run (pipeline replay). Failures raise `ShapeQueryError` saying what is missing (no CuPy/GPU, or "need about X MB"). Needs ~13 bytes a point on the GPU, +4 once cells are indexed (~2.9 GB at 168M).
  - **↳ If CPU-only machines must run these plugins:** a CPU path is feasible — `NeighborIndex` answered small shapes in 0.3-0.5 ms on the CPU, but 2-45x slower on large ones and with ~2.8 GB more RAM at 168M. It would need its own "inside" test kept identical to the kernel's rounding, or edge points could differ between machines. Reverses a recorded decision → new `DECISIONS.md` entry first.

- Double-check that we really need `NeighborIndex` (`core/services/neighbor_index.py`).
  - **Report (Open, to check):** after the linear growers moved to the shape query service, its **only** caller is the Linearity-Connected mode of `LinearRegionGrower` (`_grow_linearity_connected`), and even there it is only built when that mode is chosen. `DECISIONS.md` § 2026-09-28 already says it "is to be retired later".
  - **Why it is still there:** that mode asks one tiny question per point it grows — "radius r around this point" or "the k nearest points" — so thousands to millions of queries. The shape query service costs ~0.5 ms per query and has no k-nearest, so it is the wrong tool for this. `NeighborIndex` costs ~16 s and ~26 bytes a point to build at 60M.
  - **To check before keeping or removing it:**
    - Is Linearity-Connected mode actually used? If not, remove the mode and `NeighborIndex` with it.
    - Could the mode reuse neighbours that already exist? It needs eigenvalues, which were computed from a k-nearest search upstream — if those neighbour lists are kept (or cheap to keep), the growth needs no index at all.
    - Otherwise, could it grow one whole frontier at a time through the GPU k-nearest backend (`backend_registry.get_knn()`) instead of one point at a time?
  - If none of these work out, `NeighborIndex` stays, but only for this mode.


- ### Reviews

- Calculated Normal Z by Estimate Normals plugin are between 0 and 1, while I expected -1 to +1. Why?  
 

- ### Not issues — recorded so they are not "fixed" by mistake

- **Polygon selection includes LOD-hidden points, deliberately.** A lasso is an *area* gesture; the region exists independently of how many points were drawn. Returning only rendered points would make `Separate Selected Points` produce a subsample full of holes, with a different result at every zoom level. Storing the polygon + camera matrices makes the re-test exact and reproducible, which is what pipeline replay needs.
- **Single-point click does NOT widen, deliberately.** A click is an *identity* gesture naming one point; the pick tolerance absorbs mouse imprecision and is not a capture radius. Widening it would scale the points-per-click with the LOD factor, silently and with no feedback. The practical route to a specific LOD-hidden point is to zoom in until AUTO-LOD draws it — or, where the point is hidden for a semantic rather than a density reason, to promote it to a visible cluster first, as the line-extension window does with `pick_candidates`.
