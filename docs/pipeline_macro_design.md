# Pipeline / Macro — Design Notes

**Status:** Exploratory design from discussion (2026-06-17). In scope, low
priority, **not a milestone** — see `PROJECT.md` (In scope) and `DECISIONS.md`
2026-05-21. Build only against a real recipe (scenario-driven), never
speculatively. Nothing here is approved for construction yet.

## Goal

Save the chain of steps that produced a branch as a reusable **macro**, and
replay it on another cloud to reach the same kind of result with minimal
clicking.

## Terminology — three different things

- **A. Macro / replay** — record steps you already ran, save them, re-run. ← *this design*
- **B. Pipeline editor (DAG)** — author recipes from scratch in a node graph. *Not this.*
- **C. Batch / fan-out** — run one macro over a folder of clouds. *Later wrapper on A.*

Note: "batch" already means spatial tiling of one big cloud (`BatchProcessor`) —
a different axis. Don't overload the word.

## What the codebase already gives us

- Every analysis result node stores `tags=[plugin_name, params]` + `parent_uid`
  (`application_controller.py:144`).
- Walking `parent_uid` from a branch up to root reconstructs the ordered recipe
  for free. This is deliberate groundwork (PROJECT.md). Verified in current code.

## Capture — "Save Branch as Pipeline"

- Action plugin. Walk the selected branch's parent chain → ordered list of
  `{plugin, params, result_type}` → small **JSON** file (readable, editable).
- Drop the old UIDs — they don't transfer between clouds. Only
  `(plugin, params, order)` matter.

## Replay — "Run Pipeline"

- Action plugin. Pick a macro + a target branch.
- A **sequencer, not a for-loop**: kick step N through the existing
  `run_analysis` path; advance to N+1 from the same 100 ms completion poll.
  (Only one analysis runs at a time, so a plain loop fights the async model.)
  Output of step N becomes input of step N+1.
- Fits singleton + QTimer polling; no signals/slots.

## Selection handling (the crux)

Selection is **not** stored in params — `separate_selected_*` read it live from
the viewer at execute-time (`get_selection_mask_for`). So "select first, then
run" is already the plugins' native behavior.

| Case | Works front-loaded? |
|------|---------------------|
| No selection (pure analysis chain) | ✅ trivially |
| Selection at the **start** (crop → fixed chain) | ✅ one up-front selection |
| Selection **mid-chain** (cluster → pick clusters) | ❌ target doesn't exist yet at start |

### Seed-coordinate idea (fixes most mid-chain picks)

- Store the clicked point's **coordinate** as a param — *not* its row index, so
  it survives upstream subsampling / SOR.
- At the relevant step, pick the cluster **containing / nearest** that coordinate.
- Needs a **max-distance guard**: if the nearest point is too far, warn instead
  of silently grabbing the wrong cluster.
- Reliable on **same-cloud re-runs**; **unreliable across different clouds**
  (the coordinate frame differs — import applies a per-cloud shift).
- Example: click a ground point up front → after clustering, the cluster at that
  spot is separated as ground. (Ground also has geometric rules — lowest /
  biggest flat surface — so seeds matter most for *idiosyncratic* picks with no
  clean rule.)

## Known gap — Action plugins

- `classify_cluster`, `cut_cluster`, `merge_clusters`, `remove_clusters` are
  **Action** plugins → they leave no `(plugin, params)` lineage entry → a
  chain-of-analysis capture won't include them.
- The two `separate_selected_*` are **Analysis** → captured fine.
- Decision pending: scope target recipes to Analysis (+ seed selection), or
  extend capture to Action plugins later.

## v1 scope boundaries

- Linear chains only (DAG / branch / merge later).
- Params replayed verbatim + seed coordinates (no clever re-derivation).
- Single cloud (folder fan-out is a later wrapper).
- Semi-automated: may still pause at any step it can't auto-fill.

## Open questions

1. **First real target recipe?** Its shape decides whether the basic version
   is enough (analysis-only / start-selection) or needs seeds / pauses.
2. Retrospective capture (recommended) vs. forward authoring.
3. How Action-plugin steps in that recipe are handled.
4. Macro file format / storage — standalone JSON vs. inside the project file.

## Implementation status (2026-06-18)

Built and in the tree (`core/services/pipeline.py`, `application/pipeline_runner.py`,
`plugins/055_Pipeline/`, hooks in `gui/main_window.py`):

- **Linear analysis chains** replay via the sequencer (async path + completion poll).
- **Producer action plugins** replay too (e.g. `scale`, `estimate_normals`): run
  synchronously on the main thread, and the produced branch is recovered by
  diffing the node set, choosing the new node whose `data_type` matches the
  step's recorded `produces` (so `estimate_normals`, which emits both a normals
  and an eigenvalues branch, continues from normals).
- **Cross-branch references** (`projected_distance.reference_node`,
  `subtract.subtract_node`, `intersect.other_uid`): the **UUID is never stored**
  — capture strips it to a *hint* (the original branch's display name). At run,
  a **pre-flight** step validates all plugins exist and then opens one dialog
  (reusing `DynamicDialog`) to bind each reference to a live branch before the
  run starts. The referenced branch must already exist in the project.

- **Mid-chain selection** (`separate_selected_*`): replay **pauses** at the step
  — a non-modal prompt appears and the intermediate is made visible — the user
  selects on the real clusters, clicks Continue, and the run resumes (the worker
  reads the live selection before it's cleared). Detected via a
  `requires_selection()` hook on the plugin interface. Seed-coordinate
  front-loading was considered and **not** taken (unreliable across clouds and
  through the pipeline's scale steps).

Still deferred: full **DAG** capture (replaying a reference branch's own
sub-chain instead of binding to an existing one); folder fan-out.
