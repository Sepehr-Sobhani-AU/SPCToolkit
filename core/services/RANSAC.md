# RANSAC Module

The RANSAC infrastructure lives at `core/services/ransac/`. It implements the design ratified in `DECISIONS.md` 2026-05-26: a single-cloud `fit()` contract, model-owned refit, pluggable sampler/scorer, and an extensible primitive set. The longer-form design (layered architecture, four-primitive scope, future direction) is in Part 2 below; Part 1 captures what is built right now.

---

## Part 1 — Current state

### Primitives and backends

| Primitive | CPU `fit` | GPU `fit` (≥50k pts) | Batched `fit_many` |
|-----------|-----------|----------------------|---------------------|
| Line      | ✓ (Phase A) | ✓ (Phase B)        | ✓ (Phase B)        |
| Plane     | ✓ (Phase A) | ✓ (Phase B)        | ✓ (Phase B)        |
| Cylinder  | — (Phase C) | — (Phase C/D?)     | —                  |
| Cone      | — (Phase C) | — (Phase C/D?)     | —                  |

### Public API

```python
from core.services.ransac import fit, fit_many

model, inlier_mask = fit(
    points,                 # (N, 3) np.ndarray
    model_type,             # "line" | "plane"
    threshold,
    normals=None,
    max_iterations=1000,
    min_inlier_ratio=0.3,
    sampler=None,           # CPU only
    scorer=None,            # CPU only
    seed=None,
    backend="auto",         # "auto" | "cpu" | "gpu"
)

models, masks = fit_many(
    points_list,            # list[np.ndarray (N_b, 3)] of length B
    model_type,
    threshold,
    normals_list=None,
    max_iterations=1000,
    min_inlier_ratio=0.3,
    seed=None,
    backend="auto",
)
```

`backend="auto"` picks GPU when CUDA is available *and* (for `fit`) the input is large enough to amortise host↔device transfer — currently `N ≥ 50_000`. `fit_many` always prefers GPU when available, since the parallelism wins regardless of row size. `fit_many` falls back to a sequential CPU loop when CUDA is unavailable.

### Package layout

```
core/services/ransac/
    __init__.py
    engine.py          # fit, fit_many, backend dispatch, GPU loop
    base.py            # RansacModel / Sampler / Scorer ABCs
    samplers.py        # UniformSampler
    scorers.py         # MSACScorer (default), InlierCountScorer
    primitives/
        __init__.py
        line.py        # LineModel (CPU + GPU batched)
        plane.py       # PlaneModel (CPU + GPU batched)
```

Each primitive carries both the CPU single-cloud methods (`fit_minimal`, `distances`, `refit`) and optional batched GPU classmethods (`fit_minimal_batched_gpu`, `distances_batched_gpu`, `refit_batched_gpu`, `unpack_to_model`). Primitives that implement the GPU classmethods set `supports_gpu = True`. Refit lives on the model — SVD via `np.linalg.svd` on CPU and `torch.linalg.svd` on GPU.

### Tests

`unit_test/ransac/` — plain Python scripts:

- `test_line_fit.py`, `test_plane_fit.py` — Phase A CPU.
- `test_fit_many.py` — runs on whichever backend is available; covers recovery, mixed pass/fail, empty input.
- `test_gpu_fit_line.py`, `test_gpu_fit_plane.py` — GPU-only, skip cleanly when CUDA is unavailable.
- `test_cpu_gpu_parity.py` — same input on both backends recovers the same geometry within numerical tolerance.

---

## Part 2 — Proposed design

### Guiding principle

RANSAC's contract is **single point set in, single model out**. Region growing, voxel iteration, multi-model extraction, curve-along-segments — all of these are **orchestration** above RANSAC, not variations of it. The infrastructure draws that line explicitly.

### Layered architecture

```mermaid
graph TB
    subgraph Plugins["plugins/ — user-facing"]
        P1["e.g. power_line_detection"]
        P2["e.g. surface_region_growing"]
        P3["e.g. pipe_detection (future)"]
        P4["e.g. wall_extraction (future)"]
    end

    subgraph Orch["Orchestrators — iteration policies"]
        ONESHOT["One-shot fit<br/>user pick / cluster / ROI<br/>→ single RANSAC call"]
        GROW["Iterative region growing<br/>cable tracer, surface grower,<br/>curve-as-segments"]
        MULTI["Multi-model extraction<br/>fit dominant, remove inliers,<br/>repeat"]
    end

    subgraph Engine["core/services/ransac/ — single-cloud contract"]
        FIT["fit(points, normals=None,<br/>model_type, threshold, ...)<br/>→ (model, inlier_mask)"]
        FITMANY["fit_many(rows, ...)<br/>optional batched fast-path<br/>(performance opt-in)"]
        SAMPLER["Sampler (pluggable)<br/>• uniform (default)<br/>• NAPSAC / PROSAC (future)"]
        SCORER["Scorer (pluggable)<br/>• MSAC (default)<br/>• inlier-count<br/>• LO refinement"]
    end

    subgraph Models["core/services/ransac/primitives/"]
        LINE["LineModel<br/>min=2 pts<br/>refit: SVD (1st PC)"]
        PLANE["PlaneModel<br/>min=3 pts (or 1 pt + normal)<br/>refit: SVD (3rd PC)"]
        CYL["CylinderModel<br/>min=2 pts + 2 normals<br/>refit: iterative (LM)"]
        CONE["ConeModel<br/>min=3 pts + 3 normals<br/>refit: iterative (LM)"]
    end

    subgraph Backends["Backends"]
        CPU["CPU backend (NumPy)<br/>all models"]
        GPU["GPU backend (Torch/CUDA)<br/>vectorised hot loop<br/>refit defers to CPU<br/>for cylinder/cone"]
    end

    P1 --> GROW
    P2 --> GROW
    P3 --> ONESHOT
    P4 --> MULTI

    ONESHOT --> FIT
    GROW --> FIT
    MULTI --> FIT
    GROW -.opt-in.-> FITMANY

    FIT --> SAMPLER
    FIT --> SCORER
    FIT --> Models
    FITMANY --> Models
    Models --> Backends

    classDef plugin fill:#dcfce7,stroke:#16a34a,color:#000
    classDef orch fill:#fde68a,stroke:#b45309,color:#000
    classDef engine fill:#e0e7ff,stroke:#4338ca,color:#000
    classDef model fill:#fce7f3,stroke:#be185d,color:#000
    classDef backend fill:#cffafe,stroke:#0e7490,color:#000

    class P1,P2,P3,P4 plugin
    class ONESHOT,GROW,MULTI orch
    class FIT,FITMANY,SAMPLER,SCORER engine
    class LINE,PLANE,CYL,CONE model
    class CPU,GPU backend
```

### Layer responsibilities

| Layer | Responsibility | Does NOT concern itself with |
|-------|----------------|------------------------------|
| **Plugin** | User interaction, parameter dialogs, tree branch creation | RANSAC, iteration policy, backends |
| **Orchestrator** | Where points come from, how to iterate (region growth, multi-model loop, curve-as-segments), bookkeeping | How a single fit is computed |
| **Engine** | `fit` contract: given points, find best model + inliers | What the points represent, what happens next |
| **Model** | Minimal-sample fit, distance function, **refit on inliers**, degeneracy check | Sampling strategy, scoring strategy |
| **Backend** | NumPy or Torch tensor ops | Geometry; receives operations from primitives/engine |

### Canonical contract

```
fit(
    points: array (N, 3),
    model_type: "line" | "plane" | "cylinder" | "cone",
    threshold: float,
    normals: array (N, 3) | None,
    max_iterations: int,
    backend: "auto" | "cpu" | "gpu",
    sampler: Sampler = UniformSampler(),
    scorer: Scorer = MSACScorer(),
) -> (model, inlier_mask)  or  (None, None) on failure
```

The orchestrator is responsible for:
- selecting the points before the call (user pick, cluster filter, KD-tree neighbourhood, voxel contents, ROI, mask, …),
- mapping the returned inlier mask back to its own indexing,
- deciding what to do on failure (skip this voxel, stop growing, return partial result, …).

### Primitive set (initial scope)

Four primitives cover the vast majority of as-built point-cloud features: structural elements, MEP, and linear infrastructure.

| Primitive | Min sample (no normals) | Min sample (with normals) | Refit method |
|-----------|------------------------|--------------------------|--------------|
| Line      | 2 pts                  | 2 pts                    | SVD (1st PC) |
| Plane     | 3 pts                  | 1 pt + normal            | SVD (3rd PC) |
| Cylinder  | 5 pts (fragile)        | **2 pts + 2 normals**    | iterative (LM) |
| Cone      | 6 pts (very fragile)   | **3 pts + 3 normals**    | iterative (LM) |

Curves (arcs, splines, catenaries) are **out of scope as RANSAC primitives** — they are approximated as sequences of line segments by an iterative orchestrator, the same pattern `PowerLineTracer` already uses.

### Batched fast-path (`fit_many`)

A performance opt-in for orchestrators that need to fit thousands of small models in parallel (the surface-region-growing case). Same model contract, same threshold, same options — only the input shape changes (rows of point sets) and the output shape changes (rows of models). The basic single-cloud `fit` contract is unaffected. Most consumers will never call `fit_many`.

### Algorithmic baseline

- **Sampler:** uniform random by default. Sampler is a pluggable interface so NAPSAC (spatial locality) or PROSAC (quality-weighted) can be added later when a real scenario demands them.
- **Scorer:** MSAC (truncated-quadratic loss) by default — strictly better than binary inlier counting at no extra cost, handles noisy thresholds gracefully.
- **Refinement:** LO-style — refit best hypothesis on all its inliers before final inlier evaluation. Refit logic lives **on each model**, not in the engine.

### Orchestration patterns the engine supports

1. **One-shot fit** — single call, e.g. user selects a cluster and asks "fit a cylinder to this."
2. **Iterative region growing** — seed region → fit → expand by inlier reach → refit on expanded set → repeat. Used by cable tracing (line) and surface growing (plane). Same pattern works for pipe tracing (cylinder).
3. **Multi-model extraction** — fit dominant model, mark inliers, recurse on remaining points. Used for "find all walls" or "find all pipes in this scan."
4. **Curve-as-segments** — special case of (2): fit a short line segment, advance, fit the next segment. Sequence of `LineModel` fits captures arcs, splines, catenaries without dedicated primitives.

---

## Part 3 — Open design decisions

These belong in a Definition session, not a Construction one. Listed for visibility:

1. **Migration vs coexistence.** Does the new infrastructure replace `_batched_ransac` and `RANSAC.run` immediately, or do they coexist during a transition? Replacement is cleaner; coexistence is lower risk.
2. **`fit_many` from day one?** Defining it in the engine ABC now (even if only `PlaneModel` implements it initially) avoids a later interface change. Alternative: ship single-cloud `fit` first, add `fit_many` only when the surface plugin is migrated.
3. **Where does the curve-as-segments helper live?** It is shared between cable tracing and any future curved-pipe / curved-cable scenario. Build it as a service when the second consumer arrives, not before.
4. **Per-point weights as a first-class engine input?** Useful for MSAC scoring with point confidence, and for future PROSAC sampling. Cheap to include now, expensive to add later.

---

## Part 4 — File references (current code)

- `core/services/ransac.py` — engine + `LineModel3D` (CPU, NumPy)
- `core/services/power_line_tracer.py` — line-RANSAC consumer (region-growing orchestrator)
- `plugins/060_Infrastructure/000_power_line_detection_plugin.py` — power-line plugin entry
- `plugins/020_Points/020_Clustering/030_surface_region_growing_plugin.py` — surface-region-growing plugin and `_batched_ransac` (GPU plane RANSAC)
