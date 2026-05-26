# RANSAC Module

The RANSAC infrastructure lives at `core/services/ransac/`. It implements the design ratified in `DECISIONS.md` 2026-05-26: a single-cloud `fit()` contract, model-owned refit, pluggable sampler/scorer, and an extensible primitive set covering line, plane, cylinder, and cone.

The guiding principle: **RANSAC's contract is "single point set in, single model out."** Region growing, voxel iteration, multi-model extraction, and curve-along-segments are orchestration above RANSAC, not variations of it. The infrastructure draws that line explicitly.

---

## Architecture

```mermaid
graph TB
    subgraph Plugins["plugins/ — user-facing"]
        P1["e.g. pipe_detection"]
        P2["e.g. wall_extraction"]
        P3["e.g. region growing"]
        P4["e.g. multi-model extractor"]
    end

    subgraph Orch["Orchestrators — iteration policies"]
        ONESHOT["One-shot fit<br/>user pick / cluster / ROI<br/>→ single RANSAC call"]
        GROW["Iterative region growing<br/>seed → fit → expand → refit"]
        MULTI["Multi-model extraction<br/>fit dominant, remove inliers,<br/>repeat"]
        CURVE["Curve-as-segments<br/>sequence of short line fits<br/>(arcs, splines, catenaries)"]
    end

    subgraph Engine["core/services/ransac/ — single-cloud contract"]
        FIT["fit(points, model_type,<br/>threshold, normals=None, ...)<br/>→ (model, inlier_mask)"]
        FITMANY["fit_many(points_list, ...)<br/>batched fast-path<br/>(GPU when available)"]
        SAMPLER["Sampler<br/>• UniformSampler (default)<br/>• NAPSAC / PROSAC (future)"]
        SCORER["Scorer<br/>• MSACScorer (default)<br/>• InlierCountScorer"]
    end

    subgraph Primitives["core/services/ransac/primitives/"]
        LINE["LineModel<br/>min=2 pts<br/>refit: SVD (1st PC)"]
        PLANE["PlaneModel<br/>min=3 pts<br/>refit: SVD (3rd PC)"]
        CYL["CylinderModel<br/>min=2 pts + 2 normals<br/>refit: iterative (LM)"]
        CONE["ConeModel<br/>min=3 pts + 3 normals<br/>refit: iterative (LM)"]
    end

    subgraph Backends["Backends"]
        CPU["CPU (NumPy + scipy)<br/>all primitives"]
        GPU["GPU (Torch/CUDA)<br/>line, plane — hot loop<br/>+ batched SVD refit"]
    end

    P1 --> ONESHOT
    P2 --> MULTI
    P3 --> GROW
    P4 --> MULTI
    GROW --> CURVE

    ONESHOT --> FIT
    GROW --> FIT
    MULTI --> FIT
    CURVE --> FIT
    GROW -.opt-in.-> FITMANY

    FIT --> SAMPLER
    FIT --> SCORER
    FIT --> Primitives
    FITMANY --> Primitives
    Primitives --> Backends

    classDef plugin fill:#dcfce7,stroke:#16a34a,color:#000
    classDef orch fill:#fde68a,stroke:#b45309,color:#000
    classDef engine fill:#e0e7ff,stroke:#4338ca,color:#000
    classDef model fill:#fce7f3,stroke:#be185d,color:#000
    classDef backend fill:#cffafe,stroke:#0e7490,color:#000

    class P1,P2,P3,P4 plugin
    class ONESHOT,GROW,MULTI,CURVE orch
    class FIT,FITMANY,SAMPLER,SCORER engine
    class LINE,PLANE,CYL,CONE model
    class CPU,GPU backend
```

### Layer responsibilities

| Layer            | Responsibility                                                                                | Does NOT concern itself with                  |
|------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------|
| **Plugin**       | User interaction, parameter dialogs, tree branch creation                                     | RANSAC, iteration policy, backends            |
| **Orchestrator** | Where points come from, how to iterate, bookkeeping                                           | How a single fit is computed                  |
| **Engine**       | `fit` contract: given points, find best model + inliers                                       | What the points represent, what happens next  |
| **Model**        | Minimal-sample fit, distance function, **refit on inliers**, degeneracy reporting             | Sampling strategy, scoring strategy           |
| **Backend**      | NumPy or Torch tensor ops                                                                     | Geometry; receives operations from primitives |

---

## Primitives

### Status

| Primitive | CPU `fit`    | GPU `fit` (≥ 50 k pts) | Batched `fit_many` | Normals      |
|-----------|--------------|------------------------|---------------------|--------------|
| Line      | ✓ (Phase A)  | ✓ (Phase B)            | ✓ (Phase B, GPU)    | optional     |
| Plane     | ✓ (Phase A)  | ✓ (Phase B)            | ✓ (Phase B, GPU)    | optional     |
| Cylinder  | ✓ (Phase C)  | — (future)             | CPU fallback        | **required** |
| Cone      | ✓ (Phase C)  | — (future)             | CPU fallback        | **required** |

### Geometry

| Primitive | State                                            | Minimal fit (no normals) | Minimal fit (with normals)                 | Refit                |
|-----------|--------------------------------------------------|--------------------------|--------------------------------------------|----------------------|
| Line      | `point`, `direction` (unit)                      | 2 pts                    | 2 pts                                      | SVD (1st PC)         |
| Plane     | `point`, `normal` (unit)                         | 3 pts                    | not yet used in code                       | SVD (3rd PC)         |
| Cylinder  | `point`, `direction` (unit), `radius`            | 5 pts (not implemented)  | **2 pts + 2 normals** (closed-form)        | Levenberg–Marquardt  |
| Cone      | `apex`, `axis` (unit), `half_angle` (rad)        | 6 pts (not implemented)  | **3 pts + 3 normals** (tangent-plane LS)   | Levenberg–Marquardt  |

Cylinder and cone refit is iterative (`scipy.optimize.least_squares(..., method='lm')`), which does not vectorise across rows on GPU. A GPU batched hot loop with CPU per-row refit is plausible but unbuilt; it lands only if a real workload demands it.

Curves (arcs, splines, catenaries) are **out of scope as RANSAC primitives** — they are approximated as sequences of line segments by an iterative orchestrator (the curve-as-segments pattern).

---

## Public API

```python
from core.services.ransac import fit, fit_many

model, inlier_mask = fit(
    points,                 # (N, 3) np.ndarray
    model_type,             # "line" | "plane" | "cylinder" | "cone"
    threshold,              # float — distance threshold
    normals=None,           # (N, 3) — required for cylinder/cone
    max_iterations=1000,
    min_inlier_ratio=0.3,
    sampler=None,           # CPU only; default UniformSampler
    scorer=None,            # CPU only; default MSACScorer
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

### Backend dispatch (`backend="auto"`)

- `fit`: GPU when CUDA is available **and** `N ≥ 50_000` and the primitive sets `supports_gpu = True`. Below 50 k points host↔device transfer dominates, so CPU wins; the threshold is a constant in `engine.py` and easy to tune.
- `fit_many`: GPU when CUDA is available and the primitive sets `supports_gpu = True`, regardless of row size (the parallelism wins). Falls back to a sequential CPU loop otherwise.
- Cylinder and cone always run on CPU because `supports_gpu = False`.
- Explicit `backend="gpu"` on a model that doesn't support GPU raises a clean `RuntimeError`.

### Orchestrator contract

`fit` and `fit_many` take raw arrays, not `PointCloud` objects. The orchestrator is responsible for:

- selecting the points before the call (user pick, cluster filter, KD-tree neighbourhood, voxel contents, ROI, mask, …),
- supplying matching normals when the model requires them,
- mapping the returned inlier mask back to its own indexing,
- deciding what to do on failure (skip this voxel, stop growing, return partial result, …).

---

## Package layout

```
core/services/ransac/
    __init__.py        # public re-exports
    engine.py          # fit, fit_many, backend dispatch, GPU loop
    base.py            # RansacModel / Sampler / Scorer ABCs
    samplers.py        # UniformSampler
    scorers.py         # MSACScorer (default), InlierCountScorer
    primitives/
        __init__.py
        line.py        # LineModel       (CPU + GPU batched)
        plane.py       # PlaneModel      (CPU + GPU batched)
        cylinder.py    # CylinderModel   (CPU only; LM refit via scipy)
        cone.py        # ConeModel       (CPU only; LM refit via scipy)
```

Each primitive carries both the CPU single-cloud methods (`fit_minimal`, `distances`, `refit`) and, when `supports_gpu = True`, four batched GPU classmethods (`fit_minimal_batched_gpu`, `distances_batched_gpu`, `refit_batched_gpu`, `unpack_to_model`). Refit logic lives on the model so each primitive owns its own least-squares routine: SVD on CPU and `torch.linalg.svd` on GPU for line/plane; `scipy.optimize.least_squares` with `method='lm'` for cylinder/cone.

External primitives (added outside this package, e.g. for experiments) plug in via `register_model(name, cls)` in `engine.py`.

---

## Algorithmic baseline

- **Sampler:** uniform random by default (`UniformSampler`). Sampler is a pluggable CPU-side interface so NAPSAC (spatial locality) or PROSAC (quality-weighted) can be added when a real scenario demands them. The GPU path hardcodes uniform sampling — custom samplers are CPU-only by design.
- **Scorer:** MSAC truncated-quadratic loss by default (`MSACScorer`). Strictly better than binary inlier counting at no extra cost: residuals contribute `min(d², threshold²)`, so the score degrades smoothly across the inlier boundary rather than at a hard edge. `InlierCountScorer` is provided for parity comparisons and debugging.
- **Refinement (LO-style):** after the random-sample loop picks a best hypothesis, the engine calls `model.refit(inliers)` to refine on the full inlier set, then re-evaluates inliers against the refined model. If refit reports degeneracy, the engine falls back to the pre-refit minimal model so the caller still gets a usable result.

---

## Orchestration patterns

1. **One-shot fit** — single call, e.g. user selects a cluster and asks "fit a cylinder to this."
2. **Iterative region growing** — seed region → fit → expand by inlier reach → refit on expanded set → repeat. Used by cable tracing (line), surface growing (plane), and pipe tracing (cylinder).
3. **Multi-model extraction** — fit dominant model, mark inliers, recurse on remaining points. Used for "find all walls" or "find all pipes in this scan."
4. **Curve-as-segments** — special case of (2): fit a short line segment, advance, fit the next segment. Sequence of `LineModel` fits captures arcs, splines, and catenaries without dedicated primitives.

---

## Tests

`unit_test/ransac/` — plain Python scripts (`python unit_test/ransac/test_*.py`):

| File                          | Coverage                                                                       |
|-------------------------------|--------------------------------------------------------------------------------|
| `test_line_fit.py`            | CPU line: recovery, pure-noise rejection, reproducibility, too-few-points     |
| `test_plane_fit.py`           | CPU plane: recovery, collinear-input rejection, pure-noise, reproducibility   |
| `test_cylinder_fit.py`        | CPU cylinder: recovery (radius ~ 0.04 % error), normals-required, reproducibility |
| `test_cone_fit.py`            | CPU cone: recovery (half-angle ~ 0.006° error), cone-vs-cylinder discrimination |
| `test_fit_many.py`            | `fit_many` across mixed pass/fail rows; runs on whichever backend is available |
| `test_gpu_fit_line.py`        | GPU line; skipped cleanly without CUDA                                         |
| `test_gpu_fit_plane.py`       | GPU plane; skipped cleanly without CUDA                                        |
| `test_cpu_gpu_parity.py`      | CPU and GPU agree on direction/normal and inlier count for same input         |

---

## Open questions and future work

- **Per-point weights as a first-class engine input.** Useful for MSAC scoring with point-confidence priors and for future PROSAC sampling. Cheap to wire in now (`fit(..., weights=None)`), expensive to retrofit later. Not yet built.
- **NAPSAC / PROSAC samplers.** Both are well-understood; NAPSAC is more aligned with point-cloud data (spatial locality) than PROSAC (quality-weighted) because raw point clouds don't carry an obvious quality signal without extra precomputation.
- **GPU batched cylinder/cone.** Closed-form minimal fit and distance functions vectorise fine; the iterative LM refit does not. A workable approach: GPU hot loop + per-row CPU LM refit on the survivors. Defer until a real workload demonstrates the need.
- **Curve-as-segments helper as a shared service.** The pattern is implemented inside individual orchestrators today; promoting it to a reusable utility makes sense when a second consumer appears.
- **Single-sheet cone restriction.** Both sheets of the infinite double-cone are currently treated as one surface. Adding a "wide-end only" mask filter is a one-line change if a consumer needs it.
- **Backend-registry integration.** `core/services/ransac` currently dispatches CPU/GPU in-module via `torch.cuda.is_available()`. Routing through `global_backend_registry.get_ransac()` would only matter if something outside this package needs to choose a RANSAC backend declaratively — no consumer does today.

---

## Design lineage

- **`DECISIONS.md` 2026-05-26** — ratified the single-cloud contract, model-owned refit, pluggable sampler/scorer, four-primitive scope, and out-of-scope items (splines, catenaries, arcs as primitives).
- **Phase A** (commit `7382ff0`) — package skeleton + CPU line and plane.
- **Phase B** (commit `cecb9ab`) — GPU backend and `fit_many` for line and plane.
- **Phase C** (commit `20bf7a9`) — CPU cylinder and cone with LM refit.
