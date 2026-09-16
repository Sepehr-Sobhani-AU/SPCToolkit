# Plugin Architecture

Deep reference for SPCToolkit's plugin system: discovery, parameter schema, execution flows, backend system, runtime management, and the full plugin inventory. For high-level architecture (layers, data flows, threading, reconstruction), see **ARCHITECTURE.md** Section 7. For generated component / sequence / class / data-flow diagrams, see **docs/architecture_diagrams.md**.

## 1. Plugin Interfaces

Source: `plugins/interfaces.py`

### Plugin (Analysis Plugin)

```python
class Plugin(ABC):
    # Required
    def get_name(self) -> str: ...
    def get_parameters(self) -> Dict[str, Any]: ...
    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]: ...

    # Optional hooks (base implementations return None)
    def confirm_before_execute(self, data_node, params) -> Optional[str]: ...
    def requires_selection(self) -> Optional[str]: ...
    def build_param_dialog(self, parent, last_params=None) -> Optional[Any]: ...
```

The `execute()` return tuple contains:
- **result** — the output object (`PointCloud`, masks array, cluster labels, etc.)
- **result_type** — string identifier (e.g. `"point_cloud"`, `"masks"`, `"cluster_labels"`)
- **dependencies** — list of UIDs this result depends on (typically `[data_node.uid]`)

### Optional Hooks

| Hook | Runs on | Purpose |
|------|---------|---------|
| `confirm_before_execute(node, params)` | Main thread, **before** the worker starts | Return warning text to show a Yes/No prompt (e.g. "this falls back to a slow path, proceed?"); return `None` to run silently. Must be cheap and side-effect free. |
| `requires_selection()` | Main thread, before params are collected | Declares what the plugin consumes: `None`/`False` (nothing), `"points"`, `"branches"`, or `"either"`. Legacy `True` is read as `"points"`. Drives the selection gate (§5). |
| `build_param_dialog(parent, last_params)` | Main thread, in place of `DynamicDialog` | Return a `QDialog` exposing `get_parameters()` when the schema form can't express the input (e.g. a variable-row query builder). Returned params **must stay JSON-serialisable** so the step replays from a saved pipeline. |

### ActionPlugin

```python
class ActionPlugin(ABC):
    def get_name(self) -> str: ...
    def get_parameters(self) -> Dict[str, Any]: ...
    def execute(self, main_window, params: Dict[str, Any]) -> None: ...

    def requires_selection(self) -> Optional[str]: ...   # same contract as Plugin
```

Action plugins receive `main_window` instead of `data_node`. They perform operations directly (open dialogs, trigger I/O, modify UI state) and return nothing. `ActionPlugin` does **not** have `confirm_before_execute` or `build_param_dialog` — it collects and confirms whatever it needs inside `execute()`.

### Alias

```python
AnalysisPlugin = Plugin  # Legacy alias for backward compatibility
```

## 2. Discovery & Registration

Source: `plugins/plugin_manager.py`

### Startup Flow

```
PluginManager.__init__()
  └─ load_plugins()
       └─ os.walk(plugin_root)          # Recursive directory scan
            ├─ Skip dirs: __pycache__, hidden (.*); skip files: __-prefixed
            ├─ Compute menu_path from relative folder path
            │    root dir → menu_path = None (system plugin, not in menus)
            │    subdir   → menu_path = relative path with "/" separators
            └─ _load_plugin_file(directory, filename, menu_path, is_system_plugin)
                 ├─ importlib.import_module(package.module_name)
                 ├─ inspect.getmembers(module, isclass)
                 │    Check issubclass(obj, ActionPlugin) first (more specific)
                 │    Then  issubclass(obj, Plugin)
                 └─ _register_plugin(plugin_class, menu_path, plugin_type)
                      ├─ Instantiate → get_name() → store in registries
                      └─ Warn on duplicate names (overwrites existing)
```

### Internal Registries

| Registry | Type | Contents |
|----------|------|----------|
| `plugins` | `Dict[str, Tuple[Type, str, str]]` | `{name: (class, menu_path, "data"\|"action")}` |
| `menu_structure` | `Dict[str, List[str]]` | `{menu_path: [plugin_names]}` |
| `analysis_plugins` | `Dict[str, Type[Plugin]]` | Data-processing plugins only |
| `action_plugins` | `Dict[str, Type[ActionPlugin]]` | Action plugins only |

> **Gotcha:** `inspect.getmembers` returns *imported* classes too, not just ones defined in the
> file. A plugin module that imports another plugin class re-registers it under the importing
> file's menu path and logs `Warning: Plugin '<name>' is already registered. Overwriting.`
> Import the module, not the class, when one plugin needs another's helpers.

## 3. Menu Hierarchy

### Folder → Menu Mapping

The folder structure under `plugins/` directly defines the menu hierarchy:

```
plugins/
├── 000_File/
│   ├── 000_Import Point Cloud/    → File > Import Point Cloud (submenu)
│   │   ├── 000_e57_plugin.py      →   Import E57
│   │   └── 020_ply_plugin.py      →   Import Ply
│   ├── 010_load_project_plugin.py → File > Load Project
│   └── 040_Export Point Cloud/    → File > Export Point Cloud (submenu)
├── 020_Points/
│   ├── 000_Subsampling/           → Points > Subsampling (submenu)
│   └── 020_Clustering/            → Points > Clustering (submenu)
└── ...
```

### Numeric Prefix Convention

Folders and files use **3-digit + underscore** prefixes (`000_`, `010_`, `085_`) to control ordering. Prefixes are stripped for display only via `PluginManager._strip_prefix()` (regex: `^\d{3}_`). The raw prefixed names remain in `menu_path` keys and file names.

> **Note:** `_create_menu_hierarchy()` strips the prefix but does **not** replace underscores, so a
> folder named `070_ML_Models` renders in the menubar as `ML_Models`. Underscore→space
> conversion happens only for plugin *names*, via `_format_plugin_name()` below.

### Display Name Formatting

`MainWindow._format_plugin_name()` transforms plugin names for menu display:

1. Strip numeric prefix if present (defensive — `get_name()` values typically don't have them)
2. Replace underscores with spaces
3. Title Case each word
4. Keep known acronyms uppercase: **DBSCAN**, **HDBSCAN**, **SOR**, **MLS**, **PCA**, **ICP**

Menu building: `MainWindow.populate_menus_from_plugins()` iterates sorted menu paths, creates the hierarchy via `_create_menu_hierarchy()`, and adds actions. `rebuild_plugin_menus()` clears and rebuilds all menus from current PluginManager state.

## 4. Parameter Schema

Source: `gui/dialog_boxes/dynamic_dialog.py`

Plugins define parameters via `get_parameters()` returning a dict of `{param_name: param_info}`. The `DynamicDialog` renders appropriate Qt widgets automatically — unless the plugin overrides `build_param_dialog()` (§1), which replaces the generated form entirely.

### Supported Types

| Type | Widget | Required Keys | Optional Keys |
|------|--------|---------------|---------------|
| `int` | `QSpinBox` | `type`, `default` | `min`, `max`, `label`, `description` |
| `float` | `QDoubleSpinBox` | `type`, `default` | `min`, `max`, `decimals` (default 3), `label`, `description` |
| `string` | `QLineEdit` | `type`, `default` | `label`, `description` |
| `choice` | `QComboBox` (editable) | `type`, `options` (list) | `default`, `label`, `description` |
| `dropdown` | `QComboBox` | `type`, `options` (dict: `{value: display}`) | `default`, `label`, `description` |
| `colormap` | `QComboBox` with gradient previews | `type` | `default`, `label`, `description` |
| `bool` | `QCheckBox` | `type`, `default` | `label`, `description` |
| `info` | `QLabel` (read-only) | `type`, `default` (display text) | `label`, `description` |
| `directory` | `QLineEdit` + Browse button | `type`, `default` | `label`, `description` |

### Common Keys

All types support:
- **`type`** — one of the 9 types above (defaults to `"string"` if omitted)
- **`default`** — initial value
- **`label`** — display label in the form (defaults to param name)
- **`description`** — tooltip text

### Last-Used Value Persistence

`DialogBoxesManager._last_params` stores `{plugin_name: {param_name: value}}` in memory. When a dialog opens, `_apply_last_params()` patches the schema defaults with previously used values. Values persist for the application session (not saved to disk).

## 5. Execution Flows

### The Selection Gate (runs first, for both plugin types)

Source: `application/selection_gate.py`, `MainWindow._gate_selection_then()`

A plugin that declares `requires_selection()` is gated *before* any dialog opens and
*before* the UI is locked, so the viewer and tree are both live for picking:

```
MainWindow.open_dialog_box(plugin_name)
  └─ _gate_selection_then(plugin_name, proceed)
       ├─ kind = selection_kind(plugin_class)
       ├─ selection_present(kind)? → proceed() immediately
       └─ else → SelectionPrompt (NON-modal QDialog)
                  ├─ user selects in viewer/tree → Continue
                  │    └─ re-check; still empty? re-prompt
                  └─ Cancel → nothing runs
```

**Reading a selection.** The selection is a **boolean mask per branch, over that
branch's full-resolution cloud**, built when the gesture completes (lasso close,
click, cluster click) — not derived per plugin. It is held on the branch's item
in the tree (`TreeStructureWidget.selection_mask`), so it survives re-renders,
LOD changes and cache toggles, and is removed with the branch. A plugin reads it,
it does not re-compute it:

| Call | Returns |
|---|---|
| `selection_gate.selected_cloud_mask(viewer, uid, pc_points)` | **the boolean mask, or `None` — prefer this** |
| `selection_gate.selected_cloud_indices(viewer, uid, pc_points)` | sorted cloud rows, or `None` |
| `viewer.selection_mask_for_cloud(uid, pc_points)` | the mask, without going through the gate |
| `viewer.selected_rows(uid)` / `viewer.selection_mask_for(uid)` | same, without the length check |
| `viewer.picked_points` | ordered `(uid, cloud_row)` click picks, across branches — for "which was clicked first" |
| `viewer.first_pick(uid)` | the first clicked cloud row — for "start here" gestures |
| `viewer.selection_count()` / `viewer.has_selection()` | how many points, and whether any |
| `viewer.selection_centroid()` | mean position of the selection — for "near here" gestures |

**Take the mask unless you need positions.** What a plugin almost always does
with a selection is gather — `labels[sel]`, `points[sel]`, `annotations[sel] = x`
— and a mask does that directly. Indices gather identically but cost four bytes
per selected point to materialise (100 MB on a 25M-point selection) and can point
past the end of the array they index; a mask of the wrong length is a loud
`IndexError` instead. Reach for `selected_cloud_indices` only when you need the
row numbers themselves — to intersect the selection with another index list, or
to carry a subset of it forward as rows.

`None` means *nothing is selected in that branch*, which is deliberately distinct
from an empty array: report it to the user rather than running on nothing. The
viewer never stores an all-False mask, so `None` is the only way "nothing"
arrives.

Noise and clusters locked against selection are already excluded when the
selection is made, so there is no gate for a plugin to pass — the `allowed=`
argument callers used to have to remember is gone, along with
`picked_cloud_indices` and `get_selection_mask_for`.

Pipeline replay **bypasses** this gate and does its own pause instead (see §8), so a
replayed step always asks for a fresh selection on the newly produced intermediate
rather than reusing a stale one — and there is never a double prompt.

### Analysis Plugin Flow (Background Thread)

```
User clicks menu item
  └─ MainWindow.open_dialog_box(plugin_name)
       └─ _gate_selection_then(...)                      # selection gate, above
            └─ DialogBoxesManager.get_analysis_params(plugin_name)
                 ├─ plugin.build_param_dialog(...) if provided
                 └─ else DynamicDialog from get_parameters()
                      └─ User clicks OK → params dict (direct return, no signal)
       └─ MainWindow._start_analysis(analysis_type, params)
            ├─ Guard: no selected_branches → message box, abort
            │    (run_analysis would dispatch no work and the poll would never finish)
            ├─ controller.get_analysis_confirmation(name, params)
            │    └─ plugin.confirm_before_execute(node, params) → warning? → Yes/No
            ├─ Set global_progress = (None, "Running {analysis_type}...")
            ├─ show_progress() + show Cancel button in status bar
            ├─ disable_menus() + disable_tree()   (viewer stays live for camera)
            ├─ controller.run_analysis(plugin_name, params, on_error)
            │    └─ AnalysisExecutor.execute()
            │         ├─ guard: _is_running → on_error("Analysis already running")
            │         ├─ global_cancel_event.clear()
            │         └─ threading.Thread(target=_run_in_thread, daemon=True)
            │              ├─ [cancel check]
            │              ├─ Reconstruct if data_type != "point_cloud"
            │              │    ├─ ReconstructionService.reconstruct(uid)
            │              │    ├─ CacheService.set(uid, pc)   # auto-cache parent
            │              │    └─ wrap in a temporary DataNode carrying the same uid
            │              ├─ [cancel check]
            │              ├─ AnalysisService.execute(plugin_class, node, params)
            │              ├─ [cancel check]
            │              └─ _result_data set, _is_completed = True
            └─ _start_completion_polling()
                 └─ QTimer(100ms) → _check_analysis_completion()
                      ├─ Read global_progress → update status bar
                      ├─ check_and_process_completion() → still running? return
                      ├─ Stop timer, clear progress, hide Cancel, re-enable UI
                      ├─ Error / cancelled → report and cleanup
                      ├─ _handle_analysis_result(result_data) → add branch to tree
                      └─ _advance_pipeline_if_running(ok, error)
```

**Key constraints:**
- Only one analysis runs at a time (`_is_running` guard in `AnalysisExecutor`).
- Menus and tree are disabled during processing; the viewer stays enabled for camera moves.
- Progress: `global_variables.global_progress = (percent, message)` — `(None, msg)` for indeterminate, `(50, msg)` for 50%.
- Cancellation: the status-bar Cancel button sets `global_variables.global_cancel_event`; the worker checks it before reconstruction, before `execute()`, and after `execute()`, then reports `"Cancelled by user"` through the same error path. Long-running plugins should poll the event themselves.
- Plugins only READ data and return new objects — thread-safe without deep copies.

### Action Plugin Flow (Main Thread)

```
User clicks menu item
  └─ MainWindow.open_dialog_box(plugin_name)
       └─ _gate_selection_then(...)                      # same selection gate
            └─ execute_action_plugin(plugin_name)
                 ├─ Instantiate plugin_class()
                 ├─ get_parameters() → empty? execute immediately with {}
                 │                   → non-empty? DynamicDialog → user OK → execute
                 └─ plugin_instance.execute(main_window, params)
```

Action plugins run synchronously on the main thread. If a plugin needs long-running work,
it manages its own threading and UI state internally (several ML and region-growing plugins
do exactly this with their own progress dialogs).

## 6. Backend System

Source: `plugins/backends/`

### Abstract Base Classes

All backends extend `BaseBackend` (`plugins/backends/base.py`), which provides `name` (property),
`is_gpu` (property), and `log_execution()`. There are **eight** backend families:

| Abstract Class | Key Method(s) | Purpose |
|----------------|---------------|---------|
| `DBSCANBackend` | `run(points, eps, min_samples) → labels` | Density-based clustering |
| `HDBSCANBackend` | `run(points, min_cluster_size, min_samples, ...) → labels` | Hierarchical density clustering |
| `KNNBackend` | `query(points, k, batch_size=100_000, ...) → (distances, indices)` | K-nearest neighbour search |
| `MaskingBackend` | `apply_mask(points, mask)` / `apply_mask_to_array(array, mask)` | Boolean mask filtering (points, colors, normals) |
| `ScreenSelectionBackend` | `points_in_polygon(block, coeffs, ...) → mask` | Screen-space polygon selection |
| `SpatialGridBackend` | `cell_ids(block, lo, inv_step, ...)`, `block_bounds(block)`, `argsort(cell_ids)` | Spatial grid cell numbering (used by `core/services/spatial_grid.py`) |
| `EigenvalueBackend` | `compute_eigenvalues(points, k, ...) → (eigenvalues, eigenvectors)` | Local neighbourhood covariance |
| `NormalEstimationBackend` | `estimate_normals(points, ...) → normals` | Normal estimation |

### Concrete Implementations

| Family | GPU Class(es) | CPU Class | GPU Library |
|--------|---------------|-----------|-------------|
| DBSCAN | `CuMLDBSCAN` | `SklearnDBSCAN` (also `Open3DDBSCAN`) | RAPIDS cuML |
| HDBSCAN | `CuMLHDBSCAN` | `SklearnHDBSCAN` | RAPIDS cuML |
| KNN | `CuMLKNN` | `ScipyKNN` | RAPIDS cuML |
| Masking | `CuPyMasking` | `NumpyMasking` | CuPy |
| Selection | `CuPySelection` | `NumpySelection` | CuPy |
| Spatial grid | `CuPyGrid` | `NumpyGrid` | CuPy |
| Eigenvalues | `PyTorchCUDAEigen` | `PyTorchCPUEigen` | PyTorch CUDA |
| Normals | `PyTorchCUDANormals`, `Open3DCUDANormals` | `Open3DNormals` | PyTorch CUDA / Open3D tensor |

### BackendRegistry

Source: `plugins/backends/backend_registry.py`

`BackendRegistry(hardware_info)` classifies the machine into one of three scenarios at startup
and then picks each backend independently — the scenario alone is not the whole story, since
CuPy and PyTorch-CUDA availability are checked separately:

| Scenario | Condition |
|----------|-----------|
| **FULL GPU** | Linux + NVIDIA + RAPIDS available |
| **PARTIAL GPU** | NVIDIA without RAPIDS (Windows or Linux) |
| **CPU ONLY** | No NVIDIA GPU (or AMD / Intel) |

| Backend | GPU chosen when | Falls back to |
|---------|-----------------|---------------|
| DBSCAN / HDBSCAN / KNN | scenario is `FULL GPU` | sklearn / sklearn / scipy |
| Masking / Selection / Grid | scenario is GPU **and** `hardware.cupy_available` | NumPy |
| Eigenvalues | scenario is GPU **and** `hardware.pytorch_cuda` | PyTorch CPU |
| Normals | `pytorch_cuda` → PyTorch CUDA; else Open3D CUDA tensor if present | Open3D CPU |

Getters: `get_dbscan()`, `get_hdbscan()`, `get_knn()`, `get_masking()`, `get_selection()`,
`get_grid()`, `get_eigenvalue()`, `get_normal_estimation()`. Reporting: `get_scenario()`,
`get_status_report()`, `get_summary()`.

GPU backends (`CuMLDBSCAN`, `CuPyMasking`, …) include GPU memory pre-checks before execution.
Plugins normally don't call the registry directly — `PointCloud` methods and core services
delegate to backends through `global_variables.global_backend_registry`.

## 7. Plugin-Specific Dialogs

Source: `plugins/dialogs/`

When `DynamicDialog` is insufficient (progress bars, preview viewers, multi-step workflows),
plugins use custom `QDialog` subclasses:

| Dialog Class | File | Used By |
|-------------|------|---------|
| `ShiftDialog` | `shift_dialog.py` | Export plugins (`export_ply`, `export_las`, `export_e57`) — coordinate shift on write |
| `ClassSubsampleDialog` | `class_subsample_dialog.py` | `import_semantickitti` |
| `AnnotationWindow` | `annotation_window.py` | `annotate_points` |
| `LineExtensionWindow` | `line_extension_window.py` | `linear_region_growing`, `extend_traced_lines` |
| `TrainingProgressWindow` | `training_progress_window.py` | `train_pointnet_model`, `train_seg_model`, `train_pointnet2_seg_model` |
| `DataPreviewWindow` | `training_data_preview_window.py` | `generate_training_data`, `preview_data` |
| `DataGenerationProgressDialog` | `data_generation_progress_dialog.py` | `generate_training_data`, `import_external_dataset` |
| `ClassificationProgressDialog` | `classification_progress_dialog.py` | `classify_clusters` |
| `ClassAnalysisWindow` | `class_analysis_window.py` | `analyze_classes` |
| `ShortcutsDialog` | `shortcuts_dialog.py` | `keyboard_shortcuts`, `mouse_controls` |

**No plugin declares a custom `pyqtSignal`** — the whole `plugins/` tree is signal-free, in line
with the singleton-over-signal convention. Custom dialogs report back through direct calls and
callbacks. (The only remaining custom signal in the app is `DialogBoxesManager.analysis_params`,
kept for backward compatibility and not used by the live path, which returns params directly
from `get_analysis_params()`.)

## 8. Shared Utilities

### Coordinate Service

Source: `services/coordinate_service.py`

- `translate_and_convert(points_xyz, min_bound, colors)` — translate to origin, convert to float32 (GPU-accelerated via CuPy)
- `apply_shift(points_f32, shift)` — apply coordinate shift offset
- `find_root_translation(data_nodes, uid_str)` — walk tree to find root translation vector

### Batch Processor

Source: `core/services/batch_processor.py`

`BatchProcessor` splits a large cloud into **adaptive k-d tiles** so no leaf holds more than a
target number of primary points, processes each tile with an overlap halo, then merges the
results. Progress is reported via `global_variables.global_progress`.

| Constructor arg | Meaning |
|-----------------|---------|
| `batch_size` | Maximum **primary** points per leaf cell (the budget; the split is adaptive, so tiles vary in size but not in point count). Plugins expose this as *Target Points per Tile*. |
| `overlap_percent` | Halo width as a fraction of the tile's own extent — used only when `overlap_distance` is `None`. Relative sizing keeps a sparse tile's halo from exploding next to a dense one. |
| `overlap_distance` | Absolute halo width (e.g. DBSCAN `eps`), overriding `overlap_percent`. |
| `max_batch_size` | Hard cap on primary + halo points per tile, bounding KNN cost and VRAM however dense a neighbouring tile is. |

Key methods: `create_spatial_grid()`, `get_batch_for_grid_cell(i)`, `process_in_batches(func, ...)`,
`cluster_in_batches(func, min_points, eps, ...)`.

### Pipeline Capture & Replay

Sources: `core/services/pipeline.py` (pure logic), `application/pipeline_runner.py` (orchestration),
`plugins/055_Pipeline/` (the `save_pipeline` / `run_pipeline` plugins). Full design in
`docs/pipeline_macro_design.md`.

A *pipeline* is the ordered sequence of analysis steps that produced a branch. No extra
bookkeeping is needed to record one: every analysis result node already stores
`tags=[plugin_name, params]` and a `parent_uid`, so `capture_pipeline(branch_uid, data_nodes,
supported_plugins)` walks the parent chain back to the root and replays it forward.

| Piece | Notes |
|-------|-------|
| `PipelineStep` | `plugin`, `params`, `produces` (readability only), `bindings` |
| `Pipeline` | `name`, `steps`, `version` (`PIPELINE_VERSION = 1`) |
| `capture_pipeline()` | Returns `(pipeline, unsupported)` — steps whose plugin isn't replayable (e.g. an action-plugin edit) are dropped and reported so the user can be warned |
| `bindings` | Params that referenced another branch by UUID are stripped at capture and stored as a display-name *hint*; session UUIDs are never persisted, so they must be re-bound to a live branch before replay |
| `save_pipeline()` / `load_pipeline()` | JSON on disk |
| `PipelineRunner` | Runs steps in order against the live app, chaining each step onto the UID the previous one produced; **always pauses** at a step whose plugin `requires_selection()` so the user selects on the freshly produced intermediate |

Because replay reopens no dialogs, params captured from a custom `build_param_dialog()` must
stay JSON-serialisable.

### Global Progress

Thread-safe progress reporting via singleton:
```python
global_variables.global_progress = (None, "Reconstructing...")   # Indeterminate
global_variables.global_progress = (50, "Processing batch 5/10") # 50% determinate
global_variables.global_progress = (100, "Completed")            # Done
```

Written by background threads, read by QTimer polling on the main thread.

## 9. Runtime Management

Source: `plugins/plugin_manager.py`, `plugins/095_Plugins/000_manage_plugins_plugin.py`

### ManagePluginsPlugin

Action plugin providing a runtime UI for plugin management. Allows users to view all loaded plugins, reload, unload, and scan for new plugins without restarting the application.

### Hot Operations

| Method | Description |
|--------|-------------|
| `PluginManager.reload_plugin(name)` | `importlib.reload()` the module, re-detect class, update registries |
| `PluginManager.scan_and_load_new_plugins()` | Walk filesystem, skip already-loaded modules, load new files |
| `PluginManager.unload_plugin(name)` | Remove from `plugins`, `action_plugins`/`analysis_plugins`, and `menu_structure` |
| `MainWindow.rebuild_plugin_menus()` | Clear menubar and rebuild all menus from current PluginManager state |

Typical workflow: user adds a new `.py` file to `plugins/SomeMenu/`, clicks "Scan for New Plugins" in the Plugins menu, and the new plugin appears in menus immediately.

## 10. Complete Plugin Inventory

Generated from the source tree. **100 plugins** (69 Action, 31 Analysis) across **15 top-level menus** and 25 menu paths, listed in menu order. No system (root-level) plugins are currently registered.

### File

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| File | `load_project` | Action | `LoadProjectPlugin` |
| File | `save_project` | Action | `SaveProjectPlugin` |
| File | `save_project_as` | Action | `SaveProjectAsPlugin` |
| File > Import Point Cloud | `import_e57` | Action | `ImportE57Plugin` |
| File > Import Point Cloud | `import_las` | Action | `ImportLASPlugin` |
| File > Import Point Cloud | `import_ply` | Action | `ImportPointCloudPlugin` |
| File > Import Point Cloud | `import_npy_npz` | Action | `ImportNPYPlugin` |
| File > Import Point Cloud | `import_semantickitti` | Action | `ImportSemanticKITTIPlugin` |
| File > Export Point Cloud | `export_e57` | Action | `ExportE57Plugin` |
| File > Export Point Cloud | `export_las` | Action | `ExportLASPlugin` |
| File > Export Point Cloud | `export_ply` | Action | `ExportPointCloudPlugin` |

### View

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| View | `reset_view` | Action | `ResetViewPlugin` |
| View | `zoom_to_extent` | Action | `ZoomToExtentPlugin` |
| View | `zoom_window` | Action | `ZoomWindowPlugin` |
| View | `point_size` | Action | `PointSizePlugin` |
| View | `point_snap_tolerance` | Action | `PointSnapTolerancePlugin` |
| View | `vector_feature_thickness` | Action | `VectorFeatureThicknessPlugin` |
| View | `flythrough` | Action | `FlythroughPlugin` |
| View | `preview_data` | Action | `PreviewDataPlugin` |

### Measure

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Measure | `identify_point` | Action | `IdentifyPointPlugin` |
| Measure | `distance_measurement` | Action | `DistanceMeasurementPlugin` |

### Branch

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Branch | `delete_branch` | Action | `DeleteBranchPlugin` |
| Branch | `merge_branches` | Action | `MergeBranchesPlugin` |
| Branch | `union_branches` | Action | `UnionBranchesPlugin` |
| Branch | `subtract` | Analysis | `SubtractPlugin` |
| Branch | `intersect` | Analysis | `IntersectPlugin` |
| Branch | `duplicate_to_root` | Action | `DuplicateToRootPlugin` |

### Points

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Points > Subsampling | `subsampling` | Analysis | `SubsamplingPlugin` |
| Points > Subsampling | `density_subsampling` | Analysis | `DensitySubsamplingPlugin` |
| Points > Subsampling | `voxel_subsample` | Analysis | `VoxelSubsamplePlugin` |
| Points > Filtering | `filtering` | Analysis | `FilteringPlugin` |
| Points > Filtering | `sor` | Analysis | `SORPlugin` |
| Points > Clustering | `dbscan` | Analysis | `DBSCANPlugin` |
| Points > Clustering | `hdbscan` | Analysis | `HDBSCANPlugin` |
| Points > Clustering | `cluster_size_filter` | Analysis | `ClusterSizeFilterPlugin` |
| Points > Clustering | `surface_region_growing` | Action | `SurfaceRegionGrowingPlugin` |
| Points > Clustering | `linear_region_growing` | Action | `LinearRegionGrowingPlugin` |
| Points > Clustering | `crease_edge` | Action | `CreaseEdgePlugin` |
| Points > Clustering | `contour_growing` | Action | `ContourGrowingPlugin` |
| Points > Clustering | `extend_traced_lines` | Action | `ExtendLinesPlugin` |
| Points > Analysis | `compute_eigenvalues` | Analysis | `ComputeEigenvaluesPlugin` |
| Points > Analysis | `knn_analysis` | Analysis | `KNNAnalysisPlugin` |
| Points > Analysis | `geometric_classification` | Analysis | `GeometricClassificationPlugin` |
| Points > Analysis | `planar_classification` | Analysis | `PlanarClassificationPlugin` |
| Points > Analysis | `linear_classification` | Analysis | `LinearClassificationPlugin` |
| Points > Analysis | `scatter_classification` | Analysis | `ScatterClassificationPlugin` |
| Points > Analysis | `cylindrical_classification` | Analysis | `CylindricalClassificationPlugin` |
| Points > Analysis | `sparse_classification` | Analysis | `SparseClassificationPlugin` |
| Points > Analysis | `vegetation_classification` | Analysis | `VegetationClassificationPlugin` |
| Points > Analysis | `curvature_edge_filter` | Analysis | `CurvatureEdgeFilterPlugin` |
| Points > Analysis | `split_geometric_classes` | Action | `SplitGeometricClassesPlugin` |
| Points > Analysis | `estimate_normals` | Action | `NormalEstimationPlugin` |
| Points > Analysis | `surface_fit` | Action | `SurfaceFitPlugin` |
| Points > Analysis | `mesh_drape` | Action | `MeshDrapePlugin` |
| Points > Analysis | `projected_distance` | Analysis | `ProjectedDistancePlugin` |
| Points > Color | `color_by_branch` | Analysis | `ColorByBranchPlugin` |
| Points > Color | `rgb_color` | Analysis | `RGBColorPlugin` |
| Points > Color | `color_by_value` | Analysis | `ColorByValuePlugin` |
| Points > Transform | `scale` | Action | `ScalePlugin` |

### Selection

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Selection | `separate_selected_points` | Analysis | `SeparateSelectedPointsPlugin` |
| Selection | `separate_selected_clusters` | Analysis | `SeparateSelectedClustersPlugin` |
| Selection | `query_select` | Analysis | `QuerySelectPlugin` |

### Clusters

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Clusters | `classify_cluster` | Action | `ClassifyClusterPlugin` |
| Clusters | `merge_classified_layers` | Action | `MergeClassifiedLayersPlugin` |
| Clusters | `export_classified_clusters` | Action | `ExportClassifiedClustersPlugin` |
| Clusters | `split_classes` | Action | `SplitClassesPlugin` |
| Clusters | `cluster_by_class` | Action | `ClusterByClassPlugin` |
| Clusters | `cluster_by_value` | Action | `ClusterByValuePlugin` |
| Clusters | `undo_cluster_edit` | Action | `UndoClusterEditPlugin` |
| Clusters | `lock_unlock_clusters` | Action | `LockUnlockClustersPlugin` |
| Clusters | `color_clusters` | Action | `ColorClustersPlugin` |
| Clusters | `split_clusters` | Action | `SplitClustersPlugin` |
| Clusters | `merge_clusters` | Action | `MergeClustersPlugin` |
| Clusters | `remove_clusters` | Action | `RemoveClustersPlugin` |

### Processing

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Processing | `average_distance` | Analysis | `AverageDistancePlugin` |

### Pipeline

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Pipeline | `save_pipeline` | Action | `SavePipelinePlugin` |
| Pipeline | `run_pipeline` | Action | `RunPipelinePlugin` |

### Infrastructure

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Infrastructure | `power_line_detection` | Action | `PowerLineDetectionPlugin` |
| Infrastructure | `generate_vector_features` | Action | `GenerateVectorFeaturesPlugin` |
| Infrastructure | `cluster_boundary` | Action | `ClusterBoundaryPlugin` |
| Infrastructure | `fit_cylinder_cone` | Action | `FitCylinderConePlugin` |

### ML Models

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| ML Models > PointNet > Classification | `generate_training_data` | Action | `GenerateTrainingDataPlugin` |
| ML Models > PointNet > Classification | `analyze_classes` | Action | `AnalyzeClassesPlugin` |
| ML Models > PointNet > Classification | `train_pointnet_model` | Action | `TrainPointNetPlugin` |
| ML Models > PointNet > Classification | `classify_clusters` | Action | `ClassifyClustersMLPlugin` |
| ML Models > PointNet > Segmentation | `generate_seg_training_data` | Action | `GenerateSegTrainingDataPlugin` |
| ML Models > PointNet > Segmentation | `import_external_dataset` | Action | `ImportExternalDatasetPlugin` |
| ML Models > PointNet > Segmentation | `annotate_points` | Action | `AnnotatePointsPlugin` |
| ML Models > PointNet > Segmentation | `train_seg_model` | Action | `TrainSegModelPlugin` |
| ML Models > PointNet > Segmentation | `segment_point_cloud` | Action | `SegmentPointCloudPlugin` |
| ML Models > PointNet2 > Segmentation | `train_pointnet2_seg_model` | Action | `TrainPointNet2SegPlugin` |
| ML Models > PointNet2 > Segmentation | `segment_point_cloud_pp` | Action | `SegmentPointCloudPP` |

### CAD

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| CAD > Boundary | `convex_hull` | Analysis | `ConvexHullPlugin` |
| CAD > Boundary | `concave_hull` | Analysis | `ConcaveHullPlugin` |
| CAD > Boundary | `alpha_shape` | Analysis | `AlphaShapePlugin` |

### Tools

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Tools | `dbscan_benchmark` | Action | `DBSCANBenchmarkPlugin` |

### Plugins

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Plugins | `manage_plugins` | Action | `ManagePluginsPlugin` |

### Help

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Help | `system_info` | Action | `SystemInfoPlugin` |
| Help > Keyboard and Mouse Handling | `keyboard_shortcuts` | Action | `KeyboardShortcutsPlugin` |
| Help > Keyboard and Mouse Handling | `mouse_controls` | Action | `MouseControlsPlugin` |
