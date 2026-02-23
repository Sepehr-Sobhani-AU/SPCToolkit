# Plugin Architecture

Deep reference for SPCToolkit's plugin system: discovery, parameter schema, execution flows, backend system, runtime management, and the full plugin inventory. For high-level architecture (layers, data flows, threading, reconstruction), see **ARCHITECTURE.md** Section 7.

## 1. Plugin Interfaces

Source: `plugins/interfaces.py`

### Plugin (Analysis Plugin)

```python
class Plugin(ABC):
    def get_name(self) -> str: ...
    def get_parameters(self) -> Dict[str, Any]: ...
    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]: ...
```

The `execute()` return tuple contains:
- **result** — the output object (`PointCloud`, masks array, cluster labels, etc.)
- **result_type** — string identifier (e.g. `"point_cloud"`, `"masks"`, `"cluster_labels"`)
- **dependencies** — list of UIDs this result depends on (typically `[data_node.uid]`)

### ActionPlugin

```python
class ActionPlugin(ABC):
    def get_name(self) -> str: ...
    def get_parameters(self) -> Dict[str, Any]: ...
    def execute(self, main_window, params: Dict[str, Any]) -> None: ...
```

Action plugins receive `main_window` instead of `data_node`. They perform operations directly (open dialogs, trigger I/O, modify UI state) and return nothing.

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
            ├─ Skip: __pycache__, hidden dirs (.*), __init__.py
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

### Display Name Formatting

`MainWindow._format_plugin_name()` transforms snake_case plugin names:

1. Strip numeric prefix (`000_save_project` → `save_project`)
2. Replace underscores with spaces
3. Title Case each word
4. Keep known acronyms uppercase: **DBSCAN**, **HDBSCAN**, **SOR**, **MLS**, **PCA**, **ICP**

Menu building: `MainWindow.populate_menus_from_plugins()` iterates sorted menu paths, creates the hierarchy via `_create_menu_hierarchy()`, and adds actions. `rebuild_plugin_menus()` clears and rebuilds all menus from current PluginManager state.

## 4. Parameter Schema

Source: `gui/dialog_boxes/dynamic_dialog.py`

Plugins define parameters via `get_parameters()` returning a dict of `{param_name: param_info}`. The `DynamicDialog` renders appropriate Qt widgets automatically.

### Supported Types

| Type | Widget | Required Keys | Optional Keys |
|------|--------|---------------|---------------|
| `int` | `QSpinBox` | `type`, `default` | `min`, `max`, `label`, `description` |
| `float` | `QDoubleSpinBox` | `type`, `default` | `min`, `max`, `decimals` (default 3), `label`, `description` |
| `string` | `QLineEdit` | `type`, `default` | `label`, `description` |
| `choice` | `QComboBox` (editable) | `type`, `options` (list) | `default`, `label`, `description` |
| `dropdown` | `QComboBox` | `type`, `options` (dict: `{value: display}`) | `default`, `label`, `description` |
| `bool` | `QCheckBox` | `type`, `default` | `label`, `description` |
| `info` | `QLabel` (read-only) | `type`, `default` (display text) | `label`, `description` |
| `directory` | `QLineEdit` + Browse button | `type`, `default` | `label`, `description` |

### Common Keys

All types support:
- **`type`** — one of the 8 types above (defaults to `"string"` if omitted)
- **`default`** — initial value
- **`label`** — display label in the form (defaults to param name)
- **`description`** — tooltip text

### Last-Used Value Persistence

`DialogBoxesManager._last_params` stores `{plugin_name: {param_name: value}}` in memory. When a dialog opens, `_apply_last_params()` patches the schema defaults with previously used values. Values persist for the application session (not saved to disk).

## 5. Execution Flows

### Analysis Plugin Flow (Background Thread)

```
User clicks menu item
  └─ MainWindow.open_dialog_box(plugin_name)
       └─ DialogBoxesManager.get_analysis_params(plugin_name)
            └─ DynamicDialog (if parameters exist)
                 └─ User clicks OK → returns params dict
       └─ MainWindow._start_analysis(analysis_type, params)
            ├─ Set global_progress = (None, "Running {analysis_type}...")
            ├─ show_progress() in status bar
            ├─ disable_menus() + disable_tree()
            ├─ controller.run_analysis(plugin_name, params, on_error)
            │    └─ AnalysisExecutor.execute()
            │         └─ threading.Thread(target=_run_in_thread, daemon=True)
            │              ├─ Reconstruct if data_type != "point_cloud"
            │              │    └─ ReconstructionService.reconstruct(uid)
            │              │    └─ Auto-cache parent after reconstruction
            │              ├─ plugin.execute(data_node, params)
            │              └─ Store result in _result_data, set _is_completed = True
            └─ _start_completion_polling()
                 └─ QTimer(100ms) → _check_analysis_completion()
                      ├─ Read global_progress → update status bar
                      ├─ check_and_process_completion() → still running? return
                      ├─ Stop timer, clear progress, re-enable UI
                      ├─ Check for error → log and cleanup
                      └─ _handle_analysis_result(result_data) → add to tree → cleanup
```

**Key constraints:**
- Only one analysis runs at a time (`_is_running` guard)
- Menus and tree are disabled during processing; viewer stays enabled for camera
- Progress: `global_variables.global_progress = (percent, message)` — `(None, msg)` for indeterminate, `(50, msg)` for 50%
- Plugins only READ data and return new objects — thread-safe without deep copies

### Action Plugin Flow (Main Thread)

```
User clicks menu item
  └─ MainWindow.open_dialog_box(plugin_name)
       └─ is_action_plugin? → execute_action_plugin(plugin_name)
            ├─ Instantiate plugin_class()
            ├─ get_parameters() → empty? execute immediately with {}
            │                   → non-empty? DynamicDialog → user OK → execute
            └─ plugin_instance.execute(main_window, params)
```

Action plugins run synchronously on the main thread. If a plugin needs long-running work, it manages its own threading and UI state internally.

## 6. Backend System

Source: `plugins/backends/`

### Abstract Base Classes

All backends extend `BaseBackend` which provides `name` (property), `is_gpu` (property), and `log_execution()`.

| Abstract Class | Method Signature | Purpose |
|----------------|-----------------|---------|
| `DBSCANBackend` | `run(points, eps, min_samples) → labels` | Density-based clustering |
| `KNNBackend` | `query(points, k) → (distances, indices)` | K-nearest neighbor search |
| `MaskingBackend` | `apply_mask(points, mask) → filtered` | Boolean mask filtering |
| | `apply_mask_to_array(array, mask) → filtered` | Mask any array (colors, normals) |
| `EigenvalueBackend` | `compute_eigenvalues(points, k) → (eigenvalues, eigenvectors)` | Local neighborhood covariance |

### Concrete Implementations

| Algorithm | GPU Class | CPU Class | GPU Library |
|-----------|-----------|-----------|-------------|
| DBSCAN | `CuMLDBSCAN` | `SklearnDBSCAN` | RAPIDS cuML |
| DBSCAN (alt) | — | `Open3DDBSCAN` | — |
| KNN | `CuMLKNN` | `ScipyKNN` | RAPIDS cuML |
| Masking | `CuPyMasking` | `NumpyMasking` | CuPy |
| Eigenvalues | `PyTorchCUDAEigen` | `PyTorchCPUEigen` | PyTorch CUDA |

### BackendRegistry

Source: `plugins/backends/backend_registry.py`

Auto-detects one of three hardware scenarios at startup:

| Scenario | Condition | DBSCAN | KNN | Masking | Eigenvalues |
|----------|-----------|--------|-----|---------|-------------|
| **FULL GPU** | Linux + NVIDIA + RAPIDS | cuML | cuML | CuPy | PyTorch CUDA |
| **PARTIAL GPU** | NVIDIA without RAPIDS | sklearn | scipy | CuPy | PyTorch CUDA |
| **CPU ONLY** | No NVIDIA GPU | sklearn | scipy | NumPy | PyTorch CPU |

Getter methods: `get_dbscan()`, `get_knn()`, `get_masking()`, `get_eigenvalue()`

GPU backends (`CuMLDBSCAN`, `CuPyMasking`) include GPU memory pre-checks before execution. Plugins don't call the registry directly — `PointCloud` methods and analysis plugins delegate to backends through the registry accessed via `global_variables.global_backend_registry`.

## 7. Plugin-Specific Dialogs

Source: `plugins/dialogs/`

When `DynamicDialog` is insufficient (progress bars, preview viewers, multi-step workflows), plugins use custom `QDialog` subclasses:

| Dialog Class | File | Used By |
|-------------|------|---------|
| `ShiftDialog` | `shift_dialog.py` | Import plugins (coordinate shift on import) |
| `TrainingProgressWindow` | `training_progress_window.py` | `TrainPointNetPlugin` (training progress + cancel) |
| `DataPreviewWindow` | `training_data_preview_window.py` | `GenerateTrainingDataPlugin` (3D preview of training samples) |
| `DataGenerationProgressDialog` | `data_generation_progress_dialog.py` | `GenerateTrainingDataPlugin` (batch generation progress) |
| `ClassificationProgressDialog` | `classification_progress_dialog.py` | `ClassifyClustersMLPlugin` (inference progress) |
| `ClassAnalysisWindow` | `class_analysis_window.py` | `AnalyzeClassesPlugin` (class distribution charts) |

**Note:** `TrainingProgressWindow` uses `pyqtSignal` (`cancel_requested`) — this is a justified exception for a standalone modal dialog with a cancel button, not a violation of the singleton-over-signal convention.

## 8. Shared Utilities

### Coordinate Service

Source: `services/coordinate_service.py`

- `translate_and_convert(points_xyz, min_bound, colors)` — translate to origin, convert to float32 (GPU-accelerated via CuPy)
- `apply_shift(points_f32, shift)` — apply coordinate shift offset
- `find_root_translation(data_nodes, uid_str)` — walk tree to find root translation vector

### Batch Processor

Source: `core/services/batch_processor.py`

`BatchProcessor` — spatial batching with overlap for large point clouds that exceed GPU memory. Splits data into tiles, processes each, then merges results. Reports progress via `global_variables.global_progress`.

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

### File

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| File > Import Point Cloud | import_e57 | Action | `ImportE57Plugin` |
| File > Import Point Cloud | import_las | Action | `ImportLASPlugin` |
| File > Import Point Cloud | import_ply | Action | `ImportPointCloudPlugin` |
| File > Import Point Cloud | import_semantickitti | Action | `ImportSemanticKITTIPlugin` |
| File > Export Point Cloud | export_e57 | Action | `ExportE57Plugin` |
| File > Export Point Cloud | export_las | Action | `ExportLASPlugin` |
| File > Export Point Cloud | export_ply | Action | `ExportPLYPlugin` |
| File | load_project | Action | `LoadProjectPlugin` |
| File | save_project | Action | `SaveProjectPlugin` |
| File | save_project_as | Action | `SaveProjectAsPlugin` |

### View

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| View | zoom_to_extent | Action | `ZoomToExtentPlugin` |
| View | point_size | Action | `PointSizePlugin` |
| View | preview_data | Action | `PreviewDataPlugin` |

### Branch

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Branch | delete_branch | Action | `DeleteBranchPlugin` |
| Branch | merge_branches | Action | `MergeBranchesPlugin` |
| Branch | subtract | Analysis | `SubtractPlugin` |

### Points

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Points > Subsampling | subsampling | Analysis | `SubsamplingPlugin` |
| Points > Subsampling | density_subsampling | Analysis | `DensitySubsamplingPlugin` |
| Points > Filtering | filtering | Analysis | `FilteringPlugin` |
| Points > Filtering | sor | Analysis | `SORPlugin` |
| Points > Clustering | dbscan | Analysis | `DBSCANPlugin` |
| Points > Clustering | hdbscan | Analysis | `HDBSCANPlugin` |
| Points > Clustering | cluster_size_filter | Analysis | `ClusterSizeFilterPlugin` |
| Points > Analysis | compute_eigenvalues | Analysis | `ComputeEigenvaluesPlugin` |
| Points > Analysis | knn_analysis | Analysis | `KNNAnalysisPlugin` |
| Points > Analysis | geometric_classification | Analysis | `GeometricClassificationPlugin` |
| Points > Analysis | planar_classification | Analysis | `PlanarClassificationPlugin` |
| Points > Analysis | linear_classification | Analysis | `LinearClassificationPlugin` |
| Points > Analysis | scatter_classification | Analysis | `ScatterClassificationPlugin` |
| Points > Analysis | cylindrical_classification | Analysis | `CylindricalClassificationPlugin` |
| Points > Analysis | sparse_classification | Analysis | `SparseClassificationPlugin` |
| Points > Analysis | vegetation_classification | Analysis | `VegetationClassificationPlugin` |
| Points > Analysis | split_geometric_classes | Action | `SplitGeometricClassesPlugin` |

### Selection

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Selection | separate_selected_points | Analysis | `SeparateSelectedPointsPlugin` |
| Selection | separate_selected_clusters | Analysis | `SeparateSelectedClustersPlugin` |

### Clusters

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Clusters | classify_cluster | Action | `ClassifyClusterPlugin` |
| Clusters | merge_classified_layers | Action | `MergeClassifiedLayersPlugin` |
| Clusters | export_classified_clusters | Action | `ExportClassifiedClustersPlugin` |
| Clusters | split_classes | Action | `SplitClassesPlugin` |
| Clusters | cluster_by_class | Action | `ClusterByClassPlugin` |
| Clusters | cluster_by_value | Action | `ClusterByValuePlugin` |
| Clusters | cut_cluster | Action | `CutClusterPlugin` |
| Clusters | merge_clusters | Action | `MergeClustersPlugin` |
| Clusters | remove_clusters | Action | `RemoveClustersPlugin` |
| Clusters | undo_cluster_edit | Action | `UndoClusterEditPlugin` |
| Clusters | lock_unlock_clusters | Action | `LockUnlockClustersPlugin` |
| Clusters | color_clusters | Action | `ColorClustersPlugin` |

### Processing

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Processing | average_distance | Analysis | `AverageDistancePlugin` |

### Infrastructure

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Infrastructure | power_line_detection | Action | `PowerLineDetectionPlugin` |

### ML Models

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| ML Models > PointNet | generate_training_data | Action | `GenerateTrainingDataPlugin` |
| ML Models > PointNet | analyze_classes | Action | `AnalyzeClassesPlugin` |
| ML Models > PointNet | train_pointnet_model | Action | `TrainPointNetPlugin` |
| ML Models > PointNet | classify_clusters | Action | `ClassifyClustersMLPlugin` |

### Tools

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Tools | dbscan_benchmark | Action | `DBSCANBenchmarkPlugin` |

### Help

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Help | system_info | Action | `SystemInfoPlugin` |

### Plugins

| Menu Path | Plugin Name | Type | Class |
|-----------|-------------|------|-------|
| Plugins | manage_plugins | Action | `ManagePluginsPlugin` |

**Totals:** ~56 plugins (38 Action, 18 Analysis) across 12 menu categories.
