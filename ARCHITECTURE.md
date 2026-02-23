# SPCToolkit Architecture

This document describes the core framework architecture of SPCToolkit. It covers the main components, their relationships, and data flows.

> **Note:** Diagrams use [Mermaid](https://mermaid.js.org/) syntax. View in GitHub, VS Code with Mermaid extension, or [mermaid.live](https://mermaid.live).

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Initialization Sequence](#2-initialization-sequence)
3. [Data Flow: Loading a Point Cloud](#3-data-flow-loading-a-point-cloud)
4. [Data Flow: Running an Analysis](#4-data-flow-running-an-analysis)
5. [Data Flow: Visibility & Reconstruction](#5-data-flow-visibility--reconstruction)
6. [Component Relationships](#6-component-relationships)
7. [Plugin Integration](#7-plugin-integration)
8. [Quick Reference](#8-quick-reference)

---

## 1. High-Level Overview

The system is organized into layers: GUI, Application, Core, Infrastructure, and Plugins.

```mermaid
flowchart TD
    subgraph GUI["GUI Layer"]
        direction LR
        MW[MainWindow]
        TW[TreeStructureWidget]
        PV[PCDViewerWidget]
        DB[DialogBoxesManager]
    end

    subgraph App["Application Layer"]
        direction LR
        AC[ApplicationController]
        AE[AnalysisExecutor]
        RC[RenderingCoordinator]
        LM[LODManager]
    end

    subgraph Core["Core Layer"]
        direction LR
        subgraph CoreServices["Services"]
            RS[ReconstructionService]
            CS[CacheService]
            AS[AnalysisService]
            BP[BatchProcessor]
        end
        subgraph Transformers["Transformers x7"]
            TF["Masks, Clusters, Eigenvalues,\nValues, Colors, DistToGround,\nContainer"]
        end
        subgraph Entities
            DN["DataNodes / DataNode"]
            PC["PointCloud, Clusters,\nMasks, Values"]
        end
    end

    subgraph Infra["Infrastructure & Services"]
        direction LR
        FM[FileManager]
        HD[HardwareDetector]
        MM[MemoryManager]
    end

    subgraph Plugins["Plugin Layer"]
        direction LR
        PM[PluginManager]
        AP[AnalysisPlugins]
        ActP[ActionPlugins]
        BR[BackendRegistry]
    end

    GUI --> App --> Core
    Plugins -.-> App
    Plugins -.-> Core
    Infra -.-> Core
```

> **Singleton access:** `global_variables` (config/config.py) provides global access to MainWindow, ApplicationController, TreeStructureWidget, PCDViewerWidget, FileManager, and DataNodes. Detailed component interactions are shown in the sequence diagrams below.

### Layer Responsibilities

| Layer | Purpose | Key Files |
|-------|---------|-----------|
| **GUI** | User interaction, visualization | `gui/main_window.py`, `gui/widgets/*` |
| **Application** | Orchestration, coordination | `application/application_controller.py`, `application/analysis_executor.py`, `application/rendering_coordinator.py` |
| **Core Entities** | Data structures | `core/entities/point_cloud.py`, `core/entities/clusters.py`, `core/entities/masks.py`, `core/entities/data_node.py` |
| **Core Services** | Reconstruction, caching, analysis | `core/services/reconstruction_service.py`, `core/services/cache_service.py`, `core/services/analysis_service.py` |
| **Infrastructure** | Hardware detection, memory management | `infrastructure/hardware_detector.py`, `infrastructure/memory_manager.py` |
| **Services** | File I/O | `services/file_manager.py` |
| **Plugins** | Extensible functionality, backends | `plugins/*/`, `plugins/backends/`, `plugins/plugin_manager.py` |

### Layer Dependencies (Clean Architecture)

- **GUI** → Application → Core (inward only)
- **Infrastructure** → Core
- **Plugins** → Application + Core (via `global_variables`)
- **Core** → NOTHING (no outward dependencies)

---

## 2. Initialization Sequence

The application initializes components in a specific order to ensure dependencies are ready.

```mermaid
sequenceDiagram
    participant M as main.py
    participant MW as MainWindow
    participant AC as ApplicationController
    participant GV as global_variables

    M->>M: Configure logging
    M->>GV: Set global_hardware_info
    M->>GV: Set global_backend_registry
    M->>M: Create PluginManager

    M->>MW: Create MainWindow(plugin_manager)

    activate MW
    MW->>GV: global_file_manager = FileManager()
    MW->>GV: global_tree_structure_widget = TreeStructureWidget()
    MW->>GV: global_pcd_viewer_widget = PCDViewerWidget()
    MW->>MW: Create DialogBoxesManager
    MW->>AC: ApplicationController.create(plugin_manager, file_manager)
    AC->>AC: Create DataNodes, services, executor, coordinator
    MW->>GV: global_application_controller = controller
    MW->>GV: global_data_nodes = controller.data_nodes
    MW->>GV: global_main_window = self
    MW->>MW: Connect signals to MainWindow handlers
    MW->>MW: setup_ui()
    MW->>MW: populate_menus_from_plugins()
    deactivate MW

    M->>MW: show()
    M->>M: app.exec_()
```

### Global Variables Assignment Locations

| Variable | Assigned In |
|----------|-------------|
| `global_file_manager` | `gui/main_window.py` |
| `global_tree_structure_widget` | `gui/main_window.py` |
| `global_pcd_viewer_widget` | `gui/main_window.py` |
| `global_application_controller` | `gui/main_window.py` |
| `global_data_nodes` | `gui/main_window.py` (from controller) |
| `global_main_window` | `gui/main_window.py` |
| `global_hardware_info` | `main.py` |
| `global_backend_registry` | `main.py` |

---

## 3. Data Flow: Loading a Point Cloud

When a user opens a file, the data flows through FileManager to MainWindow, which delegates to ApplicationController.

```mermaid
sequenceDiagram
    participant User
    participant MW as MainWindow
    participant FM as FileManager
    participant AC as ApplicationController
    participant DN as DataNodes
    participant TW as TreeWidget
    participant PV as Viewer

    User->>MW: File > Open
    MW->>FM: open_point_cloud_file()
    FM->>FM: Qt File Dialog
    FM->>FM: o3d.io.read_point_cloud()
    FM->>FM: Create PointCloud object

    FM-->>MW: SIGNAL: point_cloud_loaded(path, pc)

    activate MW
    MW->>AC: add_point_cloud(pc, name)
    AC->>AC: Create DataNode, add to DataNodes
    AC-->>MW: return uid
    MW->>TW: add_branch(uid, name, is_root=True)
    MW->>AC: get_cache_memory_usage(uid)
    MW->>TW: update_cache_tooltip(uid, size)
    deactivate MW
```

### Key Points

- **FileManager** handles file dialogs and Open3D I/O
- **ApplicationController** creates DataNodes and manages the data collection
- **DataNodes** is the collection manager (UUID -> DataNode mapping)
- **TreeWidget** displays the hierarchical structure

---

## 4. Data Flow: Running an Analysis

Analysis plugins run in a background thread to keep the UI responsive.

```mermaid
sequenceDiagram
    participant User
    participant MW as MainWindow
    participant DB as DialogBoxesManager
    participant AC as ApplicationController
    participant AE as AnalysisExecutor
    participant Thread as BackgroundThread
    participant Plugin

    User->>MW: Click Analysis Menu Item
    MW->>DB: open_dialog_box(plugin_name)
    DB->>DB: Create DynamicDialog
    User->>DB: Enter params, click OK

    DB-->>MW: return params (direct call)

    activate MW
    MW->>MW: disable_menus(), disable_tree()
    MW->>MW: show_processing_overlay()
    MW->>AC: run_analysis(plugin_name, params)
    AC->>AE: execute(plugin, node, params, ...)
    MW->>MW: Start QTimer polling (100ms)
    deactivate MW

    activate Thread
    AE->>Thread: Start background thread
    Thread->>Thread: Reconstruct if needed
    Thread->>Plugin: execute(data_node, params)
    Plugin-->>Thread: return (result, type, deps)
    Thread->>AE: Mark completed (set flag)
    deactivate Thread

    Note over MW: QTimer polls every 100ms

    MW->>AE: check_completion()
    AE-->>MW: Completed! Call handle_result()

    activate MW
    MW->>AC: add_analysis_result(result, type, deps, parent, ...)
    AC->>AC: Create DataNode, add to DataNodes
    AC-->>MW: return uid
    MW->>TW: add_branch(uid, parent, name)
    MW->>TW: Update visibility
    MW->>MW: enable_menus(), enable_tree()
    MW->>MW: hide_processing_overlay()
    deactivate MW
```

### Threading Model

- **Thread Type:** Python `threading.Thread` (NOT QThread)
- **Communication:** Flag polling via QTimer (100ms) + callbacks for progress/completion/error
- **Thread Safety:** Plugins only READ data, return NEW objects
- **No Deep Copy:** Memory efficient - relies on read-only access

---

## 5. Data Flow: Visibility & Reconstruction

When a user toggles visibility, derived data (masks, clusters) must be reconstructed to PointCloud for rendering.

```mermaid
sequenceDiagram
    participant User
    participant TW as TreeWidget
    participant MW as MainWindow
    participant RC as RenderingCoordinator
    participant RS as ReconstructionService
    participant PV as Viewer

    User->>TW: Toggle checkbox
    TW->>TW: Update visibility_status dict

    TW-->>MW: SIGNAL: branch_visibility_changed(status)

    activate MW
    MW->>RC: render_visible(visibility_status)

    activate RC
    loop For each visible UID
        RC->>RC: Get DataNode
        alt Node has cached PointCloud
            RC->>RC: Use cached (fast path)
        else Need reconstruction
            RC->>RS: reconstruct(uid)
            RS->>RS: Find root or cached ancestor
            RS->>RS: Apply transformer chain
            RS-->>RC: return PointCloud
            RC->>RC: Cache result on node
        end
        RC->>RC: Apply LOD if needed
    end

    RC->>PV: set_point_vertices(combined)
    RC->>PV: update()
    deactivate RC
    MW->>MW: enable UI, hide overlay
    deactivate MW
```

### Reconstruction Process

1. **Check Cache:** If node has `cached_point_cloud`, use it immediately
2. **Find Ancestor:** Walk up tree looking for cached ancestor or root PointCloud
3. **Apply Transformers:** Use `ReconstructionService.transformer_registry` to apply transformations
4. **Cache Result:** Store reconstructed PointCloud on node for future use

### Transformer Registry

| Data Type | Transformer Class | Transformation |
|-----------|-------------------|----------------|
| `masks` | MasksTransformer | Filter points by boolean mask |
| `cluster_labels` | ClustersTransformer | Apply cluster/semantic colors |
| `eigenvalues` | EigenvaluesTransformer | Color by eigenvalue features |
| `values` | ValuesTransformer | Color by scalar values |
| `colors` | ColorsTransformer | Apply RGB colors |
| `dist_to_ground` | DistToGroundTransformer | Color by height above ground |

> **Note:** `ContainerTransformer` also exists in `core/transformers/` as a pass-through for organizational nodes but is not registered in the default `transformer_registry`.

---

## 6. Component Relationships

Static class diagram showing the main components and their relationships.

```mermaid
classDiagram
    class GlobalVariables {
        +global_main_window
        +global_application_controller
        +global_file_manager
        +global_tree_structure_widget
        +global_pcd_viewer_widget
        +global_data_nodes
    }

    class MainWindow {
        +controller: ApplicationController
        +file_manager
        +tree_widget
        +pcd_viewer_widget
        +dialog_boxes_manager
        +setup_ui()
        +render_visible_data()
        +disable_menus()
        +enable_menus()
    }

    class ApplicationController {
        +data_nodes: DataNodes
        +reconstruction_service
        +cache_service
        +analysis_executor
        +rendering_coordinator
        +selected_branches: List
        +create(plugin_manager, file_manager)$
        +add_point_cloud(pc, name)
        +add_analysis_result(result, type, deps, ...)
        +remove_node(uid)
        +get_node(uid)
        +run_analysis(plugin_name, params, ...)
        +reconstruct(uid)
        +cache_node(uid)
        +uncache_node(uid)
        +is_cached(uid)
        +load_project(data_nodes)
    }

    class DataNodes {
        +data_nodes: Dict~UUID,DataNode~
        +add_node()
        +get_node()
        +remove_node()
    }

    class DataNode {
        +uid: UUID
        +data: Any
        +data_type: str
        +parent_uid: UUID
        +depends_on: List
        +cached_point_cloud
        +is_cached: bool
        +memory_size: int
    }

    class PointCloud {
        +points: ndarray
        +colors: ndarray
        +normals: ndarray
        +attributes: Dict
        +dbscan()
        +get_subset()
        +calculate_eigenvalues()
    }

    GlobalVariables --> MainWindow
    GlobalVariables --> ApplicationController
    MainWindow --> ApplicationController
    ApplicationController --> DataNodes
    DataNodes --> DataNode
    DataNode --> PointCloud
```

### Data Type Hierarchy

```mermaid
flowchart TB
    DN[DataNode\nwrapper]
    DN --> PC[PointCloud\nprimary data]
    DN --> CL[Clusters\nlabels + names]
    DN --> MA[Masks\nboolean array]
    DN --> EV[Eigenvalues\nn,3 array]
    DN --> VA[Values\nscalars]
    DN --> CO[Colors\nRGB]
    DN --> DG[DistToGround\nheights]
    DN --> CR[ClassReference\nclass filter]
```

---

## 7. Plugin Integration

Plugins are discovered automatically from the folder structure and registered in menus.

```mermaid
flowchart LR
    subgraph Filesystem
        PF[plugins/Category/plugin.py]
    end

    subgraph Discovery
        PM[PluginManager]
        PM -->|walks| PF
        PM -->|importlib| Classes
        Classes -->|inspect.issubclass| Register
    end

    subgraph Registration
        Register --> analysis_plugins
        Register --> action_plugins
        Register --> menu_structure
    end

    subgraph MenuBuilding
        menu_structure --> MainWindow
        MainWindow -->|create QAction| MenuItem
        MenuItem -->|triggered| open_dialog_box
    end

    subgraph Execution
        open_dialog_box --> DialogBoxesManager
        DialogBoxesManager -->|returns params| MainWindow
        MainWindow --> ApplicationController
        ApplicationController --> AnalysisExecutor
        AnalysisExecutor -->|background| Plugin.execute
    end
```

### Plugin Types

| Type | Base Class | Execution | Returns |
|------|------------|-----------|---------|
| **AnalysisPlugin** | `AnalysisPlugin` | Background thread | `(result, type, deps)` |
| **ActionPlugin** | `ActionPlugin` | Main thread | `None` |

### Folder Structure = Menu Hierarchy

```
plugins/
├── 000_File/                        -> Menu: "File"
│   ├── 000_import_plugin.py         ->   "Import Point Cloud"
│   └── 010_save_plugin.py           ->   "Save Project"
├── 010_View/                        -> Menu: "View"
│   └── 000_zoom_to_extent_plugin.py ->   "Zoom To Extent"
└── 020_Points/                      -> Menu: "Points"
    └── 010_Clustering/              ->   Submenu: "Clustering"
        └── 000_dbscan_plugin.py     ->     "DBSCAN"
```

Numbering (000_, 010_, etc.) controls menu order. Folder depth controls menu nesting.

### Plugin Interfaces

```python
# Analysis Plugin (processes data, returns results)
class AnalysisPlugin(ABC):
    def get_name(self) -> str: ...
    def get_parameters(self) -> Dict[str, Any]: ...
    def execute(self, data_node, params) -> Tuple[result, type, deps]: ...

# Action Plugin (performs actions, no return)
class ActionPlugin(ABC):
    def get_name(self) -> str: ...
    def get_parameters(self) -> Dict[str, Any]: ...  # Can return {}
    def execute(self, main_window, params) -> None: ...
```

---

## 8. Quick Reference

### "I want to do X -> Look in Y"

| Task | Location |
|------|----------|
| Load a point cloud | `FileManager.open_point_cloud_file()` |
| Run an analysis plugin | `ApplicationController.run_analysis()` → `AnalysisExecutor.execute()` |
| Add a node to the tree | `MainWindow._on_point_cloud_loaded()` or `_handle_analysis_result()` |
| Render points in viewer | `PCDViewerWidget.set_point_vertices()` |
| Create a new analysis plugin | `plugins/YourCategory/your_plugin.py` (inherit `AnalysisPlugin`) |
| Create a new action plugin | `plugins/YourCategory/your_plugin.py` (inherit `ActionPlugin`) |
| Access any global manager | `from config.config import global_variables` |
| Reconstruct a branch | `ApplicationController.reconstruct(uid)` |
| Get selected tree items | `ApplicationController.selected_branches` |
| Re-render visible data | `MainWindow.render_visible_data(zoom_extent=False)` |
| Disable UI during processing | `MainWindow.disable_menus()`, `disable_tree()` |

### Signal Connections

| Signal | Source | Handler | Purpose |
|--------|--------|---------|---------|
| `point_cloud_loaded` | FileManager | MainWindow._on_point_cloud_loaded | File loaded |
| `branch_visibility_changed` | TreeStructureWidget | MainWindow._on_branch_visibility_changed | Checkbox toggled |
| `branch_selection_changed` | TreeStructureWidget | MainWindow._on_branch_selection_changed | Tree selection |
| `branch_added` | TreeStructureWidget | MainWindow._on_branch_added | New branch added |

> **Note:** Analysis parameters are passed via direct method call (`DialogBoxesManager.get_analysis_params()`), not via signal/slot.

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `gui/main_window.py` | Main window, menu building, signal handling |
| `application/application_controller.py` | Central orchestrator (factory, reconstruct, selection) |
| `application/analysis_executor.py` | Background thread analysis execution |
| `application/rendering_coordinator.py` | Visibility rendering, LOD management |
| `core/entities/data_node.py` | Single data unit wrapper |
| `core/entities/data_nodes.py` | Collection manager |
| `core/entities/point_cloud.py` | Primary data structure |
| `core/services/reconstruction_service.py` | Rebuilds PointCloud from derived data |
| `core/services/cache_service.py` | Cache management for reconstructed data |
| `core/services/analysis_service.py` | Plugin execution service |
| `core/transformers/*.py` | Data type transformers for reconstruction |
| `core/services/batch_processor.py` | Spatial batch processing for large point clouds |
| `services/file_manager.py` | File I/O operations |
| `infrastructure/hardware_detector.py` | Hardware detection (GPU, memory) |
| `infrastructure/memory_manager.py` | Memory management |
| `plugins/plugin_manager.py` | Plugin discovery and registration |
| `plugins/backends/backend_registry.py` | Backend selection (GPU/CPU) |
| `plugins/interfaces.py` | Plugin base classes |
| `config/config.py` | GlobalVariables singleton |

---

## Architectural Principles

1. **Singleton Pattern:** Use `global_variables` for inter-component communication (avoid custom signals)
2. **Background Threading:** Long operations run in threads with QTimer polling
3. **Plugin Extensibility:** Folder structure defines menu hierarchy
4. **Caching:** Reconstructed PointClouds are cached on DataNodes
5. **Read-Only Threading:** Plugins only read data, return new objects

---

## Communication Pattern

```
PREFERRED: Singleton Pattern
─────────────────────────────
global_variables.global_application_controller.reconstruct(uid)
global_variables.global_pcd_viewer_widget.update()
global_variables.global_main_window.render_visible_data()

ACCEPTABLE: Callbacks (when singleton doesn't fit)
──────────────────────────────────────────────────
component_a.process(on_complete=callback_function)

AVOID: Custom Qt Signals/Slots
──────────────────────────────
class MyClass(QObject):
    custom_signal = pyqtSignal()  # Don't do this
```

---

*Last updated: February 2026*
