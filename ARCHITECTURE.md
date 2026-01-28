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

The system is organized into layers: UI, Core Framework, Services, Data, and Plugins.

```mermaid
flowchart TB
    subgraph UI["User Interface Layer"]
        MW[MainWindow]
        TW[TreeStructureWidget]
        PV[PCDViewerWidget]
        DB[DialogBoxesManager]
    end

    subgraph Core["Core Framework Layer"]
        DM[DataManager]
        DN[DataNodes]
        NRM[NodeReconstructionManager]
        ATM[AnalysisThreadManager]
    end

    subgraph Services["Services Layer"]
        FM[FileManager]
        PM[PluginManager]
        AM[AnalysisManager]
    end

    subgraph Data["Data Layer"]
        PC[PointCloud]
        Node[DataNode]
        Clusters
        Masks
        Values
    end

    subgraph Plugins["Plugin Layer"]
        AP[AnalysisPlugins]
        ActP[ActionPlugins]
    end

    GV[global_variables\nSingleton Access]

    MW --> TW
    MW --> PV
    MW --> DB
    MW --> DM

    DM --> DN
    DM --> NRM
    DM --> ATM
    DM --> FM
    DM --> PM

    DN --> Node
    Node --> PC
    Node --> Clusters
    Node --> Masks
    Node --> Values

    PM --> AP
    PM --> ActP

    GV -.->|provides access to| DM
    GV -.->|provides access to| TW
    GV -.->|provides access to| PV
    GV -.->|provides access to| FM
    GV -.->|provides access to| MW
```

### Layer Responsibilities

| Layer | Purpose | Key Files |
|-------|---------|-----------|
| **UI** | User interaction, visualization | `gui/main_window.py`, `gui/widgets/*` |
| **Core** | Data coordination, threading | `core/data_manager.py`, `core/data_node.py` |
| **Services** | File I/O, plugin discovery | `services/file_manager.py`, `plugins/plugin_manager.py` |
| **Data** | Data structures | `core/point_cloud.py`, `core/clusters.py`, `core/masks.py` |
| **Plugins** | Extensible functionality | `plugins/*/` |

---

## 2. Initialization Sequence

The application initializes components in a specific order to ensure dependencies are ready.

```mermaid
sequenceDiagram
    participant M as main.py
    participant MW as MainWindow
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
    MW->>GV: global_data_manager = DataManager(...)
    MW->>GV: global_main_window = self
    MW->>GV: global_analysis_thread_manager = AnalysisThreadManager()
    MW->>MW: setup_ui()
    MW->>MW: populate_menus_from_plugins()
    deactivate MW

    M->>MW: show()
    M->>M: app.exec_()
```

### Global Variables Assignment Locations

| Variable | Assigned In | Line |
|----------|-------------|------|
| `global_file_manager` | `gui/main_window.py` | ~39 |
| `global_tree_structure_widget` | `gui/main_window.py` | ~43 |
| `global_pcd_viewer_widget` | `gui/main_window.py` | ~46 |
| `global_data_manager` | `gui/main_window.py` | ~56 |
| `global_main_window` | `gui/main_window.py` | ~59 |
| `global_analysis_thread_manager` | `gui/main_window.py` | ~63 |
| `global_data_nodes` | `core/data_manager.py` | ~56 |
| `global_hardware_info` | `main.py` | ~69 |
| `global_backend_registry` | `main.py` | ~77 |

---

## 3. Data Flow: Loading a Point Cloud

When a user opens a file, the data flows through FileManager to DataManager to the UI.

```mermaid
sequenceDiagram
    participant User
    participant MW as MainWindow
    participant FM as FileManager
    participant DM as DataManager
    participant DN as DataNodes
    participant TW as TreeWidget
    participant PV as Viewer

    User->>MW: File > Open
    MW->>FM: open_point_cloud_file()
    FM->>FM: Qt File Dialog
    FM->>FM: o3d.io.read_point_cloud()
    FM->>FM: Create PointCloud object

    FM-->>DM: SIGNAL: point_cloud_loaded(path, pc)

    activate DM
    DM->>DM: Create DataNode(data=pc)
    DM->>DN: add_node(data_node)
    DN-->>DM: return uid
    DM->>TW: add_branch(uid, name, is_root=True)
    DM->>DM: Calculate memory size
    DM->>TW: update_cache_tooltip(uid, size)
    deactivate DM
```

### Key Points

- **FileManager** handles file dialogs and Open3D I/O
- **DataNode** wraps the PointCloud with metadata (uid, parent, dependencies)
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
    participant DM as DataManager
    participant ATM as AnalysisThreadManager
    participant Thread as BackgroundThread
    participant Plugin

    User->>MW: Click Analysis Menu Item
    MW->>DB: open_dialog_box(plugin_name)
    DB->>DB: Create DynamicDialog
    User->>DB: Enter params, click OK

    DB-->>DM: SIGNAL: analysis_params(name, params)

    activate DM
    DM->>MW: disable_menus(), disable_tree()
    DM->>MW: show_processing_overlay()
    DM->>ATM: start_analysis(plugin, node, params)
    DM->>DM: Start QTimer polling (100ms)
    deactivate DM

    activate Thread
    ATM->>Thread: Start background thread
    Thread->>Thread: Reconstruct if needed
    Thread->>Plugin: execute(data_node, params)
    Plugin-->>Thread: return (result, type, deps)
    Thread->>ATM: Mark completed (set flag)
    deactivate Thread

    Note over DM: QTimer polls every 100ms

    DM->>ATM: check_completion()
    ATM-->>DM: Completed! Call handle_result()

    activate DM
    DM->>DM: Create result DataNode
    DM->>DN: add_node(result_node)
    DM->>TW: add_branch(uid, parent, name)
    DM->>TW: Update visibility
    DM->>MW: enable_menus(), enable_tree()
    DM->>MW: hide_processing_overlay()
    deactivate DM
```

### Threading Model

- **Thread Type:** Python `threading.Thread` (NOT QThread)
- **Communication:** Flag polling via QTimer (100ms), NOT callbacks
- **Thread Safety:** Plugins only READ data, return NEW objects
- **No Deep Copy:** Memory efficient - relies on read-only access

---

## 5. Data Flow: Visibility & Reconstruction

When a user toggles visibility, derived data (masks, clusters) must be reconstructed to PointCloud for rendering.

```mermaid
sequenceDiagram
    participant User
    participant TW as TreeWidget
    participant DM as DataManager
    participant NRM as NodeReconstructionManager
    participant PV as Viewer

    User->>TW: Toggle checkbox
    TW->>TW: Update visibility_status dict

    TW-->>DM: SIGNAL: branch_visibility_changed(status)

    activate DM
    DM->>DM: disable UI, show overlay

    loop For each visible UID
        DM->>DM: Get DataNode
        alt Node has cached PointCloud
            DM->>DM: Use cached (fast path)
        else Need reconstruction
            DM->>NRM: reconstruct_branch(uid)
            NRM->>NRM: Find root or cached ancestor
            NRM->>NRM: Apply task chain
            NRM-->>DM: return PointCloud
            DM->>DM: Cache result on node
        end
        DM->>DM: Apply LOD if needed
    end

    DM->>PV: set_point_vertices(combined)
    DM->>PV: update()
    DM->>DM: enable UI, hide overlay
    deactivate DM
```

### Reconstruction Process

1. **Check Cache:** If node has `cached_point_cloud`, use it immediately
2. **Find Ancestor:** Walk up tree looking for cached ancestor or root PointCloud
3. **Apply Tasks:** Use `NodeReconstructionManager.tasks_registry` to apply transformations
4. **Cache Result:** Store reconstructed PointCloud on node for future use

### Task Registry

| Data Type | Task Class | Transformation |
|-----------|------------|----------------|
| `masks` | ApplyMasks | Filter points (subset) |
| `cluster_labels` | ApplyClusters | Apply cluster/semantic colors |
| `eigenvalues` | ApplyEigenvalues | Color by eigenvalues |
| `values` | ApplyValues | Color by scalar values |
| `colors` | ApplyColors | Apply RGB colors |
| `dist_to_ground` | ApplyDistToGround | Color by height |
| `class_reference` | ApplyClassReference | Filter by semantic class |

---

## 6. Component Relationships

Static class diagram showing the main components and their relationships.

```mermaid
classDiagram
    class GlobalVariables {
        +global_main_window
        +global_data_manager
        +global_file_manager
        +global_tree_structure_widget
        +global_pcd_viewer_widget
        +global_data_nodes
        +global_analysis_thread_manager
    }

    class MainWindow {
        +plugin_manager
        +file_manager
        +tree_widget
        +pcd_viewer_widget
        +dialog_boxes_manager
        +data_manager
        +setup_ui()
        +open_dialog_box()
        +disable_menus()
        +enable_menus()
    }

    class DataManager {
        +data_nodes: DataNodes
        +analysis_manager
        +node_reconstruction_manager
        +selected_branches: List
        +apply_analysis()
        +reconstruct_branch()
        +handle_analysis_result()
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
    GlobalVariables --> DataManager
    MainWindow --> DataManager
    DataManager --> DataNodes
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
        DialogBoxesManager -->|SIGNAL| DataManager
        DataManager --> AnalysisThreadManager
        AnalysisThreadManager -->|background| Plugin.execute
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
| Run an analysis plugin | `DataManager.apply_analysis()` |
| Add a node to the tree | `DataManager._on_point_cloud_loaded()` or `handle_analysis_result()` |
| Render points in viewer | `PCDViewerWidget.set_point_vertices()` |
| Create a new analysis plugin | `plugins/YourCategory/your_plugin.py` (inherit `AnalysisPlugin`) |
| Create a new action plugin | `plugins/YourCategory/your_plugin.py` (inherit `ActionPlugin`) |
| Access any global manager | `from config.config import global_variables` |
| Reconstruct a branch | `DataManager.reconstruct_branch(uid)` |
| Get selected tree items | `DataManager.selected_branches` |
| Disable UI during processing | `MainWindow.disable_menus()`, `disable_tree()` |

### Signal Connections

| Signal | Source | Handler | Purpose |
|--------|--------|---------|---------|
| `point_cloud_loaded` | FileManager | DataManager._on_point_cloud_loaded | File loaded |
| `analysis_params` | DialogBoxesManager | DataManager.apply_analysis | Dialog OK clicked |
| `branch_visibility_changed` | TreeStructureWidget | DataManager._on_branch_visibility_changed | Checkbox toggled |
| `branch_selection_changed` | TreeStructureWidget | DataManager._on_branch_selection_changed | Tree selection |
| `branch_added` | TreeStructureWidget | DataManager._on_branch_added | New branch added |

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `gui/main_window.py` | Main window, menu building |
| `core/data_manager.py` | Central data coordinator |
| `core/data_node.py` | Single data unit wrapper |
| `core/data_nodes.py` | Collection manager |
| `core/point_cloud.py` | Primary data structure |
| `core/node_reconstruction_manager.py` | Rebuilds PointCloud from derived data |
| `core/analysis_thread_manager.py` | Background thread management |
| `services/file_manager.py` | File I/O operations |
| `plugins/plugin_manager.py` | Plugin discovery and registration |
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
global_variables.global_data_manager.method()
global_variables.global_pcd_viewer_widget.update()

ACCEPTABLE: Callbacks (when singleton doesn't fit)
──────────────────────────────────────────────────
component_a.process(on_complete=callback_function)

AVOID: Custom Qt Signals/Slots
──────────────────────────────
class MyClass(QObject):
    custom_signal = pyqtSignal()  # Don't do this
```

---

*Last updated: January 2026*
