# SPCToolkit Architecture Overview

This document provides a high-level overview of the SPCToolkit architecture for developers and maintainers.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Core Data Model](#3-core-data-model)
4. [Component Relationships](#4-component-relationships)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Plugin System](#6-plugin-system)
7. [Services Layer](#7-services-layer)
8. [GUI Architecture](#8-gui-architecture)
9. [Threading Model](#9-threading-model)
10. [Key Design Decisions](#10-key-design-decisions)

---

## 1. System Overview

SPCToolkit is a PyQt5-based point cloud processing application with these key characteristics:

- **Plugin-based architecture** for extensibility
- **Tree-based hierarchical data management** for organizing point cloud derivatives
- **On-demand reconstruction** for memory efficiency
- **Hardware-aware backend selection** for GPU/CPU optimization
- **Singleton communication pattern** (avoids Qt custom signals)

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           MainWindow                                 │
├─────────────┬─────────────────────────────────┬─────────────────────┤
│ TreeWidget  │         PCDViewerWidget         │    MenuBar          │
│  (Left)     │      (Center - OpenGL)          │  (from Plugins)     │
└──────┬──────┴────────────────┬────────────────┴──────────┬──────────┘
       │                       │                           │
       ▼                       ▼                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          DataManager                                  │
│  (Central Coordinator - manages data, visibility, reconstruction)     │
└───────┬──────────────┬────────────────┬────────────────┬─────────────┘
        │              │                │                │
        ▼              ▼                ▼                ▼
   ┌─────────┐   ┌──────────┐   ┌─────────────┐   ┌─────────────────┐
   │DataNodes│   │FileManager│  │AnalysisThread│  │NodeReconstruction│
   │         │   │          │   │  Manager     │   │    Manager       │
   └────┬────┘   └──────────┘   └──────┬──────┘   └────────┬────────┘
        │                              │                   │
        ▼                              ▼                   ▼
   ┌─────────┐                  ┌─────────────┐      ┌──────────┐
   │DataNode │                  │  Plugins    │      │  Tasks   │
   │(PointCloud,               │(DBSCAN, etc)│      │(Apply*)  │
   │ Clusters,                 └─────────────┘      └──────────┘
   │ Masks...)│
   └──────────┘
```

---

## 2. Directory Structure

```
SPCToolkit/
├── main.py                 # Application entry point
├── CLAUDE.md               # Developer instructions
├── ARCHITECTURE.md         # This file
│
├── core/                   # Core data structures and managers
│   ├── point_cloud.py      # PointCloud class (primary data)
│   ├── clusters.py         # Clusters class (clustering + optional semantic names)
│   ├── masks.py            # Masks class (boolean selection)
│   ├── eigenvalues.py      # Eigenvalues class
│   ├── values.py           # Values class (scalar per-point)
│   ├── colors.py           # Colors class
│   ├── class_reference.py  # ClassReference (lightweight filter by class)
│   ├── dist_to_ground.py   # DistToGround class
│   ├── data_node.py        # DataNode (tree node wrapper)
│   ├── data_nodes.py       # DataNodes collection manager
│   ├── data_manager.py     # Central coordinator
│   ├── analysis_manager.py # Plugin execution coordinator
│   ├── analysis_thread_manager.py  # Background threading
│   └── node_reconstruction_manager.py  # Reconstruction pipeline
│
├── services/               # Utility services
│   ├── file_manager.py     # Point cloud I/O, project save/load
│   ├── hardware_detector.py # GPU/CPU capability detection
│   ├── backend_registry.py # Algorithm backend selection
│   ├── batch_processor.py  # Spatial batching for large data
│   ├── memory_manager.py   # RAM/VRAM tracking
│   └── backends/           # Algorithm implementations
│       └── dbscan_backends.py
│
├── tasks/                  # Reconstruction tasks
│   ├── apply_masks.py      # Masks → filtered PointCloud
│   ├── apply_clusters.py   # Clusters → colored PointCloud (handles named clusters)
│   ├── apply_eigenvalues.py
│   ├── apply_values.py
│   ├── apply_colors.py
│   ├── apply_dist_to_ground.py
│   └── apply_class_reference.py  # Filter by semantic class
│
├── plugins/                # Plugin-based extensions
│   ├── interfaces.py       # Plugin, ActionPlugin base classes
│   ├── plugin_manager.py   # Discovery and loading
│   ├── 000_File/           # File menu plugins
│   ├── 010_View/           # View menu plugins
│   ├── 015_Branch/         # Branch menu plugins
│   ├── 020_Points/         # Points menu (Subsampling, Filtering, etc.)
│   ├── 030_Selection/      # Selection menu plugins
│   ├── 040_Clusters/       # Clusters menu plugins
│   ├── 050_Processing/     # Processing menu plugins
│   ├── 060_ML_Models/      # ML model plugins
│   └── 090_Help/           # Help menu plugins
│
├── gui/                    # Qt5 GUI components
│   ├── main_window.py      # Main application window
│   ├── widgets/
│   │   ├── tree_structure_widget.py  # Hierarchical data tree
│   │   ├── pcd_viewer_widget.py      # OpenGL 3D viewer
│   │   └── process_overlay_widget.py # Processing status overlay
│   ├── dialog_boxes/
│   │   └── dialog_boxes_manager.py   # Dynamic parameter dialogs
│   └── dialogs/            # Specialized dialog windows
│
├── config/
│   └── config.py           # GlobalVariables singleton
│
├── models/                 # ML model files
├── unit_test/              # Unit tests
└── redundant/              # Legacy code (archived)
```

---

## 3. Core Data Model

### 3.1 Data Type Hierarchy

```
                    ┌─────────────┐
                    │  DataNode   │
                    │  (wrapper)  │
                    └──────┬──────┘
                           │ contains
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │                    Data Types                         │
    ├─────────────┬───────────┬───────────┬───────────────┤
    │ PointCloud  │ Clusters  │   Masks   │  Eigenvalues  │
    │ (primary)   │ (labels + │ (boolean) │  (n,3 array)  │
    │             │  names*)  │           │               │
    ├─────────────┼───────────┼───────────┼───────────────┤
    │   Values    │  Colors   │DistToGrnd │ClassReference │
    │ (scalars)   │ (RGB)     │ (heights) │ (class filter)│
    └─────────────┴───────────┴───────────┴───────────────┘

* Clusters optionally includes cluster_names (Dict[int, str])
  and cluster_colors (Dict[str, RGB]) for semantic labeling
```

### 3.2 DataNode Structure

```python
DataNode:
├── uid: str              # Unique identifier (UUID)
├── data: Any             # PointCloud, Clusters, Masks, etc.
├── data_type: str        # "point_cloud", "cluster_labels", "masks", etc.
├── parent_uid: str       # Parent node reference (None for root)
├── depends_on: List[str] # Dependency UIDs
├── tags: List[str]       # Classification tags
├── params: str           # Human-readable description
├── is_cached: bool       # Runtime cache flag
├── cached_point_cloud    # Runtime reconstruction cache
└── memory_size: int      # Persistent memory size
```

### 3.3 Tree Structure Example

```
Project Tree:
├── scan_001.ply [PointCloud]           ← Root (loaded file)
│   ├── DBSCAN (eps=0.5) [Clusters]     ← Derived from root
│   │   └── Cluster_0 [ClassReference]  ← Derived from clusters
│   └── Ground Filter [Masks]           ← Derived from root
│       └── Eigenvalues [Eigenvalues]   ← Derived from masked points
└── scan_002.ply [PointCloud]           ← Another root
    └── ...
```

---

## 4. Component Relationships

### 4.1 Class Dependency Graph

```
┌──────────────────────────────────────────────────────────────────┐
│                         MainWindow                                │
├──────────────────────────────────────────────────────────────────┤
│  Creates and owns:                                                │
│  ├── FileManager                                                  │
│  ├── TreeStructureWidget                                          │
│  ├── PCDViewerWidget                                              │
│  ├── DataManager ─────────────────────────────────────────────┐   │
│  │   ├── Uses: DataNodes (collection)                         │   │
│  │   │         └── Contains: DataNode instances               │   │
│  │   ├── Uses: AnalysisManager → PluginManager                │   │
│  │   ├── Uses: NodeReconstructionManager → Task classes       │   │
│  │   └── Coordinates: FileManager, TreeWidget, ViewerWidget   │   │
│  ├── AnalysisThreadManager                                    │   │
│  ├── PluginManager                                            │   │
│  └── BackendRegistry → HardwareInfo                           │   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    GlobalVariables Singleton                      │
├──────────────────────────────────────────────────────────────────┤
│  global_file_manager          → FileManager instance              │
│  global_pcd_viewer_widget     → PCDViewerWidget instance          │
│  global_tree_structure_widget → TreeStructureWidget instance      │
│  global_data_nodes            → DataNodes instance                │
│  global_data_manager          → DataManager instance              │
│  global_main_window           → MainWindow instance               │
│  global_analysis_thread_manager → AnalysisThreadManager instance  │
│  global_hardware_info         → HardwareInfo instance             │
│  global_backend_registry      → BackendRegistry instance          │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Communication Pattern

```
PREFERRED: Singleton Pattern
─────────────────────────────
Any Component ──→ global_variables.global_data_manager.method()
                 global_variables.global_pcd_viewer_widget.update()

ACCEPTABLE: Callbacks (when singleton doesn't fit)
──────────────────────────────────────────────────
Component A ──callback──→ Component B

AVOID: Custom Qt Signals/Slots
──────────────────────────────
❌ class MyClass(QObject):
       custom_signal = pyqtSignal()
```

---

## 5. Data Flow Diagrams

### 5.1 Loading Point Cloud

```
User clicks: File > Import Point Cloud
                    │
                    ▼
            ┌───────────────┐
            │  FileManager  │
            │ open_point_   │
            │ cloud_file()  │
            └───────┬───────┘
                    │ Open3D reads PLY
                    │ Translate to origin
                    ▼
            ┌───────────────┐
            │  DataManager  │
            │ _on_point_    │
            │ cloud_loaded()│
            └───────┬───────┘
                    │ Create DataNode (root)
                    │ Add to DataNodes
                    ▼
    ┌───────────────┴───────────────┐
    ▼                               ▼
┌─────────────┐             ┌─────────────┐
│ TreeWidget  │             │ ViewerWidget│
│ add_branch()│             │ set_points()│
└─────────────┘             └─────────────┘
```

### 5.2 Running Analysis Plugin

```
User clicks: Points > Clustering > DBSCAN
                    │
                    ▼
          ┌─────────────────┐
          │DialogBoxesManager│
          │ open_dialog_box()│
          └────────┬────────┘
                   │ Show parameter dialog
                   │ User clicks OK
                   ▼
          ┌─────────────────┐
          │  DataManager    │
          │ apply_analysis()│
          └────────┬────────┘
                   │ Disable UI
                   │ Show overlay
                   ▼
     ┌─────────────────────────┐
     │AnalysisThreadManager    │
     │ start_analysis()        │
     └───────────┬─────────────┘
                 │ Background thread
                 ▼
         ┌───────────────┐
         │ DBSCANPlugin  │
         │  execute()    │◄─── Backend selection via
         └───────┬───────┘     BackendRegistry
                 │ Returns (Clusters, "cluster_labels", [deps])
                 ▼
     ┌─────────────────────────┐
     │ QTimer polling detects  │
     │ completion (100ms)      │
     └───────────┬─────────────┘
                 │
                 ▼
          ┌─────────────────┐
          │  DataManager    │
          │ handle_result() │
          └────────┬────────┘
                   │ Create DataNode (derived)
                   │ Enable UI, hide overlay
                   ▼
   ┌───────────────┴───────────────┐
   ▼                               ▼
┌─────────────┐             ┌─────────────┐
│ TreeWidget  │             │ (Ready for  │
│ add_branch()│             │ selection)  │
└─────────────┘             └─────────────┘
```

### 5.3 Branch Reconstruction (Visualization)

```
User selects: Clusters node in tree
                    │
                    ▼
          ┌─────────────────┐
          │  DataManager    │
          │reconstruct_branch()│
          └────────┬────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ Build path: root → target    │
    │                              │
    │ [PointCloud] → [Clusters]    │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ NodeReconstructionManager    │
    │                              │
    │ For each node in path:       │
    │   task = tasks_registry[type]│
    │   pc = task.apply(pc, data)  │
    └──────────────┬───────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  ApplyClusters  │
         │    (Task)       │
         └────────┬────────┘
                  │ Returns PointCloud with cluster colors
                  ▼
          ┌─────────────────┐
          │ ViewerWidget    │
          │ set_points()    │
          └─────────────────┘
```

### 5.4 Reconstruction Task Registry

```
NodeReconstructionManager.tasks_registry:
┌─────────────────┬───────────────────┬─────────────────────────────────┐
│   Data Type     │    Task Class     │      Transformation             │
├─────────────────┼───────────────────┼─────────────────────────────────┤
│ "masks"         │ ApplyMasks        │ Filter points (subset)          │
│ "cluster_labels"│ ApplyClusters     │ Apply cluster/semantic colors   │
│ "eigenvalues"   │ ApplyEigenvalues  │ Color by eigenvalues            │
│ "values"        │ ApplyValues       │ Color by scalar values          │
│ "colors"        │ ApplyColors       │ Apply RGB colors                │
│ "dist_to_ground"│ ApplyDistToGround │ Color by height                 │
│ "class_reference"│ApplyClassReference│ Filter by semantic class       │
└─────────────────┴───────────────────┴─────────────────────────────────┘

Note: ApplyClusters handles both simple clusters (per-point colors) and
named clusters (semantic colors via cluster_names/cluster_colors).
```

---

## 6. Plugin System

### 6.1 Folder-Based Menu Structure

```
plugins/
├── 000_File/                    → Menu: "File"
│   ├── 000_import_plugin.py     →   "Import Point Cloud"
│   └── 020_save_plugin.py       →   "Save Project"
├── 020_Points/                  → Menu: "Points"
│   ├── 000_Subsampling/         →   Submenu: "Subsampling"
│   │   └── 000_voxel_plugin.py  →     "Voxel Downsample"
│   └── 020_Clustering/          →   Submenu: "Clustering"
│       └── 000_dbscan_plugin.py →     "DBSCAN"
└── ...

Numbering (000_, 010_, etc.) controls menu order.
Folder depth controls menu nesting.
```

### 6.2 Plugin Interfaces

```python
# Analysis Plugin (processes data, returns results)
class Plugin(ABC):
    def get_name(self) -> str: ...
    def get_parameters(self) -> Dict[str, Any]: ...
    def execute(self, data_node: DataNode, params: Dict) -> Tuple[Any, str, List]: ...
    #                                                         ↑     ↑    ↑
    #                                                      result  type  deps

# Action Plugin (performs actions, no return)
class ActionPlugin(ABC):
    def get_name(self) -> str: ...
    def get_parameters(self) -> Dict[str, Any]: ...  # Can return {}
    def execute(self, main_window, params: Dict) -> None: ...
```

### 6.3 Parameter Schema

```python
def get_parameters(self) -> Dict[str, Any]:
    return {
        "eps": {
            "type": "float",
            "default": 0.5,
            "min": 0.01,
            "max": 10.0,
            "label": "Epsilon",
            "description": "Maximum neighbor distance"
        },
        "method": {
            "type": "choice",
            "options": ["Method A", "Method B"],
            "default": "Method A",
            "label": "Method"
        }
    }
```

---

## 7. Services Layer

### 7.1 BackendRegistry (Hardware-Aware Selection)

```
┌─────────────────────────────────────────────────────────────────┐
│                    HardwareDetector                              │
│  Detects: NVIDIA GPU, CUDA, CuPy, RAPIDS cuML, PyTorch CUDA     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BackendRegistry                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FULL GPU (Linux + NVIDIA + RAPIDS):                            │
│  ├── DBSCAN: sklearn (O(n log n) faster than GPU brute-force)   │
│  ├── KNN: cuML (GPU)                                            │
│  ├── Masking: CuPy (GPU)                                        │
│  └── Eigenvalues: PyTorch CUDA (GPU)                            │
│                                                                  │
│  PARTIAL GPU (NVIDIA without RAPIDS):                           │
│  ├── DBSCAN: sklearn (CPU)                                      │
│  ├── KNN: scipy (CPU)                                           │
│  ├── Masking: CuPy (GPU)                                        │
│  └── Eigenvalues: PyTorch CUDA (GPU)                            │
│                                                                  │
│  CPU ONLY:                                                       │
│  ├── DBSCAN: sklearn                                            │
│  ├── KNN: scipy                                                 │
│  ├── Masking: NumPy                                             │
│  └── Eigenvalues: NumPy                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 BatchProcessor (Spatial Batching)

```
Large Point Cloud (e.g., 50M points)
            │
            ▼
┌───────────────────────────────────────┐
│        BatchProcessor                  │
│  ┌─────┬─────┬─────┬─────┐           │
│  │ B1  │ B2  │ B3  │ B4  │  Grid     │
│  ├─────┼─────┼─────┼─────┤  with     │
│  │ B5  │ B6  │ B7  │ B8  │  10%      │
│  ├─────┼─────┼─────┼─────┤  overlap  │
│  │ B9  │ B10 │ B11 │ B12 │           │
│  └─────┴─────┴─────┴─────┘           │
└───────────────────────────────────────┘
            │
            │ Process each batch
            │ Merge results
            ▼
      Final Result
```

### 7.3 MemoryManager

```python
MemoryManager:
├── get_available_ram_mb()   # System RAM check
├── get_available_gpu_mb()   # VRAM availability
└── estimate_render_memory() # Memory for rendering
```

---

## 8. GUI Architecture

### 8.1 MainWindow Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Menu Bar: [File] [View] [Branch] [Points] [Selection] [Clusters]│
├─────────────┬───────────────────────────────────────────────────┤
│             │                                                    │
│  TreeWidget │              PCDViewerWidget                       │
│  (QTreeWidget)            (QOpenGLWidget)                       │
│             │                                                    │
│  ☑ scan.ply │         ┌─────────────────────┐                   │
│    ☑ DBSCAN │         │                     │                   │
│      □ Cls0 │         │   3D Point Cloud    │                   │
│      ☑ Cls1 │         │    Visualization    │                   │
│    □ Filter │         │                     │                   │
│             │         └─────────────────────┘                   │
│             │                                                    │
├─────────────┴───────────────────────────────────────────────────┤
│ Status Bar: GPU: 45% (3.6GB/8GB) | RAM: 32% (10GB/32GB)         │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Widget Responsibilities

```
TreeStructureWidget:
├── Display hierarchical data structure
├── Visibility checkboxes (column 0)
├── Cache status checkboxes (column 1)
├── Multi-select support (Ctrl+Click)
└── Emits: branch_added, visibility_changed, selection_changed

PCDViewerWidget:
├── OpenGL 3D rendering (VBO-based)
├── Camera controls (rotate, pan, zoom)
├── Point picking (Shift+Click)
├── Hotkeys: F (zoom extent), Ctrl+R (reset), ESC (deselect)
└── Methods: set_points(), zoom_to_extent(), reset_view()

ProcessOverlayWidget:
├── Semi-transparent status overlay
├── Shows during long operations
└── Non-blocking (visual feedback only)
```

### 8.3 UI Protection During Processing

```
During Analysis:
┌──────────────────────────────────────────────────────────────┐
│ Menu Bar: [DISABLED]                                          │
├────────────┬─────────────────────────────────────────────────┤
│            │                                                  │
│ TreeWidget │        PCDViewerWidget                          │
│ [DISABLED] │         [ENABLED]                               │
│            │                                                  │
│ ┌────────────────────┐                                       │
│ │ Running DBSCAN... │ ← ProcessOverlayWidget                │
│ └────────────────────┘                                       │
│            │                                                  │
└────────────┴─────────────────────────────────────────────────┘

✓ Viewer remains enabled for camera manipulation
✗ Menus disabled to prevent concurrent operations
✗ Tree disabled to prevent selection changes
```

---

## 9. Threading Model

### 9.1 Background Threading Architecture

```
Main Thread (Qt Event Loop)           Background Thread
─────────────────────────────         ─────────────────
        │
        │ User triggers analysis
        ▼
┌───────────────────┐
│ DataManager       │
│ apply_analysis()  │
│ • Disable UI      │
│ • Show overlay    │
└─────────┬─────────┘
          │
          │ AnalysisThreadManager.start_analysis()
          │
          ▼
┌───────────────────┐                 ┌───────────────────┐
│ Create Thread     │────────────────▶│ AnalysisThread    │
│ Start QTimer      │                 │ • Run plugin      │
│ (polling 100ms)   │                 │ • Store result    │
└─────────┬─────────┘                 │ • Set completed   │
          │                           └───────────────────┘
          │ QTimer.timeout
          ▼
┌───────────────────┐
│ Check completion  │◄─────────────── is_completed = True
│ • Hide overlay    │
│ • Enable UI       │
│ • Process result  │
└───────────────────┘
```

### 9.2 Thread Safety Rules

```
✓ SAFE:
  - Plugins READ data (read-only is thread-safe)
  - Plugins return NEW objects (no modification)
  - Reconstruction in background thread

✗ AVOID:
  - Modifying DataNodes from background thread
  - UI updates from background thread
  - Shared mutable state
```

---

## 10. Key Design Decisions

### 10.1 On-Demand Reconstruction

**Why**: Memory efficiency for large datasets

```
Traditional Approach:          SPCToolkit Approach:
────────────────────          ────────────────────
Store each view as            Store only:
separate PointCloud:          • Root PointCloud
• 50M points × 3 views        • Lightweight derived data
• 450MB × 3 = 1.35GB          • Reconstruct on demand
                              • ~500MB total
```

### 10.2 Singleton Over Signals

**Why**: Simpler debugging, explicit control flow

```
Signal/Slot (AVOID):           Singleton (PREFER):
────────────────────           ────────────────────
self.signal.emit(data)         global_variables.global_data_manager.method(data)
# Who receives this?           # Explicit destination
# Hard to trace                # Easy to trace
```

### 10.3 Plugin-Based Architecture

**Why**: Easy extensibility without core changes

```
Adding new analysis:
1. Create plugins/Category/your_plugin.py
2. Implement Plugin interface
3. Restart application
4. Menu item appears automatically
```

### 10.4 Functional Task Pattern

**Why**: Immutability enables safe concurrency

```python
# Tasks NEVER modify input
def apply(self, point_cloud: PointCloud, data: Any) -> PointCloud:
    # Create and return NEW PointCloud
    return PointCloud(
        points=point_cloud.points[mask],
        colors=new_colors
    )
```

### 10.5 Hardware-Aware Backends

**Why**: Optimal performance across different systems

```
User's GPU: RTX 3080 (8GB)
├── CuPy available → Use GPU for masking
├── PyTorch CUDA available → Use GPU for eigenvalues
└── RAPIDS not available → Use sklearn for DBSCAN
```

---

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| Entry Point | `main.py` | Application startup |
| Data Model | `core/point_cloud.py` | Primary data structure |
| Central Coordinator | `core/data_manager.py` | Orchestrates everything |
| Plugin Discovery | `plugins/plugin_manager.py` | Loads plugins |
| Plugin Interface | `plugins/interfaces.py` | Base classes |
| Reconstruction | `core/node_reconstruction_manager.py` | Builds visualizations |
| 3D Viewer | `gui/widgets/pcd_viewer_widget.py` | OpenGL rendering |
| Tree View | `gui/widgets/tree_structure_widget.py` | Data hierarchy |
| Global Access | `config/config.py` | Singleton pattern |
| Hardware Detection | `services/hardware_detector.py` | GPU/CPU capabilities |
| Backend Selection | `services/backend_registry.py` | Algorithm selection |

---

*Last updated: January 2026*
