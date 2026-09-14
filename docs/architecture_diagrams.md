# SPCToolkit — Development Diagrams

Four diagrams generated from the current source tree (branch `refactor-linear-region-growing`).
They complement **ARCHITECTURE.md** (layers, flows) and **PLUGIN_ARCHITECTURE.md** (plugin reference).

1. [Component Diagram — Plugin System](#1-component-diagram--plugin-system)
2. [Sequence Diagram — Analysis Plugin Execution & Threading](#2-sequence-diagram--analysis-plugin-execution--threading)
3. [Class Diagram — DataNode / PointCloud Entities](#3-class-diagram--datanode--pointcloud-entities)
4. [Data Flow Diagram — Import → Reconstruct → Plugin → Render](#4-data-flow-diagram--import--reconstruct--plugin--render)

---

## 1. Component Diagram — Plugin System

Static structure of the plugin subsystem: discovery from the filesystem, the two
interfaces, the four registries, menu building, and how a running plugin reaches
data and hardware backends.

```mermaid
flowchart TB
    subgraph FS["Filesystem — plugins/"]
        DIRS["15 numbered menu dirs<br/>000_File … 100_Help<br/>(153 .py files)"]
        SYS["plugins/*.py at root<br/>menu_path = None"]
    end

    subgraph PMGR["PluginManager — plugins/plugin_manager.py"]
        WALK["load_plugins()<br/>os.walk + importlib"]
        INSP["_load_plugin_file()<br/>inspect.getmembers → issubclass"]
        REG["_register_plugin()<br/>instantiate → get_name()"]
        HOT["reload_plugin() / unload_plugin()<br/>scan_and_load_new_plugins()"]
    end

    subgraph REGS["Registries (dicts)"]
        R1["plugins<br/>{name: (class, menu_path, kind)}"]
        R2["menu_structure<br/>{menu_path: [names]}"]
        R3["analysis_plugins"]
        R4["action_plugins"]
    end

    subgraph IFACE["plugins/interfaces.py"]
        P["Plugin (AnalysisPlugin)<br/>get_name / get_parameters / execute<br/>confirm_before_execute<br/>requires_selection<br/>build_param_dialog"]
        AP["ActionPlugin<br/>get_name / get_parameters / execute<br/>requires_selection"]
    end

    subgraph UI["GUI"]
        MW["MainWindow<br/>populate_menus_from_plugins()<br/>_create_menu_hierarchy()<br/>rebuild_plugin_menus()"]
        GATE["selection_gate<br/>selection_kind / selection_present"]
        DBM["DialogBoxesManager<br/>_last_params"]
        DD["DynamicDialog<br/>8 param types"]
        CUST["plugins/dialogs/*<br/>custom QDialogs"]
    end

    subgraph RUN["Execution"]
        AC["ApplicationController.run_analysis()"]
        AE["AnalysisExecutor<br/>threading.Thread"]
        PR["PipelineRunner<br/>replay captured steps"]
    end

    subgraph BE["plugins/backends/"]
        BR["BackendRegistry<br/>FULL GPU / PARTIAL GPU / CPU"]
        B2["dbscan · hdbscan · knn · masking<br/>eigenvalue · normals · grid · selection"]
    end

    DIRS --> WALK
    SYS --> WALK
    WALK --> INSP --> REG --> REGS
    HOT --> REGS
    INSP -. "issubclass check" .-> IFACE
    R2 --> MW
    MW -->|QAction triggered| GATE
    GATE --> DBM
    DBM --> DD
    DBM -. "build_param_dialog()" .-> CUST
    DBM -->|params dict| AC
    AC --> AE
    PR --> AC
    R3 --> AE
    R4 -->|main thread| MW
    AE -->|"execute(data_node, params)"| P
    MW -->|"execute(main_window, params)"| AP
    P --> BR
    BR --> B2
    HOT -.-> MW
```

**Key facts**

| Item | Value |
|------|-------|
| Discovery | `os.walk` over `plugins/`, folder path → menu path, `000_` prefixes order menus |
| Plugin kinds | `Plugin` (background thread, returns `(result, type, deps)`) and `ActionPlugin` (main thread, returns `None`) |
| Registries | `plugins`, `menu_structure`, `analysis_plugins`, `action_plugins` |
| Hot reload | `ManagePluginsPlugin` → `reload/unload/scan` → `rebuild_plugin_menus()` |
| Backend choice | `global_variables.global_backend_registry`, picked once at startup from detected hardware |

---

## 2. Sequence Diagram — Analysis Plugin Execution & Threading

One full run of an analysis plugin, including the selection gate, the pre-flight
confirmation, the worker thread, and the 100 ms QTimer poll that brings the
result back to the main thread.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant MW as MainWindow
    participant SG as selection_gate
    participant DBM as DialogBoxesManager
    participant AC as ApplicationController
    participant AE as AnalysisExecutor
    participant TH as Worker Thread
    participant RS as ReconstructionService
    participant PL as Plugin
    participant GV as global_variables
    participant TW as TreeStructureWidget

    User->>MW: Click menu action
    MW->>SG: selection_kind(plugin_class)
    alt selection required and absent
        MW->>User: SelectionPrompt (non-modal)
        User->>MW: select points/branch → Continue
    end
    MW->>DBM: get_analysis_params(name)
    DBM->>DBM: DynamicDialog or build_param_dialog()
    DBM-->>MW: params dict (direct return, no signal)

    MW->>AC: get_analysis_confirmation(name, params)
    AC->>PL: confirm_before_execute(node, params)
    PL-->>AC: warning or None
    alt warning returned
        MW->>User: Yes/No question
    end

    MW->>MW: disable_menus(), disable_tree(), show Cancel
    MW->>AC: run_analysis(name, params, on_error)
    AC->>AE: execute(plugin_class, node, params, type)
    AE->>GV: global_cancel_event.clear()
    AE->>TH: threading.Thread(daemon=True).start()
    MW->>MW: _start_completion_polling() — QTimer 100 ms

    activate TH
    alt node.data_type != "point_cloud"
        TH->>GV: global_progress = (None, "Reconstructing branch")
        TH->>RS: reconstruct(uid)
        RS-->>TH: PointCloud
        TH->>TH: CacheService.set(uid, pc) — auto-cache parent
        TH->>TH: wrap in temporary DataNode
    end
    TH->>GV: global_progress = (None, "Running plugin...")
    TH->>PL: execute(data_node, params)
    Note over PL: read-only access,<br/>returns NEW objects<br/>(no deep copy needed)
    PL-->>TH: (result, result_type, dependencies)
    TH->>GV: global_progress = (100, "Completed")
    TH->>AE: _result_data set, _is_completed = True
    deactivate TH

    loop every 100 ms
        MW->>GV: read global_progress
        MW->>MW: show_progress(message, percent)
        MW->>AE: check_and_process_completion()
    end
    AE-->>MW: True

    MW->>MW: timer.stop(), clear_progress(), enable_menus/tree
    alt error or cancelled
        MW->>User: error message
    else success
        MW->>AC: add_analysis_result(result, type, deps, parent, params)
        AC-->>MW: new uid
        MW->>TW: add_branch(uid, parent, name)
        MW->>MW: render_visible_data()
        MW->>MW: _advance_pipeline_if_running()
    end
```

**Threading contract**

- `threading.Thread` (daemon), **never** `QThread`; one analysis at a time (`_is_running` guard).
- Cross-thread state: `global_variables.global_progress` (write in thread, read in timer) and
  `global_variables.global_cancel_event` (set by Cancel button, polled in thread).
- Cancellation is checked before reconstruction, before `execute()`, and after `execute()`.
- Menus and tree are disabled during the run; the viewer stays live for camera moves.

---

## 3. Class Diagram — DataNode / PointCloud Entities

The data model: one `DataNodes` collection of `DataNode` wrappers, each holding
either a full `PointCloud` or a lightweight derived payload that a matching
transformer replays onto a parent cloud.

```mermaid
classDiagram
    direction LR

    class DataNodes {
        +dict data_nodes
        +add_node(node) UUID
        +remove_node(uid) bool
        +get_node(uid) DataNode
        +update_parent(uid, new_parent) bool
        +validate_dependency(uid) bool
        +list_nodes() List
    }

    class DataNode {
        +UUID uid
        +str params
        +str alias
        +Any data
        +str data_type
        +str data_name
        +UUID parent_uid
        +List~UUID~ depends_on
        +List~str~ tags
        +bool is_cached
        +Any cached_point_cloud
        +float cache_timestamp
        +str memory_size
    }

    class PointCloud {
        +ndarray points
        +ndarray colors
        +ndarray normals
        +ndarray translation
        +dict attributes
        +dict metadata
        +size()
        +merge(clouds)$
        +get_subset(mask)
        +add_attribute(name, values)
        +dbscan(eps, min_points)
        +hdbscan(min_cluster_size)
        +get_eigenvalues(k)
        +density_downsample(voxel)
        +get_obb()
    }

    class Masks {
        +ndarray mask
    }
    class Clusters {
        +ndarray labels
        +names
    }
    class Eigenvalues
    class Values
    class Colors
    class Normals
    class DistToGround
    class ClassReference
    class TransformMatrix
    class VectorFeature {
        +str symbol_type
        +str geometry_type
        +geometry
        +ndarray transform_matrix
        +ndarray dimensions
        +cluster_reference
        +color
    }
    class CADObject

    class ReconstructionService {
        +data_nodes
        +Dict transformer_registry
        +reconstruct(uid) PointCloud
        +get_reconstruction_path(uid) List
        -_apply_transformer(pc, node)
    }

    class Transformer {
        <<interface>>
        +execute() PointCloud
    }

    DataNodes "1" o-- "*" DataNode : owns
    DataNode "0..1" --> "1" DataNode : parent_uid
    DataNode "*" ..> "*" DataNode : depends_on
    DataNode --> PointCloud : data (root) / cached_point_cloud
    DataNode --> Masks
    DataNode --> Clusters
    DataNode --> Eigenvalues
    DataNode --> Values
    DataNode --> Colors
    DataNode --> Normals
    DataNode --> DistToGround
    DataNode --> ClassReference
    DataNode --> TransformMatrix
    DataNode --> VectorFeature
    DataNode --> CADObject

    ReconstructionService --> DataNodes
    ReconstructionService --> Transformer : registry lookup by data_type
    Transformer --> PointCloud : produces
```

**`transformer_registry` — `data_type` → transformer** (`core/services/reconstruction_service.py`)

| data_type | Transformer | Effect on the parent PointCloud |
|-----------|-------------|---------------------------------|
| `masks` | `MasksTransformer` | keep points where mask is True |
| `cluster_labels` | `ClustersTransformer` | apply cluster / class colors |
| `values` | `ValuesTransformer` | color by scalar field |
| `eigenvalues` | `EigenvaluesTransformer` | color by eigen-feature |
| `colors` | `ColorsTransformer` | replace RGB |
| `dist_to_ground` | `DistToGroundTransformer` | color by height above ground |
| `class_reference` | `ClassReferenceTransformer` | filter/color by class |
| `normals` | `NormalsTransformer` | attach normals |
| `transform_matrix` | `TransformMatrixTransformer` | apply 4×4 transform |

`ContainerTransformer` exists as a pass-through for organizational nodes and is not in the registry.
`VectorFeature` and `CADObject` are render-only payloads — they are drawn as wireframe branches, not reconstructed into clouds.

---

## 4. Data Flow Diagram — Import → Reconstruct → Plugin → Render

Where data enters, what stores it, what transforms it, and what leaves. Solid
arrows are point data; dashed arrows are control/metadata.

```mermaid
flowchart TB
    FILE[("Disk<br/>.ply .las/.laz .e57<br/>.npy/.npz · SemanticKITTI")]
    PROJ[("Project file<br/>saved DataNodes")]
    PIPE[("Saved pipeline<br/>JSON steps")]

    IMP["Import ActionPlugin<br/>plugins/000_File/…"]
    CS["coordinate_service<br/>translate_and_convert()<br/>→ shift to origin, float32"]
    FM["FileManager<br/>open_point_cloud_file()<br/>save_project / load_project"]

    ACADD["ApplicationController<br/>add_point_cloud() / add_analysis_result()"]
    STORE[("DataNodes<br/>UUID → DataNode tree")]
    CACHE[("CacheService<br/>cached_point_cloud on node")]

    RS["ReconstructionService.reconstruct(uid)<br/>walk to nearest cached ancestor<br/>replay transformer chain"]

    AE["AnalysisExecutor<br/>background thread"]
    PLUG["Analysis Plugin.execute()"]
    BP["BatchProcessor<br/>adaptive point-budget k-d tiles"]
    BR["BackendRegistry<br/>GPU ↔ CPU implementations"]

    RC["RenderingCoordinator<br/>prepare_branches → per-branch slices<br/>prepare_mesh_lines → line geometry"]
    LOD["LODManager<br/>dynamic point budget + subsample"]
    VIEW["PCDViewerWidget<br/>set_branches / set_lines<br/>OpenGL draw"]
    TREE["TreeStructureWidget<br/>visibility · selection"]

    EXP["Export ActionPlugin<br/>.ply .las .e57 · classified clusters"]
    OUT[("Disk — exported cloud")]

    FILE --> IMP --> CS --> ACADD
    PROJ --> FM --> ACADD
    ACADD --> STORE
    PIPE -.-> PR["PipelineRunner<br/>replay steps"] -.-> ACADD

    STORE --> RS
    CACHE <--> RS
    RS --> AE
    STORE --> AE
    AE --> PLUG
    PLUG <--> BP
    BP <--> BR
    PLUG <--> BR
    PLUG -->|"(result, result_type, deps)"| ACADD

    STORE --> RC
    RS --> RC
    RC --> LOD --> VIEW
    TREE -.->|visibility_status| RC
    TREE -.->|selected_branches| ACADD
    VIEW -.->|picked points / polygon| ACADD

    STORE --> EXP --> OUT
    STORE --> FM --> PROJ

    classDef store fill:#eef,stroke:#557
    class STORE,CACHE store
```

**Notes**

- Import plugins shift every cloud to the origin and cast to **float32** — the project has no
  large-coordinate code paths downstream.
- Derived nodes (masks, labels, values…) store only the small payload; the full cloud is
  rebuilt on demand by `ReconstructionService` and then cached on the node.
- Reconstruction always starts from the nearest **cached ancestor**, not from the root, so a
  deep branch chain does not re-run every transformer.
- The viewer never receives raw branch data — `RenderingCoordinator` turns each visible branch
  into its own `Nx6 float32` slice (it deliberately does **not** concatenate; the viewer does that
  lazily only when picking or zoom needs a global index), and `LODManager` subsamples to a
  budget derived from free RAM and VRAM first.

---

*Generated 2026-08-23 from the source tree; regenerate if the plugin or entity layout changes.*
*Companion references: **ARCHITECTURE.md** (layers, flows) and **PLUGIN_ARCHITECTURE.md** (plugin reference).*
