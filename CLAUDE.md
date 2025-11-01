# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPCToolkit is a PyQt5-based point cloud processing application with a plugin-based architecture. The application provides interactive visualization and analysis of 3D point cloud data through a tree-based hierarchical data management system.

## Important Architectural Principles

### Communication Pattern: Prefer Singleton Over Signal/Slot
**CRITICAL**: This project avoids Qt's custom signal/slot mechanism. Use the following priority order:

**Priority Order:**
1. **FIRST CHOICE - Singleton Pattern**: Use `global_variables` to access instances and call methods directly
2. **SECOND CHOICE - Callbacks**: Use callback functions when singleton pattern doesn't make sense
3. **NEVER USE - Custom Signals/Slots**: Avoid creating custom pyqtSignal declarations

**What to AVOID:**
- ❌ Custom `pyqtSignal()` declarations
- ❌ `.connect()` for custom signals
- ❌ `.emit()` for custom signals
- ❌ Signal-based communication between managers/widgets

**What to USE (in priority order):**
1. ✅ **Singleton pattern** (global_variables) - PREFER THIS
2. ✅ Callback functions - Use when singleton is not appropriate
3. ✅ Direct method calls
4. ✅ Polling patterns for checking async status

**Exception:** Built-in Qt widget signals (like `QThread.finished`, `QTimer.timeout`) may be used when absolutely necessary.

**Example - WRONG:**
```python
# WRONG: Using custom signals
class MyManager(QObject):
    data_updated = pyqtSignal(str)
    def process(self):
        self.data_updated.emit("done")
```

**Example - BEST (Singleton Pattern):**
```python
# BEST: Using singleton pattern with direct calls
class MyManager:
    def process(self):
        # Do work
        # Directly call method on global instance
        global_variables.global_main_window.update_display()
        global_variables.global_data_manager.refresh_data()
```

**Example - ACCEPTABLE (Callbacks):**
```python
# ACCEPTABLE: Using callbacks when singleton doesn't fit
class MyManager:
    def process(self, completion_callback):
        # Do work
        result = calculate_something()
        # Call the callback
        completion_callback(result)

# Usage
def on_complete(result):
    print(f"Got result: {result}")

manager.process(on_complete)
```

## Running the Application

```bash
# Start the main application
python main.py

# Run unit tests
python unit_test/plugins_discovery_and_loading_test.py
python unit_test/analysis_plugins_execution_test.py
python unit_test/menu_plugin_test.py
```

## Core Architecture

### Plugin System

The application uses a dual-plugin architecture that automatically discovers and loads plugins from designated directories:

- **Analysis Plugins** (`plugins/analysis/`): Implement point cloud processing algorithms (DBSCAN, filtering, eigenvalue analysis, etc.)
- **Menu Plugins** (`plugins/menus/`): Define UI menu items and their actions

Both plugin types inherit from abstract base classes in `plugins/interfaces.py`:
- `AnalysisPlugin`: Requires `get_name()`, `get_parameters()`, and `execute()` methods
- `MenuPlugin`: Requires `get_menu_location()`, `get_menu_items()`, and `handle_action()` methods

The `PluginManager` (plugins/plugin_manager.py) scans plugin directories on startup, imports modules, and registers classes that implement the plugin interfaces.

### Data Management System

The core data architecture revolves around a tree-based hierarchical system:

**DataNode** (core/data_node.py): Represents a single unit of data with:
- `uid`: Unique identifier (UUID)
- `data`: The actual data object (PointCloud, Masks, Clusters, etc.)
- `data_type`: Type identifier for reconstruction
- `parent_uid`: Parent node reference
- `depends_on`: List of dependency UIDs
- `tags`: Classification tags

**DataNodes** (core/data_nodes.py): Manages the collection of all DataNode instances.

**DataManager** (core/data_manager.py): Central coordinator that:
- Manages interactions between UI widgets (tree view, 3D viewer) and data
- Handles point cloud loading via FileManager
- Applies analyses through AnalysisManager
- Reconstructs derived data nodes back into PointCloud instances for visualization

### Branch Reconstruction

The reconstruction system allows viewing derived data (masks, clusters, etc.) as point clouds:

1. When a non-PointCloud node needs visualization, `DataManager.reconstruct_branch()` traverses up to the root PointCloud
2. It builds a list of DataNode UIDs from root to target
3. `NodeReconstructionManager` applies each node's transformation sequentially using task classes from `tasks/` directory
4. Tasks (ApplyMasks, ApplyClusters, ApplyValues, etc.) convert derived data types back into filtered/modified PointCloud instances

### Key Data Types

**PointCloud** (core/point_cloud.py): Primary data structure containing:
- `points`: (n, 3) numpy array of XYZ coordinates
- `colors`, `normals`: Optional (n, 3) arrays
- `attributes`: Dictionary for arbitrary per-point data
- Methods for DBSCAN, subsampling, eigenvalue calculation, SOR filtering, etc.
- GPU acceleration support via CuPy when available

**Clusters** (core/clusters.py): Stores clustering results as:
- `labels`: Integer array assigning points to clusters (-1 for noise)
- `colors`: Optional color array for visualization

**Masks** (core/masks.py): Boolean array for point selection/filtering.

## Creating New Plugins

### Analysis Plugin Template

Place in `plugins/analysis/your_plugin.py`:

```python
from typing import Dict, Any, List, Tuple
from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud

class YourPlugin(AnalysisPlugin):
    def get_name(self) -> str:
        return "your_plugin_name"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "param1": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "label": "Parameter 1",
                "description": "Description of parameter"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud: PointCloud = data_node.data
        # Process point cloud
        result = ...  # Your analysis result
        return result, "result_type", [data_node.uid]
```

### Menu Plugin Template

Place in `plugins/menus/your_menu_plugin.py`:

```python
from typing import Dict, Any, List
from plugins.interfaces import MenuPlugin

class YourMenuPlugin(MenuPlugin):
    def get_menu_location(self) -> str:
        return "Action/SubMenu"  # Or just "TopLevel"

    def get_menu_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Menu Item",
                "action": "action_name",
                "tooltip": "Description"
            }
        ]

    def handle_action(self, action_name: str, main_window):
        main_window.open_dialog_box(action_name)
```

## Global Variables (Singleton Pattern)

The application uses a singleton pattern for global access to core manager and widget instances (config/config.py):

### GlobalVariables Class
```python
class GlobalVariables:
    def __init__(self):
        self.global_file_manager = None
        self.global_pcd_viewer_widget = None
        self.global_tree_structure_widget = None
        self.global_data_nodes = None
        self.global_data_manager = None
        self.global_main_window = None

# Singleton instance
global_variables = GlobalVariables()
```

### Important Implementation Rules:
1. **Store only manager/widget class instances** - NOT individual variables or primitive values
2. **Pattern**: Each attribute should reference a class instance (FileManager, TreeStructureWidget, DataManager, etc.)
3. **Avoid**: Storing individual variables like `is_processing`, `current_operation`, etc.
4. **Access**: Any module can access via `from config.config import global_variables`
5. **Assignment**: Instances are assigned when created (usually in MainWindow.__init__)

### Current Global Instances:
- `global_variables.global_file_manager` - FileManager instance
- `global_variables.global_pcd_viewer_widget` - PCDViewerWidget instance
- `global_variables.global_tree_structure_widget` - TreeStructureWidget instance
- `global_variables.global_data_nodes` - DataNodes collection manager
- `global_variables.global_data_manager` - DataManager instance
- `global_variables.global_main_window` - MainWindow instance
- `global_variables.global_analysis_thread_manager` - AnalysisThreadManager instance

### Usage Example:
```python
from config.config import global_variables

# Access any global manager
data_manager = global_variables.global_data_manager
main_window = global_variables.global_main_window

# Call methods on global instances
main_window.disable_menus()
global_variables.global_tree_structure_widget.add_branch(...)
```

## GUI Architecture

**MainWindow** (gui/main_window.py):
- Dynamically builds menus from folder-based plugin structure
- Contains QSplitter with TreeStructureWidget (left) and PCDViewerWidget (right)
- Coordinates FileManager, DataManager, and DialogBoxesManager
- Provides methods to disable/enable menus and tree during processing:
  - `disable_menus()` / `enable_menus()` - Controls menu bar availability
  - `disable_tree()` / `enable_tree()` - Controls tree widget availability
- Contains ProcessOverlayWidget instances for visual feedback during operations

**TreeStructureWidget** (gui/widgets/tree_structure_widget.py):
- Displays hierarchical data structure
- Emits signals for visibility changes, selection, branch additions
- Disabled during processing to prevent user interaction

**PCDViewerWidget** (gui/widgets/pcd_viewer_widget.py):
- Custom 3D visualization using PyOpenGL (QOpenGLWidget)
- Renders point clouds using OpenGL VBOs for performance
- Supports interactive rotation, panning, zooming, and point picking
- Remains enabled during processing to allow camera manipulation
- Updates based on visibility status from tree widget
- Hotkeys:
  - Left Click: Rotate around X and Y axes
  - Ctrl + Left Click: Rotate around Z-axis
  - Right/Middle Click: Pan along X and Y axes
  - Ctrl + Right/Middle Click: Pan along Z-axis
  - Mouse Wheel: Zoom in and out
  - Double Left Click: Update center of rotation to clicked point
  - Shift + Left Click: Select a point
  - Shift + Right Click: Deselect a point
  - ESC: Deselect all selected points (with confirmation)
  - Ctrl + R: Reset camera view to default state

**ProcessOverlayWidget** (gui/widgets/process_overlay_widget.py):
- Semi-transparent overlay that displays processing status messages
- Positioned over tree widget during operations for visual feedback
- Does NOT block user interactions (protection handled by disabling widgets)
- Shows messages like "Running DBSCAN...", "Updating visibility...", etc.

**DialogBoxesManager** (gui/dialog_boxes/dialog_boxes_manager.py):
- Creates dynamic parameter input dialogs based on plugin `get_parameters()` schema
- Emits parameters back to DataManager for analysis execution

## Important Implementation Details

### Background Threading for Long Operations

The application uses background threading to keep the UI responsive during long-running operations:

**AnalysisThreadManager** (core/analysis_thread_manager.py):
- Manages background thread execution using Python's `threading.Thread` (NOT QThread)
- Uses singleton pattern for communication - NO callbacks or custom signals
- QTimer polling checks for completion every 100ms
- When complete, calls methods directly on global instances via singleton pattern
- Handles reconstruction in background thread if needed (memory efficient - no deep copy)

**Thread Safety:**
- Plugins only READ data, never modify it (read-only access is thread-safe)
- No deep copy needed - plugins return new objects instead of modifying input
- Reconstruction happens in background thread to avoid blocking UI

**UI Protection During Processing:**
- Menu bar is disabled via `main_window.disable_menus()`
- Tree widget is disabled via `main_window.disable_tree()`
- ProcessOverlayWidget shows status message (visual feedback only, doesn't block)
- Viewer remains enabled for camera manipulation (rotate, pan, zoom)

### Batch Processing

For large point clouds, use `BatchProcessor` (services/batch_processor.py) which provides spatial batching with automatic overlap handling. See `DBSCANPlugin` for reference implementation.

### GPU Acceleration

PointCloud methods (especially `get_subset()`) automatically use CuPy for GPU acceleration when available, falling back to NumPy if not installed.

### Coplanar Point Handling

When points are coplanar (e.g., after draping operations), the PointCloud class automatically uses 2D bounding box calculations instead of 3D OBB to avoid numerical issues.

### Adding New Data Types

To support a new derived data type:
1. Create the data class in `core/`
2. Create a reconstruction task class in `tasks/` inheriting from a base task pattern
3. Register the data type in `NodeReconstructionManager.tasks_registry`
4. Create an analysis plugin that returns the new data type

## Dependencies

Core dependencies (based on imports):
- PyQt5: GUI framework
- PyOpenGL: OpenGL bindings for 3D visualization in the viewer widget
- numpy: Numerical operations
- open3d: 3D geometry processing (OBB, point cloud I/O, DBSCAN)
- tensorflow: Used in eigenvalue utilities
- scipy: Scientific computing (KDTree, statistics)
- pandas: Data manipulation
- sklearn (optional): Alternative DBSCAN implementation
- cupy (optional): GPU acceleration for mask operations

## Testing

Unit tests are located in `unit_test/` directory. Tests verify:
- Plugin discovery and loading
- Analysis plugin execution
- Menu plugin registration
- Individual plugin functionality
- Do not use signal/slot method