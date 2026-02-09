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
        global_variables.global_application_controller.reconstruct(uid)
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
python unit_test/action_plugin_test.py
```

## Core Architecture

### Plugin System

The application uses a folder-based plugin architecture that automatically discovers and loads plugins from designated directories:

- **Analysis Plugins** (`plugins/*/` folders): Implement point cloud processing algorithms (DBSCAN, filtering, eigenvalue analysis, etc.)
- **Action Plugins** (`plugins/*/` folders): Implement menu actions without data processing (save project, load project, zoom to extent, etc.)

The folder structure defines the menu hierarchy automatically:
- `plugins/Points/Clustering/dbscan_plugin.py` → Menu: **Points > Clustering > DBSCAN**
- `plugins/File/save_project_plugin.py` → Menu: **File > Save Project**
- `plugins/View/zoom_to_extent_plugin.py` → Menu: **View > Zoom To Extent**

Both plugin types inherit from abstract base classes in `plugins/interfaces.py`:
- `AnalysisPlugin`: Requires `get_name()`, `get_parameters()`, and `execute()` methods - processes data and returns results
- `ActionPlugin`: Requires `get_name()`, `get_parameters()`, and `execute()` methods - performs actions without returning data

The `PluginManager` (plugins/plugin_manager.py) scans plugin directories on startup, imports modules, and registers classes that implement the plugin interfaces.

### Data Management System

The core data architecture revolves around a tree-based hierarchical system:

**DataNode** (core/entities/data_node.py): Represents a single unit of data with:
- `uid`: Unique identifier (UUID)
- `data`: The actual data object (PointCloud, Masks, Clusters, etc.)
- `data_type`: Type identifier for reconstruction
- `parent_uid`: Parent node reference
- `depends_on`: List of dependency UIDs
- `tags`: Classification tags

**DataNodes** (core/entities/data_nodes.py): Manages the collection of all DataNode instances.

**ApplicationController** (application/application_controller.py): Central coordinator that:
- Orchestrates interactions between UI widgets (tree view, 3D viewer) and data
- Manages branch selection and reconstruction via ReconstructionService
- Applies analyses through AnalysisExecutor
- Reconstructs derived data nodes back into PointCloud instances for visualization
- Created via `ApplicationController.create(plugin_manager, file_manager)` factory method

### Branch Reconstruction

The reconstruction system allows viewing derived data (masks, clusters, etc.) as point clouds:

1. When a non-PointCloud node needs visualization, `ApplicationController.reconstruct(uid)` delegates to `ReconstructionService`
2. ReconstructionService traverses up to the root PointCloud and builds a list of DataNode UIDs from root to target
3. Task classes from `tasks/` directory apply each node's transformation sequentially
4. Tasks (ApplyMasks, ApplyClusters, ApplyValues, etc.) convert derived data types back into filtered/modified PointCloud instances

### Key Data Types

**PointCloud** (core/entities/point_cloud.py): Primary data structure containing:
- `points`: (n, 3) numpy array of XYZ coordinates
- `colors`, `normals`: Optional (n, 3) arrays
- `attributes`: Dictionary for arbitrary per-point data
- Methods for DBSCAN, subsampling, eigenvalue calculation, SOR filtering, etc.
- GPU acceleration support via CuPy when available

**Clusters** (core/entities/clusters.py): Stores clustering results as:
- `labels`: Integer array assigning points to clusters (-1 for noise)
- `colors`: Optional color array for visualization

**Masks** (core/entities/masks.py): Boolean array for point selection/filtering.

## Creating New Plugins

### Analysis Plugin Template

Place in `plugins/YourCategory/your_analysis_plugin.py`:

```python
from typing import Dict, Any, List, Tuple
from plugins.interfaces import AnalysisPlugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud

class YourAnalysisPlugin(AnalysisPlugin):
    def get_name(self) -> str:
        return "your_analysis_name"

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

### Action Plugin Template

Place in `plugins/YourMenu/your_action_plugin.py`:

```python
from typing import Dict, Any
from plugins.interfaces import ActionPlugin
from config.config import global_variables

class YourActionPlugin(ActionPlugin):
    def get_name(self) -> str:
        return "your_action_name"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return parameter schema for dynamic dialog.
        Return empty dict {} if no parameters needed.
        """
        return {
            "option": {
                "type": "choice",
                "options": ["Option1", "Option2"],
                "default": "Option1",
                "label": "Choose Option",
                "description": "Select an option"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the action.

        Args:
            main_window: The main application window
            params: Parameters from the dialog (or empty dict)
        """
        # Access global instances via singleton pattern
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Common operations:
        # controller.selected_branches  - get selected branch UIDs
        # controller.reconstruct(uid)   - reconstruct a branch to PointCloud
        # main_window.render_visible_data(zoom_extent=False)  - re-render visible data
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
        self.global_application_controller = None
        self.global_main_window = None

# Singleton instance
global_variables = GlobalVariables()
```

### Important Implementation Rules:
1. **Store only manager/widget class instances** - NOT individual variables or primitive values
2. **Pattern**: Each attribute should reference a class instance (FileManager, TreeStructureWidget, ApplicationController, etc.)
3. **Avoid**: Storing individual variables like `is_processing`, `current_operation`, etc.
4. **Access**: Any module can access via `from config.config import global_variables`
5. **Assignment**: Instances are assigned when created (usually in MainWindow.__init__)

### Current Global Instances:
- `global_variables.global_file_manager` - FileManager instance
- `global_variables.global_pcd_viewer_widget` - PCDViewerWidget instance
- `global_variables.global_tree_structure_widget` - TreeStructureWidget instance
- `global_variables.global_data_nodes` - DataNodes collection manager
- `global_variables.global_application_controller` - ApplicationController instance
- `global_variables.global_main_window` - MainWindow instance

### Usage Example:
```python
from config.config import global_variables

# Access any global manager
controller = global_variables.global_application_controller
main_window = global_variables.global_main_window

# Common plugin operations
selected_uid = controller.selected_branches[0]
point_cloud = controller.reconstruct(selected_uid)
main_window.render_visible_data(zoom_extent=False)
```

## GUI Architecture

**MainWindow** (gui/main_window.py):
- Dynamically builds menus from folder-based plugin structure
- Contains QSplitter with TreeStructureWidget (left) and PCDViewerWidget (right)
- Creates ApplicationController via `ApplicationController.create()` factory
- Coordinates ApplicationController and DialogBoxesManager
- Provides methods to disable/enable menus and tree during processing:
  - `disable_menus()` / `enable_menus()` - Controls menu bar availability
  - `disable_tree()` / `enable_tree()` - Controls tree widget availability
  - `render_visible_data(zoom_extent=False)` - Re-render visible branches
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
- Key methods:
  - `set_points(points, colors)` - Set point cloud data for visualization
  - `zoom_to_extent()` - Frame all visible points optimally with 20% padding
  - `reset_view()` - Reset camera to default position and orientation
  - `update()` - Trigger view refresh
- Hotkeys:
  - Left Click: Rotate around X and Y axes
  - Ctrl + Left Click: Rotate around Z-axis
  - Right/Middle Click: Pan along X and Y axes
  - Ctrl + Right/Middle Click: Pan along Z-axis
  - Mouse Wheel: Zoom in and out
  - Double Left Click: Update center of rotation to clicked point
  - Shift + Left Click: Select a point
  - Shift + Right Click: Deselect a point
  - P: Enter polygon selection mode (click vertices, right-click/double-click to close and select)
  - ESC: Cancel polygon mode (if active), or deselect all selected points (with confirmation)
  - Ctrl + R: Reset camera view to default state
  - F: Zoom to extent (fit all visible points in viewport)

**ProcessOverlayWidget** (gui/widgets/process_overlay_widget.py):
- Semi-transparent overlay that displays processing status messages
- Positioned over tree widget during operations for visual feedback
- Does NOT block user interactions (protection handled by disabling widgets)
- Shows messages like "Running DBSCAN...", "Updating visibility...", etc.

**DialogBoxesManager** (gui/dialog_boxes/dialog_boxes_manager.py):
- Creates dynamic parameter input dialogs based on plugin `get_parameters()` schema
- Passes parameters to MainWindow for analysis execution via ApplicationController

## Important Implementation Details

### Background Threading for Long Operations

The application uses background threading to keep the UI responsive during long-running operations:

**AnalysisExecutor** (application/analysis_executor.py):
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

**CRITICAL**: Always maximize GPU usage. Never accept CPU fallbacks when GPU is available.

**User Preference**: The user has an NVIDIA RTX 3080 Laptop GPU (8GB VRAM) and wants ALL computationally intensive operations to use GPU acceleration. When writing or modifying code:
- Always prefer GPU implementations over CPU
- Use CuPy instead of NumPy for array operations when possible
- Use TensorFlow with GPU for ML operations
- Use Open3D GPU methods when available
- Never silently fall back to CPU - if GPU fails, report the issue

**GPU Libraries in Use:**
- CuPy: GPU-accelerated NumPy operations
- TensorFlow: ML operations on GPU
- Open3D: Some 3D operations support GPU

PointCloud methods (especially `get_subset()`) automatically use CuPy for GPU acceleration when available.

### Coplanar Point Handling

When points are coplanar (e.g., after draping operations), the PointCloud class automatically uses 2D bounding box calculations instead of 3D OBB to avoid numerical issues.

### Adding New Data Types

To support a new derived data type:
1. Create the data class in `core/entities/`
2. Create a reconstruction task class in `tasks/` inheriting from a base task pattern
3. Register the data type in `ReconstructionService.tasks_registry`
4. Create an analysis plugin that returns the new data type

## Common Plugin Examples

### Example: Zoom To Extent Plugin (Action Plugin)
Located in `plugins/View/zoom_to_extent_plugin.py`:

```python
from typing import Dict, Any
from plugins.interfaces import ActionPlugin
from config.config import global_variables

class ZoomToExtentPlugin(ActionPlugin):
    def get_name(self) -> str:
        return "zoom_to_extent"

    def get_parameters(self) -> Dict[str, Any]:
        return {}  # No parameters needed

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        viewer_widget = global_variables.global_pcd_viewer_widget
        viewer_widget.zoom_to_extent()
```

### Example: Save Project Plugins (Action Plugins)

**Save Project** (`plugins/File/save_project_plugin.py`):
- Standard "Save" behavior (Ctrl+S)
- First save: prompts for filename
- Subsequent saves: overwrites current file silently (no confirmation)
- Uses `file_manager.save_project(new_file=False)`
- Only shows error messages (no success popup)

**Save Project As** (`plugins/File/save_project_as_plugin.py`):
- Standard "Save As" behavior (Ctrl+Shift+S)
- Always prompts for a new filename
- Qt file dialog automatically shows overwrite warning if file exists
- Uses `file_manager.save_project(new_file=True)`
- Only shows error messages (no success popup)

**FileManager Path Tracking:**
- `FileManager.current_project_path` tracks the last saved/loaded project file path
- When loading a project, the path is automatically stored for future saves
- When saving with `new_file=False`, uses the stored path (or prompts if none exists)
- When saving with `new_file=True`, always prompts and updates the stored path

### Example: Label Clusters Plugin (Action Plugin with Parameters)
Located in `plugins/Training/label_clusters_plugin.py`:

```python
from typing import Dict, Any
from plugins.interfaces import ActionPlugin
from config.config import global_variables

class LabelClustersPlugin(ActionPlugin):
    def get_name(self) -> str:
        return "label_clusters"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "class_name": {
                "type": "choice",
                "options": ["Tree", "Pole", "Building", "Ground"],
                "default": "Tree",
                "label": "Class Label"
            },
            "save_directory": {
                "type": "string",
                "default": "training_data",
                "label": "Save Directory"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget

        # Get selected branch and reconstruct
        selected_uid = controller.selected_branches[0]
        point_cloud = controller.reconstruct(selected_uid)

        # Get selected clusters from picked points
        selected_indices = viewer_widget.picked_points_indices
        # ... process and save clusters
```

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