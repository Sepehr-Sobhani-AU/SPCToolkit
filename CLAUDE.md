# CLAUDE.md

Guidance for Claude Code when working with this repository. For full architecture details (data flows, threading, reconstruction, component diagrams), see **ARCHITECTURE.md**.

## Project Overview

SPCToolkit is a PyQt5-based point cloud processing application with a plugin-based architecture. It provides interactive 3D visualization and analysis of point cloud data through a tree-based hierarchical data management system.

## Running & Testing

```bash
# Start the application
python main.py

# Run unit tests
python unit_test/plugins_discovery_and_loading_test.py
python unit_test/analysis_plugins_execution_test.py
python unit_test/action_plugin_test.py
```

## Rules & Conventions

### Communication Pattern: Prefer Singleton Over Signal/Slot

**CRITICAL**: This project avoids Qt's custom signal/slot mechanism.

**Priority order:**
1. **Singleton pattern** — use `global_variables` to access instances and call methods directly (PREFER THIS)
2. **Callbacks** — use when singleton doesn't fit (e.g. `on_progress`, `on_complete`, `on_error`)
3. **Direct method calls**
4. **QTimer polling** — for checking async status

**NEVER** create custom `pyqtSignal()` declarations, `.connect()` for custom signals, or `.emit()` for custom signals. Built-in Qt widget signals (e.g. `QTimer.timeout`) are acceptable when necessary.

**Example — WRONG:**
```python
class MyManager(QObject):
    data_updated = pyqtSignal(str)
    def process(self):
        self.data_updated.emit("done")
```

**Example — BEST (Singleton):**
```python
class MyManager:
    def process(self):
        global_variables.global_main_window.update_display()
        global_variables.global_application_controller.reconstruct(uid)
```

**Example — ACCEPTABLE (Callback):**
```python
class MyManager:
    def process(self, on_complete):
        result = calculate_something()
        on_complete(result)
```

### GPU Acceleration

**CRITICAL**: Always maximize GPU usage. Never accept CPU fallbacks silently.

- Prefer CuPy over NumPy for array operations when possible
- Use TensorFlow with GPU for ML operations
- Use Open3D GPU methods when available
- If GPU fails, report the error — do not silently fall back to CPU

### Background Threading

- Use Python `threading.Thread` (NOT QThread)
- Use QTimer polling (100ms) to check completion status
- Use callbacks for progress reporting (`on_progress`), completion (`on_complete`), and errors (`on_error`)
- Plugins only READ data, never modify it — they return new objects (thread-safe, no deep copy needed)
- Disable menus and tree during processing; viewer stays enabled for camera manipulation

## Creating New Plugins

Both plugin types inherit from abstract base classes in `plugins/interfaces.py`. Note: `AnalysisPlugin` is an alias for the base class `Plugin`.

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
        """Return parameter schema for dynamic dialog. Return {} if none needed."""
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
        controller = global_variables.global_application_controller
        viewer_widget = global_variables.global_pcd_viewer_widget
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Common operations:
        # controller.selected_branches  — get selected branch UIDs
        # controller.reconstruct(uid)   — reconstruct a branch to PointCloud
        # main_window.render_visible_data(zoom_extent=False)  — re-render visible data
```

## Global Variables (Singleton Pattern)

Access via `from config.config import global_variables`. Defined in `config/config.py`.

**Rules:**
- Store only manager/widget class instances — NOT primitives or flags
- Exception: `global_progress` (tuple) and `training_data_folder` (string) are lightweight shared state

### All Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `global_file_manager` | `FileManager` | File I/O operations |
| `global_data_nodes` | `DataNodes` | Collection of all DataNode instances |
| `global_application_controller` | `ApplicationController` | Central coordinator |
| `global_pcd_viewer_widget` | `PCDViewerWidget` | 3D point cloud viewer |
| `global_tree_structure_widget` | `TreeStructureWidget` | Hierarchical data tree |
| `global_main_window` | `MainWindow` | Main application window |
| `global_hardware_info` | `HardwareInfo` | Detected hardware capabilities |
| `global_backend_registry` | `BackendRegistry` | Algorithm backend selection |
| `global_progress` | `tuple` | `(None, "msg")` indeterminate, `(50, "msg")` 50% determinate |
| `training_data_folder` | `str` | Default training data directory (`"training_data"`) |

### Usage Example

```python
from config.config import global_variables

controller = global_variables.global_application_controller
main_window = global_variables.global_main_window

selected_uid = controller.selected_branches[0]
point_cloud = controller.reconstruct(selected_uid)
main_window.render_visible_data(zoom_extent=False)
```

## Dependencies

Core dependencies (based on imports):
- PyQt5: GUI framework
- PyOpenGL: 3D visualization
- numpy: Numerical operations
- open3d: 3D geometry processing (OBB, I/O, DBSCAN)
- tensorflow: Eigenvalue utilities (GPU)
- scipy: KDTree, statistics
- pandas: Data manipulation
- sklearn (optional): Alternative DBSCAN
- cupy (optional): GPU-accelerated array operations

## Testing

Unit tests are in `unit_test/`. Tests verify:
- Plugin discovery and loading
- Analysis plugin execution
- Menu plugin registration
- Individual plugin functionality
- Do not use signal/slot method
