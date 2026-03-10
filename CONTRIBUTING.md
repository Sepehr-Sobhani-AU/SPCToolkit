# Contributing to SPCToolkit

Thank you for your interest in contributing to SPCToolkit! This guide will help you get started.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SPCToolkit.git
   cd SPCToolkit
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
5. **Make your changes**, then commit and push:
   ```bash
   git add <files>
   git commit -m "Description of changes"
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** on GitHub

## Code Conventions

### Communication Pattern: Singleton Over Signal/Slot

This project avoids Qt's custom signal/slot mechanism. Use the singleton pattern via `global_variables` to access instances and call methods directly.

```python
# CORRECT — use singleton
from config.config import global_variables

controller = global_variables.global_application_controller
controller.reconstruct(uid)
global_variables.global_main_window.render_visible_data(zoom_extent=False)

# WRONG — do not create custom signals
class MyManager(QObject):
    data_updated = pyqtSignal(str)  # Never do this
```

Built-in Qt widget signals (e.g., `QTimer.timeout`) are acceptable.

### GPU Acceleration

- Prefer CuPy over NumPy for array operations when possible
- Use PyTorch with CUDA for ML operations
- If GPU operations fail, report the error — do not silently fall back to CPU

### Background Threading

- Use Python `threading.Thread`, not `QThread`
- Use `QTimer` polling (100ms) to check completion
- Use callbacks for progress (`on_progress`), completion (`on_complete`), and errors (`on_error`)
- Plugins only read data — they return new objects, never modify input

## Creating Plugins

The fastest way to contribute is by creating a new plugin. See [Plugin Architecture](PLUGIN_ARCHITECTURE.md) for the full guide.

**Quick version:** Create a `.py` file in the appropriate `plugins/` subdirectory implementing either `AnalysisPlugin` or `ActionPlugin` from `plugins/interfaces.py`. The plugin is automatically discovered and added to the menu.

## Running Tests

```bash
python unit_test/plugins_discovery_and_loading_test.py
python unit_test/analysis_plugins_execution_test.py
python unit_test/action_plugin_test.py
```

## Reporting Issues

- Use [GitHub Issues](https://github.com/Sepehr-Sobhani-AU/SPCToolkit/issues) to report bugs or request features
- Include steps to reproduce, expected behavior, and your environment (OS, GPU, Python version)
