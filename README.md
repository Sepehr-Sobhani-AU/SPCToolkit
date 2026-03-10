# SPCToolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Smart Point Cloud Toolkit** — An interactive desktop application for 3D point cloud processing, analysis, and machine learning.

<!-- TODO: Add screenshot of the application here -->
<!-- ![SPCToolkit Screenshot](docs/screenshot.png) -->

## Features

- **3D Visualization** — OpenGL-based viewer with real-time rendering, point picking, polygon selection, and dynamic level-of-detail
- **Plugin Architecture** — 40+ plugins organized into automatic menu hierarchies; create new plugins by dropping a Python file into a folder
- **GPU Acceleration** — Automatic hardware detection with CUDA/CuPy/RAPIDS backends; falls back gracefully when GPU is unavailable
- **Machine Learning** — PointNet and PointNet++ models for point cloud classification and semantic segmentation with built-in training pipelines
- **Multi-Format I/O** — Import/export PLY, LAS/LAZ, E57, NumPy, and SemanticKITTI formats
- **Clustering & Analysis** — DBSCAN/HDBSCAN clustering, eigenvalue computation, geometric classification, normal estimation, and more
- **Interactive Editing** — Cut, merge, remove, and classify clusters in real-time with full undo support
- **Hierarchical Data Tree** — Track data derivation relationships with parent/child branches, visibility toggling, and per-branch memory tracking
- **Hardware Monitoring** — Live status bar showing RAM, VRAM, GPU utilization, and temperature

## Installation

### Prerequisites

- Python 3.9+
- A GPU with CUDA support is recommended but not required

### Setup

```bash
# Clone the repository
git clone https://github.com/Sepehr-Sobhani-AU/SPCToolkit.git
cd SPCToolkit

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Optional: GPU Acceleration

For NVIDIA GPU users, install additional packages for full GPU acceleration:

```bash
# CuPy for GPU array operations
pip install cupy-cuda12x

# pynvml for GPU monitoring
pip install pynvml

# RAPIDS cuML for GPU-accelerated DBSCAN/KNN (Linux only)
# See https://docs.rapids.ai/install for installation instructions
pip install cuml
```

## Quick Start

```bash
python main.py
```

1. **Import a point cloud**: File > Import Point Cloud > choose format (PLY, LAS, E57, etc.)
2. **Navigate**: Left-click to rotate, right-click to pan, scroll to zoom, press **F** to fit view
3. **Analyze**: Points > Analysis > Compute Eigenvalues for geometric features
4. **Cluster**: Points > Clustering > DBSCAN to segment the point cloud
5. **Classify**: Use geometric classification or train a PointNet model under ML Models
6. **Save**: File > Save Project to preserve your work as a `.pcdtk` project file

For a detailed walkthrough, see the [Getting Started Guide](GETTING_STARTED.md).

## Viewer Controls

| Input | Action |
|-------|--------|
| Left Click + Drag | Rotate |
| Ctrl + Left Click | Rotate around Z axis |
| Right/Middle Click + Drag | Pan |
| Double Left Click | Set rotation center on clicked point |
| Scroll Wheel | Zoom |
| **F** | Zoom to extent |
| **Ctrl + R** | Reset camera |
| **Shift + Left Click** | Select point |
| **P** | Polygon selection mode |
| **Z** | Zoom window mode |
| **C** | Cut cluster |
| **M** | Merge clusters |
| **Delete** | Remove clusters |
| **+/-** | Increase/decrease point size |

## Plugin System

SPCToolkit uses a folder-based plugin architecture. Plugins are automatically discovered and organized into menus based on their directory structure:

```
plugins/
  010_File/
    000_import_ply_plugin.py      # File > Import Point Cloud > PLY
    100_save_project_plugin.py    # File > Save Project
  020_Points/
    010_Subsampling/
      000_subsampling_plugin.py   # Points > Subsampling > Subsampling
    030_Analysis/
      000_eigenvalues_plugin.py   # Points > Analysis > Compute Eigenvalues
```

Creating a new plugin is as simple as implementing the `AnalysisPlugin` or `ActionPlugin` interface and placing the file in the appropriate directory.

For full details, see [Plugin Architecture](PLUGIN_ARCHITECTURE.md).

## Architecture

SPCToolkit follows a layered architecture with clear separation of concerns:

- **GUI Layer** — PyQt5 widgets, OpenGL viewer, dialog management
- **Application Layer** — Controllers, executors, rendering coordination
- **Core Layer** — Data entities, services, transformers
- **Plugin Layer** — Analysis and action plugins with backend abstraction

For the complete architecture documentation, see [Architecture](ARCHITECTURE.md).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
