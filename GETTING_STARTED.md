# Getting Started with SPCToolkit

This guide walks you through the basic workflows in SPCToolkit.

## 1. Launching the Application

```bash
python main.py
```

On startup, SPCToolkit detects your hardware (GPU, CUDA availability) and displays the result on the splash screen. The status bar at the bottom of the main window shows your compute mode:

- **FULL GPU** — All algorithms use GPU acceleration (requires NVIDIA GPU + RAPIDS)
- **PARTIAL GPU** — PyTorch/CuPy operations use GPU; clustering uses CPU
- **CPU ONLY** — All operations run on CPU

## 2. Loading a Point Cloud

Go to **File > Import Point Cloud** and select the format:

| Format | Extensions | Typical Use |
|--------|-----------|-------------|
| PLY | `.ply` | General-purpose polygon/point cloud format |
| LAS/LAZ | `.las`, `.laz` | Aerial LiDAR, survey data |
| E57 | `.e57` | Terrestrial laser scanning (ASTM standard) |
| NumPy | `.npy`, `.npz` | Batch processing, programmatic workflows |
| SemanticKITTI | `.bin` + `.label` | Autonomous driving datasets |

After loading, the point cloud appears in the **3D viewer** (right panel) and as a branch in the **data tree** (left panel).

## 3. Navigating the 3D Viewer

### Mouse Controls

| Input | Action |
|-------|--------|
| Left Click + Drag | Rotate around X/Y axes |
| Ctrl + Left Click + Drag | Rotate around Z axis |
| Right Click + Drag | Pan |
| Ctrl + Right Click + Drag | Pan along Z axis |
| Scroll Wheel | Zoom in/out |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **F** | Zoom to extent — fit all visible data in the viewport |
| **Ctrl + R** | Reset camera to default position |
| **+** / **-** | Increase / decrease point size |
| **Z** | Toggle zoom window mode (drag a rectangle to zoom in) |
| **Esc** | Exit current mode or deselect all points |

## 4. Selecting Points

SPCToolkit offers two selection modes:

### Point Selection
- **Shift + Left Click** — Select a single point
- **Shift + Right Click** — Deselect a single point

### Polygon Selection
1. Press **P** to enter polygon selection mode
2. Click to place polygon vertices
3. Right-click or double-click to close the polygon
4. All points inside the polygon are selected
5. Press **Shift + P** for polygon deselection mode

### Extracting Selections
After selecting points, go to **Selection > Separate Selected Points** to extract them into a new branch.

## 5. Running Analysis

### Eigenvalue Computation
Eigenvalues describe the local geometric structure around each point (planar, linear, scattered).

1. Select a branch in the data tree
2. Go to **Points > Analysis > Compute Eigenvalues**
3. Set the **K** parameter (number of neighbors, typically 20-50)
4. Results appear as a child branch in the data tree

### Geometric Classification
Automatically classifies points based on their geometric structure:

1. First compute eigenvalues (see above)
2. Go to **Points > Analysis > Geometric Classification**
3. Configure thresholds for planar, linear, and scatter features
4. Results assign a class label to each point

### Normal Estimation
1. Select a branch
2. Go to **Points > Analysis > Estimate Normals**
3. Set K neighbors and orientation method

## 6. Clustering

### DBSCAN Clustering
Segments the point cloud into spatial clusters:

1. Select a branch in the data tree
2. Go to **Points > Clustering > DBSCAN**
3. Set parameters:
   - **Epsilon** — Maximum distance between neighboring points
   - **Min Points** — Minimum points to form a cluster
4. Clusters appear as a result branch with color-coded groups

### Working with Clusters
Once you have clusters, you can:
- **C** — Cut a cluster (split it)
- **M** — Merge selected clusters
- **Delete** — Remove selected clusters
- **Clusters > Lock/Unlock Clusters** — Protect clusters from edits
- **Clusters > Color Clusters** — Assign custom colors
- **Clusters > Classify Cluster** — Assign semantic class labels

## 7. Machine Learning

### PointNet Classification
Train a model to classify clusters by type:

1. **Prepare training data**: Classify several clusters manually using **Clusters > Classify Cluster**
2. **Generate training data**: ML Models > PointNet > Classification > Generate Training Data
3. **Train the model**: ML Models > PointNet > Classification > Train PointNet Model
4. **Apply to new data**: ML Models > PointNet > Classification > Classify Clusters

### PointNet Segmentation
Per-point semantic segmentation:

1. **Annotate points**: ML Models > PointNet > Segmentation > Annotate Points (brush-based labeling)
2. **Generate training data**: ML Models > PointNet > Segmentation > Generate Seg Training Data
3. **Train**: ML Models > PointNet > Segmentation > Train Seg Model
4. **Segment**: ML Models > PointNet > Segmentation > Segment Point Cloud

### PointNet++ Segmentation
For improved segmentation with hierarchical feature learning:

1. Prepare training data (same as PointNet segmentation)
2. **Train**: ML Models > PointNet++ > Segmentation > Train PointNet++ Model
3. **Segment**: ML Models > PointNet++ > Segmentation > Segment Point Cloud

## 8. Saving and Exporting

### Save Project
**File > Save Project** saves the entire workspace (all branches, tree structure, visibility state) as a `.pcdtk` file. Projects are auto-versioned with incremental backups.

### Export Point Cloud
**File > Export Point Cloud** exports data in standard formats:
- **PLY** — Widely supported, includes colors and normals
- **LAS** — Standard for LiDAR data with classification fields
- **E57** — ASTM standard for 3D imaging data

## 9. The Data Tree

The left panel shows a hierarchical tree of all your data:

- **Top-level branches** are imported point clouds
- **Child branches** are analysis results, clusters, or extracted selections
- **Checkboxes** control visibility in the 3D viewer
- **Tooltips** show memory usage per branch
- Each branch maintains a reference to its parent, preserving the data derivation chain

## Next Steps

- Explore the [Plugin Architecture](PLUGIN_ARCHITECTURE.md) to create custom plugins
- Read the [Architecture](ARCHITECTURE.md) documentation for a deep understanding of the system
- Check [CONTRIBUTING.md](CONTRIBUTING.md) if you'd like to contribute
