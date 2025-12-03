# KNN Analysis Plugin

## Overview

The **KNN Analysis Plugin** computes various statistical measures based on the K-Nearest Neighbors for each point in a point cloud. This plugin is useful for:

- **Outlier detection**: Points with unusually high/low neighbor distances
- **Point density analysis**: Understanding local point cloud density
- **Surface roughness estimation**: Analyzing geometric variations
- **Feature extraction**: Creating features for machine learning pipelines
- **Local geometry characterization**: Understanding neighborhood structure

## Location

- **File**: `plugins/020_Points/030_Analysis/010_knn_analysis_plugin.py`
- **Menu Path**: `Points > Analysis > KNN Analysis`

## Parameters

### k_neighbors (Integer)
- **Description**: Number of nearest neighbors to analyze for each point
- **Default**: 10
- **Range**: 1-100
- **Recommendation**:
  - Use 5-15 for local density analysis
  - Use 20-50 for smoother feature extraction
  - Higher values capture broader neighborhood characteristics

### statistic (Choice)
The statistical measure to compute from neighbor distances:

1. **Average Distance**
   - Computes the mean distance to k-nearest neighbors
   - Best for: General density analysis and outlier detection
   - High values indicate sparse regions, low values indicate dense regions

2. **Maximum Distance**
   - Returns the distance to the farthest of the k neighbors
   - Best for: Understanding neighborhood extent
   - Useful for adaptive radius selection

3. **Minimum Distance**
   - Returns the distance to the closest neighbor
   - Best for: Detecting very close point pairs
   - Useful for identifying duplicate or near-duplicate points

4. **Std Deviation**
   - Computes the standard deviation of distances to k neighbors
   - Best for: Detecting irregular point distributions
   - High values indicate non-uniform local density

5. **Distance to Kth Neighbor**
   - Returns the distance specifically to the k-th nearest neighbor
   - Best for: Consistent neighborhood radius estimation
   - Often used in density-based algorithms

6. **Sum of Distances**
   - Computes the total sum of distances to k neighbors
   - Best for: Weighted density measures
   - Scales with k, useful for comparative analysis

## Output

The plugin returns a **Values** object containing one floating-point value per point in the point cloud. These values can be:
- Visualized using the color mapping system
- Used as input for further analysis
- Exported for external processing
- Used to create masks for filtering

## Usage Example

1. Load a point cloud into SPCToolkit
2. Select the point cloud branch in the tree view
3. Navigate to: **Points > Analysis > KNN Analysis**
4. Set parameters:
   - `k_neighbors`: 10
   - `statistic`: "Average Distance"
5. Execute the plugin
6. The result will appear as a child node in the tree structure
7. Visualize the results by selecting the color mode

## Use Cases

### Example 1: Outlier Detection
```
Parameters:
- k_neighbors: 15
- statistic: "Average Distance"

High average distance values indicate potential outliers or edge points.
```

### Example 2: Density-Based Filtering
```
Parameters:
- k_neighbors: 10
- statistic: "Distance to Kth Neighbor"

Use the result to create masks separating dense from sparse regions.
```

### Example 3: Surface Roughness
```
Parameters:
- k_neighbors: 20
- statistic: "Std Deviation"

High standard deviation indicates rough or irregular surfaces.
```

### Example 4: Duplicate Detection
```
Parameters:
- k_neighbors: 2
- statistic: "Minimum Distance"

Very small values (near 0) indicate duplicate or near-duplicate points.
```

## Implementation Details

- Uses the `PointCloud.KNN()` method for efficient neighbor search
- Automatically excludes the point itself from neighbor calculations
- Returns float32 values for memory efficiency
- Compatible with the reconstruction system for branch operations

## Performance

- **Speed**: Fast for most point clouds (< 1M points)
- **Memory**: Efficient, stores only one value per point
- **Optimization**: Uses KDTree internally for O(n log n) performance

## Related Plugins

- **Compute Eigenvalues**: For geometric feature extraction
- **Average Distance**: Simplified version computing only average distances
- **DBSCAN**: Clustering algorithm that uses similar neighbor concepts
- **SOR Filter**: Statistical outlier removal using neighbor distances