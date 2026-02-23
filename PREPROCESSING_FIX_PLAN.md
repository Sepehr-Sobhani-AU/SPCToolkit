# PointNet Preprocessing Fix Implementation Plan

## Problem Summary
Model achieves 98% validation accuracy but fails during inference on the SAME data used for training. Root cause: **preprocessing mismatch**.

### Diagnostic Results
Running `diagnose_mismatch.py` on training data showed:
- **Data type mismatch**: Training uses float64, inference uses float32
- **Random sampling differences**: Each run samples different 1024 points from full cluster
- **Mean Squared Error**: 0.20 (SIGNIFICANT difference)

## Root Cause
Current pipeline samples AFTER computing features, resulting in:
1. Training data: Fixed 1024 points (one random sample, saved to .npy)
2. Inference: Different 1024 points each time → different features → model fails

## Solution: Correct Pipeline

### New Preprocessing Pipeline
```
1. Load cluster (N points, raw coordinates)
2. Center at origin (for float32 precision)
3. Compute normals on ALL N points (full neighborhood)
4. Compute eigenvalues on ALL N points (full neighborhood)
5. Normalize to unit sphere
6. Save ALL N points with features → (N, 9) not (1024, 9)
7. During training: randomly sample 1024 per epoch (augmentation!)
8. During inference: deterministically sample 1024 (or random seed)
```

## Implementation Status

### ✅ COMPLETED
**Step 1**: `convert_ply_to_training_data.py` updated
- Uses new pipeline: center → features → normalize
- Saves ALL points (variable size)
- Uses float32 throughout

### 🚧 IN PROGRESS
**Step 2**: Training pipeline needs update to handle variable-size inputs

Current blockers:
- `load_training_data()` tries to stack variable-size arrays → fails
- `PointNetClassifier.train()` expects fixed-size numpy arrays
- Need PyTorch dataset with random sampling per epoch

### ⏳ PENDING
**Step 3**: Update `inference.py` to match training pipeline
**Step 4**: Ensure float32 consistency everywhere
**Step 5**: Regenerate all training data
**Step 6**: Retrain and validate

## Next Steps (Manual Implementation Required)

### A. Update Plugin Data Loading

Modify `plugins/060_ML_Models/000_PointNet/010_train_model_plugin.py`:

```python
def load_training_data(self, data_dir):
    # ... existing code to line 805 ...

    # DON'T stack into fixed array - keep as list!
    # data = np.array(all_data, dtype=np.float32)  # ❌ REMOVE THIS

    # Determine max points and features from data
    point_counts = [sample.shape[0] for sample in all_data]
    num_features = all_data[0].shape[1]  # Should be 9

    print(f"\nVariable-size data loaded:")
    print(f"  Samples: {len(all_data)}")
    print(f"  Points range: {min(point_counts)} - {max(point_counts)}")
    print(f"  Features: {num_features}")

    labels = np.array(all_labels, dtype=np.int32)

    # Return lists, not arrays!
    return all_data, labels, class_mapping, metadata, num_features
```

### B. Create PyTorch Dataset with Sampling

Add to plugin:

```python
def create_variable_size_dataset(data_list, labels, num_points=1024,
                                  shuffle=True, augment=False):
    """
    Create PyTorch dataset from variable-size point clouds.

    Args:
        data_list: List of numpy arrays with shape (N_i, 9)
        labels: numpy array of labels
        num_points: Target points to sample (1024)
        shuffle: Shuffle data
        augment: Apply augmentation

    Returns:
        torch.utils.data.DataLoader
    """
    def generator():
        indices = list(range(len(data_list)))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            sample = data_list[idx].astype(np.float32)
            label = labels[idx]

            # Random sample to num_points
            n = sample.shape[0]
            if n >= num_points:
                # Randomly sample
                chosen = np.random.choice(n, num_points, replace=False)
                sampled = sample[chosen]
            else:
                # Pad by random duplication
                sampled = sample
                deficit = num_points - n
                pad_idx = np.random.choice(n, deficit, replace=True)
                sampled = np.vstack([sampled, sample[pad_idx]])

            yield torch.tensor(sampled, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    return generator
```

### C. Update Training Call

Replace lines ~336-340 in plugin's execute():

```python
# OLD:
# history = classifier.train(X_train, y_train, X_val, y_val, ...)

# NEW:
# Create datasets and dataloaders
from torch.utils.data import DataLoader

train_dataset = VariableSizeDataset(X_train, y_train, num_points=1024, augment=augment_enable)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

val_dataset = VariableSizeDataset(X_val, y_val, num_points=1024, augment=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Train with dataloaders
history = classifier.train(
    train_loader,
    val_loader=val_loader,
    epochs=epochs,
    class_weight=class_weights,
    verbose=1
)
```

### D. Update Inference

Modify `models/pointnet/inference.py` → `_process_cluster_for_inference()`:

```python
def _process_cluster_for_inference(cluster_xyz, num_points, use_normals=True, use_eigenvalues=True):
    # Step 1: Center at origin (FLOAT32 precision)
    centroid = np.mean(cluster_xyz, axis=0)
    centered = (cluster_xyz - centroid).astype(np.float32)

    # Step 2: Create PointCloud
    point_cloud = PointCloud(points=centered)

    # Step 3: Compute features on ALL points
    if use_normals:
        point_cloud.estimate_normals(k=30)
        normals = point_cloud.normals.astype(np.float32)
    else:
        normals = np.zeros((len(centered), 3), dtype=np.float32)

    if use_eigenvalues:
        eigenvalues = point_cloud.get_eigenvalues(k=30, smooth=True).astype(np.float32)
    else:
        eigenvalues = np.zeros((len(centered), 3), dtype=np.float32)

    # Step 4: Normalize to unit sphere
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist > 0:
        normalized_xyz = (centered / max_dist).astype(np.float32)
    else:
        normalized_xyz = centered

    # Step 5: Stack features
    features = np.hstack([normalized_xyz, normals, eigenvalues]).astype(np.float32)

    # Step 6: Sample to num_points (deterministic for consistency)
    if len(features) >= num_points:
        # Use deterministic sampling based on spatial distribution
        np.random.seed(42)  # Fixed seed for reproducibility
        indices = np.random.choice(len(features), num_points, replace=False)
        features = features[indices]
    else:
        # Pad
        deficit = num_points - len(features)
        pad_idx = np.random.choice(len(features), deficit, replace=True)
        features = np.vstack([features, features[pad_idx]])

    return features.astype(np.float32)
```

## Testing Plan

1. **Regenerate training data**:
   ```bash
   python convert_ply_to_training_data.py
   ```

2. **Verify .npy files are variable-size**:
   ```python
   import numpy as np
   data = np.load('training_data/Car/car_1.npy')
   print(data.shape)  # Should be (N, 9) where N varies
   ```

3. **Train new model** with updated code

4. **Test on same data used for training** - should work perfectly now!

5. **Run diagnostic** again to verify MSE < 1e-5

## Why This Fixes Everything

✅ **Training**: Samples different 1024 points each epoch → better augmentation
✅ **Inference**: Uses SAME preprocessing pipeline → consistent features
✅ **Float32**: Consistent dtype throughout
✅ **Deterministic**: Same cluster always gives same result (if using fixed seed)

## Estimated Implementation Time

- Code changes: 1-2 hours
- Regenerate training data: 30 minutes
- Retrain model: 2-4 hours
- Testing: 30 minutes

**Total: ~4-7 hours**
