"""
PointNet Inference Utilities (PyTorch)

Functions for loading trained PointNet models and performing cluster classification.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple, List, Optional, Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from core.pointnet_model import PointNet
from core.entities.point_cloud import PointCloud


def load_model_with_metadata(model_dir: str, device: str = None) -> Tuple[PointNet, Dict[int, str], Dict]:
    """
    Load a trained PointNet model along with its metadata.

    Args:
        model_dir: Directory containing model files:
            - pointnet_best.pt: The trained model checkpoint
            - class_mapping.json: Class ID to class name mapping
            - training_metadata.json: Training configuration and metadata
        device: Device to load model on ('cuda', 'cpu', or None for auto-detect)

    Returns:
        Tuple of (model, class_mapping, metadata) where:
        - model: Loaded PyTorch model (in eval mode)
        - class_mapping: Dict mapping class IDs (int) to class names (str)
        - metadata: Dict with training metadata (num_points, num_features, etc.)

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If metadata is invalid
    """
    # Check model directory exists
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Build file paths - check for both .pt and .keras (for backward compatibility check)
    model_path = model_dir / 'pointnet_best.pt'
    if not model_path.exists():
        # Try alternative name
        model_path = model_dir / 'pointnet_best.pth'

    class_mapping_path = model_dir / 'class_mapping.json'
    metadata_path = model_dir / 'training_metadata.json'

    # Check required files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not class_mapping_path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load metadata first to get model architecture
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Validate metadata has required fields
    required_fields = ['num_points', 'num_features', 'num_classes']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Metadata missing required field: {field}")

    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping_raw = json.load(f)

    # Convert string keys to integers
    class_mapping = {int(k): v for k, v in class_mapping_raw.items()}

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model with correct architecture
    model = PointNet(
        num_points=metadata['num_points'],
        num_features=metadata['num_features'],
        num_classes=metadata['num_classes'],
        use_tnet=metadata.get('use_tnet', True)
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict (for compatibility)
        model.load_state_dict(checkpoint)

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Store device in metadata for later use
    metadata['_device'] = device

    return model, class_mapping, metadata


def classify_clusters_batch(
    point_cloud: PointCloud,
    model: PointNet,
    metadata: Dict,
    clusters_to_classify: List[int],
    confidence_threshold: float = 0.5,
    skip_small_clusters: bool = True,
    max_points_per_cluster: int = 20000,
    progress_callback: Optional[Callable[[int, int, int, str, float], None]] = None,
    batch_size: int = 16
) -> Tuple[Dict[int, int], Dict]:
    """
    Classify multiple clusters using a trained PointNet model.

    Args:
        point_cloud: PointCloud with cluster_labels attribute
        model: Trained PyTorch model (in eval mode)
        metadata: Model metadata dict with num_points, num_features, etc.
        clusters_to_classify: List of cluster IDs to classify
        confidence_threshold: Minimum confidence for classification (0-1)
        skip_small_clusters: Skip clusters smaller than num_points
        max_points_per_cluster: Maximum points per cluster (for feature computation)
        progress_callback: Optional callback(current, total, cluster_id, class_name, confidence)
        batch_size: Number of clusters to process in parallel (GPU batching)

    Returns:
        Tuple of (class_ids, stats) where:
        - class_ids: Dict mapping cluster IDs to predicted class IDs
        - stats: Dict with classification statistics

    Raises:
        ValueError: If point_cloud doesn't have cluster_labels
    """
    # Validate point cloud has cluster labels
    cluster_labels = point_cloud.get_attribute("cluster_labels")
    if cluster_labels is None:
        raise ValueError("PointCloud must have cluster_labels attribute")

    # Extract metadata
    num_points = metadata['num_points']
    num_features = metadata['num_features']
    if not torch.cuda.is_available() and '_device' not in metadata:
        print("WARNING: CUDA not available — inference will fall back to CPU and be significantly slower.")
    device = metadata.get('_device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Determine which features are expected
    use_normals = num_features >= 6
    use_eigenvalues = num_features >= 9

    # Initialize results
    class_ids = {}

    # Statistics tracking
    stats = {
        'total_clusters': len(clusters_to_classify),
        'classified': 0,
        'skipped_small': 0,
        'skipped_low_confidence': 0,
        'unclassified': 0,
        'confidences': []
    }

    # Prepare clusters for batch processing
    cluster_data = []
    cluster_indices = []

    for idx, cluster_id in enumerate(clusters_to_classify):
        # Extract cluster points
        cluster_mask = cluster_labels == cluster_id
        cluster_xyz = point_cloud.points[cluster_mask]

        # Check cluster size
        if len(cluster_xyz) < num_points:
            if skip_small_clusters:
                stats['skipped_small'] += 1
                class_ids[cluster_id] = -1

                if progress_callback:
                    progress_callback(idx + 1, len(clusters_to_classify),
                                    cluster_id, "Skipped (too small)", 0.0)
                continue

        # Subsample if cluster is too large
        if len(cluster_xyz) > max_points_per_cluster:
            indices = np.random.choice(len(cluster_xyz), max_points_per_cluster, replace=False)
            cluster_xyz = cluster_xyz[indices]

        try:
            # Process cluster to model input format
            features = _process_cluster_for_inference(
                cluster_xyz,
                num_points=num_points,
                metadata=metadata,
                use_normals=use_normals,
                use_eigenvalues=use_eigenvalues
            )

            cluster_data.append(features)
            cluster_indices.append((idx, cluster_id))

        except Exception as e:
            print(f"Error preprocessing cluster {cluster_id}: {str(e)}")
            stats['skipped_low_confidence'] += 1
            class_ids[cluster_id] = -1

            if progress_callback:
                progress_callback(idx + 1, len(clusters_to_classify),
                                cluster_id, "Error", 0.0)

    # Process in batches for efficiency
    if len(cluster_data) > 0:
        with torch.no_grad():
            for batch_start in range(0, len(cluster_data), batch_size):
                batch_end = min(batch_start + batch_size, len(cluster_data))
                batch_features = np.stack(cluster_data[batch_start:batch_end])
                batch_indices = cluster_indices[batch_start:batch_end]

                # Convert to tensor and move to device
                batch_tensor = torch.FloatTensor(batch_features).to(device)

                # Forward pass
                logits = model(batch_tensor)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()

                # Process each result in the batch
                for i, (orig_idx, cluster_id) in enumerate(batch_indices):
                    probs = probabilities[i]
                    predicted_class_id = int(np.argmax(probs))
                    confidence = float(np.max(probs))

                    # Check confidence threshold
                    if confidence < confidence_threshold:
                        stats['skipped_low_confidence'] += 1
                        class_ids[cluster_id] = -1
                        stats['unclassified'] += 1

                        if progress_callback:
                            progress_callback(orig_idx + 1, len(clusters_to_classify),
                                            cluster_id, "Unclassified (low confidence)", confidence)
                    else:
                        class_ids[cluster_id] = predicted_class_id
                        stats['classified'] += 1
                        stats['confidences'].append(confidence)

                        # Get class name for callback
                        class_name = metadata.get('class_mapping', {}).get(
                            str(predicted_class_id), f"Class_{predicted_class_id}"
                        )

                        if progress_callback:
                            progress_callback(orig_idx + 1, len(clusters_to_classify),
                                            cluster_id, class_name, confidence)

    # Calculate average confidence
    if stats['confidences']:
        stats['avg_confidence'] = float(np.mean(stats['confidences']))
    else:
        stats['avg_confidence'] = 0.0

    return class_ids, stats


def classify_single_cluster(
    cluster_xyz: np.ndarray,
    model: PointNet,
    metadata: Dict,
    class_mapping: Dict[int, str]
) -> Tuple[str, float, Dict]:
    """
    Classify a single cluster.

    Args:
        cluster_xyz: Cluster points (n, 3)
        model: Trained PyTorch model
        metadata: Model metadata
        class_mapping: Dict mapping class IDs to class names

    Returns:
        Tuple of (class_name, confidence, all_probabilities)
    """
    num_points = metadata['num_points']
    num_features = metadata['num_features']
    device = metadata.get('_device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    use_normals = num_features >= 6
    use_eigenvalues = num_features >= 9

    # Process cluster
    features = _process_cluster_for_inference(
        cluster_xyz,
        num_points=num_points,
        metadata=metadata,
        use_normals=use_normals,
        use_eigenvalues=use_eigenvalues
    )

    # Add batch dimension and convert to tensor
    batch = torch.FloatTensor(features).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    predicted_class_id = int(np.argmax(probs))
    confidence = float(np.max(probs))
    class_name = class_mapping.get(predicted_class_id, f"Class_{predicted_class_id}")

    # Create probability dict
    all_probs = {class_mapping.get(i, f"Class_{i}"): float(probs[i])
                 for i in range(len(probs))}

    return class_name, confidence, all_probs


def _process_cluster_for_inference(
    cluster_xyz: np.ndarray,
    num_points: int,
    metadata: Dict,
    use_normals: bool = True,
    use_eigenvalues: bool = True
) -> np.ndarray:
    """
    Process a cluster's XYZ points into model input format.

    This function replicates the preprocessing done during training:
    1. Create PointCloud
    2. Normalize (if enabled in training)
    3. Compute normals (if needed, using training KNN)
    4. Compute eigenvalues (if needed, using training KNN and smooth settings)
    5. Stack features
    6. Subsample to num_points

    Args:
        cluster_xyz: Cluster points (n, 3)
        num_points: Target number of points for model input
        metadata: Training metadata containing preprocessing parameters
        use_normals: Whether to compute normals
        use_eigenvalues: Whether to compute eigenvalues

    Returns:
        Feature array (num_points, num_features)
    """
    # Create PointCloud from cluster
    point_cloud = PointCloud(points=cluster_xyz.copy())

    # Extract training preprocessing parameters from metadata
    source_metadata = metadata.get('source_metadata', {})
    processing = source_metadata.get('processing', {}) if source_metadata else {}

    # Get normalization setting (default to True for safety)
    normalize_enabled = processing.get('normalization', {}).get('enabled', True)

    # Get feature computation parameters
    normals_config = processing.get('features', {}).get('normals', {})
    eigenvalues_config = processing.get('features', {}).get('eigenvalues', {})

    normals_knn = normals_config.get('knn', 30)
    eigenvalues_knn = eigenvalues_config.get('knn', 30)
    eigenvalues_smooth = eigenvalues_config.get('smooth', True)

    # Normalize (only if enabled in training)
    if normalize_enabled:
        point_cloud.normalise(
            apply_scaling=True,
            apply_centering=True,
            rotation_axes=(False, False, False)
        )

    # Build feature list
    features = []

    # Add XYZ (normalized or not, depending on training settings)
    features.append(point_cloud.points)

    # Compute normals if requested (using training KNN value)
    if use_normals:
        if len(point_cloud.points) >= normals_knn:
            point_cloud.estimate_normals(k=normals_knn)
            features.append(point_cloud.normals)
        else:
            # Not enough points for normals, use zeros
            features.append(np.zeros((len(point_cloud.points), 3), dtype=np.float32))

    # Compute eigenvalues if requested (using training KNN and smooth values)
    if use_eigenvalues:
        if len(point_cloud.points) >= eigenvalues_knn:
            eigenvalues = point_cloud.get_eigenvalues(k=eigenvalues_knn, smooth=eigenvalues_smooth)
            features.append(eigenvalues)
        else:
            # Not enough points for eigenvalues, use zeros
            features.append(np.zeros((len(point_cloud.points), 3), dtype=np.float32))

    # Stack features horizontally
    combined = np.hstack(features).astype(np.float32)

    # Subsample to target num_points
    if len(combined) > num_points:
        indices = np.random.choice(len(combined), num_points, replace=False)
        combined = combined[indices]
    elif len(combined) < num_points:
        # Pad with duplicates if too small
        deficit = num_points - len(combined)
        if len(combined) > 0:
            pad_indices = np.random.choice(len(combined), deficit, replace=True)
            pad_data = combined[pad_indices]
            combined = np.vstack([combined, pad_data])
        else:
            # Edge case: empty cluster, return zeros
            num_features = 3 + (3 if use_normals else 0) + (3 if use_eigenvalues else 0)
            combined = np.zeros((num_points, num_features), dtype=np.float32)

    return combined


def get_model_info(model_dir: str) -> Dict:
    """
    Get information about a trained model without loading the weights.

    Args:
        model_dir: Directory containing model files

    Returns:
        Dict with model information
    """
    model_dir = Path(model_dir)

    metadata_path = model_dir / 'training_metadata.json'
    class_mapping_path = model_dir / 'class_mapping.json'

    info = {
        'model_dir': str(model_dir),
        'exists': model_dir.exists()
    }

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        info['num_points'] = metadata.get('num_points')
        info['num_features'] = metadata.get('num_features')
        info['num_classes'] = metadata.get('num_classes')
        info['use_tnet'] = metadata.get('use_tnet')
        info['best_val_accuracy'] = metadata.get('best_val_accuracy')
        info['epochs_completed'] = metadata.get('epochs_completed')
        info['framework'] = metadata.get('framework', 'TensorFlow')  # Default for old models

    if class_mapping_path.exists():
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        info['classes'] = list(class_mapping.values())

    # Check which model files exist
    info['has_best_model'] = (model_dir / 'pointnet_best.pt').exists()
    info['has_final_model'] = (model_dir / 'pointnet_final.pt').exists()

    return info
