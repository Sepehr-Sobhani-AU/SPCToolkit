"""
PointNet Segmentation Inference Utilities (PyTorch)

Functions for loading trained segmentation models and performing blockwise
inference on large point clouds.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from core.entities.point_cloud import PointCloud
from core.services.strided_spatial_hash import StridedSpatialHash
from models.pointnet.pointnet_seg_model import PointNetSegmentation


def load_seg_model_with_metadata(
    model_dir: str,
    device: str = None
) -> Tuple[PointNetSegmentation, Dict[int, str], Dict]:
    """
    Load a trained segmentation model along with its metadata.

    Args:
        model_dir: Directory containing model files:
            - seg_model_best.pt: Trained model checkpoint
            - class_mapping.json: Class ID to name mapping
            - training_metadata.json: Training configuration
        device: Device to load model on (None for auto-detect)

    Returns:
        Tuple of (model, class_mapping, metadata)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_path = model_dir / 'seg_model_best.pt'
    if not model_path.exists():
        model_path = model_dir / 'seg_model_best.pth'

    class_mapping_path = model_dir / 'class_mapping.json'
    metadata_path = model_dir / 'training_metadata.json'

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not class_mapping_path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    required_fields = ['num_points', 'num_features', 'num_classes']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Metadata missing required field: {field}")

    with open(class_mapping_path, 'r') as f:
        class_mapping_raw = json.load(f)
    class_mapping = {int(k): v for k, v in class_mapping_raw.items()}

    checkpoint = torch.load(model_path, map_location=device)

    # Auto-detect model type (backward compatible: default to PointNet)
    model_type = metadata.get('model_type', 'PointNet')
    metadata['model_type'] = model_type

    if model_type == 'PointNet++ SSG':
        from models.pointnet2.pointnet2_seg_model import PointNet2SSGSegmentation
        model = PointNet2SSGSegmentation(
            num_points=metadata['num_points'],
            num_features=metadata['num_features'],
            num_classes=metadata['num_classes'],
            block_size=metadata.get('block_size', 10.0),
            use_fps=True  # Always use FPS for inference (better coverage)
        )
    else:
        model = PointNetSegmentation(
            num_points=metadata['num_points'],
            num_features=metadata['num_features'],
            num_classes=metadata['num_classes'],
            use_tnet=metadata.get('use_tnet', True)
        )

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    metadata['_device'] = device

    # Load per-feature standardization stats (backward compat: None if absent)
    if 'feature_mean' in metadata:
        metadata['_feature_mean'] = np.array(metadata['feature_mean'], dtype=np.float32)
        metadata['_feature_std'] = np.array(metadata['feature_std'], dtype=np.float32)
    else:
        metadata['_feature_mean'] = None
        metadata['_feature_std'] = None

    return model, class_mapping, metadata


def segment_point_cloud_blockwise(
    point_cloud: PointCloud,
    model: PointNetSegmentation,
    metadata: Dict,
    block_size: float = 10.0,
    overlap: float = 1.0,
    batch_size: int = 8,
    confidence_threshold: float = 0.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    full_normals: Optional[np.ndarray] = None,
    full_eigenvalues: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Segment a large point cloud by dividing it into spatial blocks.

    Uses the same strided spatial hashing as training data generation to
    ensure identical block geometry. Block size and stride are read from
    the model's training metadata (source_metadata).

    Pipeline:
    1. Build strided spatial hash (matching training)
    2. Per block: compute features, subsample/pad to num_points, run model
    3. Average softmax probabilities across overlapping blocks
    4. Return (N,) label array

    Args:
        point_cloud: Input PointCloud to segment
        model: Trained segmentation model (in eval mode)
        metadata: Model metadata with num_points, num_features, etc.
        block_size: Fallback block size if not in training metadata
        overlap: Unused (kept for API compatibility). Overlap is determined
            by the training stride (stride < block_size → natural overlap).
        batch_size: Number of chunks to process in parallel on GPU
        confidence_threshold: Points with max probability below this are
            labeled num_classes (mapped to "Unclassified" by the plugin).
            Set to 0.0 to disable.
        progress_callback: Optional callback(current_block, total_blocks, status_msg)
        full_normals: Optional pre-computed (N,3) normals for the full cloud.
            If None and model needs normals, they are auto-computed.
        full_eigenvalues: Optional pre-computed (N,3) eigenvalues for the full cloud.
            If None and model needs eigenvalues, they are auto-computed.

    Returns:
        numpy array of shape (N,) with per-point class labels
    """
    num_points = metadata['num_points']
    num_features = metadata['num_features']
    if not torch.cuda.is_available() and '_device' not in metadata:
        print("WARNING: CUDA not available — inference will fall back to CPU and be significantly slower.")
    device = metadata.get('_device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Per-feature standardization stats (None for old models without them)
    feature_mean = metadata.get('_feature_mean', None)
    feature_std = metadata.get('_feature_std', None)

    # Classes ignored during training (e.g. "unlabeled", "outlier") — their
    # output neurons were never trained, so mask them out during inference.
    ignore_classes = metadata.get('ignore_classes', [])
    if ignore_classes:
        class_mapping = metadata.get('class_mapping', {})
        ignored_names = [class_mapping.get(str(c), class_mapping.get(c, f"Class_{c}"))
                         for c in ignore_classes]
        print(f"Masking {len(ignore_classes)} ignored classes: {', '.join(ignored_names)}")

    # Determine feature configuration
    has_mask_channel = metadata.get('has_mask_channel', False)
    raw_features = num_features - 1 if has_mask_channel else num_features
    use_normals = raw_features >= 6
    use_eigenvalues = raw_features >= 9

    # Get preprocessing parameters from metadata
    source_metadata = metadata.get('source_metadata', {})
    processing = source_metadata.get('processing', {}) if source_metadata else {}
    normalize_enabled = processing.get('normalization', {}).get('enabled', True)
    normals_config = processing.get('features', {}).get('normals', {})
    eigenvalues_config = processing.get('features', {}).get('eigenvalues', {})
    normals_knn = normals_config.get('knn', 30)
    eigenvalues_knn = eigenvalues_config.get('knn', 30)
    eigenvalues_smooth = eigenvalues_config.get('smooth', True)

    # Read training block parameters — use same geometry as training
    training_block_size = source_metadata.get('block_size', block_size)
    training_stride = source_metadata.get('stride', block_size)

    if training_block_size != block_size:
        print(f"Using training block_size={training_block_size}m "
              f"(overriding requested {block_size}m to match training)")
    print(f"Strided spatial hash: block_size={training_block_size}m, stride={training_stride}m")

    points = point_cloud.points
    N = len(points)
    num_classes = metadata['num_classes']

    # Normals/eigenvalues should be pre-computed on the full cloud (matching
    # training where they come from global branches). Auto-compute only if
    # not provided and the model needs them.
    if use_normals and full_normals is None:
        if progress_callback:
            progress_callback(0, 1, "Computing normals on full cloud...")
        pc_full = PointCloud(points=points.copy())
        pc_full.estimate_normals(k=normals_knn)
        full_normals = pc_full.normals

    if use_eigenvalues and full_eigenvalues is None:
        if progress_callback:
            progress_callback(0, 1, "Computing eigenvalues on full cloud...")
        pc_full_eig = PointCloud(points=points.copy())
        full_eigenvalues = pc_full_eig.get_eigenvalues(k=eigenvalues_knn, smooth=eigenvalues_smooth)

    # Accumulate softmax probabilities and counts for averaging.
    # float32 halves memory vs float64: (55M, 29) goes from ~13 GB to ~6.4 GB.
    prob_accum = np.zeros((N, num_classes), dtype=np.float32)
    count_accum = np.zeros(N, dtype=np.float32)

    # Build strided spatial hash (identical algorithm to training data generation)
    spatial_hash = StridedSpatialHash(points, training_block_size, training_stride)
    valid_positions = spatial_hash.enumerate_blocks(min_points=3)
    total_blocks = len(valid_positions)

    print(f"Grid: {spatial_hash.nx} x {spatial_hash.ny}, "
          f"cells_per_block: {spatial_hash.cells_per_block}, "
          f"valid blocks: {total_blocks:,}")

    if progress_callback:
        progress_callback(0, total_blocks, f"Processing {total_blocks} blocks...")

    # Stream: process each block → infer → accumulate → free.
    for block_idx, (ix, iy) in enumerate(valid_positions):
        block_indices = spatial_hash.get_block_indices(ix, iy)
        if len(block_indices) == 0:
            continue

        block_xyz = points[block_indices]

        # Slice pre-computed features for this block
        block_normals = full_normals[block_indices] if full_normals is not None else None
        block_eigenvalues = full_eigenvalues[block_indices] if full_eigenvalues is not None else None

        features = _compute_full_block_features(
            block_xyz,
            normalize=normalize_enabled,
            block_normals=block_normals,
            block_eigenvalues=block_eigenvalues,
            feature_mean=feature_mean,
            feature_std=feature_std,
            has_mask_channel=has_mask_channel
        )

        if features is None:
            continue

        chunks, perm_indices, n_original = _create_chunks(features, num_points)
        del features

        # Accumulate chunk predictions into block-level probabilities
        block_probs = np.zeros((n_original, num_classes), dtype=np.float32)

        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            batch_probs = _process_chunks(batch_chunks, model, device, ignore_classes)

            for j in range(batch_end - batch_start):
                chunk_idx = batch_start + j
                perm_start = chunk_idx * num_points
                perm_end = min(perm_start + num_points, n_original)
                n_real = perm_end - perm_start
                original_positions = perm_indices[perm_start:perm_end]
                block_probs[original_positions] += batch_probs[j][:n_real]

            del batch_probs

        del chunks

        # Equal weighting — matches training (no primary/overlap distinction).
        # With stride < block_size, points naturally appear in multiple blocks
        # and their probabilities are averaged.
        prob_accum[block_indices] += block_probs
        count_accum[block_indices] += 1.0

        del block_probs

        if progress_callback:
            progress_callback(block_idx + 1, total_blocks,
                              f"Block {block_idx + 1}/{total_blocks}")

    del spatial_hash

    # Assign labels in chunks to avoid a temporary (N, C) allocation.
    # Points below confidence_threshold get label num_classes ("Unclassified").
    unclassified_label = np.int32(num_classes)
    labels = np.full(N, unclassified_label, dtype=np.int32)
    valid_indices = np.where(count_accum > 0)[0]
    LABEL_CHUNK = 1_000_000
    n_low_conf = 0
    for start in range(0, len(valid_indices), LABEL_CHUNK):
        end = min(start + LABEL_CHUNK, len(valid_indices))
        idx = valid_indices[start:end]
        avg = prob_accum[idx] / count_accum[idx, np.newaxis]
        labels[idx] = np.argmax(avg, axis=1).astype(np.int32)
        if confidence_threshold > 0:
            max_prob = np.max(avg, axis=1)
            low_conf = max_prob < confidence_threshold
            labels[idx[low_conf]] = unclassified_label
            n_low_conf += int(np.sum(low_conf))

    if confidence_threshold > 0 and n_low_conf > 0:
        print(f"Confidence threshold {confidence_threshold}: "
              f"{n_low_conf:,} points ({n_low_conf/N*100:.1f}%) marked Unclassified")

    del prob_accum, count_accum

    return labels


def _compute_full_block_features(
    block_xyz: np.ndarray,
    normalize: bool,
    block_normals: Optional[np.ndarray] = None,
    block_eigenvalues: Optional[np.ndarray] = None,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None,
    has_mask_channel: bool = False
) -> Optional[np.ndarray]:
    """
    Assemble features for all points in a block from pre-computed data.

    Normalization matches training: XY centered to block centroid,
    Z ground-relative (Z - min_Z), no scaling.

    Args:
        block_xyz: (n, 3) raw coordinates for this block
        normalize: Whether to apply spatial normalization
        block_normals: (n, 3) pre-computed normals (from full cloud), or None
        block_eigenvalues: (n, 3) pre-computed eigenvalues (from full cloud), or None
        feature_mean: Per-feature means from training for z-score standardization
        feature_std: Per-feature stds from training for z-score standardization
        has_mask_channel: If True, append a column of 1.0 (all real points at inference)

    Returns:
        (n, F) feature array or None if block too small
    """
    n = len(block_xyz)
    if n < 3:
        return None

    # Pre-allocate single contiguous array
    n_features = 3
    if block_normals is not None:
        n_features += 3
    if block_eigenvalues is not None:
        n_features += 3

    combined = np.empty((n, n_features), dtype=np.float32)

    # Fill XYZ with in-place normalization (matching training: XY centered, Z ground-relative)
    combined[:, :3] = block_xyz
    if normalize:
        centroid_xy = np.mean(combined[:, :2], axis=0)
        min_z = np.min(combined[:, 2])
        combined[:, 0] -= centroid_xy[0]
        combined[:, 1] -= centroid_xy[1]
        combined[:, 2] -= min_z

    # Fill normals and eigenvalues
    col = 3
    if block_normals is not None:
        combined[:, col:col + 3] = block_normals
        col += 3
    if block_eigenvalues is not None:
        combined[:, col:col + 3] = block_eigenvalues

    # In-place standardization
    if feature_mean is not None and feature_std is not None:
        combined -= feature_mean
        combined /= feature_std

    # Append binary mask channel (all 1.0 — every point is real at inference)
    if has_mask_channel:
        mask_col = np.ones((n, 1), dtype=np.float32)
        combined = np.hstack([combined, mask_col])

    return combined


def _create_chunks(
    features: np.ndarray,
    num_points: int
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """
    Split a variable-size feature array into fixed-size chunks for model input.

    Uses random permutation so each chunk has spatially diverse points,
    matching the random subsampling used during training.

    Args:
        features: (n, F) feature array for a block
        num_points: Fixed chunk size expected by the model

    Returns:
        (chunks, perm_indices, n_original) where:
        - chunks: list of (num_points, F) arrays
        - perm_indices: (n,) array mapping permuted positions back to original
        - n_original: number of real points
    """
    n = len(features)

    if n <= num_points:
        # Single chunk, pad with duplicated points
        perm_indices = np.arange(n)
        if n < num_points:
            pad_indices = np.random.choice(n, num_points - n, replace=True)
            chunk = np.vstack([features, features[pad_indices]])
        else:
            chunk = features
        return [chunk], perm_indices, n

    # Multiple chunks: randomly permute all points, split into chunks
    perm_indices = np.random.permutation(n)
    permuted = features[perm_indices]

    n_chunks = int(np.ceil(n / num_points))
    chunks = []
    for c in range(n_chunks):
        start = c * num_points
        end = min(start + num_points, n)
        chunk_data = permuted[start:end]

        # Pad last chunk if needed
        if len(chunk_data) < num_points:
            deficit = num_points - len(chunk_data)
            pad_indices = np.random.choice(len(chunk_data), deficit, replace=True)
            chunk_data = np.vstack([chunk_data, chunk_data[pad_indices]])

        chunks.append(chunk_data)

    return chunks, perm_indices, n


def _process_chunks(
    chunk_batch: List[np.ndarray],
    model: PointNetSegmentation,
    device: torch.device,
    ignore_classes: Optional[List[int]] = None
) -> np.ndarray:
    """
    Run a batch of fixed-size chunks through the model.

    Args:
        chunk_batch: List of (num_points, F) arrays
        model: Segmentation model
        device: Torch device
        ignore_classes: Class indices to mask out before softmax

    Returns:
        (B, num_points, C) softmax probability array
    """
    batch_tensor = torch.FloatTensor(np.stack(chunk_batch)).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        if ignore_classes:
            logits[:, :, ignore_classes] = float('-inf')
        probs = F.softmax(logits, dim=2).cpu().numpy()

    del batch_tensor, logits
    torch.cuda.empty_cache()

    return probs
