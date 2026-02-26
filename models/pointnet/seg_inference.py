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
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> np.ndarray:
    """
    Segment a large point cloud by dividing it into spatial blocks.

    Pipeline:
    1. Divide point cloud into spatial grid blocks (XY plane)
    2. Add overlap between blocks to handle boundary artifacts
    3. Per block: subsample/pad to num_points, compute features, run model
    4. Merge overlapping predictions via softmax probability averaging
    5. Return (N,) label array

    Args:
        point_cloud: Input PointCloud to segment
        model: Trained segmentation model (in eval mode)
        metadata: Model metadata with num_points, num_features, etc.
        block_size: Size of spatial blocks in XY plane (meters)
        overlap: Overlap between blocks (meters)
        batch_size: Number of blocks to process in parallel on GPU
        progress_callback: Optional callback(current_block, total_blocks, status_msg)

    Returns:
        numpy array of shape (N,) with per-point class labels
    """
    num_points = metadata['num_points']
    num_features = metadata['num_features']
    device = metadata.get('_device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Per-feature standardization stats (None for old models without them)
    feature_mean = metadata.get('_feature_mean', None)
    feature_std = metadata.get('_feature_std', None)

    # Determine feature configuration
    use_normals = num_features >= 6
    use_eigenvalues = num_features >= 9

    # Get preprocessing parameters from metadata
    source_metadata = metadata.get('source_metadata', {})
    processing = source_metadata.get('processing', {}) if source_metadata else {}
    normalize_enabled = processing.get('normalization', {}).get('enabled', True)
    normals_config = processing.get('features', {}).get('normals', {})
    eigenvalues_config = processing.get('features', {}).get('eigenvalues', {})
    normals_knn = normals_config.get('knn', 30)
    eigenvalues_knn = eigenvalues_config.get('knn', 30)
    eigenvalues_smooth = eigenvalues_config.get('smooth', True)

    points = point_cloud.points
    N = len(points)
    num_classes = metadata['num_classes']

    # Accumulate softmax probabilities and counts for averaging
    prob_accum = np.zeros((N, num_classes), dtype=np.float64)
    count_accum = np.zeros(N, dtype=np.float64)

    # Create spatial blocks (XY plane)
    blocks = _create_spatial_blocks(points, block_size, overlap)
    total_blocks = len(blocks)

    if progress_callback:
        progress_callback(0, total_blocks, f"Processing {total_blocks} blocks...")

    # --- Phase 1: Prepare chunks for all blocks ---
    # Each entry: (chunk_array, block_id) for flat batching
    all_chunks = []
    # Per-block metadata: (block_indices, primary_mask, perm_indices, n_original, n_chunks)
    block_meta = []

    for block_idx, (block_point_indices, block_primary_mask) in enumerate(blocks):
        if len(block_point_indices) == 0:
            continue

        block_xyz = points[block_point_indices]

        features = _compute_full_block_features(
            block_xyz,
            normalize=normalize_enabled,
            use_normals=use_normals,
            use_eigenvalues=use_eigenvalues,
            normals_knn=normals_knn,
            eigenvalues_knn=eigenvalues_knn,
            eigenvalues_smooth=eigenvalues_smooth,
            feature_mean=feature_mean,
            feature_std=feature_std
        )

        if features is None:
            continue

        chunks, perm_indices, n_original = _create_chunks(features, num_points)

        bid = len(block_meta)
        block_meta.append((block_point_indices, block_primary_mask,
                           perm_indices, n_original, len(chunks)))

        for chunk in chunks:
            all_chunks.append((chunk, bid))

        if progress_callback:
            progress_callback(block_idx + 1, total_blocks,
                              f"Preparing block {block_idx + 1}/{total_blocks}")

    # --- Phase 2: Infer in GPU batches, map back to block positions ---
    # Allocate per-block probability accumulators
    block_probs_list = [
        np.zeros((meta[3], num_classes), dtype=np.float64)  # (n_original, C)
        for meta in block_meta
    ]

    total_chunks = len(all_chunks)

    # Precompute: for each chunk, its index within its block
    block_chunk_counters = {}  # bid -> next chunk index
    chunk_within_block_map = []  # parallel to all_chunks
    for chunk_array, bid in all_chunks:
        idx = block_chunk_counters.get(bid, 0)
        chunk_within_block_map.append(idx)
        block_chunk_counters[bid] = idx + 1

    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        chunk_arrays = [all_chunks[i][0] for i in range(batch_start, batch_end)]

        batch_probs = _process_chunks(chunk_arrays, model, device)  # (B, num_points, C)

        for j in range(len(chunk_arrays)):
            chunk_idx_global = batch_start + j
            _, bid = all_chunks[chunk_idx_global]
            _, _, perm_indices, n_original, n_block_chunks = block_meta[bid]

            chunk_probs = batch_probs[j]  # (num_points, C)
            chunk_within_block = chunk_within_block_map[chunk_idx_global]

            perm_start = chunk_within_block * num_points
            perm_end = min(perm_start + num_points, n_original)
            n_real = perm_end - perm_start  # real points in this chunk (rest is padding)

            # Map predictions back: permuted positions -> original block positions
            original_positions = perm_indices[perm_start:perm_end]
            block_probs_list[bid][original_positions] += chunk_probs[:n_real]

    # --- Phase 3: Accumulate into global arrays with overlap weighting ---
    for bid, (block_indices, primary_mask, _, _, _) in enumerate(block_meta):
        block_probs = block_probs_list[bid]

        # Primary points get full weight
        primary_indices = block_indices[primary_mask]
        primary_probs = block_probs[primary_mask]
        prob_accum[primary_indices] += primary_probs
        count_accum[primary_indices] += 1.0

        # Overlap points get partial weight for smooth blending
        overlap_mask = ~primary_mask
        overlap_indices = block_indices[overlap_mask]
        if len(overlap_indices) > 0:
            overlap_probs = block_probs[overlap_mask]
            prob_accum[overlap_indices] += overlap_probs * 0.5
            count_accum[overlap_indices] += 0.5

    # Assign labels from accumulated probabilities
    labels = np.zeros(N, dtype=np.int32)
    valid_mask = count_accum > 0
    if np.any(valid_mask):
        avg_probs = prob_accum[valid_mask] / count_accum[valid_mask, np.newaxis]
        labels[valid_mask] = np.argmax(avg_probs, axis=1).astype(np.int32)

    # Points never seen by any block get label 0 (or could be -1)
    # Label 0 is typically the most common class, which is reasonable

    return labels


def _create_spatial_blocks(
    points: np.ndarray,
    block_size: float,
    overlap: float
) -> list:
    """
    Divide point cloud into spatial grid blocks on XY plane.

    Args:
        points: (N, 3) point coordinates
        block_size: Size of each block in XY
        overlap: Overlap between blocks in meters

    Returns:
        List of (point_indices, primary_mask) tuples
    """
    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)

    # Calculate grid dimensions
    extent = max_xy - min_xy
    nx = max(1, int(np.ceil(extent[0] / block_size)))
    ny = max(1, int(np.ceil(extent[1] / block_size)))

    blocks = []

    for ix in range(nx):
        for iy in range(ny):
            # Primary block bounds
            x_min = min_xy[0] + ix * block_size
            x_max = min_xy[0] + (ix + 1) * block_size
            y_min = min_xy[1] + iy * block_size
            y_max = min_xy[1] + (iy + 1) * block_size

            # Extended bounds with overlap
            x_min_ext = x_min - overlap
            x_max_ext = x_max + overlap
            y_min_ext = y_min - overlap
            y_max_ext = y_max + overlap

            # Find points in extended block
            mask_ext = (
                (points[:, 0] >= x_min_ext) & (points[:, 0] < x_max_ext) &
                (points[:, 1] >= y_min_ext) & (points[:, 1] < y_max_ext)
            )
            block_indices = np.where(mask_ext)[0]

            if len(block_indices) == 0:
                continue

            # Find primary points (within non-overlapping bounds)
            block_points = points[block_indices]
            primary_mask = (
                (block_points[:, 0] >= x_min) & (block_points[:, 0] < x_max) &
                (block_points[:, 1] >= y_min) & (block_points[:, 1] < y_max)
            )

            blocks.append((block_indices, primary_mask))

    return blocks


def _compute_full_block_features(
    block_xyz: np.ndarray,
    normalize: bool,
    use_normals: bool,
    use_eigenvalues: bool,
    normals_knn: int,
    eigenvalues_knn: int,
    eigenvalues_smooth: bool,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Compute features for all points in a block (no subsampling).

    Returns features array of shape (n_block_points, num_features) or None if failed.
    """
    if len(block_xyz) < 3:
        return None

    pc = PointCloud(points=block_xyz.copy())

    if normalize:
        pc.normalise(
            apply_scaling=True,
            apply_centering=True,
            rotation_axes=(False, False, False)
        )

    features = [pc.points]

    if use_normals:
        if len(pc.points) >= normals_knn:
            pc.estimate_normals(k=normals_knn)
            features.append(pc.normals)
        else:
            features.append(np.zeros((len(pc.points), 3), dtype=np.float32))

    if use_eigenvalues:
        if len(pc.points) >= eigenvalues_knn:
            eigenvalues = pc.get_eigenvalues(k=eigenvalues_knn, smooth=eigenvalues_smooth)
            features.append(eigenvalues)
        else:
            features.append(np.zeros((len(pc.points), 3), dtype=np.float32))

    combined = np.hstack(features).astype(np.float32)

    # Apply per-feature standardization if stats are available (from training)
    if feature_mean is not None and feature_std is not None:
        combined = (combined - feature_mean) / feature_std

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
    device: torch.device
) -> np.ndarray:
    """
    Run a batch of fixed-size chunks through the model.

    Args:
        chunk_batch: List of (num_points, F) arrays
        model: Segmentation model
        device: Torch device

    Returns:
        (B, num_points, C) softmax probability array
    """
    batch_tensor = torch.FloatTensor(np.stack(chunk_batch)).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        probs = F.softmax(logits, dim=2).cpu().numpy()

    return probs
