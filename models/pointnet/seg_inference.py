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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    full_normals: Optional[np.ndarray] = None,
    full_eigenvalues: Optional[np.ndarray] = None
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
        full_normals: Optional pre-computed (N,3) normals for the full cloud.
            If None and model needs normals, they are auto-computed.
        full_eigenvalues: Optional pre-computed (N,3) eigenvalues for the full cloud.
            If None and model needs eigenvalues, they are auto-computed.

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

    # Accumulate softmax probabilities and counts for overlap averaging.
    # float32 halves memory vs float64: (55M, 29) goes from ~13 GB to ~6.4 GB.
    prob_accum = np.zeros((N, num_classes), dtype=np.float32)
    count_accum = np.zeros(N, dtype=np.float32)

    # Create spatial blocks (XY plane)
    blocks = _create_spatial_blocks(points, block_size, overlap)
    total_blocks = len(blocks)

    if progress_callback:
        progress_callback(0, total_blocks, f"Processing {total_blocks} blocks...")

    # Stream: process each block → infer → accumulate → free.
    # The old 3-phase approach (buffer all chunks, then infer, then accumulate)
    # stored all_chunks + block_probs_list + prob_accum simultaneously,
    # easily exceeding 30 GB for large clouds. Streaming keeps peak memory
    # at prob_accum + one block's worth of temporaries.
    for block_idx, (block_point_indices, block_primary_mask) in enumerate(blocks):
        if len(block_point_indices) == 0:
            continue

        block_xyz = points[block_point_indices]

        # Slice pre-computed features for this block
        block_normals = full_normals[block_point_indices] if full_normals is not None else None
        block_eigenvalues = full_eigenvalues[block_point_indices] if full_eigenvalues is not None else None

        features = _compute_full_block_features(
            block_xyz,
            normalize=normalize_enabled,
            block_normals=block_normals,
            block_eigenvalues=block_eigenvalues,
            feature_mean=feature_mean,
            feature_std=feature_std
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
            batch_probs = _process_chunks(batch_chunks, model, device)

            for j in range(batch_end - batch_start):
                chunk_idx = batch_start + j
                perm_start = chunk_idx * num_points
                perm_end = min(perm_start + num_points, n_original)
                n_real = perm_end - perm_start
                original_positions = perm_indices[perm_start:perm_end]
                block_probs[original_positions] += batch_probs[j][:n_real]

            del batch_probs

        del chunks

        # Accumulate into global arrays with overlap weighting
        primary_indices = block_point_indices[block_primary_mask]
        prob_accum[primary_indices] += block_probs[block_primary_mask]
        count_accum[primary_indices] += 1.0

        overlap_mask = ~block_primary_mask
        overlap_indices = block_point_indices[overlap_mask]
        if len(overlap_indices) > 0:
            prob_accum[overlap_indices] += block_probs[overlap_mask] * 0.5
            count_accum[overlap_indices] += 0.5

        del block_probs

        if progress_callback:
            progress_callback(block_idx + 1, total_blocks,
                              f"Block {block_idx + 1}/{total_blocks}")

    del blocks

    # Assign labels in chunks to avoid a temporary (N, C) allocation
    labels = np.zeros(N, dtype=np.int32)
    valid_indices = np.where(count_accum > 0)[0]
    LABEL_CHUNK = 1_000_000
    for start in range(0, len(valid_indices), LABEL_CHUNK):
        end = min(start + LABEL_CHUNK, len(valid_indices))
        idx = valid_indices[start:end]
        avg = prob_accum[idx] / count_accum[idx, np.newaxis]
        labels[idx] = np.argmax(avg, axis=1).astype(np.int32)

    del prob_accum, count_accum

    return labels


def _create_spatial_blocks(
    points: np.ndarray,
    block_size: float,
    overlap: float
) -> list:
    """
    Divide point cloud into spatial grid blocks on XY plane using spatial hashing.

    Uses O(N log N) sort + O(1) per-cell lookup instead of O(N) per block,
    matching the spatial hash pattern from training data generation.

    Args:
        points: (N, 3) point coordinates
        block_size: Size of each block in XY
        overlap: Overlap between blocks in meters

    Returns:
        List of (point_indices, primary_mask) tuples
    """
    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)

    extent = max_xy - min_xy
    nx = max(1, int(np.ceil(extent[0] / block_size)))
    ny = max(1, int(np.ceil(extent[1] / block_size)))

    # O(N): assign each point to its block-sized cell (int32 saves ~880 MB vs int64)
    cx = np.floor((points[:, 0] - min_xy[0]) / block_size).astype(np.int32)
    cy = np.floor((points[:, 1] - min_xy[1]) / block_size).astype(np.int32)
    np.clip(cx, 0, nx - 1, out=cx)
    np.clip(cy, 0, ny - 1, out=cy)

    # O(N log N): sort by cell for O(1) lookup via searchsorted splits
    cell_id = (cx * ny + cy).astype(np.int32)
    del cx, cy
    order = np.argsort(cell_id)
    sorted_ids = cell_id[order]
    del cell_id
    splits = np.searchsorted(sorted_ids, np.arange(nx * ny + 1))
    del sorted_ids

    # How many neighboring cells does the overlap reach?
    overlap_cells = max(1, int(np.ceil(overlap / block_size)))

    blocks = []

    for ix in range(nx):
        for iy in range(ny):
            x_min = min_xy[0] + ix * block_size
            x_max = min_xy[0] + (ix + 1) * block_size
            y_min = min_xy[1] + iy * block_size
            y_max = min_xy[1] + (iy + 1) * block_size

            # Gather candidate indices from primary + neighboring cells
            parts = []
            for dx in range(-overlap_cells, overlap_cells + 1):
                sx = ix + dx
                if sx < 0 or sx >= nx:
                    continue
                for dy in range(-overlap_cells, overlap_cells + 1):
                    sy = iy + dy
                    if sy < 0 or sy >= ny:
                        continue
                    linear = sx * ny + sy
                    start, end = splits[linear], splits[linear + 1]
                    if start < end:
                        parts.append(order[start:end])

            if not parts:
                continue

            candidates = np.concatenate(parts) if len(parts) > 1 else parts[0]

            # Filter candidates to exact extended bounds
            cand_pts = points[candidates]
            ext_mask = (
                (cand_pts[:, 0] >= x_min - overlap) & (cand_pts[:, 0] < x_max + overlap) &
                (cand_pts[:, 1] >= y_min - overlap) & (cand_pts[:, 1] < y_max + overlap)
            )
            block_indices = candidates[ext_mask]

            if len(block_indices) == 0:
                continue

            # Primary mask on block subset only
            block_pts = points[block_indices]
            primary_mask = (
                (block_pts[:, 0] >= x_min) & (block_pts[:, 0] < x_max) &
                (block_pts[:, 1] >= y_min) & (block_pts[:, 1] < y_max)
            )

            blocks.append((block_indices, primary_mask))

    return blocks


def _compute_full_block_features(
    block_xyz: np.ndarray,
    normalize: bool,
    block_normals: Optional[np.ndarray] = None,
    block_eigenvalues: Optional[np.ndarray] = None,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None
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

    del batch_tensor, logits
    torch.cuda.empty_cache()

    return probs
