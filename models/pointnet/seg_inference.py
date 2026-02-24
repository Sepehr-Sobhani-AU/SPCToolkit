"""
PointNet Segmentation Inference Utilities (PyTorch)

Functions for loading trained segmentation models and performing blockwise
inference on large point clouds.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple, Optional, Callable
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

    # Prepare all block data
    block_batch = []
    block_indices_batch = []

    for block_idx, (block_point_indices, block_primary_mask) in enumerate(blocks):
        if len(block_point_indices) == 0:
            continue

        block_xyz = points[block_point_indices]

        # Compute features for this block
        features = _compute_block_features(
            block_xyz,
            num_points=num_points,
            normalize=normalize_enabled,
            use_normals=use_normals,
            use_eigenvalues=use_eigenvalues,
            normals_knn=normals_knn,
            eigenvalues_knn=eigenvalues_knn,
            eigenvalues_smooth=eigenvalues_smooth
        )

        if features is None:
            continue

        # Store original -> subsampled mapping
        block_batch.append((features, block_point_indices, block_primary_mask))

        # Process in GPU batches
        if len(block_batch) >= batch_size or block_idx == total_blocks - 1:
            _process_batch(
                block_batch, model, device, num_points,
                num_classes, prob_accum, count_accum
            )
            block_batch.clear()

        if progress_callback:
            progress_callback(block_idx + 1, total_blocks,
                            f"Block {block_idx + 1}/{total_blocks}")

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


def _compute_block_features(
    block_xyz: np.ndarray,
    num_points: int,
    normalize: bool,
    use_normals: bool,
    use_eigenvalues: bool,
    normals_knn: int,
    eigenvalues_knn: int,
    eigenvalues_smooth: bool
) -> Optional[np.ndarray]:
    """
    Compute features for a single block of points.

    Returns features array of shape (num_points, num_features) or None if failed.
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

    # Subsample or pad to num_points
    if len(combined) > num_points:
        indices = np.random.choice(len(combined), num_points, replace=False)
        combined = combined[indices]
    elif len(combined) < num_points:
        deficit = num_points - len(combined)
        if len(combined) > 0:
            pad_indices = np.random.choice(len(combined), deficit, replace=True)
            combined = np.vstack([combined, combined[pad_indices]])
        else:
            num_feat = combined.shape[1] if combined.ndim == 2 else 9
            combined = np.zeros((num_points, num_feat), dtype=np.float32)

    return combined


def _process_batch(
    block_batch: list,
    model: PointNetSegmentation,
    device: torch.device,
    num_points: int,
    num_classes: int,
    prob_accum: np.ndarray,
    count_accum: np.ndarray
):
    """
    Process a batch of blocks through the model and accumulate probabilities.

    Args:
        block_batch: List of (features, block_indices, primary_mask) tuples
        model: Segmentation model
        device: Torch device
        num_points: Points per block for model
        num_classes: Number of classes
        prob_accum: (N, C) accumulator for softmax probs
        count_accum: (N,) count accumulator
    """
    # Stack features into batch tensor
    batch_features = np.stack([item[0] for item in block_batch])
    batch_tensor = torch.FloatTensor(batch_features).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)  # (B, num_points, C)
        probs = F.softmax(logits, dim=2).cpu().numpy()  # (B, num_points, C)

    for i, (features, block_indices, primary_mask) in enumerate(block_batch):
        block_probs = probs[i]  # (num_points, C)

        # We subsampled/padded the block to num_points.
        # Map predictions back to original points.
        n_original = len(block_indices)

        if n_original <= num_points:
            # We used all original points (possibly padded)
            # Take only the first n_original predictions
            original_probs = block_probs[:n_original]
        else:
            # We subsampled - we need to map back
            # For subsampled blocks, we can only assign to the subsampled points
            # Use uniform assignment to all block points as approximation
            # Since subsampling was random, distribute model's average prediction
            avg_prob = np.mean(block_probs, axis=0, keepdims=True)
            original_probs = np.tile(avg_prob, (n_original, 1))

        # Only accumulate for primary points (not in overlap zone)
        primary_indices = block_indices[primary_mask]
        primary_probs = original_probs[primary_mask]

        # Also accumulate overlap points but with lower weight for smooth blending
        overlap_mask = ~primary_mask
        overlap_indices = block_indices[overlap_mask]
        overlap_probs = original_probs[overlap_mask]

        # Primary points get full weight
        prob_accum[primary_indices] += primary_probs
        count_accum[primary_indices] += 1.0

        # Overlap points get partial weight for smooth blending
        if len(overlap_indices) > 0:
            prob_accum[overlap_indices] += overlap_probs * 0.5
            count_accum[overlap_indices] += 0.5
