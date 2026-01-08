# train_pointnet.py
"""
Training script for PointNet cluster classification model (PyTorch).

This script loads training data from a directory structure where:
- Each subfolder represents a class (folder name = class name)
- Each .npy file contains a cluster sample (num_points, num_features)

Usage:
    python train_pointnet.py --data_dir training_data --epochs 100 --batch_size 32
"""

import os
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from core.pointnet_model import PointNet, PointNetClassifier


class PointCloudDataset(Dataset):
    """
    PyTorch Dataset for variable-size point cloud data.

    Handles random sampling of points during training for data augmentation.
    """

    def __init__(self, data_list, labels, num_points, augment=False):
        """
        Args:
            data_list: List of numpy arrays, each (num_points_i, num_features)
            labels: Numpy array of labels (num_samples,)
            num_points: Target number of points to sample
            augment: Whether to apply data augmentation
        """
        self.data_list = data_list
        self.labels = labels
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        points = self.data_list[idx]
        label = self.labels[idx]

        # Random sampling to fixed number of points
        num_available = points.shape[0]
        if num_available >= self.num_points:
            # Random sample without replacement
            indices = np.random.choice(num_available, self.num_points, replace=False)
        else:
            # Sample with replacement if not enough points
            indices = np.random.choice(num_available, self.num_points, replace=True)

        points = points[indices]

        # Apply data augmentation if enabled
        if self.augment:
            points = self._augment(points)

        return torch.FloatTensor(points), torch.LongTensor([label])[0]

    def _augment(self, points):
        """Apply random rotation around Z-axis and jitter."""
        # Random rotation around Z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Apply rotation only to XYZ (first 3 features)
        xyz = points[:, :3]
        other_features = points[:, 3:]

        rotated_xyz = xyz @ rotation_matrix.T

        # Add small random jitter
        jitter = np.random.normal(0, 0.01, rotated_xyz.shape).astype(np.float32)
        rotated_xyz = rotated_xyz + jitter

        # Recombine
        augmented_points = np.concatenate([rotated_xyz, other_features], axis=1)

        return augmented_points


def load_training_data(data_dir, verbose=True):
    """
    Load training data from directory structure.

    Expected structure:
        data_dir/
            ClassA/
                sample1.npy
                sample2.npy
            ClassB/
                sample1.npy
            metadata.json (optional)

    Args:
        data_dir: Root directory containing class subdirectories
        verbose: Print loading information

    Returns:
        Tuple of (data, labels, class_mapping, metadata)
        - data: list of numpy arrays (each: num_points_i, num_features)
        - labels: numpy array (n_samples,) with integer class IDs
        - class_mapping: dict {class_id: class_name}
        - metadata: dict with dataset information
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Try to load metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if verbose:
            print(f"Loaded metadata from {metadata_path}")
            print(f"  Created: {metadata.get('dataset_info', {}).get('created_at', 'Unknown')}")
            print(f"  Features: {metadata.get('data_format', {}).get('feature_order', [])}")

    # Find all class directories
    class_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            class_dirs.append(item)

    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir}")

    # Sort class names for consistent ordering
    class_dirs = sorted(class_dirs)

    # Create class mapping
    class_mapping = {i: class_name for i, class_name in enumerate(class_dirs)}

    if verbose:
        print(f"\nFound {len(class_dirs)} classes:")
        for class_id, class_name in class_mapping.items():
            print(f"  {class_id}: {class_name}")

    # Load all samples
    all_data = []
    all_labels = []
    class_counts = {class_name: 0 for class_name in class_dirs}

    for class_id, class_name in class_mapping.items():
        class_dir = os.path.join(data_dir, class_name)

        # Find all .npy files
        npy_files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]

        if len(npy_files) == 0:
            print(f"WARNING: No .npy files found in {class_dir}")
            continue

        for npy_file in npy_files:
            filepath = os.path.join(class_dir, npy_file)
            try:
                sample = np.load(filepath).astype(np.float32)
                all_data.append(sample)
                all_labels.append(class_id)
                class_counts[class_name] += 1
            except Exception as e:
                print(f"ERROR loading {filepath}: {e}")

    if len(all_data) == 0:
        raise ValueError("No valid samples loaded!")

    labels = np.array(all_labels, dtype=np.int64)

    if verbose:
        print(f"\nLoaded {len(all_data)} samples:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")

        # Check point counts
        point_counts = [sample.shape[0] for sample in all_data]
        num_features = all_data[0].shape[1]
        print(f"\nData info:")
        print(f"  num_samples: {len(all_data)}")
        print(f"  num_features: {num_features}")
        print(f"  num_points per sample:")
        print(f"    min: {min(point_counts)}")
        print(f"    max: {max(point_counts)}")
        print(f"    mean: {np.mean(point_counts):.1f}")

    return all_data, labels, class_mapping, metadata


def train_one_epoch(model, train_loader, optimizer, criterion, device, class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_data, batch_labels in train_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(batch_data)

        # Apply class weights if provided
        if class_weights is not None:
            weights = class_weights[batch_labels]
            loss = F.cross_entropy(logits, batch_labels, reduction='none')
            loss = (loss * weights).mean()
        else:
            loss = criterion(logits, batch_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * batch_data.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_data)
            loss = criterion(logits, batch_labels)

            total_loss += loss.item() * batch_data.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_data.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train PointNet cluster classification model (PyTorch)')
    parser.add_argument('--data_dir', type=str, default='training_data',
                        help='Directory containing training data (default: training_data)')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained model (default: models)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--use_tnet', action='store_true', default=True,
                        help='Use T-Net transformations (default: True)')
    parser.add_argument('--no_tnet', action='store_true', default=False,
                        help='Disable T-Net transformations')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Apply data augmentation (default: False)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')

    args = parser.parse_args()

    # Handle tnet flag
    use_tnet = args.use_tnet and not args.no_tnet

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("PointNet Cluster Classification Training (PyTorch)")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load training data
    print("\n[1/6] Loading training data...")
    data, labels, class_mapping, metadata = load_training_data(args.data_dir, verbose=True)

    num_samples = len(data)
    num_features = data[0].shape[1]
    num_classes = len(class_mapping)

    # Determine num_points from metadata or data
    if metadata and 'processing' in metadata and 'sampling' in metadata['processing']:
        num_points = metadata['processing']['sampling'].get('min_points_per_sample', 1024)
    else:
        num_points = min(sample.shape[0] for sample in data)

    print(f"\nModel will use {num_points} points per sample (randomly sampled during training)")

    # Split train/validation
    print(f"\n[2/6] Splitting data (validation ratio: {args.val_split})...")
    train_indices, val_indices = train_test_split(
        np.arange(num_samples),
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels
    )

    X_train = [data[i] for i in train_indices]
    y_train = labels[train_indices]
    X_val = [data[i] for i in val_indices]
    y_val = labels[val_indices]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Compute class weights
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.FloatTensor(class_weights_array).to(device)

    print(f"\n  Class weights (for imbalanced data):")
    for class_id, weight in enumerate(class_weights_array):
        print(f"    {class_mapping[class_id]}: {weight:.3f}")

    # Create datasets and dataloaders
    print(f"\n[3/6] Creating data loaders...")
    print(f"  Random sampling: ENABLED (samples {num_points} points per sample each epoch)")
    print(f"  Data augmentation: {'ENABLED' if args.augment else 'DISABLED'}")

    train_dataset = PointCloudDataset(X_train, y_train, num_points, augment=args.augment)
    val_dataset = PointCloudDataset(X_val, y_val, num_points, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    print(f"\n[4/6] Creating PointNet model...")
    print(f"  Input shape: ({num_points}, {num_features})")
    print(f"  Output classes: {num_classes}")
    print(f"  Use T-Net: {use_tnet}")

    model = PointNet(
        num_points=num_points,
        num_features=num_features,
        num_classes=num_classes,
        use_tnet=use_tnet
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Setup optimizer, scheduler, criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # Create output directory with run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_pytorch_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "pointnet_best.pt"
    final_model_path = run_dir / "pointnet_final.pt"

    # Training loop
    print(f"\n[5/6] Training model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output directory: {run_dir}")
    print("-" * 80)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, class_weights
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
              f"lr: {current_lr:.6f}")

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_mapping': class_mapping,
                'num_points': num_points,
                'num_features': num_features,
                'num_classes': num_classes,
                'use_tnet': use_tnet
            }, best_model_path)
            print(f"  -> Saved best model (val_acc: {val_acc:.5f})")
        else:
            epochs_without_improvement += 1

        # Update scheduler
        scheduler.step(val_acc)

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    print(f"\n[6/6] Saving final model...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'class_mapping': class_mapping,
        'num_points': num_points,
        'num_features': num_features,
        'num_classes': num_classes,
        'use_tnet': use_tnet,
        'history': history
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save class mapping
    mapping_path = run_dir / "class_mapping.json"
    with open(mapping_path, 'w') as f:
        # Convert int keys to strings for JSON
        json.dump({str(k): v for k, v in class_mapping.items()}, f, indent=2)
    print(f"Class mapping saved to {mapping_path}")

    # Save training metadata
    training_metadata = {
        'framework': 'PyTorch',
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'num_points': int(num_points),
        'num_features': int(num_features),
        'num_classes': int(num_classes),
        'class_mapping': {str(k): v for k, v in class_mapping.items()},
        'training_samples': int(len(X_train)),
        'validation_samples': int(len(X_val)),
        'epochs_completed': epoch + 1,
        'best_val_accuracy': float(best_val_acc),
        'final_val_accuracy': float(val_acc),
        'use_tnet': use_tnet,
        'data_augmentation': args.augment,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'random_sampling': {
            'enabled': True,
            'num_points': int(num_points),
            'note': 'Each epoch randomly samples different subsets of points from variable-size training data'
        },
        'source_metadata': metadata
    }

    metadata_path = run_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    print(f"Training metadata saved to {metadata_path}")

    # Save training history
    history_path = run_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Print final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Models saved to: {run_dir}/")
    print(f"  - Best model: pointnet_best.pt")
    print(f"  - Final model: pointnet_final.pt")
    print(f"  - Class mapping: class_mapping.json")
    print(f"  - Metadata: training_metadata.json")
    print(f"  - History: training_history.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
