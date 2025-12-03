# train_pointnet.py
"""
Training script for PointNet cluster classification model.

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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

from core.pointnet_model import PointNetClassifier


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
        - data: numpy array (n_samples, num_points, num_features)
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
                sample = np.load(filepath)
                all_data.append(sample)
                all_labels.append(class_id)
                class_counts[class_name] += 1
            except Exception as e:
                print(f"ERROR loading {filepath}: {e}")

    if len(all_data) == 0:
        raise ValueError("No valid samples loaded!")

    # Convert to numpy arrays
    data = np.array(all_data, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)

    if verbose:
        print(f"\nLoaded {len(data)} samples:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
        print(f"\nData shape: {data.shape}")
        print(f"  num_samples: {data.shape[0]}")
        print(f"  num_points: {data.shape[1]}")
        print(f"  num_features: {data.shape[2]}")

    return data, labels, class_mapping, metadata


def create_augmentation_fn():
    """
    Create data augmentation function for point clouds.

    Applies random rotations around Z-axis and random jitter.

    Returns:
        Function that takes (points, label) and returns augmented (points, label)
    """
    def augment(points, label):
        # Random rotation around Z-axis
        theta = tf.random.uniform([], 0, 2 * np.pi)
        cos_theta = tf.cos(theta)
        sin_theta = tf.sin(theta)

        # Rotation matrix around Z-axis
        rotation_matrix = tf.stack([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])

        # Apply rotation only to XYZ (first 3 features)
        xyz = points[:, :3]
        other_features = points[:, 3:]

        rotated_xyz = tf.matmul(xyz, rotation_matrix)

        # Add small random jitter
        jitter = tf.random.normal(tf.shape(rotated_xyz), mean=0.0, stddev=0.01)
        rotated_xyz = rotated_xyz + jitter

        # Recombine
        augmented_points = tf.concat([rotated_xyz, other_features], axis=1)

        return augmented_points, label

    return augment


def main():
    parser = argparse.ArgumentParser(description='Train PointNet cluster classification model')
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
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Apply data augmentation (default: False)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("=" * 80)
    print("PointNet Cluster Classification Training")
    print("=" * 80)

    # Load training data
    print("\n[1/6] Loading training data...")
    data, labels, class_mapping, metadata = load_training_data(args.data_dir, verbose=True)

    num_samples, num_points, num_features = data.shape
    num_classes = len(class_mapping)

    # Split train/validation
    print(f"\n[2/6] Splitting data (validation ratio: {args.val_split})...")
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Compute class weights for imbalanced datasets
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

    print(f"\n  Class weights (for imbalanced data):")
    for class_id, weight in class_weights.items():
        print(f"    {class_mapping[class_id]}: {weight:.3f}")

    # Create model
    print(f"\n[3/6] Creating PointNet model...")
    print(f"  Input shape: ({num_points}, {num_features})")
    print(f"  Output classes: {num_classes}")
    print(f"  Use T-Net: {args.use_tnet}")

    classifier = PointNetClassifier(
        num_points=num_points,
        num_features=num_features,
        num_classes=num_classes,
        use_tnet=args.use_tnet
    )
    classifier.class_mapping = class_mapping

    classifier.compile_model(learning_rate=args.learning_rate)

    # Print model summary
    classifier.summary()

    # Setup callbacks
    print(f"\n[4/6] Setting up training callbacks...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(args.output_dir, 'pointnet_best.keras'),
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        patience=20,
        mode='max',
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    callbacks = [model_checkpoint, early_stopping, reduce_lr]

    # Data augmentation (optional)
    if args.augment:
        print(f"  Data augmentation: ENABLED")
    else:
        print(f"  Data augmentation: DISABLED")

    # Train model
    print(f"\n[5/6] Training model...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print("-" * 80)

    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )

    # Save final model
    print(f"\n[6/6] Saving final model...")
    final_model_path = os.path.join(args.output_dir, 'pointnet_final.keras')
    classifier.save(final_model_path)

    # Save class mapping
    mapping_path = os.path.join(args.output_dir, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Class mapping saved to {mapping_path}")

    # Save training metadata
    training_metadata = {
        'num_points': int(num_points),
        'num_features': int(num_features),
        'num_classes': int(num_classes),
        'class_mapping': class_mapping,
        'training_samples': int(len(X_train)),
        'validation_samples': int(len(X_val)),
        'epochs_completed': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_sparse_categorical_accuracy'])),
        'final_val_accuracy': float(history.history['val_sparse_categorical_accuracy'][-1]),
        'use_tnet': args.use_tnet,
        'data_augmentation': args.augment,
        'source_metadata': metadata
    }

    metadata_path = os.path.join(args.output_dir, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    print(f"Training metadata saved to {metadata_path}")

    # Print final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation accuracy: {training_metadata['best_val_accuracy']:.4f}")
    print(f"Final validation accuracy: {training_metadata['final_val_accuracy']:.4f}")
    print(f"Models saved to: {args.output_dir}/")
    print(f"  - Best model: pointnet_best.keras")
    print(f"  - Final model: pointnet_final.keras")
    print(f"  - Class mapping: class_mapping.json")
    print(f"  - Metadata: training_metadata.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
