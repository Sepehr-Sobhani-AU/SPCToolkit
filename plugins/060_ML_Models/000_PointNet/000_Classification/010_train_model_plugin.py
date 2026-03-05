"""
Train PointNet Model Plugin (PyTorch)

Trains a PointNet model for cluster classification using training data from a specified directory.
The directory should contain subfolders where each subfolder name is a class label and contains .npy files.
"""

import gc
import os
import json
import csv
import time
import numpy as np
from typing import Dict, Any
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtWidgets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.pointnet_model import PointNet
from plugins.dialogs.training_progress_window import TrainingProgressWindow


class PointCloudDataset(Dataset):
    """PyTorch Dataset for variable-size point cloud data with random sampling."""

    def __init__(self, data_list, labels, num_points, augment=False):
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
            indices = np.random.choice(num_available, self.num_points, replace=False)
        else:
            indices = np.random.choice(num_available, self.num_points, replace=True)

        points = points[indices]

        if self.augment:
            points = self._augment(points)

        return torch.FloatTensor(points), torch.LongTensor([label])[0]

    def _augment(self, points):
        """Apply random rotation around Z-axis and jitter."""
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        xyz = points[:, :3]
        other_features = points[:, 3:]

        rotated_xyz = xyz @ rotation_matrix.T
        jitter = np.random.normal(0, 0.01, rotated_xyz.shape).astype(np.float32)
        rotated_xyz = rotated_xyz + jitter

        return np.concatenate([rotated_xyz, other_features], axis=1)


class TrainPointNetPlugin(ActionPlugin):
    """
    Action plugin for training PointNet model for cluster classification.
    """

    # Class variables to store last used parameters
    last_params = {
        "training_data_dir": "training_data",
        "output_dir": "models",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "val_split": 0.2,
        "use_tnet": True,
        "early_stopping_patience": 20,
        "repetitions": 1
    }

    def get_name(self) -> str:
        return "train_pointnet_model"

    def _get_next_run_number(self, output_dir: str) -> int:
        """Get the next run number by scanning existing folders."""
        if not os.path.exists(output_dir):
            return 1

        run_numbers = []
        for item in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, item)) and item.startswith('run_'):
                try:
                    parts = item.split('_')
                    if len(parts) >= 2:
                        run_num = int(parts[1])
                        run_numbers.append(run_num)
                except (ValueError, IndexError):
                    continue

        return max(run_numbers) + 1 if run_numbers else 1

    def _write_to_tracking_csv(self, output_dir: str, training_data: Dict[str, Any]):
        """Write training results to central tracking CSV file."""
        csv_path = os.path.join(output_dir, 'training_history.csv')

        fieldnames = [
            'folder_name', 'timestamp', 'run_number', 'epochs', 'batch_size',
            'learning_rate', 'val_split', 'use_tnet', 'early_stopping_patience',
            'repetitions', 'random_seed', 'best_val_acc', 'final_val_acc',
            'epochs_completed', 'training_samples', 'validation_samples',
            'num_classes', 'was_cancelled'
        ]

        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(training_data)

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "training_data_dir": {
                "type": "directory",
                "default": self.last_params["training_data_dir"],
                "label": "Training Data Directory",
                "description": "Directory containing class subfolders with .npy files"
            },
            "output_dir": {
                "type": "directory",
                "default": self.last_params["output_dir"],
                "label": "Output Directory",
                "description": "Directory to save trained model"
            },
            "epochs": {
                "type": "int",
                "default": self.last_params["epochs"],
                "min": 1,
                "max": 1000,
                "label": "Epochs",
                "description": "Number of training epochs"
            },
            "batch_size": {
                "type": "int",
                "default": self.last_params["batch_size"],
                "min": 1,
                "max": 128,
                "label": "Batch Size",
                "description": "Training batch size"
            },
            "learning_rate": {
                "type": "float",
                "default": self.last_params["learning_rate"],
                "min": 0.0000001,
                "max": 0.1,
                "decimals": 7,
                "label": "Learning Rate",
                "description": "Initial learning rate"
            },
            "val_split": {
                "type": "float",
                "default": self.last_params["val_split"],
                "min": 0.1,
                "max": 0.5,
                "label": "Validation Split",
                "description": "Fraction of data to use for validation"
            },
            "use_tnet": {
                "type": "bool",
                "default": self.last_params["use_tnet"],
                "label": "Use T-Net",
                "description": "Use spatial and feature transformation networks"
            },
            "early_stopping_patience": {
                "type": "int",
                "default": self.last_params["early_stopping_patience"],
                "min": 1,
                "max": 100,
                "label": "Early Stopping Patience",
                "description": "Number of epochs with no improvement before stopping training"
            },
            "repetitions": {
                "type": "int",
                "default": self.last_params["repetitions"],
                "min": 1,
                "max": 100,
                "label": "Training Repetitions",
                "description": "Number of times to train with different random initializations"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Execute the training process."""
        # Get parameters
        data_dir = params['training_data_dir'].strip()
        output_dir = params['output_dir'].strip()
        epochs = int(params['epochs'])
        batch_size = int(params['batch_size'])
        learning_rate = float(params['learning_rate'])
        val_split = float(params['val_split'])
        use_tnet = params['use_tnet']
        early_stopping_patience = int(params['early_stopping_patience'])
        repetitions = int(params['repetitions'])

        # Store parameters for next time
        TrainPointNetPlugin.last_params = {
            "training_data_dir": data_dir,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_split": val_split,
            "use_tnet": use_tnet,
            "early_stopping_patience": early_stopping_patience,
            "repetitions": repetitions
        }

        # Validate directories
        if not os.path.exists(data_dir):
            QMessageBox.critical(
                main_window,
                "Invalid Directory",
                f"Training data directory does not exist:\n{data_dir}"
            )
            return

        # Setup device
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available — training will fall back to CPU and be significantly slower.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Track results from all repetitions
        all_run_results = []
        best_overall_accuracy = 0.0
        best_overall_run = None

        base_run_number = self._get_next_run_number(output_dir)

        print(f"\n{'='*80}")
        print(f"Starting {repetitions} training run(s) with different random initializations")
        print(f"Starting from run #{base_run_number}")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*80}")

        for rep_index in range(repetitions):
            run_number = base_run_number + rep_index

            print(f"\n{'='*80}")
            print(f"TRAINING RUN #{run_number} ({rep_index + 1}/{repetitions})")
            print(f"{'='*80}")

            # Set random seed
            random_seed = int(time.time() * 1000) % (2**32)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
            print(f"Random seed: {random_seed}")

            time.sleep(0.01)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"run_{run_number:03d}_{timestamp}"
            unique_output_dir = os.path.join(output_dir, folder_name)
            os.makedirs(unique_output_dir, exist_ok=True)

            print(f"Model will be saved to: {unique_output_dir}")

            try:
                main_window.disable_menus()
                main_window.disable_tree()
                main_window.tree_overlay.show_processing("Loading training data...")

                # Load training data
                data, labels, class_mapping, metadata = self.load_training_data(data_dir)

                num_samples = len(data)
                num_features = data[0].shape[1]
                num_classes = len(class_mapping)

                if metadata and 'processing' in metadata and 'sampling' in metadata['processing']:
                    num_points = metadata['processing']['sampling'].get('min_points_per_sample', 1024)
                else:
                    num_points = min(sample.shape[0] for sample in data)

                print(f"\nModel will use {num_points} points per sample")

                class_counts = {class_mapping[i]: np.sum(labels == i) for i in class_mapping.keys()}
                info_msg = f"Loaded {num_samples} samples from {num_classes} classes:\n"
                for class_name, count in class_counts.items():
                    info_msg += f"  {class_name}: {count} samples\n"

                print("\n" + "="*80)
                print("PointNet Training Started (PyTorch)")
                print("="*80)
                print(info_msg)

                # Split train/validation
                main_window.tree_overlay.show_processing(f"Splitting data ({val_split*100:.0f}% validation)...")

                train_indices, val_indices = train_test_split(
                    np.arange(num_samples),
                    test_size=val_split,
                    random_state=random_seed,
                    stratify=labels
                )

                X_train = [data[i] for i in train_indices]
                y_train = labels[train_indices]
                X_val = [data[i] for i in val_indices]
                y_val = labels[val_indices]

                print(f"\nTraining samples: {len(X_train)}")
                print(f"Validation samples: {len(X_val)}")

                # Compute class weights
                class_weights_array = compute_class_weight(
                    'balanced',
                    classes=np.unique(y_train),
                    y=y_train
                )
                class_weights = torch.FloatTensor(class_weights_array).to(device)

                print("\nClass weights:")
                for class_id, weight in enumerate(class_weights_array):
                    print(f"  {class_mapping[class_id]}: {weight:.3f}")

                # Create datasets and dataloaders
                main_window.tree_overlay.show_processing("Creating PointNet model...")

                train_dataset = PointCloudDataset(X_train, y_train, num_points, augment=False)
                val_dataset = PointCloudDataset(X_val, y_val, num_points, augment=False)

                # num_workers=0: data is pre-loaded in memory so there's no I/O benefit
                # from multiprocessing, and forked workers cause segfaults with CUDA
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=True, drop_last=False
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=True
                )

                print(f"\nModel configuration:")
                print(f"  Input: ({num_points}, {num_features})")
                print(f"  Classes: {num_classes}")
                print(f"  T-Net: {use_tnet}")

                # Create model
                model = PointNet(
                    num_points=num_points,
                    num_features=num_features,
                    num_classes=num_classes,
                    use_tnet=use_tnet
                ).to(device)

                total_params = sum(p.numel() for p in model.parameters())
                print(f"  Total parameters: {total_params:,}")

                # Setup optimizer and scheduler
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=10
                )
                criterion = nn.CrossEntropyLoss()

                # Create progress window
                progress_window = TrainingProgressWindow(parent=main_window, total_epochs=epochs)
                if repetitions > 1:
                    progress_window.setWindowTitle(f"PointNet Training Progress - Run #{run_number} ({rep_index + 1}/{repetitions})")
                progress_window.show()
                progress_window.training_started()

                screen_geometry = QtWidgets.QApplication.desktop().screenGeometry()
                x = (screen_geometry.width() - progress_window.width()) // 2
                y = (screen_geometry.height() - progress_window.height()) // 2
                progress_window.move(x, y)

                print(f"\nTraining configuration:")
                print(f"  Epochs: {epochs}")
                print(f"  Batch size: {batch_size}")
                print(f"  Learning rate: {learning_rate}")
                print("-"*80)

                # Training loop
                main_window.tree_overlay.show_processing("Training model...")

                history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
                best_val_acc = 0.0
                best_model_state = None
                best_per_class_acc = {}
                final_per_class_acc = {}
                epochs_without_improvement = 0
                was_cancelled = False

                for epoch in range(epochs):
                    # Check cancellation
                    if progress_window.training_cancelled:
                        print("\nUser requested training cancellation. Stopping...")
                        was_cancelled = True
                        break

                    # Training phase
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0

                    for batch_data, batch_labels in train_loader:
                        batch_data = batch_data.to(device)
                        batch_labels = batch_labels.to(device)

                        optimizer.zero_grad()
                        logits = model(batch_data)

                        # Apply class weights
                        weights = class_weights[batch_labels]
                        loss = F.cross_entropy(logits, batch_labels, reduction='none')
                        loss = (loss * weights).mean()

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item() * batch_data.size(0)
                        predictions = torch.argmax(logits, dim=1)
                        train_correct += (predictions == batch_labels).sum().item()
                        train_total += batch_data.size(0)

                    train_loss /= train_total
                    train_acc = train_correct / train_total

                    # Validation phase
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    val_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

                    with torch.no_grad():
                        for batch_data, batch_labels in val_loader:
                            batch_data = batch_data.to(device)
                            batch_labels = batch_labels.to(device)

                            logits = model(batch_data)
                            loss = criterion(logits, batch_labels)

                            val_loss += loss.item() * batch_data.size(0)
                            predictions = torch.argmax(logits, dim=1)
                            val_correct += (predictions == batch_labels).sum().item()
                            val_total += batch_data.size(0)

                            # Update confusion matrix
                            p = predictions.cpu().numpy()
                            l = batch_labels.cpu().numpy()
                            for pred_i, true_i in zip(p, l):
                                if 0 <= true_i < num_classes and 0 <= pred_i < num_classes:
                                    val_confusion[true_i, pred_i] += 1

                    val_loss /= val_total
                    val_acc = val_correct / val_total

                    # Compute per-class accuracy from confusion matrix
                    val_per_class_acc = {}
                    for c in range(num_classes):
                        class_total = val_confusion[c].sum()
                        if class_total > 0:
                            val_per_class_acc[c] = float(val_confusion[c, c]) / class_total
                    final_per_class_acc = val_per_class_acc.copy()

                    # Update history
                    history['loss'].append(train_loss)
                    history['acc'].append(train_acc)
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)

                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']

                    # Update progress window
                    progress_window.update_epoch(
                        epoch + 1,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        learning_rate=current_lr
                    )
                    # Classification runs on the main thread — explicit event processing
                    # keeps the UI responsive (segmentation doesn't need this since it
                    # uses a background thread with QTimer polling).
                    QtWidgets.QApplication.processEvents()

                    print(f"Epoch {epoch+1:3d}/{epochs} - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
                          f"lr: {current_lr:.6f}")

                    # Check for improvement
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = model.state_dict().copy()
                        best_per_class_acc = val_per_class_acc.copy()
                        epochs_without_improvement = 0
                        print(f"  -> New best model (val_acc: {val_acc:.5f})")

                        # Save best model
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': val_acc,
                            'class_mapping': class_mapping,
                            'num_points': num_points,
                            'num_features': num_features,
                            'num_classes': num_classes,
                            'use_tnet': use_tnet
                        }, os.path.join(unique_output_dir, 'pointnet_best.pt'))
                    else:
                        epochs_without_improvement += 1

                    # Update scheduler
                    scheduler.step(val_acc)

                    # Early stopping
                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        break

                # Save final model
                main_window.tree_overlay.show_processing("Saving model...")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_mapping': class_mapping,
                    'num_points': num_points,
                    'num_features': num_features,
                    'num_classes': num_classes,
                    'use_tnet': use_tnet,
                    'history': history
                }, os.path.join(unique_output_dir, 'pointnet_final.pt'))

                # Save class mapping
                mapping_path = os.path.join(unique_output_dir, 'class_mapping.json')
                with open(mapping_path, 'w') as f:
                    json.dump(class_mapping, f, indent=2)

                # Mark training as completed and save screenshot BEFORE
                # metadata/CSV so the dialog is always closeable
                progress_window.training_completed(best_val_acc, cancelled=was_cancelled)
                progress_window.save_snapshot(
                    os.path.join(unique_output_dir, 'training_progress.png'))

                # Save training metadata
                training_metadata = {
                    'framework': 'PyTorch',
                    'pytorch_version': torch.__version__,
                    'folder_name': folder_name,
                    'timestamp': timestamp,
                    'num_points': int(num_points),
                    'num_features': int(num_features),
                    'num_classes': int(num_classes),
                    'class_mapping': class_mapping,
                    'training_samples': int(len(X_train)),
                    'validation_samples': int(len(X_val)),
                    'epochs_completed': len(history['loss']),
                    'best_val_accuracy': float(best_val_acc),
                    'final_val_accuracy': float(history['val_acc'][-1]) if history['val_acc'] else 0.0,
                    'use_tnet': use_tnet,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'early_stopping_patience': early_stopping_patience,
                    'validation_split': val_split,
                    'random_seed': random_seed,
                    'run_number': run_number,
                    'total_repetitions': repetitions,
                    'random_sampling': {
                        'enabled': True,
                        'num_points': int(num_points),
                        'note': 'Each epoch randomly samples different subsets of points'
                    },
                    'source_metadata': metadata,
                    'best_per_class_accuracy': {
                        class_mapping.get(cid, f"Class_{cid}"): float(acc)
                        for cid, acc in best_per_class_acc.items()
                    },
                    'final_per_class_accuracy': {
                        class_mapping.get(cid, f"Class_{cid}"): float(acc)
                        for cid, acc in final_per_class_acc.items()
                    },
                }

                metadata_path = os.path.join(unique_output_dir, 'training_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(training_metadata, f, indent=2)

                # Write per-class accuracy CSV
                csv_path = os.path.join(unique_output_dir, 'per_class_accuracy.csv')
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['class_id', 'class_name', 'best_accuracy', 'final_accuracy'])
                    for cid in sorted(set(list(best_per_class_acc.keys()) + list(final_per_class_acc.keys()))):
                        name = class_mapping.get(cid, f"Class_{cid}")
                        writer.writerow([cid, name,
                                        f"{best_per_class_acc.get(cid, 0.0):.6f}",
                                        f"{final_per_class_acc.get(cid, 0.0):.6f}"])
                    # Summary row
                    best_mean = sum(best_per_class_acc.values()) / max(len(best_per_class_acc), 1)
                    final_mean = sum(final_per_class_acc.values()) / max(len(final_per_class_acc), 1)
                    writer.writerow(['', 'Mean', f"{best_mean:.6f}", f"{final_mean:.6f}"])
                print(f"Per-class accuracy saved to: {csv_path}")

                print("\n" + "="*80)
                if was_cancelled:
                    print("Training Cancelled by User")
                else:
                    print("Training Complete!")
                print("="*80)
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                print(f"Epochs completed: {training_metadata['epochs_completed']}")
                print(f"Models saved to: {unique_output_dir}/")
                print("="*80)

                if repetitions == 1:
                    if was_cancelled:
                        QMessageBox.information(
                            progress_window,
                            "Training Cancelled",
                            f"Training was cancelled by user.\n\n"
                            f"Best validation accuracy: {best_val_acc:.2%}\n"
                            f"Epochs completed: {training_metadata['epochs_completed']}\n\n"
                            f"Best model saved to:\n{unique_output_dir}/"
                        )
                    else:
                        QMessageBox.information(
                            progress_window,
                            "Training Complete",
                            f"PointNet training completed successfully!\n\n"
                            f"Best validation accuracy: {best_val_acc:.2%}\n"
                            f"Epochs completed: {training_metadata['epochs_completed']}\n\n"
                            f"Models saved to:\n{unique_output_dir}/"
                        )

                # Track results
                run_result = {
                    'run_number': run_number,
                    'random_seed': random_seed,
                    'best_val_accuracy': best_val_acc,
                    'epochs_completed': training_metadata['epochs_completed'],
                    'output_dir': unique_output_dir,
                    'was_cancelled': was_cancelled
                }
                all_run_results.append(run_result)

                if best_val_acc > best_overall_accuracy:
                    best_overall_accuracy = best_val_acc
                    best_overall_run = run_number

                # Write to tracking CSV
                csv_data = {
                    'folder_name': folder_name,
                    'timestamp': timestamp,
                    'run_number': run_number,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'val_split': val_split,
                    'use_tnet': use_tnet,
                    'early_stopping_patience': early_stopping_patience,
                    'repetitions': repetitions,
                    'random_seed': random_seed,
                    'best_val_acc': best_val_acc,
                    'final_val_acc': training_metadata['final_val_accuracy'],
                    'epochs_completed': training_metadata['epochs_completed'],
                    'training_samples': training_metadata['training_samples'],
                    'validation_samples': training_metadata['validation_samples'],
                    'num_classes': training_metadata['num_classes'],
                    'was_cancelled': was_cancelled
                }
                self._write_to_tracking_csv(output_dir, csv_data)

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(f"\nERROR during training run {run_number}:\n{error_msg}")

                try:
                    if 'progress_window' in locals():
                        progress_window.training_completed(0.0, cancelled=False)
                        progress_window.status_label.setText("Training failed - see error message")
                except:
                    pass

                error_parent = progress_window if 'progress_window' in locals() else main_window
                QMessageBox.critical(
                    error_parent,
                    f"Training Error (Run {run_number}/{repetitions})",
                    f"An error occurred during training:\n\n{str(e)}"
                )

            finally:
                main_window.tree_overlay.hide_processing()
                main_window.enable_menus()
                main_window.enable_tree()

                # Release RAM and VRAM — drop references to large training objects.
                # Assignment to None is safe even if the variable was never bound
                # (an early exception just means it becomes a fresh local).
                model = None
                optimizer = None
                scheduler = None
                criterion = None
                train_dataset = None
                val_dataset = None
                train_loader = None
                val_loader = None
                class_weights = None
                data = None
                X_train = None
                X_val = None
                best_model_state = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print("Training memory released (model, datasets, CUDA cache).")

        # Print summary
        if repetitions > 1 and len(all_run_results) > 0:
            print(f"\n{'='*80}")
            print(f"TRAINING SUMMARY - {len(all_run_results)} RUN(S) COMPLETED")
            print(f"{'='*80}")
            for result in all_run_results:
                status = "CANCELLED" if result['was_cancelled'] else "COMPLETE"
                best_marker = " <- BEST" if result['run_number'] == best_overall_run else ""
                print(f"Run #{result['run_number']}: {result['best_val_accuracy']:.2%} [{status}]{best_marker}")
            print(f"\nBest run: #{best_overall_run} with {best_overall_accuracy:.2%}")
            print(f"{'='*80}")

            summary_msg = f"Completed {len(all_run_results)} training run(s)\n\n"
            summary_msg += f"BEST RESULT: Run #{best_overall_run} - {best_overall_accuracy:.2%}\n\n"
            for result in all_run_results:
                status = "✓" if not result['was_cancelled'] else "✗"
                best_marker = " <- BEST" if result['run_number'] == best_overall_run else ""
                summary_msg += f"{status} Run #{result['run_number']}: {result['best_val_accuracy']:.2%}{best_marker}\n"

            QMessageBox.information(main_window, "Training Runs Complete", summary_msg)

    def load_training_data(self, data_dir):
        """Load training data from directory structure."""
        metadata_path = os.path.join(data_dir, 'metadata.json')
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        class_dirs = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                class_dirs.append(item)

        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {data_dir}")

        class_dirs = sorted(class_dirs)
        class_mapping = {i: class_name for i, class_name in enumerate(class_dirs)}

        all_data = []
        all_labels = []

        for class_id, class_name in class_mapping.items():
            class_dir = os.path.join(data_dir, class_name)
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
                except Exception as e:
                    print(f"ERROR loading {filepath}: {e}")

        if len(all_data) == 0:
            raise ValueError("No valid samples loaded!")

        labels = np.array(all_labels, dtype=np.int64)

        point_counts = [sample.shape[0] for sample in all_data]
        num_features = all_data[0].shape[1]
        print(f"\nLoaded {len(all_data)} samples:")
        print(f"  num_features: {num_features}")
        print(f"  num_points: min={min(point_counts)}, max={max(point_counts)}, mean={np.mean(point_counts):.1f}")

        return all_data, labels, class_mapping, metadata
