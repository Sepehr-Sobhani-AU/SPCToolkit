"""
Train PointNet Segmentation Model Plugin (PyTorch)

Trains a PointNet segmentation model for per-point semantic labeling using
training data (.npz files) from a specified directory.

Training runs in a background thread with QTimer polling to keep the UI responsive.
"""

import os
import json
import csv
import time
import threading
import numpy as np
from typing import Dict, Any
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtWidgets, QtCore
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from models.pointnet.pointnet_seg_model import PointNetSegmentation
from plugins.dialogs.training_progress_window import TrainingProgressWindow


class SegPointCloudDataset(Dataset):
    """PyTorch Dataset for segmentation training data (.npz files with features + labels)."""

    def __init__(self, file_paths, num_points, num_classes, augment=False):
        """
        Args:
            file_paths: List of .npz file paths, each containing 'features' (N,F) and 'labels' (N,)
            num_points: Target number of points per sample (random subsample)
            num_classes: Number of valid classes (labels clamped to [0, num_classes-1])
            augment: Whether to apply data augmentation
        """
        self.num_points = num_points
        self.num_classes = num_classes
        self.augment = augment
        # Pre-load all data into memory to avoid per-access disk I/O
        # (concurrent np.load in forked workers causes segfaults with large datasets)
        self.all_features = []
        self.all_labels = []
        for fp in file_paths:
            data = np.load(fp)
            self.all_features.append(data['features'].astype(np.float32))
            self.all_labels.append(data['labels'].astype(np.int64))

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, idx):
        features = self.all_features[idx]
        labels = self.all_labels[idx]

        # Clamp labels to valid range for CrossEntropyLoss
        np.clip(labels, 0, self.num_classes - 1, out=labels)

        # Random subsample to fixed number of points
        num_available = features.shape[0]
        if num_available >= self.num_points:
            indices = np.random.choice(num_available, self.num_points, replace=False)
        else:
            indices = np.random.choice(num_available, self.num_points, replace=True)

        features = features[indices]
        labels = labels[indices]

        if self.augment:
            features = self._augment(features)

        return torch.FloatTensor(features), torch.LongTensor(labels)

    def _augment(self, features):
        """Apply random rotation around Z-axis and jitter to XYZ."""
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        xyz = features[:, :3]
        other = features[:, 3:]

        rotated_xyz = xyz @ rotation_matrix.T
        jitter = np.random.normal(0, 0.01, rotated_xyz.shape).astype(np.float32)
        rotated_xyz = rotated_xyz + jitter

        return np.concatenate([rotated_xyz, other], axis=1)


class TrainSegModelPlugin(ActionPlugin):
    """Action plugin for training PointNet segmentation model."""

    last_params = {
        "training_data_dir": "training_data_seg",
        "output_dir": "models",
        "num_points": 4096,
        "epochs": 100,
        "batch_size": 4,
        "learning_rate": 0.001,
        "val_split": 0.2,
        "use_tnet": True,
        "early_stopping_patience": 20,
    }

    def get_name(self) -> str:
        return "train_seg_model"

    def _get_next_run_number(self, output_dir: str) -> int:
        """Get the next run number by scanning existing folders."""
        if not os.path.exists(output_dir):
            return 1
        run_numbers = []
        for item in os.listdir(output_dir):
            if os.path.isdir(os.path.join(output_dir, item)) and item.startswith('seg_run_'):
                try:
                    parts = item.split('_')
                    if len(parts) >= 3:
                        run_num = int(parts[2])
                        run_numbers.append(run_num)
                except (ValueError, IndexError):
                    continue
        return max(run_numbers) + 1 if run_numbers else 1

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "training_data_dir": {
                "type": "directory",
                "default": self.last_params["training_data_dir"],
                "label": "Training Data Directory",
                "description": "Directory containing .npz files with 'features' and 'labels' arrays"
            },
            "output_dir": {
                "type": "directory",
                "default": self.last_params["output_dir"],
                "label": "Output Directory",
                "description": "Directory to save trained model"
            },
            "num_points": {
                "type": "int",
                "default": self.last_params["num_points"],
                "min": 512,
                "max": 16384,
                "label": "Points Per Block",
                "description": "Number of points per training block (must match inference)"
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
                "max": 64,
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
                "description": "Fraction of data for validation"
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
                "description": "Epochs without improvement before stopping"
            },
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Execute the segmentation training process."""
        data_dir = params['training_data_dir'].strip()
        output_dir = params['output_dir'].strip()
        num_points = int(params['num_points'])
        epochs = int(params['epochs'])
        batch_size = int(params['batch_size'])
        learning_rate = float(params['learning_rate'])
        val_split = float(params['val_split'])
        use_tnet = params['use_tnet']
        early_stopping_patience = int(params['early_stopping_patience'])

        TrainSegModelPlugin.last_params = params.copy()

        if not os.path.exists(data_dir):
            QMessageBox.critical(main_window, "Invalid Directory",
                               f"Training data directory does not exist:\n{data_dir}")
            return

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        run_number = self._get_next_run_number(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"seg_run_{run_number:03d}_{timestamp}"
        unique_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(unique_output_dir, exist_ok=True)

        random_seed = int(time.time() * 1000) % (2**32)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)

        print(f"\n{'='*80}")
        print(f"PointNet Segmentation Training - Run #{run_number}")
        print(f"{'='*80}")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Random seed: {random_seed}")
        print(f"Output: {unique_output_dir}")

        # --- Setup phase (main thread, fast) ---
        try:
            main_window.disable_menus()
            main_window.disable_tree()
            main_window.tree_overlay.show_processing("Loading training data...")

            # Load training data
            npz_files, class_mapping, metadata, num_features = self._load_training_data(data_dir)

            if len(npz_files) == 0:
                QMessageBox.critical(main_window, "No Data",
                                   f"No .npz files found in:\n{data_dir}")
                main_window.tree_overlay.hide_processing()
                main_window.enable_menus()
                main_window.enable_tree()
                return

            num_classes = len(class_mapping)

            # Validate label ranges in a sample of files
            label_min, label_max = 0, 0
            for f in npz_files[:min(20, len(npz_files))]:
                sample_labels = np.load(f)['labels']
                label_min = min(label_min, sample_labels.min())
                label_max = max(label_max, sample_labels.max())
            if label_min < 0 or label_max >= num_classes:
                print(f"WARNING: Labels out of range! Found [{label_min}, {label_max}] "
                      f"but num_classes={num_classes}. Labels will be clamped to [0, {num_classes-1}].")

            print(f"\nDataset: {len(npz_files)} samples")
            print(f"Classes: {num_classes} - {class_mapping}")
            print(f"Features per point: {num_features}")
            print(f"Points per block: {num_points}")
            print(f"Label range in data: [{label_min}, {label_max}]")

            # Split train/val
            main_window.tree_overlay.show_processing("Splitting data...")

            train_files, val_files = train_test_split(
                npz_files,
                test_size=val_split,
                random_state=random_seed
            )

            print(f"Training samples: {len(train_files)}")
            print(f"Validation samples: {len(val_files)}")

            # Compute class weights from training set
            main_window.tree_overlay.show_processing("Computing class weights...")
            class_weights = self._compute_class_weights(train_files, num_classes)
            class_weights_tensor = torch.FloatTensor(class_weights).to(device)

            print(f"\nClass weights:")
            for cid, weight in enumerate(class_weights):
                name = class_mapping.get(cid, f"Class_{cid}")
                print(f"  {name}: {weight:.3f}")

            # Create datasets and dataloaders
            train_dataset = SegPointCloudDataset(train_files, num_points, num_classes, augment=True)
            val_dataset = SegPointCloudDataset(val_files, num_points, num_classes, augment=False)

            # num_workers=0: data is pre-loaded in memory so there's no I/O benefit
            # from multiprocessing, and forked workers cause segfaults with large datasets
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )

            # Create model
            main_window.tree_overlay.show_processing("Creating model...")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = PointNetSegmentation(
                num_points=num_points,
                num_features=num_features,
                num_classes=num_classes,
                use_tnet=use_tnet
            ).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nModel: {total_params:,} parameters")
            print(f"T-Net: {use_tnet}")

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=10
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

            # Create progress window
            progress_window = TrainingProgressWindow(parent=main_window, total_epochs=epochs)
            progress_window.setWindowTitle(f"Segmentation Training - Run #{run_number}")
            progress_window.show()
            progress_window.training_started()

            screen_geometry = QtWidgets.QApplication.desktop().screenGeometry()
            x = (screen_geometry.width() - progress_window.width()) // 2
            y = (screen_geometry.height() - progress_window.height()) // 2
            progress_window.move(x, y)

            main_window.tree_overlay.show_processing("Training model...")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(main_window, "Setup Error",
                               f"Failed to set up training:\n\n{str(e)}")
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()
            return

        # --- Background training thread ---
        # Shared state between thread and main thread (GIL-safe for simple append/read)
        training_state = {
            'epoch_results': [],   # thread appends, main thread pops
            'is_running': True,
            'error': None,
            'history': {'loss': [], 'acc': [], 'miou': [], 'val_loss': [], 'val_acc': [], 'val_miou': []},
            'best_val_miou': 0.0,
            'was_cancelled': False,
            'last_epoch': 0,
            'last_val_miou': 0.0,
            'last_val_acc': 0.0,
        }

        def _train_loop():
            """Training loop running in background thread."""
            history = training_state['history']
            best_val_miou = 0.0
            epochs_without_improvement = 0

            try:
                for epoch in range(epochs):
                    if progress_window.training_cancelled:
                        training_state['was_cancelled'] = True
                        break

                    # Training phase
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    train_iou_sum = 0.0
                    train_iou_count = 0

                    for batch_features, batch_labels in train_loader:
                        if progress_window.training_cancelled:
                            training_state['was_cancelled'] = True
                            break

                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)

                        optimizer.zero_grad()
                        logits = model(batch_features)  # (B, N, C)

                        B, N, C = logits.shape
                        loss = criterion(logits.reshape(B * N, C), batch_labels.reshape(B * N))

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item() * B
                        preds = torch.argmax(logits, dim=2)
                        train_correct += (preds == batch_labels).sum().item()
                        train_total += B * N

                        miou = TrainSegModelPlugin._compute_miou(preds, batch_labels, num_classes)
                        if miou is not None:
                            train_iou_sum += miou * B
                            train_iou_count += B

                    if training_state['was_cancelled']:
                        break

                    num_batches = max(train_iou_count, 1)
                    train_loss /= num_batches
                    train_acc = train_correct / max(train_total, 1)
                    train_miou = train_iou_sum / num_batches

                    # Validation phase
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    val_iou_sum = 0.0
                    val_iou_count = 0

                    with torch.no_grad():
                        for batch_features, batch_labels in val_loader:
                            batch_features = batch_features.to(device)
                            batch_labels = batch_labels.to(device)

                            logits = model(batch_features)
                            B, N, C = logits.shape
                            loss = criterion(logits.reshape(B * N, C), batch_labels.reshape(B * N))

                            val_loss += loss.item() * B
                            preds = torch.argmax(logits, dim=2)
                            val_correct += (preds == batch_labels).sum().item()
                            val_total += B * N

                            miou = TrainSegModelPlugin._compute_miou(preds, batch_labels, num_classes)
                            if miou is not None:
                                val_iou_sum += miou * B
                                val_iou_count += B

                    val_batches = max(val_iou_count, 1)
                    val_loss /= val_batches
                    val_acc = val_correct / max(val_total, 1)
                    val_miou = val_iou_sum / val_batches

                    # Update history
                    history['loss'].append(train_loss)
                    history['acc'].append(train_acc)
                    history['miou'].append(train_miou)
                    history['val_loss'].append(val_loss)
                    history['val_acc'].append(val_acc)
                    history['val_miou'].append(val_miou)

                    current_lr = optimizer.param_groups[0]['lr']

                    # Queue epoch result for main thread to display
                    training_state['epoch_results'].append({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_miou': train_miou,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_miou': val_miou,
                        'learning_rate': current_lr,
                    })

                    training_state['last_epoch'] = epoch
                    training_state['last_val_miou'] = val_miou
                    training_state['last_val_acc'] = val_acc

                    # Check improvement (track mIoU)
                    if val_miou > best_val_miou:
                        best_val_miou = val_miou
                        epochs_without_improvement = 0

                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_miou': val_miou,
                            'val_acc': val_acc,
                            'class_mapping': class_mapping,
                            'num_points': num_points,
                            'num_features': num_features,
                            'num_classes': num_classes,
                            'use_tnet': use_tnet,
                            'task_type': 'segmentation'
                        }, os.path.join(unique_output_dir, 'seg_model_best.pt'))
                    else:
                        epochs_without_improvement += 1

                    scheduler.step(val_miou)

                    if epochs_without_improvement >= early_stopping_patience:
                        print(f"\nEarly stopping after {epoch+1} epochs")
                        break

                training_state['best_val_miou'] = best_val_miou

                # Save final model
                torch.save({
                    'epoch': training_state['last_epoch'],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_miou': training_state['last_val_miou'],
                    'val_acc': training_state['last_val_acc'],
                    'class_mapping': class_mapping,
                    'num_points': num_points,
                    'num_features': num_features,
                    'num_classes': num_classes,
                    'use_tnet': use_tnet,
                    'task_type': 'segmentation',
                    'history': history
                }, os.path.join(unique_output_dir, 'seg_model_final.pt'))

                # Save class mapping
                with open(os.path.join(unique_output_dir, 'class_mapping.json'), 'w') as f:
                    json.dump(class_mapping, f, indent=2)

                # Save training metadata
                training_metadata = {
                    'framework': 'PyTorch',
                    'pytorch_version': torch.__version__,
                    'task_type': 'segmentation',
                    'folder_name': folder_name,
                    'timestamp': timestamp,
                    'num_points': num_points,
                    'num_features': num_features,
                    'num_classes': num_classes,
                    'class_mapping': class_mapping,
                    'training_samples': len(train_files),
                    'validation_samples': len(val_files),
                    'epochs_completed': len(history['loss']),
                    'best_val_miou': float(best_val_miou),
                    'final_val_miou': float(history['val_miou'][-1]) if history['val_miou'] else 0.0,
                    'final_val_acc': float(history['val_acc'][-1]) if history['val_acc'] else 0.0,
                    'use_tnet': use_tnet,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'early_stopping_patience': early_stopping_patience,
                    'validation_split': val_split,
                    'random_seed': random_seed,
                    'run_number': run_number,
                    'source_metadata': metadata
                }

                with open(os.path.join(unique_output_dir, 'training_metadata.json'), 'w') as f:
                    json.dump(training_metadata, f, indent=2)

            except RuntimeError as e:
                err_str = str(e)
                is_cuda_error = (
                    "out of memory" in err_str.lower() or
                    "CUDNN" in err_str.upper() or
                    "device-side assert" in err_str.lower()
                )
                if is_cuda_error:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if "device-side assert" in err_str.lower():
                        training_state['error'] = (
                            f"CUDA device-side assert at epoch {training_state['last_epoch']+1}.\n\n"
                            f"This usually means label values are out of range for "
                            f"the number of classes ({num_classes}).\n\n"
                            f"Check that training data labels are in [0, {num_classes-1}]."
                        )
                    else:
                        training_state['error'] = (
                            f"GPU memory exceeded.\n\n"
                            f"Current batch_size={batch_size}, num_points={num_points}.\n"
                            f"Try reducing batch size or points per block."
                        )
                else:
                    training_state['error'] = str(e)
            except Exception as e:
                import traceback
                traceback.print_exc()
                training_state['error'] = str(e)
            finally:
                training_state['is_running'] = False

        # Launch training thread
        thread = threading.Thread(target=_train_loop, daemon=True)
        thread.start()

        # --- QTimer polling (main thread) ---
        poll_timer = QtCore.QTimer()

        def _on_poll():
            """Poll training thread state and update UI."""
            # Process queued epoch results
            while training_state['epoch_results']:
                result = training_state['epoch_results'].pop(0)
                epoch = result['epoch']

                progress_window.update_epoch(
                    epoch, result['train_loss'], result['train_acc'],
                    result['val_loss'], result['val_acc'],
                    learning_rate=result['learning_rate'],
                    train_miou=result['train_miou'],
                    val_miou=result['val_miou']
                )

                is_new_best = result['val_miou'] >= training_state['best_val_miou']
                best_marker = f"  -> New best model (val_mIoU: {result['val_miou']:.5f})" if is_new_best else ""

                print(f"Epoch {epoch:3d}/{epochs} - "
                      f"loss: {result['train_loss']:.4f} - acc: {result['train_acc']:.4f} - mIoU: {result['train_miou']:.4f} - "
                      f"val_loss: {result['val_loss']:.4f} - val_acc: {result['val_acc']:.4f} - val_mIoU: {result['val_miou']:.4f}"
                      f"{best_marker}")

            # Check if thread finished
            if not training_state['is_running']:
                poll_timer.stop()
                _on_training_finished()

        def _on_training_finished():
            """Handle training thread completion on main thread."""
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

            history = training_state['history']
            best_val_miou = training_state['best_val_miou']
            was_cancelled = training_state['was_cancelled']
            error = training_state['error']
            epochs_completed = len(history['loss'])

            if error:
                print(f"\nERROR during training:\n{error}")
                try:
                    progress_window.training_complete = True
                    progress_window.cancel_button.setVisible(False)
                    progress_window.close_button.setVisible(True)
                    progress_window.status_label.setText("Training failed")
                except:
                    pass
                QMessageBox.critical(main_window, "Training Error",
                                   f"An error occurred:\n\n{error}")
                return

            progress_window.training_completed(best_val_miou, cancelled=was_cancelled)

            print(f"\n{'='*80}")
            print("Training Complete!" if not was_cancelled else "Training Cancelled!")
            print(f"{'='*80}")
            print(f"Best val mIoU: {best_val_miou:.4f}")
            print(f"Epochs completed: {epochs_completed}")
            print(f"Model saved to: {unique_output_dir}/")
            print(f"{'='*80}")

            if was_cancelled:
                QMessageBox.information(main_window, "Training Cancelled",
                    f"Training cancelled.\n\n"
                    f"Best val mIoU: {best_val_miou:.2%}\n"
                    f"Epochs: {epochs_completed}\n\n"
                    f"Saved to:\n{unique_output_dir}/")
            else:
                QMessageBox.information(main_window, "Training Complete",
                    f"Segmentation training completed!\n\n"
                    f"Best val mIoU: {best_val_miou:.2%}\n"
                    f"Epochs: {epochs_completed}\n\n"
                    f"Saved to:\n{unique_output_dir}/")

        poll_timer.timeout.connect(_on_poll)
        poll_timer.start(100)

    def _load_training_data(self, data_dir):
        """
        Load training data from directory of .npz files.

        Expected structure:
            data_dir/
                *.npz files (each with 'features' and 'labels' arrays)
                metadata.json (optional, with class_mapping)

        Returns:
            (npz_files, class_mapping, metadata, num_features)
        """
        metadata = {}
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Find all .npz files
        npz_files = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.npz'):
                    npz_files.append(os.path.join(root, f))

        npz_files.sort()

        if len(npz_files) == 0:
            return [], {}, metadata, 0

        # Get class mapping from metadata or infer from data
        class_mapping = metadata.get('class_mapping', {})
        if isinstance(class_mapping, dict):
            class_mapping = {int(k): v for k, v in class_mapping.items()}

        # Determine num_features and num_classes from first file
        sample = np.load(npz_files[0])
        num_features = sample['features'].shape[1]

        if not class_mapping:
            # Infer from data - scan a few files for unique labels
            all_labels = set()
            for f in npz_files[:min(20, len(npz_files))]:
                data = np.load(f)
                all_labels.update(np.unique(data['labels']).tolist())
            class_mapping = {i: f"Class_{i}" for i in sorted(all_labels)}

        return npz_files, class_mapping, metadata, num_features

    def _compute_class_weights(self, file_paths, num_classes):
        """Compute class weights from subset of training files for balanced loss."""
        # Sample a subset for efficiency
        sample_files = file_paths[:min(100, len(file_paths))]
        all_labels = []

        for f in sample_files:
            data = np.load(f)
            all_labels.append(data['labels'].flatten())

        all_labels = np.concatenate(all_labels)
        unique_classes = np.unique(all_labels)

        weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=all_labels
        )

        # Create full weight array (some classes might be missing in sample)
        full_weights = np.ones(num_classes, dtype=np.float32)
        for cls, weight in zip(unique_classes, weights):
            if 0 <= cls < num_classes:
                full_weights[cls] = weight

        return full_weights

    @staticmethod
    def _compute_miou(predictions, labels, num_classes):
        """Compute mean IoU across classes present in batch."""
        iou_list = []
        for cls in range(num_classes):
            pred_mask = (predictions == cls)
            label_mask = (labels == cls)
            intersection = (pred_mask & label_mask).sum().item()
            union = (pred_mask | label_mask).sum().item()
            if union > 0:
                iou_list.append(intersection / union)
        if len(iou_list) == 0:
            return None
        return sum(iou_list) / len(iou_list)
