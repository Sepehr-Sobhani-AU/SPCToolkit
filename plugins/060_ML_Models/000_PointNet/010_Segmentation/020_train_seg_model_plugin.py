"""
Train PointNet Segmentation Model Plugin (PyTorch)

Trains a PointNet segmentation model for per-point semantic labeling using
training data (.npz files) from a specified directory.

Training runs in a background thread with QTimer polling to keep the UI responsive.
"""

import gc
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from models.pointnet.pointnet_seg_model import PointNetSegmentation
from plugins.dialogs.training_progress_window import TrainingProgressWindow


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance in segmentation.

    Down-weights well-classified examples so the model focuses on hard/rare points.
    """

    def __init__(self, weight=None, gamma=2.0, ignore_index=-100, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction='none', ignore_index=self.ignore_index,
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        # Average only over non-ignored elements
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return focal.sum() * 0.0
        return focal[valid_mask].mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation — optimizes region overlap directly.

    Robust to class imbalance since it measures per-class overlap rather than
    per-pixel correctness.
    """

    def __init__(self, ignore_index=-100, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes).float()  # (N, C)

        intersection = (probs * one_hot).sum(dim=0)
        cardinality = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedFocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss — balances pixel-level and region-level learning."""

    def __init__(self, weight=None, gamma=2.0, ignore_index=-100, label_smoothing=0.1,
                 focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(weight=weight, gamma=gamma,
                               ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.focal_weight * self.focal(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


class SegPointCloudDataset(Dataset):
    """PyTorch Dataset for segmentation training data (.npz files with features + labels)."""

    def __init__(self, file_paths, num_points, num_classes, augment=False,
                 feature_mean=None, feature_std=None, ignore_classes=None,
                 label_remap=None):
        """
        Args:
            file_paths: List of .npz file paths, each containing 'features' (N,F) and 'labels' (N,)
            num_points: Target number of points per sample (random subsample)
            num_classes: Number of valid dense classes (after remap)
            augment: Whether to apply data augmentation
            feature_mean: Optional per-feature mean array (F,) for standardization
            feature_std: Optional per-feature std array (F,) for standardization
            ignore_classes: Optional list of dense class IDs to remap to -100 (ignored by loss)
            label_remap: Dict {original_id: dense_id} to remap sparse labels to dense 0..N-1
        """
        self.num_points = num_points
        self.num_classes = num_classes
        self.augment = augment
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.ignore_classes = set(ignore_classes) if ignore_classes else set()
        self.label_remap = label_remap
        # Pre-load all data into memory to avoid per-access disk I/O
        # (concurrent np.load in forked workers causes segfaults with large datasets)
        self.all_features = []
        self.all_labels = []
        for fp in file_paths:
            data = np.load(fp)
            self.all_features.append(data['features'].astype(np.float32))
            labels = data['labels'].astype(np.int64)
            # Remap sparse original IDs to dense 0..N-1 during pre-load
            remapped = np.full_like(labels, -100)  # unmapped → ignored
            for src, dst in self.label_remap.items():
                remapped[labels == src] = dst
            self.all_labels.append(remapped)

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, idx):
        features = self.all_features[idx].copy()
        labels = self.all_labels[idx].copy()

        # Remap ignored classes to -100 (PyTorch ignore_index)
        if self.ignore_classes:
            for cls_id in self.ignore_classes:
                labels[labels == cls_id] = -100

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

        # Per-feature standardization
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std

        return torch.FloatTensor(features), torch.LongTensor(labels)

    def _augment(self, features):
        """Apply random rotation, scaling, and jitter to XYZ and normals."""
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Random isotropic scaling
        scale = np.random.uniform(0.9, 1.1)

        xyz = features[:, :3]
        rotated_xyz = (xyz @ rotation_matrix.T) * scale
        jitter = np.random.normal(0, 0.02, rotated_xyz.shape).astype(np.float32)
        rotated_xyz = rotated_xyz + jitter

        parts = [rotated_xyz]

        if features.shape[1] >= 6:
            # Rotate normals (columns 3:6) by the same rotation (no scaling — unit vectors)
            normals = features[:, 3:6]
            rotated_normals = normals @ rotation_matrix.T
            parts.append(rotated_normals)
            # Eigenvalues are rotation-invariant but scale with point scale
            if features.shape[1] > 6:
                eigenvalues = features[:, 6:]
                parts.append(eigenvalues * (scale ** 2))
        elif features.shape[1] > 3:
            parts.append(features[:, 3:])

        return np.concatenate(parts, axis=1)


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
        "loss_function": "Focal + Class Weights",
        "scheduler": "OneCycleLR",
        "use_scene_validation": True,
        "val_data_dir": "training_data_seg_val",
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
                "description": "Fraction of data for validation. When scene validation is on, selects this fraction from the validation folder."
            },
            "use_scene_validation": {
                "type": "bool",
                "default": self.last_params["use_scene_validation"],
                "label": "Use Scene Validation",
                "description": "Use a separate scene folder for validation instead of random split"
            },
            "val_data_dir": {
                "type": "directory",
                "default": self.last_params["val_data_dir"],
                "label": "Validation Data Directory",
                "description": "Directory containing .npz files from a different scene for validation",
                "enabled_by": "use_scene_validation"
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
            "loss_function": {
                "type": "dropdown",
                "options": {
                    "Cross Entropy": "Cross Entropy (class weighted)",
                    "Focal Loss": "Focal Loss (gamma=2)",
                    "Focal + Class Weights": "Focal Loss + Class Weights (gamma=2)",
                    "Dice Loss": "Dice Loss",
                    "Focal + Dice": "Focal + Dice (combined)",
                },
                "default": self.last_params["loss_function"],
                "label": "Loss Function",
                "description": "Loss function for training"
            },
            "scheduler": {
                "type": "dropdown",
                "options": {
                    "OneCycleLR": "OneCycleLR (warmup + cosine decay, no restarts)",
                    "CosineAnnealing": "Cosine Annealing (smooth decay to min LR)",
                    "CosineWarmRestarts": "Cosine Annealing + Warm Restarts (periodic LR resets)",
                    "ReduceLROnPlateau": "Reduce on Plateau (halve LR when val mIoU stalls)",
                },
                "default": self.last_params["scheduler"],
                "label": "LR Scheduler",
                "description": "Learning rate scheduling strategy"
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
        loss_function = params.get('loss_function', 'Focal + Class Weights')
        scheduler_name = params.get('scheduler', 'OneCycleLR')
        use_scene_validation = params.get('use_scene_validation', True)
        val_data_dir = params.get('val_data_dir', '').strip()

        TrainSegModelPlugin.last_params = params.copy()

        if not os.path.exists(data_dir):
            QMessageBox.critical(main_window, "Invalid Directory",
                               f"Training data directory does not exist:\n{data_dir}")
            return

        if use_scene_validation and (not val_data_dir or not os.path.exists(val_data_dir)):
            QMessageBox.critical(main_window, "Invalid Directory",
                               f"Validation data directory does not exist:\n{val_data_dir}")
            return

        if not torch.cuda.is_available():
            print("WARNING: CUDA not available — training will fall back to CPU and be significantly slower.")
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

            # Split train/val
            main_window.tree_overlay.show_processing("Splitting data...")

            if use_scene_validation:
                # Scene-based: all training .npz for train, sample from separate folder for val
                train_files = npz_files

                val_npz_files, val_class_mapping, val_metadata, val_num_features = \
                    self._load_training_data(val_data_dir)

                if len(val_npz_files) == 0:
                    QMessageBox.critical(main_window, "No Validation Data",
                                       f"No .npz files found in:\n{val_data_dir}")
                    main_window.tree_overlay.hide_processing()
                    main_window.enable_menus()
                    main_window.enable_tree()
                    return

                if val_num_features != num_features:
                    QMessageBox.critical(main_window, "Feature Mismatch",
                        f"Training data has {num_features} features but "
                        f"validation data has {val_num_features} features.\n\n"
                        f"Both must use the same feature configuration.")
                    main_window.tree_overlay.hide_processing()
                    main_window.enable_menus()
                    main_window.enable_tree()
                    return

                # Merge train + val class mappings (union of classes)
                merged_mapping = dict(class_mapping)
                for vid, vname in val_class_mapping.items():
                    if vid not in merged_mapping:
                        merged_mapping[vid] = vname
                class_mapping = merged_mapping

                # Randomly sample val_split fraction from validation folder
                n_val = max(1, int(len(val_npz_files) * val_split))
                np.random.shuffle(val_npz_files)
                val_files = val_npz_files[:n_val]

                print(f"Validation mode: Scene-based (separate folder)")
                print(f"  Train dir: {data_dir} ({len(train_files)} samples)")
                print(f"  Val dir:   {val_data_dir} ({len(val_npz_files)} total, "
                      f"{len(val_files)} sampled at {val_split:.0%})")
            else:
                # Random split from single folder
                train_files, val_files = train_test_split(
                    npz_files,
                    test_size=val_split,
                    random_state=random_seed
                )
                print(f"Validation mode: Random split ({val_split:.0%})")
                print(f"  Training samples: {len(train_files)}")
                print(f"  Validation samples: {len(val_files)}")

            # Build dense remap: original (possibly sparse) IDs -> sequential [0, N-1]
            sorted_class_ids = sorted(class_mapping.keys())
            original_to_dense = {orig_id: dense_id for dense_id, orig_id in enumerate(sorted_class_ids)}
            dense_class_mapping = {dense_id: class_mapping[orig_id] for orig_id, dense_id in original_to_dense.items()}
            num_classes = len(dense_class_mapping)
            # Replace class_mapping with dense version for all downstream use
            class_mapping = dense_class_mapping

            print(f"\nDataset: {len(npz_files)} samples")
            print(f"Classes: {num_classes} - {class_mapping}")
            print(f"Features per point: {num_features}")
            print(f"Points per block: {num_points}")
            if sorted_class_ids != list(range(num_classes)):
                print(f"Label remap: {original_to_dense}")

            # Compute per-feature standardization stats from training set
            main_window.tree_overlay.show_processing("Computing feature statistics...")
            feature_mean, feature_std = self._compute_feature_stats(train_files)
            print(f"\nPer-feature stats (mean / std):")
            feature_names = ['X_norm', 'Y_norm', 'Z_norm', 'Nx', 'Ny', 'Nz', 'E1', 'E2', 'E3']
            for i in range(len(feature_mean)):
                fname = feature_names[i] if i < len(feature_names) else f"F{i}"
                print(f"  {fname}: mean={feature_mean[i]:.6f}  std={feature_std[i]:.6f}")

            # Find and configure ignored classes
            ignore_classes = self._find_ignore_classes(class_mapping)
            if ignore_classes:
                ignored_names = [class_mapping.get(c, f"Class_{c}") for c in ignore_classes]
                print(f"\nIgnoring classes: {ignore_classes} ({', '.join(ignored_names)})")
                print(f"  These will be excluded from loss computation and mIoU evaluation.")
            else:
                print(f"\nNo classes to ignore.")

            # Compute class weights from training set
            main_window.tree_overlay.show_processing("Computing class weights...")
            class_weights = self._compute_class_weights(train_files, num_classes, label_remap=original_to_dense)
            # Zero out weights for ignored classes
            for cls_id in ignore_classes:
                if 0 <= cls_id < num_classes:
                    class_weights[cls_id] = 0.0
            class_weights_tensor = torch.FloatTensor(class_weights).to(device)

            print(f"\nClass weights:")
            for cid, weight in enumerate(class_weights):
                name = class_mapping.get(cid, f"Class_{cid}")
                ignored_tag = " (IGNORED)" if cid in ignore_classes else ""
                print(f"  {name}: {weight:.3f}{ignored_tag}")

            # Create datasets and dataloaders
            train_dataset = SegPointCloudDataset(
                train_files, num_points, num_classes, augment=True,
                feature_mean=feature_mean, feature_std=feature_std,
                ignore_classes=ignore_classes, label_remap=original_to_dense)
            val_dataset = SegPointCloudDataset(
                val_files, num_points, num_classes, augment=False,
                feature_mean=feature_mean, feature_std=feature_std,
                ignore_classes=ignore_classes, label_remap=original_to_dense)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
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

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
            if scheduler_name == "OneCycleLR":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=learning_rate, epochs=epochs,
                    steps_per_epoch=len(train_loader), pct_start=0.05,
                    anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0)
                step_scheduler_per_batch = True
            elif scheduler_name == "CosineAnnealing":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)
                step_scheduler_per_batch = False
            elif scheduler_name == "CosineWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=20, T_mult=2, eta_min=1e-6)
                step_scheduler_per_batch = False
            else:  # ReduceLROnPlateau
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-5)
                step_scheduler_per_batch = False
            # Build loss function based on user selection
            if loss_function == "Cross Entropy":
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights_tensor, ignore_index=-100, label_smoothing=0.1)
            elif loss_function == "Focal Loss":
                criterion = FocalLoss(
                    weight=None, gamma=2.0, ignore_index=-100, label_smoothing=0.1)
            elif loss_function == "Dice Loss":
                criterion = DiceLoss(ignore_index=-100)
            elif loss_function == "Focal + Dice":
                criterion = CombinedFocalDiceLoss(
                    weight=class_weights_tensor, gamma=2.0, ignore_index=-100,
                    label_smoothing=0.1)
            else:  # "Focal + Class Weights" (default)
                criterion = FocalLoss(
                    weight=class_weights_tensor, gamma=2.0, ignore_index=-100,
                    label_smoothing=0.1)
            print(f"Loss function: {loss_function}")
            print(f"LR Scheduler: {scheduler_name}")

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
            'best_val_per_class_iou': {},
            'final_val_per_class_iou': {},
        }

        def _train_loop():
            """Training loop running in background thread."""
            torch.backends.cudnn.benchmark = True
            scaler = torch.amp.GradScaler()
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
                    train_batch_count = 0
                    train_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

                    for batch_features, batch_labels in train_loader:
                        if progress_window.training_cancelled:
                            training_state['was_cancelled'] = True
                            break

                        batch_features = batch_features.to(device, non_blocking=True)
                        batch_labels = batch_labels.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast('cuda'):
                            logits = model(batch_features)  # (B, N, C)
                            B, N, C = logits.shape
                            loss = criterion(logits.reshape(B * N, C), batch_labels.reshape(B * N))

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        if step_scheduler_per_batch:
                            scheduler.step()

                        train_loss += loss.item() * B
                        train_batch_count += B
                        preds = torch.argmax(logits.detach(), dim=2)
                        valid = batch_labels != -100
                        train_correct += (preds[valid] == batch_labels[valid]).sum().item()
                        train_total += valid.sum().item()

                        TrainSegModelPlugin._update_confusion_matrix(
                            train_confusion, preds, batch_labels, num_classes)

                        del logits, loss, preds, valid

                    if training_state['was_cancelled']:
                        break

                    train_loss /= max(train_batch_count, 1)
                    train_acc = train_correct / max(train_total, 1)
                    train_miou, train_per_class_iou = TrainSegModelPlugin._miou_from_confusion(train_confusion, ignore_classes)

                    # Validation phase
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    val_batch_count = 0
                    val_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        for batch_features, batch_labels in val_loader:
                            batch_features = batch_features.to(device, non_blocking=True)
                            batch_labels = batch_labels.to(device, non_blocking=True)

                            logits = model(batch_features)
                            B, N, C = logits.shape
                            loss = criterion(logits.reshape(B * N, C), batch_labels.reshape(B * N))

                            val_loss += loss.item() * B
                            val_batch_count += B
                            preds = torch.argmax(logits, dim=2)
                            valid = batch_labels != -100
                            val_correct += (preds[valid] == batch_labels[valid]).sum().item()
                            val_total += valid.sum().item()

                            TrainSegModelPlugin._update_confusion_matrix(
                                val_confusion, preds, batch_labels, num_classes)

                            del logits, loss, preds, valid

                    val_loss /= max(val_batch_count, 1)
                    val_acc = val_correct / max(val_total, 1)
                    val_miou, val_per_class_iou = TrainSegModelPlugin._miou_from_confusion(val_confusion, ignore_classes)

                    # Free GPU cache between epochs
                    torch.cuda.empty_cache()

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

                    # Print per-class IoU breakdown
                    iou_parts = []
                    for cid in sorted(val_per_class_iou.keys()):
                        name = class_mapping.get(cid, f"C{cid}")
                        iou_parts.append(f"{name}: {val_per_class_iou[cid]:.3f}")
                    if iou_parts:
                        print(f"  Per-class IoU: {' | '.join(iou_parts)}")

                    # Always track final per-class IoU
                    training_state['final_val_per_class_iou'] = val_per_class_iou.copy()

                    # Check improvement (track mIoU)
                    if val_miou > best_val_miou:
                        best_val_miou = val_miou
                        training_state['best_val_miou'] = best_val_miou
                        training_state['best_val_per_class_iou'] = val_per_class_iou.copy()
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
                            'task_type': 'segmentation',
                            'feature_mean': feature_mean.tolist(),
                            'feature_std': feature_std.tolist(),
                            'ignore_classes': ignore_classes,
                            'original_to_dense': original_to_dense,
                        }, os.path.join(unique_output_dir, 'seg_model_best.pt'))
                    else:
                        epochs_without_improvement += 1

                    if not step_scheduler_per_batch:
                        if scheduler_name == "ReduceLROnPlateau":
                            scheduler.step(val_miou)
                        else:
                            scheduler.step()

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
                    'history': history,
                    'feature_mean': feature_mean.tolist(),
                    'feature_std': feature_std.tolist(),
                    'ignore_classes': ignore_classes,
                    'original_to_dense': original_to_dense,
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
                    'source_metadata': metadata,
                    'feature_mean': feature_mean.tolist(),
                    'feature_std': feature_std.tolist(),
                    'ignore_classes': ignore_classes,
                    'original_to_dense': {str(k): v for k, v in original_to_dense.items()},
                    'best_per_class_iou': {
                        class_mapping.get(cid, f"Class_{cid}"): float(iou)
                        for cid, iou in training_state['best_val_per_class_iou'].items()
                    },
                    'final_per_class_iou': {
                        class_mapping.get(cid, f"Class_{cid}"): float(iou)
                        for cid, iou in training_state['final_val_per_class_iou'].items()
                    },
                }

                with open(os.path.join(unique_output_dir, 'training_metadata.json'), 'w') as f:
                    json.dump(training_metadata, f, indent=2)

                # Write per-class IoU CSV (non-fatal — don't block training completion)
                try:
                    best_pci = training_state['best_val_per_class_iou']
                    final_pci = training_state['final_val_per_class_iou']
                    csv_path = os.path.join(unique_output_dir, 'per_class_iou.csv')
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['class_id', 'class_name', 'iou_at_best_miou', 'final_iou'])
                        for cid in sorted(set(list(best_pci.keys()) + list(final_pci.keys()))):
                            if cid in ignore_classes:
                                continue
                            name = class_mapping.get(cid, f"Class_{cid}")
                            writer.writerow([cid, name,
                                            f"{best_pci.get(cid, 0.0):.6f}",
                                            f"{final_pci.get(cid, 0.0):.6f}"])
                        # Summary row
                        best_miou = sum(best_pci.values()) / max(len(best_pci), 1)
                        final_miou = sum(final_pci.values()) / max(len(final_pci), 1)
                        writer.writerow(['', 'mIoU', f"{best_miou:.6f}", f"{final_miou:.6f}"])
                    print(f"Per-class IoU saved to: {csv_path}")
                except Exception as csv_err:
                    print(f"Warning: Failed to save per-class IoU CSV: {csv_err}")

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

                try:
                    progress_window.update_epoch(
                        epoch, result['train_loss'], result['train_acc'],
                        result['val_loss'], result['val_acc'],
                        learning_rate=result['learning_rate'],
                        train_miou=result['train_miou'],
                        val_miou=result['val_miou']
                    )
                except Exception as e:
                    print(f"Warning: progress window update failed at epoch {epoch}: {e}")

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
            # Drain any remaining epoch results the poll timer missed
            while training_state['epoch_results']:
                result = training_state['epoch_results'].pop(0)
                epoch = result['epoch']
                try:
                    progress_window.update_epoch(
                        epoch, result['train_loss'], result['train_acc'],
                        result['val_loss'], result['val_acc'],
                        learning_rate=result['learning_rate'],
                        train_miou=result['train_miou'],
                        val_miou=result['val_miou']
                    )
                except Exception as e:
                    print(f"Warning: progress window update failed at epoch {epoch}: {e}")
                is_new_best = result['val_miou'] >= training_state['best_val_miou']
                best_marker = f"  -> New best model (val_mIoU: {result['val_miou']:.5f})" if is_new_best else ""
                print(f"Epoch {epoch:3d}/{epochs} - "
                      f"loss: {result['train_loss']:.4f} - acc: {result['train_acc']:.4f} - mIoU: {result['train_miou']:.4f} - "
                      f"val_loss: {result['val_loss']:.4f} - val_acc: {result['val_acc']:.4f} - val_mIoU: {result['val_miou']:.4f}"
                      f"{best_marker}")

            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

            history = training_state['history']
            best_val_miou = training_state['best_val_miou']
            was_cancelled = training_state['was_cancelled']
            error = training_state['error']
            epochs_completed = len(history['loss'])

            # Always mark dialog as completed first so it's closeable
            if error:
                progress_window.training_completed(0.0, cancelled=False)
                progress_window.status_label.setText("Training failed")
            else:
                progress_window.training_completed(best_val_miou, cancelled=was_cancelled)

            # Save screenshot (after dialog shows final state)
            progress_window.save_snapshot(
                os.path.join(unique_output_dir, 'training_progress.png'))

            if error:
                print(f"\nERROR during training:\n{error}")
                QMessageBox.critical(progress_window, "Training Error",
                                   f"An error occurred:\n\n{error}")
            elif was_cancelled:
                print(f"\n{'='*80}")
                print("Training Cancelled!")
                print(f"{'='*80}")
                print(f"Best val mIoU: {best_val_miou:.4f}")
                print(f"Epochs completed: {epochs_completed}")
                print(f"Model saved to: {unique_output_dir}/")
                print(f"{'='*80}")
                QMessageBox.information(progress_window, "Training Cancelled",
                    f"Training cancelled.\n\n"
                    f"Best val mIoU: {best_val_miou:.2%}\n"
                    f"Epochs: {epochs_completed}\n\n"
                    f"Saved to:\n{unique_output_dir}/")
            else:
                print(f"\n{'='*80}")
                print("Training Complete!")
                print(f"{'='*80}")
                print(f"Best val mIoU: {best_val_miou:.4f}")
                print(f"Epochs completed: {epochs_completed}")
                print(f"Model saved to: {unique_output_dir}/")
                print(f"{'='*80}")
                QMessageBox.information(progress_window, "Training Complete",
                    f"Segmentation training completed!\n\n"
                    f"Best val mIoU: {best_val_miou:.2%}\n"
                    f"Epochs: {epochs_completed}\n\n"
                    f"Saved to:\n{unique_output_dir}/")

            # Release RAM and VRAM — drop closure references to large objects
            nonlocal model, optimizer, scheduler, criterion
            nonlocal train_dataset, val_dataset, train_loader, val_loader
            nonlocal class_weights_tensor
            model = None
            optimizer = None
            scheduler = None
            criterion = None
            train_dataset = None
            val_dataset = None
            train_loader = None
            val_loader = None
            class_weights_tensor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("Training memory released (model, datasets, CUDA cache).")

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

    def _compute_feature_stats(self, file_paths):
        """Compute per-feature mean and std from a subset of training files.

        Returns:
            (feature_mean, feature_std) as numpy arrays of shape (F,)
        """
        sample_files = file_paths[:min(200, len(file_paths))]
        all_features = []

        for fp in sample_files:
            data = np.load(fp)
            all_features.append(data['features'].astype(np.float32))

        combined = np.vstack(all_features)
        feature_mean = np.mean(combined, axis=0).astype(np.float32)
        feature_std = np.std(combined, axis=0).astype(np.float32)
        feature_std = np.maximum(feature_std, 1e-6)

        return feature_mean, feature_std

    def _find_ignore_classes(self, class_mapping):
        """Find class IDs for semantically meaningless classes like 'unlabeled' or 'outlier'.

        Returns:
            List of class IDs to ignore during training.
        """
        ignore_names = {'unlabeled', 'outlier'}
        ignore_classes = []
        for cls_id, name in class_mapping.items():
            if name.lower().strip() in ignore_names:
                ignore_classes.append(cls_id)
        return sorted(ignore_classes)

    def _compute_class_weights(self, file_paths, num_classes, label_remap=None):
        """Compute class weights from training files for balanced loss.

        Args:
            file_paths: List of .npz file paths
            num_classes: Number of dense classes
            label_remap: Dict {original_id: dense_id} to remap labels before counting
        """
        # Count labels across ALL files (reading just labels is cheap)
        class_counts = np.zeros(num_classes, dtype=np.int64)
        for f in file_paths:
            data = np.load(f)
            labels = data['labels'].flatten()
            remapped = np.full_like(labels, -1)
            for src, dst in label_remap.items():
                remapped[labels == src] = dst
            labels = remapped
            for c in range(num_classes):
                class_counts[c] += np.sum(labels == c)

        total = class_counts.sum()
        present_mask = class_counts > 0
        n_present = present_mask.sum()

        # Balanced weights for classes that exist: n_samples / (n_classes * count)
        full_weights = np.ones(num_classes, dtype=np.float32)
        full_weights[present_mask] = total / (n_present * class_counts[present_mask])

        # Cap extreme weights — very rare classes get weights of 100+
        # which causes gradient instability. Use sqrt-dampened weights instead.
        median_w = np.median(full_weights[present_mask])
        full_weights = np.sqrt(full_weights / median_w) * median_w

        # Classes missing from the entire dataset get the max computed weight
        if not np.all(present_mask):
            max_w = full_weights[present_mask].max()
            full_weights[~present_mask] = max_w

        full_weights = np.clip(full_weights, 0.1, 20.0).astype(np.float32)

        return full_weights

    @staticmethod
    def _update_confusion_matrix(confusion, preds, labels, num_classes):
        """Add batch predictions to a running (num_classes, num_classes) confusion matrix.

        Args:
            confusion: np.ndarray of shape (num_classes, num_classes), modified in-place
            preds: torch.Tensor of predicted class indices
            labels: torch.Tensor of ground-truth class indices
            num_classes: int
        """
        p = preds.cpu().numpy().flatten()
        l = labels.cpu().numpy().flatten()
        # Only count valid labels
        mask = (l >= 0) & (l < num_classes) & (p >= 0) & (p < num_classes)
        indices = l[mask] * num_classes + p[mask]
        confusion += np.bincount(
            indices, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

    @staticmethod
    def _miou_from_confusion(confusion, ignore_classes=None):
        """Compute per-class IoU from a confusion matrix and return (mIoU, per_class_iou).

        IoU_c = TP_c / (TP_c + FP_c + FN_c)

        Args:
            confusion: np.ndarray of shape (num_classes, num_classes)
                       confusion[i, j] = # points with true class i predicted as class j
            ignore_classes: Optional set of class IDs to exclude from mIoU computation

        Returns:
            (miou, per_class_iou) where per_class_iou is a dict {class_id: iou} for classes
            with non-zero support. miou is the mean across those classes.
        """
        ignore_set = set(ignore_classes) if ignore_classes else set()
        num_classes = confusion.shape[0]
        per_class_iou = {}
        for c in range(num_classes):
            if c in ignore_set:
                continue
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            denom = tp + fp + fn
            if denom > 0:
                per_class_iou[c] = tp / denom
        if len(per_class_iou) == 0:
            return 0.0, {}
        miou = sum(per_class_iou.values()) / len(per_class_iou)
        return miou, per_class_iou
