"""
PointNet++ SSG Semantic Segmentation Model (PyTorch)

Hierarchical point cloud segmentation using Set Abstraction (encoder) and
Feature Propagation (decoder) layers with Single-Scale Grouping.

Based on: PointNet++: Deep Hierarchical Feature Learning on Point Sets
          in a Metric Space, Charles R. Qi et al., NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from core.pointnet_model import ConvBNReLU
from models.pointnet2.pointnet2_ops import (
    furthest_point_sample,
    random_sample,
    ball_query,
    three_nn,
    three_interpolate,
)


class SharedMLP(nn.Module):
    """Shared MLP applied per-point (2D convolution with kernel 1x1)."""

    def __init__(self, channels: list):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 1, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SetAbstraction(nn.Module):
    """
    Set Abstraction layer — the building block of PointNet++ encoder.

    1. Farthest Point Sampling: selects npoint centroids
    2. Ball Query: groups nsample neighbors within radius for each centroid
    3. Shared MLP + MaxPool: extracts a feature vector per group

    When npoint is None, applies global pooling (for the deepest SA layer).
    """

    def __init__(self, npoint, radius, nsample, in_channels, mlp_channels, use_fps=True):
        """
        Args:
            npoint: number of centroids (None = global aggregation)
            radius: ball query radius
            nsample: max neighbors per centroid
            in_channels: input feature channels (excluding XYZ)
            mlp_channels: list of output channels for the shared MLP
            use_fps: if True use Farthest Point Sampling, else random sampling (faster)
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_fps = use_fps

        # MLP input: grouped XYZ (3) + features (in_channels)
        self.mlp = SharedMLP([in_channels + 3] + mlp_channels)

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, C, N) point features (or None)

        Returns:
            new_xyz: (B, npoint, 3) centroid coordinates
            new_features: (B, mlp_out, npoint) aggregated features
        """
        B, N, _ = xyz.shape

        if self.npoint is None:
            # Global aggregation: all points → single output
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            # Group all points: (B, 1, N, 3)
            grouped_xyz = xyz.unsqueeze(1)
            if features is not None:
                # (B, C, N) → (B, C, 1, N) → (B, 3+C, 1, N)
                grouped_features = features.unsqueeze(2)  # (B, C, 1, N)
                grouped = torch.cat([grouped_xyz.permute(0, 3, 1, 2), grouped_features], dim=1)
            else:
                grouped = grouped_xyz.permute(0, 3, 1, 2)  # (B, 3, 1, N)
        else:
            # Select centroids via FPS or random sampling
            if self.use_fps:
                fps_idx = furthest_point_sample(xyz, self.npoint)  # (B, npoint)
            else:
                fps_idx = random_sample(xyz, self.npoint)  # (B, npoint)

            # Gather centroid coordinates
            fps_idx_expand = fps_idx.unsqueeze(-1).expand(-1, -1, 3)
            new_xyz = torch.gather(xyz, 1, fps_idx_expand)  # (B, npoint, 3)

            # Ball query for neighbors
            group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B, npoint, nsample)

            # Group XYZ and normalize to local coordinates
            group_idx_flat = group_idx.reshape(B, -1)  # (B, npoint*nsample)
            grouped_xyz = torch.gather(
                xyz, 1,
                group_idx_flat.unsqueeze(-1).expand(-1, -1, 3)
            ).reshape(B, self.npoint, self.nsample, 3)  # (B, npoint, nsample, 3)

            # Relative coordinates: subtract centroid
            grouped_xyz -= new_xyz.unsqueeze(2)  # (B, npoint, nsample, 3)

            # Group features
            if features is not None:
                C = features.shape[1]
                grouped_feats = torch.gather(
                    features, 2,
                    group_idx_flat.unsqueeze(1).expand(-1, C, -1)
                ).reshape(B, C, self.npoint, self.nsample)  # (B, C, npoint, nsample)
                # Concat: (B, 3+C, npoint, nsample)
                grouped = torch.cat([grouped_xyz.permute(0, 3, 1, 2), grouped_feats], dim=1)
            else:
                grouped = grouped_xyz.permute(0, 3, 1, 2)  # (B, 3, npoint, nsample)

        # Shared MLP: (B, 3+C, npoint, nsample) → (B, mlp_out, npoint, nsample)
        new_features = self.mlp(grouped)

        # Max pool over neighbors: (B, mlp_out, npoint)
        new_features = new_features.max(dim=-1)[0]

        return new_xyz, new_features


class FeaturePropagation(nn.Module):
    """
    Feature Propagation layer — upsamples features for the decoder.

    1. Interpolate features from downsampled points to original resolution
       using inverse-distance weighted 3-NN
    2. Concatenate with skip connection features from the encoder
    3. Apply shared MLP (Conv1d)
    """

    def __init__(self, in_channels, mlp_channels):
        """
        Args:
            in_channels: interpolated channels + skip channels
            mlp_channels: list of output channels for the MLP
        """
        super().__init__()
        layers = []
        prev = in_channels
        for out in mlp_channels:
            layers.append(nn.Conv1d(prev, out, 1, bias=False))
            layers.append(nn.BatchNorm1d(out))
            layers.append(nn.ReLU(inplace=True))
            prev = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz_up, xyz_down, features_up, features_down):
        """
        Args:
            xyz_up: (B, N, 3) higher-resolution point coordinates
            xyz_down: (B, M, 3) lower-resolution point coordinates (M < N)
            features_up: (B, C_skip, N) skip connection features (or None)
            features_down: (B, C_down, M) features to interpolate

        Returns:
            (B, mlp_out, N) upsampled and refined features
        """
        B, N, _ = xyz_up.shape
        M = xyz_down.shape[1]

        if M == 1:
            # Single point (global feature): broadcast to all points
            interpolated = features_down.expand(-1, -1, N)  # (B, C_down, N)
        else:
            # 3-NN interpolation
            dist_sq, idx = three_nn(xyz_up, xyz_down)  # (B, N, 3) each

            # Inverse distance weights
            dist_sq = dist_sq.clamp(min=1e-10)
            weight = 1.0 / dist_sq  # (B, N, 3)
            weight = weight / weight.sum(dim=-1, keepdim=True)  # normalize

            # Interpolate
            interpolated = three_interpolate(features_down, idx, weight)  # (B, C_down, N)

        # Skip connection
        if features_up is not None:
            combined = torch.cat([interpolated, features_up], dim=1)
        else:
            combined = interpolated

        # MLP
        return self.mlp(combined)


class PointNet2SSGSegmentation(nn.Module):
    """
    PointNet++ SSG semantic segmentation model.

    Encoder: 4 Set Abstraction layers (progressively downsample)
    Decoder: 4 Feature Propagation layers (upsample back to original resolution)
    Head: Conv1d classifier

    Args:
        num_points: number of points per sample (for reference, not enforced)
        num_features: number of input features per point (including XYZ)
        num_classes: number of output semantic classes
        block_size: spatial block size in meters (scales ball query radii)
    """

    def __init__(self, num_points=4096, num_features=9, num_classes=6,
                 block_size=10.0, use_fps=True):
        super().__init__()
        self.num_points = num_points
        self.num_features = num_features
        self.num_classes = num_classes
        self.block_size = block_size
        self.use_fps = use_fps

        # Extra features beyond XYZ (normals, eigenvalues, etc.)
        extra_feat = num_features - 3

        # Scale radii proportionally to block size
        scale = block_size / 10.0

        # Encoder — Set Abstraction layers
        self.sa1 = SetAbstraction(
            npoint=num_points // 4,
            radius=0.5 * scale,
            nsample=32,
            in_channels=extra_feat,
            mlp_channels=[64, 64, 128],
            use_fps=use_fps
        )
        self.sa2 = SetAbstraction(
            npoint=num_points // 16,
            radius=1.0 * scale,
            nsample=64,
            in_channels=128,
            mlp_channels=[128, 128, 256],
            use_fps=use_fps
        )
        self.sa3 = SetAbstraction(
            npoint=num_points // 64,
            radius=2.0 * scale,
            nsample=128,
            in_channels=256,
            mlp_channels=[256, 256, 512],
            use_fps=use_fps
        )
        self.sa4 = SetAbstraction(
            npoint=None,  # global aggregation
            radius=None,
            nsample=None,
            in_channels=512,
            mlp_channels=[512, 512, 1024]
        )

        # Decoder — Feature Propagation layers
        self.fp4 = FeaturePropagation(
            in_channels=1024 + 512,  # interpolated + skip from SA3
            mlp_channels=[512, 512]
        )
        self.fp3 = FeaturePropagation(
            in_channels=512 + 256,  # interpolated + skip from SA2
            mlp_channels=[256, 256]
        )
        self.fp2 = FeaturePropagation(
            in_channels=256 + 128,  # interpolated + skip from SA1
            mlp_channels=[128, 128]
        )
        self.fp1 = FeaturePropagation(
            in_channels=128 + extra_feat,  # interpolated + original features
            mlp_channels=[128, 128]
        )

        # Classification head
        self.head_conv = nn.Conv1d(128, 128, 1, bias=False)
        self.head_bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (B, N, F) input points with features

        Returns:
            (B, N, C) per-point class logits
        """
        B, N, num_feat = x.shape

        xyz = x[:, :, :3].contiguous()  # (B, N, 3)

        if num_feat > 3:
            features = x[:, :, 3:].transpose(1, 2).contiguous()  # (B, extra_feat, N)
        else:
            features = None

        # ---- Encoder ----
        xyz0 = xyz
        feat0 = features  # original features for skip connection

        xyz1, feat1 = self.sa1(xyz0, feat0)   # (B, N/4, 3), (B, 128, N/4)
        xyz2, feat2 = self.sa2(xyz1, feat1)   # (B, N/16, 3), (B, 256, N/16)
        xyz3, feat3 = self.sa3(xyz2, feat2)   # (B, N/64, 3), (B, 512, N/64)
        xyz4, feat4 = self.sa4(xyz3, feat3)   # (B, 1, 3), (B, 1024, 1)

        # ---- Decoder ----
        up3 = self.fp4(xyz3, xyz4, feat3, feat4)  # (B, 512, N/64)
        up2 = self.fp3(xyz2, xyz3, feat2, up3)    # (B, 256, N/16)
        up1 = self.fp2(xyz1, xyz2, feat1, up2)    # (B, 128, N/4)
        up0 = self.fp1(xyz0, xyz1, feat0, up1)    # (B, 128, N)

        # ---- Head ----
        out = F.relu(self.head_bn(self.head_conv(up0)))  # (B, 128, N)
        out = self.dropout(out)
        out = self.classifier(out)  # (B, C, N)

        # Transpose to (B, N, C) to match PointNet interface
        return out.transpose(1, 2)

    def predict(self, x):
        """Predict per-point class labels."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=2)

    def predict_proba(self, x):
        """Predict per-point class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=2)


class PointNet2Segmenter:
    """
    Wrapper for PointNet++ SSG segmentation with training and inference.

    Mirrors PointNetSegmenter interface for seamless integration with
    the existing training and inference pipeline.
    """

    def __init__(self, num_points=4096, num_features=9, num_classes=6,
                 block_size=10.0, use_fps=True, device=None):
        self.num_points = num_points
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_fps = use_fps
        self.block_size = block_size

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = PointNet2SSGSegmentation(
            num_points=num_points,
            num_features=num_features,
            num_classes=num_classes,
            block_size=block_size,
            use_fps=use_fps
        ).to(self.device)

        self.optimizer = None
        self.scheduler = None
        self.class_mapping = None
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_miou': [],
            'val_loss': [], 'val_acc': [], 'val_miou': []
        }

    def compile_model(self, learning_rate=0.001):
        """Set up optimizer and scheduler."""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )

    def train(self, train_loader, val_loader=None, epochs=100,
              class_weights=None, callbacks=None, verbose=1):
        """
        Train the segmentation model.

        Args:
            train_loader: DataLoader yielding (features, labels) batches
                features: (B, N, F) float32
                labels: (B, N) int64
            val_loader: Optional validation DataLoader
            epochs: Number of training epochs
            class_weights: Optional class weight tensor for imbalanced data
            callbacks: List of callback objects
            verbose: Verbosity level

        Returns:
            Training history dict
        """
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        best_val_miou = 0.0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_iou_sum = 0.0
            train_iou_count = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()

                logits = self.model(batch_features)  # (B, N, C)

                B_size, N_size, C_size = logits.shape
                loss = criterion(
                    logits.reshape(B_size * N_size, C_size),
                    batch_labels.reshape(B_size * N_size)
                )

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * B_size
                predictions = torch.argmax(logits, dim=2)
                train_correct += (predictions == batch_labels).sum().item()
                train_total += B_size * N_size

                miou = self._compute_miou(predictions, batch_labels, self.num_classes)
                if miou is not None:
                    train_iou_sum += miou * B_size
                    train_iou_count += B_size

            train_loss /= max(train_iou_count, 1)
            train_acc = train_correct / max(train_total, 1)
            train_miou = train_iou_sum / max(train_iou_count, 1)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_miou'].append(train_miou)

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            val_miou = 0.0
            if val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0
                val_iou_sum = 0.0
                val_iou_count = 0

                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        logits = self.model(batch_features)
                        B_size, N_size, C_size = logits.shape
                        loss = criterion(
                            logits.reshape(B_size * N_size, C_size),
                            batch_labels.reshape(B_size * N_size)
                        )

                        val_loss_sum += loss.item() * B_size
                        predictions = torch.argmax(logits, dim=2)
                        val_correct += (predictions == batch_labels).sum().item()
                        val_total += B_size * N_size

                        miou = self._compute_miou(predictions, batch_labels, self.num_classes)
                        if miou is not None:
                            val_iou_sum += miou * B_size
                            val_iou_count += B_size

                val_loss = val_loss_sum / max(val_iou_count, 1)
                val_acc = val_correct / max(val_total, 1)
                val_miou = val_iou_sum / max(val_iou_count, 1)

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_miou'].append(val_miou)

                self.scheduler.step(val_miou)

                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    if callbacks:
                        for cb in callbacks:
                            if hasattr(cb, 'on_improvement'):
                                cb.on_improvement(epoch, val_miou, self.model)

            if verbose:
                if val_loader is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - mIoU: {train_miou:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_mIoU: {val_miou:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - mIoU: {train_miou:.4f}")

            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, 'on_epoch_end'):
                        cb.on_epoch_end(epoch, self.history)

        return self.history

    @staticmethod
    def _compute_miou(predictions, labels, num_classes):
        """Compute mean Intersection over Union across classes present in batch."""
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

    def predict(self, point_clouds):
        """Predict per-point class labels from numpy array."""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(point_clouds).to(self.device)
            logits = self.model(x)
            predictions = torch.argmax(logits, dim=2)
            return predictions.cpu().numpy()

    def predict_proba(self, point_clouds):
        """Predict per-point class probabilities from numpy array."""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(point_clouds).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=2)
            return probs.cpu().numpy()

    def save(self, filepath):
        """Save model checkpoint to disk."""
        filepath = Path(filepath)
        if filepath.suffix not in ['.pt', '.pth']:
            filepath = filepath.with_suffix('.pt')

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'num_points': self.num_points,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'block_size': self.block_size,
            'use_fps': self.use_fps,
            'model_type': 'PointNet++ SSG',
            'class_mapping': self.class_mapping,
            'history': self.history,
            'task_type': 'segmentation'
        }

        torch.save(checkpoint, filepath)
        print(f"PointNet++ SSG model saved to {filepath}")

    def load(self, filepath):
        """Load a trained model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.num_points = checkpoint.get('num_points', self.num_points)
        self.num_features = checkpoint.get('num_features', self.num_features)
        self.num_classes = checkpoint.get('num_classes', self.num_classes)
        self.block_size = checkpoint.get('block_size', self.block_size)
        self.use_fps = checkpoint.get('use_fps', True)

        self.model = PointNet2SSGSegmentation(
            num_points=self.num_points,
            num_features=self.num_features,
            num_classes=self.num_classes,
            block_size=self.block_size,
            use_fps=self.use_fps
        ).to(self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.class_mapping = checkpoint.get('class_mapping')
        self.history = checkpoint.get('history', self.history)

        print(f"PointNet++ SSG model loaded from {filepath}")

    def summary(self):
        """Print model summary."""
        print(f"PointNet++ SSG Segmentation Model Summary")
        print(f"=" * 50)
        print(f"Input: ({self.num_points}, {self.num_features})")
        print(f"Classes: {self.num_classes}")
        print(f"Block size: {self.block_size}m")
        print(f"Device: {self.device}")
        print(f"=" * 50)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"=" * 50)
