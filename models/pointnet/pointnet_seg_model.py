"""
PointNet Semantic Segmentation Model (PyTorch)

Per-point classification architecture for semantic segmentation of point clouds.
Reuses ConvBNReLU and TNet from the classification model.

Based on: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
Charles R. Qi et al., CVPR 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from core.pointnet_model import ConvBNReLU, TNet


class PointNetSegmentation(nn.Module):
    """
    PointNet segmentation model for per-point classification.

    Key difference from classification: concatenates local features (64-dim) with
    global feature (1024-dim) for each point, then applies per-point MLP.

    Architecture:
        Input(B,N,F) -> TNet -> Conv(F->64->64) -> [save local]
        -> Conv(64->64->128->1024) -> GlobalMaxPool -> expand(B,1024,N)
        -> concat with local(B,64,N) -> (B,1088,N)
        -> Conv(1088->512->256->128->C) -> Output(B,N,C)

    Args:
        num_points: Number of points per sample
        num_features: Number of input features per point
        num_classes: Number of output semantic classes
        use_tnet: Whether to use T-Net transformations
    """

    def __init__(self, num_points=4096, num_features=9, num_classes=6, use_tnet=True):
        super().__init__()
        self.num_points = num_points
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_tnet = use_tnet

        # Input transform (T-Net for 3D spatial features)
        if use_tnet:
            self.input_tnet = TNet(3)
            self.feature_tnet = TNet(64)

        # Shared MLP to get local features (64-dim)
        self.conv1 = ConvBNReLU(num_features, 64)
        self.conv2 = ConvBNReLU(64, 64)

        # Shared MLP for global features (64 -> 128 -> 1024)
        self.conv3 = ConvBNReLU(64, 64)
        self.conv4 = ConvBNReLU(64, 128)
        self.conv5 = ConvBNReLU(128, 1024)

        # Per-point segmentation MLP (1088 = 64 local + 1024 global)
        self.seg_conv1 = ConvBNReLU(1088, 512)
        self.seg_drop1 = nn.Dropout(0.3)
        self.seg_conv2 = ConvBNReLU(512, 256)
        self.seg_drop2 = nn.Dropout(0.3)
        self.seg_conv3 = ConvBNReLU(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.seg_output = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, num_points, num_features)

        Returns:
            Per-point class logits (batch, num_points, num_classes)
        """
        batch_size = x.size(0)
        num_points = x.size(1)

        # Transpose to (batch, num_features, num_points) for Conv1d
        x = x.transpose(1, 2)

        # Input transform (only on spatial features: x, y, z)
        if self.use_tnet:
            spatial_features = x[:, :3, :]
            transformed_spatial = self.input_tnet(spatial_features)

            if self.num_features > 3:
                other_features = x[:, 3:, :]
                x = torch.cat([transformed_spatial, other_features], dim=1)
            else:
                x = transformed_spatial

        # Shared MLP (F -> 64 -> 64)
        x = self.conv1(x)
        x = self.conv2(x)

        # Feature transform
        if self.use_tnet:
            x = self.feature_tnet(x)

        # Save local features (B, 64, N)
        local_features = x

        # Shared MLP (64 -> 64 -> 128 -> 1024)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Global max pooling -> (B, 1024)
        global_feature = torch.max(x, dim=2)[0]

        # Expand global feature to (B, 1024, N)
        global_feature_expanded = global_feature.unsqueeze(2).expand(-1, -1, num_points)

        # Concatenate local + global -> (B, 1088, N)
        combined = torch.cat([local_features, global_feature_expanded], dim=1)

        # Per-point segmentation MLP
        x = self.seg_conv1(combined)
        x = self.seg_drop1(x)
        x = self.seg_conv2(x)
        x = self.seg_drop2(x)
        x = self.seg_conv3(x)
        x = self.dropout(x)
        x = self.seg_output(x)  # (B, C, N)

        # Transpose to (B, N, C)
        x = x.transpose(1, 2)

        return x

    def predict(self, x):
        """
        Predict per-point class labels.

        Args:
            x: Input tensor (batch, num_points, num_features)

        Returns:
            Predicted class IDs (batch, num_points)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=2)

    def predict_proba(self, x):
        """
        Predict per-point class probabilities.

        Args:
            x: Input tensor (batch, num_points, num_features)

        Returns:
            Class probabilities (batch, num_points, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=2)


class PointNetSegmenter:
    """
    Wrapper class for PointNet segmentation model with training and inference capabilities.

    Mirrors the PointNetClassifier interface but adapted for per-point segmentation.
    """

    def __init__(self, num_points=4096, num_features=9, num_classes=6, use_tnet=True, device=None):
        self.num_points = num_points
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_tnet = use_tnet

        if device is None:
            if not torch.cuda.is_available():
                print("WARNING: CUDA not available — model will run on CPU and be significantly slower.")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = PointNetSegmentation(
            num_points=num_points,
            num_features=num_features,
            num_classes=num_classes,
            use_tnet=use_tnet
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

                # Forward: (B, N, C)
                logits = self.model(batch_features)

                # Reshape for CrossEntropyLoss: (B*N, C) vs (B*N,)
                B, N, C = logits.shape
                loss = criterion(logits.reshape(B * N, C), batch_labels.reshape(B * N))

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * B
                predictions = torch.argmax(logits, dim=2)
                train_correct += (predictions == batch_labels).sum().item()
                train_total += B * N

                # Per-batch mIoU
                miou = self._compute_miou(predictions, batch_labels, self.num_classes)
                if miou is not None:
                    train_iou_sum += miou * B
                    train_iou_count += B

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
                        B, N, C = logits.shape
                        loss = criterion(logits.reshape(B * N, C), batch_labels.reshape(B * N))

                        val_loss_sum += loss.item() * B
                        predictions = torch.argmax(logits, dim=2)
                        val_correct += (predictions == batch_labels).sum().item()
                        val_total += B * N

                        miou = self._compute_miou(predictions, batch_labels, self.num_classes)
                        if miou is not None:
                            val_iou_sum += miou * B
                            val_iou_count += B

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
        """
        Predict per-point class labels.

        Args:
            point_clouds: numpy array (n_samples, num_points, num_features)

        Returns:
            Predicted class IDs numpy array (n_samples, num_points)
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(point_clouds).to(self.device)
            logits = self.model(x)
            predictions = torch.argmax(logits, dim=2)
            return predictions.cpu().numpy()

    def predict_proba(self, point_clouds):
        """
        Predict per-point class probabilities.

        Args:
            point_clouds: numpy array (n_samples, num_points, num_features)

        Returns:
            Class probabilities numpy array (n_samples, num_points, num_classes)
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(point_clouds).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=2)
            return probs.cpu().numpy()

    def save(self, filepath):
        """Save the model to disk."""
        filepath = Path(filepath)
        if filepath.suffix not in ['.pt', '.pth']:
            filepath = filepath.with_suffix('.pt')

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'num_points': self.num_points,
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'use_tnet': self.use_tnet,
            'class_mapping': self.class_mapping,
            'history': self.history,
            'task_type': 'segmentation'
        }

        torch.save(checkpoint, filepath)
        print(f"Segmentation model saved to {filepath}")

    def load(self, filepath):
        """Load a trained model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)

        if (checkpoint.get('num_points') != self.num_points or
            checkpoint.get('num_features') != self.num_features or
            checkpoint.get('num_classes') != self.num_classes or
            checkpoint.get('use_tnet') != self.use_tnet):

            self.num_points = checkpoint.get('num_points', self.num_points)
            self.num_features = checkpoint.get('num_features', self.num_features)
            self.num_classes = checkpoint.get('num_classes', self.num_classes)
            self.use_tnet = checkpoint.get('use_tnet', self.use_tnet)

            self.model = PointNetSegmentation(
                num_points=self.num_points,
                num_features=self.num_features,
                num_classes=self.num_classes,
                use_tnet=self.use_tnet
            ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_mapping = checkpoint.get('class_mapping')
        self.history = checkpoint.get('history', self.history)

        print(f"Segmentation model loaded from {filepath}")

    def summary(self):
        """Print model summary."""
        print(f"PointNet Segmentation Model Summary")
        print(f"=" * 50)
        print(f"Input: ({self.num_points}, {self.num_features})")
        print(f"Classes: {self.num_classes}")
        print(f"T-Net: {self.use_tnet}")
        print(f"Device: {self.device}")
        print(f"=" * 50)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"=" * 50)
