# core/pointnet_model_pytorch.py
"""
PointNet model architecture for point cloud classification - PyTorch implementation.

Based on: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
Charles R. Qi et al., CVPR 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class ConvBNReLU(nn.Module):
    """
    1D Convolutional block with Batch Normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (batch, channels, num_points)
        return F.relu(self.bn(self.conv(x)))


class DenseBNReLU(nn.Module):
    """
    Dense (Linear) block with Batch Normalization and ReLU activation.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # x: (batch, features)
        return F.relu(self.bn(self.fc(x)))


class TNet(nn.Module):
    """
    T-Net: Transformation network for spatial/feature alignment.

    Predicts a transformation matrix to align input points/features.
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        # Shared MLP
        self.conv1 = ConvBNReLU(num_features, 64)
        self.conv2 = ConvBNReLU(64, 128)
        self.conv3 = ConvBNReLU(128, 1024)

        # FC layers
        self.fc1 = DenseBNReLU(1024, 512)
        self.fc2 = DenseBNReLU(512, 256)

        # Transform matrix prediction
        self.transform = nn.Linear(256, num_features * num_features)

        # Initialize transform layer to predict identity matrix
        nn.init.zeros_(self.transform.weight)
        nn.init.constant_(self.transform.bias, 0)
        # Set bias to identity matrix
        self.transform.bias.data = torch.eye(num_features).flatten()

    def forward(self, x):
        # x: (batch, num_features, num_points)
        batch_size = x.size(0)

        # Shared MLP
        x_feat = self.conv1(x)
        x_feat = self.conv2(x_feat)
        x_feat = self.conv3(x_feat)

        # Global max pooling
        x_feat = torch.max(x_feat, dim=2)[0]  # (batch, 1024)

        # FC layers
        x_feat = self.fc1(x_feat)
        x_feat = self.fc2(x_feat)

        # Predict transformation matrix
        transform = self.transform(x_feat)
        transform = transform.view(batch_size, self.num_features, self.num_features)

        # Add identity to help with initial training
        identity = torch.eye(self.num_features, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        transform = transform + identity

        # Apply transformation: (batch, num_features, num_points) @ (batch, num_features, num_features)
        # Need to transpose for batch matrix multiply
        x_transformed = torch.bmm(transform, x)

        return x_transformed


class PointNet(nn.Module):
    """
    PointNet classification model.

    Args:
        num_points: Number of points per sample (default: 1024)
        num_features: Number of input features per point (default: 9)
        num_classes: Number of output classes
        use_tnet: Whether to use T-Net transformations (default: True)
    """

    def __init__(self, num_points=1024, num_features=9, num_classes=3, use_tnet=True):
        super().__init__()
        self.num_points = num_points
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_tnet = use_tnet

        # Input transform (T-Net for 3D spatial features)
        if use_tnet:
            self.input_tnet = TNet(3)
            self.feature_tnet = TNet(64)

        # Shared MLP (64, 64)
        self.conv1 = ConvBNReLU(num_features, 64)
        self.conv2 = ConvBNReLU(64, 64)

        # Shared MLP (64, 128, 1024)
        self.conv3 = ConvBNReLU(64, 64)
        self.conv4 = ConvBNReLU(64, 128)
        self.conv5 = ConvBNReLU(128, 1024)

        # Classification MLP
        self.fc1 = DenseBNReLU(1024, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = DenseBNReLU(512, 256)
        self.dropout2 = nn.Dropout(0.3)

        # Output layer
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, num_points, num_features)

        Returns:
            Class logits (batch, num_classes)
        """
        batch_size = x.size(0)

        # Transpose to (batch, num_features, num_points) for Conv1d
        x = x.transpose(1, 2)

        # Input transform (only on spatial features: x, y, z)
        if self.use_tnet:
            spatial_features = x[:, :3, :]  # (batch, 3, num_points)
            transformed_spatial = self.input_tnet(spatial_features)

            if self.num_features > 3:
                other_features = x[:, 3:, :]  # (batch, num_features-3, num_points)
                x = torch.cat([transformed_spatial, other_features], dim=1)
            else:
                x = transformed_spatial

        # Shared MLP (64, 64)
        x = self.conv1(x)
        x = self.conv2(x)

        # Feature transform
        if self.use_tnet:
            x = self.feature_tnet(x)

        # Shared MLP (64, 128, 1024)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Global max pooling
        global_feature = torch.max(x, dim=2)[0]  # (batch, 1024)

        # Classification MLP
        x = self.fc1(global_feature)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        # Output (logits, no softmax - will use CrossEntropyLoss)
        logits = self.output(x)

        return logits

    def predict(self, x):
        """
        Predict class labels.

        Args:
            x: Input tensor (batch, num_points, num_features)

        Returns:
            Predicted class IDs (batch,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x):
        """
        Predict class probabilities.

        Args:
            x: Input tensor (batch, num_points, num_features)

        Returns:
            Class probabilities (batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class PointNetClassifier:
    """
    Wrapper class for PointNet model with training and inference capabilities.

    Provides a similar interface to the TensorFlow version for easy migration.
    """

    def __init__(self, num_points=1024, num_features=9, num_classes=3, use_tnet=True, device=None):
        """
        Initialize PointNet classifier.

        Args:
            num_points: Number of points per sample
            num_features: Number of input features per point
            num_classes: Number of output classes
            use_tnet: Whether to use T-Net transformations
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.num_points = num_points
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_tnet = use_tnet

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = PointNet(
            num_points=num_points,
            num_features=num_features,
            num_classes=num_classes,
            use_tnet=use_tnet
        ).to(self.device)

        self.optimizer = None
        self.scheduler = None
        self.class_mapping = None  # Will be set during training
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def compile_model(self, learning_rate=0.001):
        """
        Set up optimizer and scheduler.

        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_data, train_labels, val_data=None, val_labels=None,
              epochs=100, batch_size=32, callbacks=None, verbose=1):
        """
        Train the model.

        Args:
            train_data: Training data numpy array (n_samples, num_points, num_features)
            train_labels: Training labels numpy array (n_samples,)
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of callback objects (optional)
            verbose: Verbosity level (0=silent, 1=progress)

        Returns:
            Training history dict
        """
        # Convert to tensors
        train_data = torch.FloatTensor(train_data).to(self.device)
        train_labels = torch.LongTensor(train_labels).to(self.device)

        has_validation = val_data is not None and val_labels is not None
        if has_validation:
            val_data = torch.FloatTensor(val_data).to(self.device)
            val_labels = torch.LongTensor(val_labels).to(self.device)

        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        best_val_acc = 0.0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_data, batch_labels in train_loader:
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(batch_data)
                loss = self.criterion(logits, batch_labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                train_loss += loss.item() * batch_data.size(0)
                predictions = torch.argmax(logits, dim=1)
                train_correct += (predictions == batch_labels).sum().item()
                train_total += batch_data.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(val_data)
                    val_loss = self.criterion(logits, val_labels).item()
                    predictions = torch.argmax(logits, dim=1)
                    val_acc = (predictions == val_labels).float().mean().item()

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Update scheduler
                self.scheduler.step(val_acc)

                # Check for improvement
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if callbacks:
                        for cb in callbacks:
                            if hasattr(cb, 'on_improvement'):
                                cb.on_improvement(epoch, val_acc, self.model)

            # Print progress
            if verbose:
                if has_validation:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f}")

            # Callbacks
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, 'on_epoch_end'):
                        cb.on_epoch_end(epoch, self.history)

        return self.history

    def predict(self, point_clouds):
        """
        Predict class labels for point clouds.

        Args:
            point_clouds: Input point clouds numpy array (n_samples, num_points, num_features)

        Returns:
            Predicted class IDs numpy array (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(point_clouds).to(self.device)
            logits = self.model(x)
            predictions = torch.argmax(logits, dim=1)
            return predictions.cpu().numpy()

    def predict_proba(self, point_clouds):
        """
        Predict class probabilities for point clouds.

        Args:
            point_clouds: Input point clouds numpy array (n_samples, num_points, num_features)

        Returns:
            Class probabilities numpy array (n_samples, num_classes)
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(point_clouds).to(self.device)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def save(self, filepath):
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model (.pt or .pth format)
        """
        filepath = Path(filepath)

        # Ensure .pt extension
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
            'history': self.history
        }

        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Recreate model with correct architecture if needed
        if (checkpoint.get('num_points') != self.num_points or
            checkpoint.get('num_features') != self.num_features or
            checkpoint.get('num_classes') != self.num_classes or
            checkpoint.get('use_tnet') != self.use_tnet):

            self.num_points = checkpoint.get('num_points', self.num_points)
            self.num_features = checkpoint.get('num_features', self.num_features)
            self.num_classes = checkpoint.get('num_classes', self.num_classes)
            self.use_tnet = checkpoint.get('use_tnet', self.use_tnet)

            self.model = PointNet(
                num_points=self.num_points,
                num_features=self.num_features,
                num_classes=self.num_classes,
                use_tnet=self.use_tnet
            ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_mapping = checkpoint.get('class_mapping')
        self.history = checkpoint.get('history', self.history)

        print(f"Model loaded from {filepath}")

    def summary(self):
        """Print model summary."""
        print(f"PointNet Model Summary")
        print(f"=" * 50)
        print(f"Input: ({self.num_points}, {self.num_features})")
        print(f"Classes: {self.num_classes}")
        print(f"T-Net: {self.use_tnet}")
        print(f"Device: {self.device}")
        print(f"=" * 50)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"=" * 50)

        # Print layer info
        print("\nLayers:")
        for name, module in self.model.named_modules():
            if name and not any(c.isdigit() for c in name.split('.')[-1] if len(name.split('.')[-1]) == 1):
                params = sum(p.numel() for p in module.parameters(recurse=False))
                if params > 0:
                    print(f"  {name}: {params:,} params")


# Callback classes for compatibility
class ModelCheckpoint:
    """Save the best model during training."""

    def __init__(self, filepath, monitor='val_acc', verbose=1):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.verbose = verbose
        self.best_value = 0.0

    def on_improvement(self, epoch, val_acc, model):
        if val_acc > self.best_value:
            self.best_value = val_acc

            # Ensure directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save model
            torch.save(model.state_dict(), self.filepath)

            if self.verbose:
                print(f"Epoch {epoch+1}: val_acc improved to {val_acc:.5f}, saving model to {self.filepath}")


class EarlyStopping:
    """Stop training when validation accuracy stops improving."""

    def __init__(self, patience=20, min_delta=0.001, verbose=1):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_value = 0.0
        self.should_stop = False

    def on_epoch_end(self, epoch, history):
        val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0

        if val_acc > self.best_value + self.min_delta:
            self.best_value = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
