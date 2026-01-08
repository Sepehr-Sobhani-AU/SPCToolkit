# core/pointnet_model.py
"""
PointNet model architecture for point cloud classification.

Based on: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
Charles R. Qi et al., CVPR 2017
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def conv_bn_relu(x, filters, kernel_size=1, name=None):
    """
    Convolutional block with Batch Normalization and ReLU activation.

    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size (default: 1)
        name: Layer name prefix

    Returns:
        Output tensor after Conv -> BN -> ReLU
    """
    x = layers.Conv1D(filters, kernel_size, padding='valid', name=f'{name}_conv' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_bn' if name else None)(x)
    x = layers.Activation('relu', name=f'{name}_relu' if name else None)(x)
    return x


def dense_bn_relu(x, units, name=None):
    """
    Dense block with Batch Normalization and ReLU activation.

    Args:
        x: Input tensor
        units: Number of units
        name: Layer name prefix

    Returns:
        Output tensor after Dense -> BN -> ReLU
    """
    x = layers.Dense(units, name=f'{name}_dense' if name else None)(x)
    x = layers.BatchNormalization(name=f'{name}_bn' if name else None)(x)
    x = layers.Activation('relu', name=f'{name}_relu' if name else None)(x)
    return x


def transformation_net(inputs, num_features, name='tnet'):
    """
    T-Net: Transformation network for spatial/feature alignment.

    Predicts a transformation matrix to align input points/features.

    Args:
        inputs: Input tensor (batch_size, num_points, num_features)
        num_features: Number of features (3 for spatial, 64 for feature transform)
        name: Layer name prefix

    Returns:
        Transformed input tensor
    """
    # Shared MLP
    x = conv_bn_relu(inputs, 64, name=f'{name}_conv1')
    x = conv_bn_relu(x, 128, name=f'{name}_conv2')
    x = conv_bn_relu(x, 1024, name=f'{name}_conv3')

    # Global max pooling
    x = layers.GlobalMaxPooling1D(name=f'{name}_maxpool')(x)

    # FC layers
    x = dense_bn_relu(x, 512, name=f'{name}_fc1')
    x = dense_bn_relu(x, 256, name=f'{name}_fc2')

    # Transform matrix
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer='zeros',
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        name=f'{name}_transform'
    )(x)

    # Reshape to matrix
    transform = layers.Reshape((num_features, num_features), name=f'{name}_reshape')(x)

    # Apply transformation
    return layers.Dot(axes=(2, 1), name=f'{name}_matmul')([inputs, transform])


def create_pointnet_model(num_points=1024, num_features=9, num_classes=3, use_tnet=True):
    """
    Create PointNet classification model.

    Args:
        num_points: Number of points per sample (default: 1024)
        num_features: Number of input features per point (default: 9)
        num_classes: Number of output classes
        use_tnet: Whether to use T-Net transformations (default: True)

    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=(num_points, num_features), name='input')

    # Input transform (optional)
    if use_tnet:
        # Only transform spatial features (first 3 dimensions: X, Y, Z)
        spatial_features = layers.Lambda(lambda x: x[:, :, :3], name='extract_spatial')(inputs)
        transformed_spatial = transformation_net(spatial_features, num_features=3, name='input_tnet')

        # Concatenate transformed spatial with other features
        if num_features > 3:
            other_features = layers.Lambda(lambda x: x[:, :, 3:], name='extract_other')(inputs)
            x = layers.Concatenate(axis=2, name='concat_features')([transformed_spatial, other_features])
        else:
            x = transformed_spatial
    else:
        x = inputs

    # Shared MLP (64, 64)
    x = conv_bn_relu(x, 64, name='conv1')
    x = conv_bn_relu(x, 64, name='conv2')

    # Feature transform (optional)
    if use_tnet:
        x = transformation_net(x, num_features=64, name='feature_tnet')

    # Shared MLP (64, 128, 1024)
    x = conv_bn_relu(x, 64, name='conv3')
    x = conv_bn_relu(x, 128, name='conv4')
    x = conv_bn_relu(x, 1024, name='conv5')

    # Global feature (max pooling)
    global_feature = layers.GlobalMaxPooling1D(name='global_maxpool')(x)

    # Classification MLP (512, 256, num_classes)
    x = dense_bn_relu(global_feature, 512, name='fc1')
    x = layers.Dropout(0.3, name='dropout1')(x)
    x = dense_bn_relu(x, 256, name='fc2')
    x = layers.Dropout(0.3, name='dropout2')(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='pointnet')
    return model


class PointNetClassifier:
    """
    Wrapper class for PointNet model with training and inference capabilities.
    """

    def __init__(self, num_points=1024, num_features=9, num_classes=3, use_tnet=True):
        """
        Initialize PointNet classifier.

        Args:
            num_points: Number of points per sample
            num_features: Number of input features per point
            num_classes: Number of output classes
            use_tnet: Whether to use T-Net transformations
        """
        self.num_points = num_points
        self.num_features = num_features
        self.num_classes = num_classes
        self.use_tnet = use_tnet

        self.model = create_pointnet_model(
            num_points=num_points,
            num_features=num_features,
            num_classes=num_classes,
            use_tnet=use_tnet
        )

        self.class_mapping = None  # Will be set during training

    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss.

        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

    def train(self, train_data, train_labels, val_data=None, val_labels=None,
              epochs=100, batch_size=32, callbacks=None):
        """
        Train the model.

        Args:
            train_data: Training data (n_samples, num_points, num_features)
            train_labels: Training labels (n_samples,)
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks

        Returns:
            Training history
        """
        validation_data = None
        if val_data is not None and val_labels is not None:
            validation_data = (val_data, val_labels)

        history = self.model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, point_clouds):
        """
        Predict class labels for point clouds.

        Args:
            point_clouds: Input point clouds (n_samples, num_points, num_features)

        Returns:
            Predicted class IDs (n_samples,)
        """
        predictions = self.model.predict(point_clouds, verbose=0)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, point_clouds):
        """
        Predict class probabilities for point clouds.

        Args:
            point_clouds: Input point clouds (n_samples, num_points, num_features)

        Returns:
            Class probabilities (n_samples, num_classes)
        """
        return self.model.predict(point_clouds, verbose=0)

    def save(self, filepath):
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model (.keras format)
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def summary(self):
        """Print model summary."""
        return self.model.summary()
