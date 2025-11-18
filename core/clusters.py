"""
A class for a DataNode's 'data' attribute containing cluster labels and optional colors.
"""

import numpy as np


class Clusters:
    """
    A class for a DataNode's 'data' attribute containing cluster labels and optional colors.

    This class stores clustering results produced by clustering algorithms like DBSCAN.
    It maintains cluster labels for each point and optional RGB colors for visualization.

    Args:
        labels (np.ndarray): Integer array of cluster labels with shape (n_points,).
                            Label -1 typically indicates noise points.
        colors (np.ndarray, optional): Array of RGB color values with shape (n_points, 3).
                                      Values should be in the range [0, 1].
                                      Defaults to None.
    """

    def __init__(self, labels: np.ndarray, colors: np.ndarray = None):
        # Type conversion
        self.labels = labels.astype(np.int32)
        self.colors = colors.astype(np.float32) if colors is not None else None

        # Validate labels
        if not isinstance(self.labels, np.ndarray):
            raise ValueError("The labels must be a numpy array.")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("The labels must be an integer array.")
        if len(self.labels.shape) != 1:
            raise ValueError("The labels must be a 1D array.")
        if len(self.labels) == 0:
            raise ValueError("The labels must have at least one value.")

        # Validate colors if provided
        if self.colors is not None:
            if not isinstance(self.colors, np.ndarray):
                raise ValueError("The colors must be a numpy array.")
            if not np.issubdtype(self.colors.dtype, np.floating):
                raise ValueError("The colors must be a floating-point array.")
            if len(self.colors.shape) != 2 or self.colors.shape[1] != 3:
                raise ValueError("The colors must be a 2D array with shape (n_points, 3).")
            if len(self.colors) != len(self.labels):
                raise ValueError(
                    f"The number of colors ({len(self.colors)}) must match "
                    f"the number of labels ({len(self.labels)})."
                )

            # Ensure color values are in range [0, 1]
            if np.min(self.colors) < 0 or np.max(self.colors) > 1:
                print("Warning: Some color values are outside the range [0, 1]. Clipping values.")
                self.colors = np.clip(self.colors, 0, 1)

    def set_random_color(self, noise_color: np.ndarray = np.array([0.2, 0.2, 0.2])):
        """
        Generates random colors for each cluster and assigns them to points.

        This method creates a unique random color for each cluster label and
        assigns a specified color to noise points (label -1).

        Args:
            noise_color (np.ndarray): The RGB color to assign to noise points.
                                     Defaults to [0.2, 0.2, 0.2] (dark gray).
        """
        # Get unique labels and their indexes
        unique_labels, label_indexes = np.unique(self.labels, return_inverse=True)

        # Generate random colors for all labels (including -1)
        np.random.seed(42)  # Optional: Set a seed for reproducibility
        cluster_colors = np.random.rand(len(unique_labels), 3)

        # Assign a color for noise (-1 labels)
        noise_labels = unique_labels == -1
        cluster_colors[noise_labels] = noise_color

        # Assign colors to each point based on their label_indexes
        point_colors = cluster_colors[label_indexes]

        self.colors = np.float32(point_colors)

    def __repr__(self):
        n_clusters = len(np.unique(self.labels[self.labels != -1]))
        n_noise = np.sum(self.labels == -1)
        has_colors = self.colors is not None
        return (f"Clusters(n_points={len(self.labels)}, n_clusters={n_clusters}, "
                f"n_noise={n_noise}, has_colors={has_colors})")

    def __str__(self):
        return self.__repr__()





