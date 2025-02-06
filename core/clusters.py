# A class for cluster data nodes.
import numpy as np


class Clusters:
    """
    A class for Clusters objects with cluster labels and optional colors.
    """
    def __init__(self, labels: np.ndarray, colors: np.ndarray = None):
        self.labels = labels
        self.colors = colors

    # A method to generate random colors for the clusters.
    def set_random_color(self, noise_color: np.ndarray = [0.2, 0.2, 0.2]):
        """
        Sets a random color for each point based on its cluster labels'
        Also assign a color for noise points, the default is [0.2, 0.2, 0.2].

        Arg:
            - noise_color (np.ndarray): The color to assign to noise points.
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
        return f"Clusters(labels={self.labels}, colors={self.colors})"

    def __str__(self):
        return f"Clusters(labels={self.labels}, colors={self.colors})"





