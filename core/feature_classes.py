# core/feature_classes.py
"""
A class for representing feature classification labels for point cloud clusters.
"""

import numpy as np
from typing import Dict, List


class FeatureClasses:
    """
    A class for storing feature classification labels for point cloud clusters.

    This class maintains the mapping between cluster IDs and feature classes,
    along with visualization colors for each feature class. It works with
    cluster labels produced by clustering algorithms like DBSCAN.

    Args:
        labels (np.ndarray): Array of feature class labels for each point.
        class_mapping (Dict[int, str]): Mapping from cluster IDs to feature class names.
        class_colors (Dict[str, np.ndarray]): Mapping from feature class names to RGB colors.
    """

    def __init__(
            self,
            labels: np.ndarray,
            class_mapping: Dict[int, str],
            class_colors: Dict[str, np.ndarray]
    ):
        self.labels = labels.astype(np.int32)
        self.class_mapping = class_mapping
        self.class_colors = class_colors

        # Validate input data
        if not isinstance(self.labels, np.ndarray):
            raise ValueError("The labels must be a numpy array.")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("The labels must be an integer array.")
        if len(self.labels.shape) != 1:
            raise ValueError("The labels must be a 1D array.")
        if len(self.labels) == 0:
            raise ValueError("The labels must have at least one value.")

    def get_unique_classes(self) -> List[str]:
        """
        Get the list of unique feature classes in the dataset.

        Returns:
            List[str]: List of unique feature class names.
        """
        return list(set(self.class_mapping.values()))

    def get_points_by_class(self, feature_class: str) -> np.ndarray:
        """
        Get the indices of points belonging to a specific feature class.

        Args:
            feature_class (str): The feature class name to filter by.

        Returns:
            np.ndarray: Boolean mask identifying points of the requested class.
        """
        if feature_class not in self.get_unique_classes():
            raise ValueError(f"Feature class '{feature_class}' not found in the mapping.")

        # Create mask for all points
        mask = np.zeros(len(self.labels), dtype=bool)

        # Find all cluster IDs that map to this feature class
        for cluster_id, cls in self.class_mapping.items():
            if cls == feature_class:
                mask |= (self.labels == cluster_id)

        return mask

    def get_points_color(self) -> np.ndarray:
        """
        Get color values for all points based on their feature class.

        Returns:
            np.ndarray: Array of RGB color values for each point.
        """
        # Initialize with a default color (gray)
        colors = np.ones((len(self.labels), 3), dtype=np.float32) * 0.7

        # Assign colors based on feature class
        for cluster_id, class_name in self.class_mapping.items():
            if class_name in self.class_colors:
                mask = self.labels == cluster_id
                colors[mask] = self.class_colors[class_name]

        return colors

    def __repr__(self):
        """
        String representation of the FeatureClasses object.

        Returns:
            str: Description of the object.
        """
        class_counts = {}
        for cls in self.get_unique_classes():
            mask = self.get_points_by_class(cls)
            class_counts[cls] = np.sum(mask)

        return f"FeatureClasses(classes={len(self.get_unique_classes())}, " \
               f"points={len(self.labels)}, " \
               f"counts={class_counts})"

    def __str__(self):
        """
        Human-readable string representation.

        Returns:
            str: Description of the object.
        """
        return self.__repr__()
    