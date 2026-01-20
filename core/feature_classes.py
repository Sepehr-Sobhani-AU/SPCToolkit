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

        # Find all cluster IDs that map to this feature class
        cluster_ids = [cid for cid, cls in self.class_mapping.items() if cls == feature_class]

        if not cluster_ids:
            return np.zeros(len(self.labels), dtype=bool)

        # Use np.isin for vectorized membership test (much faster than loop)
        return np.isin(self.labels, cluster_ids)

    def get_points_color(self) -> np.ndarray:
        """
        Get color values for all points based on their feature class.

        Uses vectorized lookup table for O(n) performance instead of O(n*k).

        Returns:
            np.ndarray: Array of RGB color values for each point.
        """
        n_points = len(self.labels)
        default_color = np.array([0.7, 0.7, 0.7], dtype=np.float32)

        if len(self.class_mapping) == 0:
            return np.tile(default_color, (n_points, 1))

        # Find the range of label values for lookup table sizing
        min_label = int(np.min(self.labels))
        max_label = int(np.max(self.labels))

        # Offset to handle negative labels (e.g., -1 for noise)
        offset = -min_label if min_label < 0 else 0
        table_size = max_label + offset + 1

        # Build color lookup table (O(k) where k = number of mappings)
        color_lut = np.tile(default_color, (table_size, 1))

        for label_id, class_name in self.class_mapping.items():
            if class_name in self.class_colors:
                idx = label_id + offset
                if 0 <= idx < table_size:
                    color_lut[idx] = self.class_colors[class_name]

        # Apply lookup using vectorized indexing (O(n))
        adjusted_labels = self.labels + offset
        colors = color_lut[adjusted_labels]

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
    