# core/class_reference.py
"""
A lightweight reference class for filtering point clouds by semantic class.

This class stores only a reference to a class (ID, name, color) rather than
duplicating point data, enabling memory-efficient class-based views.
"""

import numpy as np


class ClassReference:
    """
    A lightweight reference for filtering points by semantic class.

    This class stores only metadata about a class, not the actual point data.
    Used to create class-specific views without data duplication.

    Args:
        class_id (int): Integer ID of the semantic class (e.g., 0, 1, 2).
        class_name (str): Name of the class (e.g., "Tree", "Car", "Building").
        color (np.ndarray): RGB color for this class with shape (3,), values in [0, 1].
    """

    def __init__(self, class_id: int, class_name: str, color: np.ndarray):
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.color = np.asarray(color, dtype=np.float32)

        # Validate inputs
        if not isinstance(self.class_id, (int, np.integer)):
            raise ValueError("class_id must be an integer.")
        if not isinstance(self.class_name, str) or not self.class_name:
            raise ValueError("class_name must be a non-empty string.")
        if self.color.shape != (3,):
            raise ValueError("color must be an array of shape (3,) representing RGB.")
        if np.any((self.color < 0) | (self.color > 1)):
            raise ValueError("color values must be in the range [0, 1].")

    def __repr__(self):
        return (f"ClassReference(class_id={self.class_id}, "
                f"class_name='{self.class_name}', "
                f"color={self.color})")

    def __str__(self):
        return f"ClassReference: {self.class_name} (ID={self.class_id})"
