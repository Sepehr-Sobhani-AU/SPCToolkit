# core/dist_to_ground.py
"""
A class for a DataNode's 'data' attribute containing vertical distances to ground.
"""

import numpy as np


class DistToGround:
    """
    A class for a DataNode's 'data' attribute containing vertical distances to ground.

    This class stores distance values representing the vertical (Z-axis)
    distance from each point to its nearest ground point. Positive values
    indicate points above ground, negative values indicate points below ground.

    Args:
        distances (np.ndarray): Array of vertical distance values with shape (n_points,).
    """

    def __init__(self, distances: np.ndarray):
        self.distances = distances.astype(np.float32)

        # Validate input data
        if not isinstance(self.distances, np.ndarray):
            raise ValueError("The input data must be a numpy array.")
        if not np.issubdtype(self.distances.dtype, np.floating):
            raise ValueError("The input data must be a floating-point array.")
        if len(self.distances.shape) != 1:
            raise ValueError("The input data must be a 1D array.")
        if len(self.distances) == 0:
            raise ValueError("The input data must have at least one value.")

    def __repr__(self):
        return f"DistToGround(shape={self.distances.shape})"

    def __str__(self):
        return f"DistToGround(shape={self.distances.shape})"