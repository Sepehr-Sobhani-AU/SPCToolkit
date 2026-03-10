"""
A class for a DataNode's 'data' attribute containing surface normals.
"""

import numpy as np


class Normals:
    """
    A class for a DataNode's 'data' attribute containing surface normals.

    Args:
        normals (np.ndarray): Array of unit normal vectors with shape (n_points, 3).
    """

    def __init__(self, normals: np.ndarray):
        self.normals = normals.astype(np.float32)

        if not isinstance(self.normals, np.ndarray):
            raise ValueError("The input data must be a numpy array.")
        if not np.issubdtype(self.normals.dtype, np.floating):
            raise ValueError("The input data must be a floating-point array.")
        if len(self.normals.shape) != 2 or self.normals.shape[1] != 3:
            raise ValueError("The input data must be a 2D array with shape (n_points, 3).")
        if len(self.normals) == 0:
            raise ValueError("The input data must have at least one value.")

    def __repr__(self):
        return f"Normals(shape={self.normals.shape})"

    def __str__(self):
        return f"Normals(shape={self.normals.shape})"
