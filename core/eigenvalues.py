"""
A class for a DataNode's 'data' attribute containing eigenvalues.
"""

import numpy as np


class Eigenvalues:
    """
    A class for a DataNode's 'data' attribute containing eigenvalues.

    Args:
        eigenvalues (np.ndarray): Array of eigenvalues with shape (n_points, 3).
    """

    def __init__(self, eigenvalues: np.ndarray):
        self.eigenvalues = eigenvalues.astype(np.float32)

        # Validate input data
        if not isinstance(self.eigenvalues, np.ndarray):
            raise ValueError("The input data must be a numpy array.")
        if not np.issubdtype(self.eigenvalues.dtype, np.floating):
            raise ValueError("The input data must be a floating-point array.")
        if len(self.eigenvalues.shape) != 2 or self.eigenvalues.shape[1] != 3:
            raise ValueError("The input data must be a 2D array with shape (n_points, 3).")
        if len(self.eigenvalues) == 0:
            raise ValueError("The input data must have at least one value.")

    def __repr__(self):
        return f"Eigenvalues(shape={self.eigenvalues.shape})"

    def __str__(self):
        return f"Eigenvalues(shape={self.eigenvalues.shape})"