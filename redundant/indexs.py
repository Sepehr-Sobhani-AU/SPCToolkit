"""
A class for a DataNode's 'data' attribute as a list of point indexes.
"""

import numpy as np


class Indexs:
    """
    A class for a DataNode's 'data' attribute as a list of point indexes.

    Args:
        index (np.ndarray): Integer array of point indexes.
    """

    def __init__(self, index: np.ndarray):
        self.index = index.astype(np.uint32)

        # Validate input data
        if not isinstance(self.index, np.ndarray):
            raise ValueError("The input data must be a numpy array.")
        if not np.issubdtype(self.index.dtype, np.integer):
            raise ValueError("The input data must be an integer array.")
        if len(self.index.shape) != 1:
            raise ValueError("The input data must be a 1D array.")
        if np.any(self.index < 0):
            raise ValueError("The input data must be non-negative.")
        if len(np.unique(self.index)) != len(self.index):
            raise ValueError("The input data must have unique values.")
        if len(self.index) == 0:
            raise ValueError("The input data must have at least one value.")

    def __repr__(self):
        return f"Indexes(index={self.index})"

    def __str__(self):
        return f"Indexes(index={self.index})"
