"""
A class for a DataNode's 'data' attribute as a list of cluster labels.
"""

import numpy as np


class Labels:
    """
    A class for a DataNode's 'data' attribute as a list of cluster labels.

    Args:
        labels (np.ndarray): Integer array of cluster labels.
    """

    def __init__(self, labels: np.ndarray):
        self.labels = labels.astype(np.int32)

        # Validate input data
        if not isinstance(self.labels, np.ndarray):
            raise ValueError("The input data must be a numpy array.")
        if not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("The input data must be an integer array.")
        if len(self.labels.shape) != 1:
            raise ValueError("The input data must be a 1D array.")
        if len(self.labels) == 0:
            raise ValueError("The input data must have at least one value.")

    def __repr__(self):
        return f"Labels(label={self.labels})"

    def __str__(self):
        return f"Labels(label={self.labels})"
