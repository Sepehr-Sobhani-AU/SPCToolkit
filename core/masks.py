"""
A class for a DataNode's 'data' attribute as a boolean mask array.
"""

import numpy as np


class Masks:
    """
    A class for a DataNode's 'data' attribute as a boolean mask array.

    Args:
        mask (np.ndarray): A boolean array.
    """

    def __init__(self, mask: np.ndarray):
        self.mask = mask.astype(np.bool)

        # Validate input data
        if not isinstance(self.mask, np.ndarray):
            raise ValueError("The input data must be a numpy array.")
        if self.mask.dtype != np.bool:
            raise ValueError("The input data must be a boolean array.")
        if len(self.mask.shape) != 1:
            raise ValueError("The input data must be a 1D, boolean array.")

    def __repr__(self):
        return f"Masks(mask={self.mask})"

    def __str__(self):
        return f"Masks(mask={self.mask})"
