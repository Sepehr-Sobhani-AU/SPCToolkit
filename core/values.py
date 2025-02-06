"""
A class for a DataNode's data attribute as a list of point's attribute values.
"""

import numpy as np


class Values:
    """
    A class for a DataNode's data attribute as a list of point's attribute values.

    Args:
        values (np.ndarray): DataNode's data attribute.
    """

    def __init__(self, values: np.ndarray):
        self.values = values.dtype(np.float32)

        # Validate input data
        if not isinstance(self.values, np.ndarray):
            raise ValueError("The input data must be a numpy array.")
        if not np.issubdtype(self.values.dtype, np.float32):
            raise ValueError("The input data must be an integer array.")
        if len(self.values) == 0:
            raise ValueError("The input data must have at least one value.")

    def __repr__(self):
        return f"Values(value={self.values})"

    def __str__(self):
        return f"Values(value={self.values})"
