"""
A class for a DataNode's 'data' attribute containing RGB colors.
"""

import numpy as np


class Colors:
    """
    A class for a DataNode's 'data' attribute containing RGB colors.

    Args:
        colors (np.ndarray): Array of RGB color values with shape (n_points, 3).
                            Values should be in the range [0, 1].
    """

    def __init__(self, colors: np.ndarray):
        self.colors = colors.astype(np.float32)

        # Validate input data
        if not isinstance(self.colors, np.ndarray):
            raise ValueError("The input data must be a numpy array.")
        if not np.issubdtype(self.colors.dtype, np.floating):
            raise ValueError("The input data must be a floating-point array.")
        if len(self.colors.shape) != 2 or self.colors.shape[1] != 3:
            raise ValueError("The input data must be a 2D array with shape (n_points, 3).")
        if len(self.colors) == 0:
            raise ValueError("The input data must have at least one value.")

        # Ensure color values are in range [0, 1]
        if np.min(self.colors) < 0 or np.max(self.colors) > 1:
            print("Warning: Some color values are outside the range [0, 1]. Clipping values.")
            self.colors = np.clip(self.colors, 0, 1)

    def __repr__(self):
        return f"Colors(shape={self.colors.shape})"

    def __str__(self):
        return f"Colors(shape={self.colors.shape})"