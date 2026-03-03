"""
Stride-based spatial hashing for block-wise point cloud processing.

Builds an O(N log N) spatial index over XY coordinates. Blocks of
block_size x block_size are enumerated at stride-step intervals.
When stride < block_size, blocks overlap naturally (each point
appears in multiple blocks).
"""

import numpy as np


class StridedSpatialHash:
    """
    Stride-based 2D spatial hash for efficient block lookups.

    Points are assigned to stride-sized grid cells via integer floor
    division, then sorted by cell ID so that per-cell lookups are O(1)
    via searchsorted splits. Blocks spanning cells_per_block x
    cells_per_block stride cells are assembled by unioning the
    relevant cell slices.

    Args:
        points: (N, 3) point coordinates (only XY used for hashing)
        block_size: Block window size in meters (XY)
        stride: Step size between block positions in meters.
            stride < block_size -> overlapping blocks
            stride == block_size -> non-overlapping blocks
    """

    def __init__(self, points: np.ndarray, block_size: float, stride: float):
        self.block_size = block_size
        self.stride = stride
        self.cells_per_block = int(round(block_size / stride))

        min_xy = np.min(points[:, :2], axis=0)
        extent = np.max(points[:, :2], axis=0) - min_xy
        self.nx = max(1, int(np.ceil(extent[0] / stride)))
        self.ny = max(1, int(np.ceil(extent[1] / stride)))

        # Assign each point to its stride cell (int32 saves memory on large clouds)
        cx = np.floor((points[:, 0] - min_xy[0]) / stride).astype(np.int32)
        cy = np.floor((points[:, 1] - min_xy[1]) / stride).astype(np.int32)
        np.clip(cx, 0, self.nx - 1, out=cx)
        np.clip(cy, 0, self.ny - 1, out=cy)

        # Sort by cell ID for O(1) per-cell lookup via searchsorted splits
        cell_id = (cx * self.ny + cy).astype(np.int32)
        del cx, cy
        self._order = np.argsort(cell_id)
        sorted_ids = cell_id[self._order]
        del cell_id
        self._splits = np.searchsorted(sorted_ids, np.arange(self.nx * self.ny + 1))
        del sorted_ids

    def get_block_indices(self, ix: int, iy: int) -> np.ndarray:
        """
        Get point indices for the block at grid position (ix, iy).

        Gathers indices from cells_per_block x cells_per_block stride
        cells starting at (ix, iy). O(1) per cell via split lookup.

        Returns:
            (M,) int64 array of point indices, or empty array if no points.
        """
        parts = []
        for dx in range(self.cells_per_block):
            sx = ix + dx
            if sx >= self.nx:
                break
            for dy in range(self.cells_per_block):
                sy = iy + dy
                if sy >= self.ny:
                    break
                linear = sx * self.ny + sy
                start, end = self._splits[linear], self._splits[linear + 1]
                if start < end:
                    parts.append(self._order[start:end])
        if not parts:
            return np.array([], dtype=np.int64)
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def enumerate_blocks(self, min_points: int = 3) -> list:
        """
        Find all valid block positions with at least min_points points.

        Scans the full grid, counting points per block from cell splits.

        Args:
            min_points: Minimum points required for a block to be valid.

        Returns:
            List of (ix, iy) grid position tuples.
        """
        valid = []
        cpb = self.cells_per_block
        for ix in range(self.nx):
            for iy in range(self.ny):
                count = 0
                for dx in range(cpb):
                    sx = ix + dx
                    if sx >= self.nx:
                        break
                    for dy in range(cpb):
                        sy = iy + dy
                        if sy >= self.ny:
                            break
                        linear = sx * self.ny + sy
                        count += self._splits[linear + 1] - self._splits[linear]
                if count >= min_points:
                    valid.append((ix, iy))
        return valid

    @property
    def grid_shape(self) -> tuple:
        """(nx, ny) grid dimensions in stride cells."""
        return (self.nx, self.ny)
