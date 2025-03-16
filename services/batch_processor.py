# services/batch_processor.py
"""
Spatial grid-based batch processor for efficient point cloud processing.

This module provides functionality to divide large point clouds into smaller batches
for efficient processing, using a spatial grid approach with overlapping regions.
"""

import numpy as np
from typing import Callable, Any, Dict, List, Tuple, Optional


class BatchProcessor:
    """
    Processes large point clouds in spatial batches with optional overlap between adjacent cells.

    This class divides a point cloud into a spatial grid, processes each grid cell (with optional
    overlapping regions), and combines the results. It's particularly useful for computationally
    intensive operations on large point clouds that can be processed independently in batches.

    Attributes:
        points (np.ndarray): The input point cloud data with shape (n, 3).
        batch_size (int): Target number of points per batch.
        overlap_percent (float): Percentage of overlap between adjacent grid cells (0-1).
        grid_indices (np.ndarray): Parallel array mapping each point to its grid cell.
        grid_dimensions (tuple): The dimensions of the grid (nx, ny, nz).
        cell_size (np.ndarray): Size of each grid cell in each dimension.
        grid_bounds (Dict): Min/max coordinates of the grid.
    """

    def __init__(
            self,
            points: np.ndarray,
            batch_size: int = 100000,
            overlap_percent: float = 0.1
    ):
        """
        Initialize the BatchProcessor with point cloud and configuration.

        Args:
            points (np.ndarray): Point cloud data of shape (n, 3).
            batch_size (int, optional): Target number of points per batch. Defaults to 100000.
            overlap_percent (float, optional): Percentage of overlap between adjacent cells (0-1).
                Defaults to 0.1 (10%).
        """
        self.points = points
        self.batch_size = batch_size
        self.overlap_percent = overlap_percent

        # Will be computed in create_spatial_grid
        self.grid_indices = None
        self.grid_dimensions = None
        self.cell_size = None
        self.grid_bounds = None

        # Initialize the spatial grid
        self.create_spatial_grid()

    def create_spatial_grid(self):
        """
        Create a spatial grid and assign points to grid cells.

        This method:
        1. Calculates the bounding box of the point cloud
        2. Determines optimal grid dimensions based on point density and target batch size
        3. Creates a mapping from each point to its grid cell using a hash-based approach
        """
        # Calculate bounding box
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)

        # Calculate overall volume and estimate grid dimensions
        total_volume = np.prod(max_bounds - min_bounds)
        point_density = len(self.points) / total_volume

        # Estimate cell size to achieve target batch size
        cell_volume = self.batch_size / point_density
        cell_length = np.cbrt(cell_volume)

        # Calculate grid dimensions
        extents = max_bounds - min_bounds
        grid_dims = np.ceil(extents / cell_length).astype(int)
        self.grid_dimensions = tuple(grid_dims)

        # Calculate actual cell size
        self.cell_size = extents / grid_dims

        # Store grid bounds
        self.grid_bounds = {
            'min': min_bounds,
            'max': max_bounds
        }

        # Compute grid indices for each point
        normalized_points = (self.points - min_bounds) / self.cell_size
        grid_indices = np.floor(normalized_points).astype(int)

        # Clamp indices to valid range
        for i in range(3):
            np.clip(grid_indices[:, i], 0, grid_dims[i] - 1, out=grid_indices[:, i])

        # Convert 3D indices to 1D hash
        self.grid_indices = (
                grid_indices[:, 0] * grid_dims[1] * grid_dims[2] +
                grid_indices[:, 1] * grid_dims[2] +
                grid_indices[:, 2]
        )

        # TODO: Add validation and error handling for edge cases (empty point clouds, etc.)

    def get_batch_for_grid_cell(self, cell_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of points for a given grid cell, including overlapping regions.

        Args:
            cell_idx (int): The 1D index of the grid cell.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Indices of points in the batch
                - Boolean mask identifying primary (non-overlap) points
        """
        # Convert 1D cell index to 3D coordinates
        nx, ny, nz = self.grid_dimensions
        cell_z = cell_idx % nz
        cell_y = (cell_idx // nz) % ny
        cell_x = cell_idx // (ny * nz)

        # Find primary points (those belonging directly to this cell)
        primary_mask = self.grid_indices == cell_idx
        primary_indices = np.where(primary_mask)[0]

        if len(primary_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=bool)

        # Calculate extended bounds with overlap
        min_bounds = self.grid_bounds['min'] + np.array([cell_x, cell_y, cell_z]) * self.cell_size
        max_bounds = min_bounds + self.cell_size

        overlap_size = self.cell_size * self.overlap_percent
        extended_min = min_bounds - overlap_size
        extended_max = max_bounds + overlap_size

        # Find all points within the extended bounds
        mask = np.ones(len(self.points), dtype=bool)
        for dim in range(3):
            mask &= (self.points[:, dim] >= extended_min[dim])
            mask &= (self.points[:, dim] <= extended_max[dim])

        batch_indices = np.where(mask)[0]

        # Create mask identifying which points are primary to this cell
        is_primary = np.zeros(len(batch_indices), dtype=bool)
        for i, idx in enumerate(batch_indices):
            if primary_mask[idx]:
                is_primary[i] = True

        return batch_indices, is_primary

    def process_in_batches(self, processing_func: Callable,
                           callback: Optional[Callable] = None,
                           **kwargs) -> Any:
        """
        Process the point cloud in batches using the provided function.

        Args:
            processing_func (Callable): Function that takes a batch of points and returns a result.
                The function should accept: (points, **kwargs) and return a result.
            callback (Optional[Callable], optional): Function to call after each batch with
                progress information. Defaults to None.
            **kwargs: Additional arguments to pass to the processing function.

        Returns:
            Any: The combined results from all batches.
        """
        # Get list of unique cell indices that contain points
        unique_cells = np.unique(self.grid_indices)
        total_cells = len(unique_cells)

        # Create a results list to store output for each batch
        results = np.zeros((len(self.points), 3))

        # Process each grid cell
        for i, cell_idx in enumerate(unique_cells):
            # Get batch points and primary mask
            batch_indices, is_primary = self.get_batch_for_grid_cell(cell_idx)

            if len(batch_indices) == 0:
                continue

            # Get the actual points for this batch
            batch_points = self.points[batch_indices]

            # Process this batch
            batch_result = processing_func(batch_points, **kwargs)

            # Store results for primary points only
            primary_batch_indices = batch_indices[is_primary]
            primary_batch_results = batch_result[is_primary] if hasattr(batch_result, '__getitem__') else batch_result

            if hasattr(batch_result, '__getitem__'):
                results[primary_batch_indices] = primary_batch_results

            # Report progress
            print(f"Processed batch {i + 1}/{total_cells}, {len(batch_points)} points")

        # TODO: Implement advanced result merging strategies for different analysis types
        # TODO: Handle different result formats (arrays, lists, objects, etc.)

        # Convert results to appropriate format
        if all(r is None for r in results):
            return None

        # Try to convert to numpy array if possible
        try:
            return np.array(results)
        except:
            return results

    # TODO: Add method for merging results from different batches (e.g., for DBSCAN clusters)
    # TODO: Add specialized batch processing methods for different analysis types
    # TODO: Add validation and error handling for the processing function


