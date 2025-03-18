# services/batch_processor.py
"""
Spatial grid-based batch processor for efficient point cloud processing.

This module provides functionality to divide large point clouds into smaller batches
for efficient processing, using a spatial grid approach with overlapping regions.
"""

import numpy as np
from typing import Callable, Any, Dict, Tuple, Optional


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

    # services/batch_processor.py

    # services/batch_processor.py

    def cluster_in_batches(self, clustering_func, min_points=5, eps=0.05, progress_callback=None, **kwargs):
        """
        Process large point clouds in batches using the provided clustering function.

        This method divides the point cloud into overlapping spatial grid cells,
        clusters each cell independently, and then reconciles the results to maintain
        consistent global cluster labeling, with special handling to avoid
        incorrectly classifying boundary points as noise.

        Args:
            clustering_func (Callable): Function that performs clustering on a batch of points.
                Should accept (points, eps, min_points, **kwargs) and return cluster labels.
            min_points (int): Minimum number of points required to form a cluster
            eps (float): Maximum distance between two points to be considered neighbors
            progress_callback (Callable, optional): Function to report progress.
                Should accept (current_step, total_steps, stage_name) arguments.
            **kwargs: Additional arguments to pass to the clustering function

        Returns:
            numpy.ndarray: Array of cluster labels for each point in the original point cloud
        """
        # Get list of unique cell indices that contain points
        unique_cells = np.unique(self.grid_indices)
        total_cells = len(unique_cells)

        # Total steps for progress reporting
        total_steps = total_cells + 4

        # Initialize arrays to store results
        all_labels = np.full(len(self.points), -1, dtype=np.int32)  # Default to noise

        # Initialize progress
        if progress_callback:
            progress_callback(0, total_steps, "Initializing batch clustering")
        print(f"Processing {total_cells} grid cells for clustering...")

        # PHASE 1: Perform clustering on each batch independently
        # -------------------------------------------------------

        # Dictionary to track which points are in which cells (for overlapping regions)
        # Maps point_idx -> list of (cell_idx, local_cluster_id) assignments
        point_assignments = {}

        # Lists to store cell data for efficient processing
        cell_data = []  # Will store (cell_idx, batch_indices, is_primary, batch_labels)

        # Process each cell
        for batch_idx, cell_idx in enumerate(unique_cells):
            # Get batch points and primary mask
            batch_indices, is_primary = self.get_batch_for_grid_cell(cell_idx)

            if len(batch_indices) == 0:
                continue

            # Get the actual points for this batch
            batch_points = self.points[batch_indices]

            # Process this batch with the provided clustering function
            batch_labels = clustering_func(batch_points, eps=eps, min_points=min_points, **kwargs)

            # Store the cell data for later processing
            cell_data.append((cell_idx, batch_indices, is_primary, batch_labels))

            # Track ALL points and their cluster assignments
            for i, point_idx in enumerate(batch_indices):
                label = batch_labels[i]
                if point_idx not in point_assignments:
                    point_assignments[point_idx] = []

                # Store both cluster and noise assignments - we'll sort them out later
                point_assignments[point_idx].append((cell_idx, label))

            # Report progress
            if progress_callback:
                progress_callback(batch_idx + 1, total_steps, f"Processed cell {batch_idx + 1}/{total_cells}")

            # Count non-noise clusters in this cell for logging
            num_clusters = len(np.unique(batch_labels[batch_labels >= 0]))
            print(f"Processed cell {batch_idx + 1}/{total_cells}, found {num_clusters} local clusters")

        # PHASE 2: Resolve noise vs. cluster conflicts
        # --------------------------------------------
        if progress_callback:
            progress_callback(total_cells + 1, total_steps, "Resolving noise classifications")

        print("Resolving noise classifications...")

        # For each point, prefer any cluster assignment over noise
        # This is critical for points near cell boundaries that might be
        # incorrectly classified as noise in some cells
        preferred_assignments = {}  # Maps point_idx -> (cell_idx, preferred_local_id)

        for point_idx, assignments in point_assignments.items():
            # First, try to find any non-noise assignment
            non_noise_assignments = [(cell, label) for cell, label in assignments if label >= 0]

            if non_noise_assignments:
                # If we have non-noise assignments, pick the one with the highest label
                # (this is a heuristic, as higher labels often represent larger clusters in DBSCAN)
                preferred_assignments[point_idx] = max(non_noise_assignments, key=lambda x: x[1])
            else:
                # If all assignments are noise, keep it as noise
                preferred_assignments[point_idx] = (-1, -1)  # Special marker for noise

        # PHASE 3: Build cluster relationship graph
        # -----------------------------------------
        if progress_callback:
            progress_callback(total_cells + 2, total_steps, "Building cluster relationship graph")

        print("Building cluster relationship graph...")

        # Maps (cell_idx, local_id) -> set of points in this cluster
        cluster_points = {}

        # Populate cluster_points from preferred assignments
        for point_idx, (cell_idx, local_id) in preferred_assignments.items():
            if local_id >= 0:  # Only include non-noise clusters
                key = (cell_idx, local_id)
                if key not in cluster_points:
                    cluster_points[key] = set()
                cluster_points[key].add(point_idx)

        # Build the graph of connected clusters
        # Two clusters are connected if they share points in the overlapping regions
        cluster_connections = {}  # Maps cluster key -> set of connected cluster keys

        # For each point, find all cells where it's part of a cluster
        for point_idx, assignments in point_assignments.items():
            # Get all cluster assignments for this point (excluding noise)
            cluster_assignments = [(cell, label) for cell, label in assignments if label >= 0]

            # If the point is in multiple clusters
            if len(cluster_assignments) > 1:
                # Create connections between all pairs of clusters
                for i in range(len(cluster_assignments)):
                    for j in range(i + 1, len(cluster_assignments)):
                        cluster1 = cluster_assignments[i]
                        cluster2 = cluster_assignments[j]

                        # Add bidirectional connection
                        if cluster1 not in cluster_connections:
                            cluster_connections[cluster1] = set()
                        if cluster2 not in cluster_connections:
                            cluster_connections[cluster2] = set()

                        cluster_connections[cluster1].add(cluster2)
                        cluster_connections[cluster2].add(cluster1)

        # PHASE 4: Find connected components (to form global clusters)
        # -----------------------------------------------------------
        if progress_callback:
            progress_callback(total_cells + 3, total_steps, "Finding global clusters")

        print("Finding global clusters...")

        # Use a breadth-first search to find connected components
        visited = set()
        global_cluster_map = {}  # Maps (cell_idx, local_id) -> global_id
        next_global_id = 0

        # Find connected components
        for cluster_key in cluster_points.keys():
            if cluster_key in visited:
                continue

            # Start a new connected component
            component = []
            queue = [cluster_key]

            # BFS to find all connected clusters
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.append(current)

                # Add all connected clusters to the queue
                for neighbor in cluster_connections.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            # For this component, calculate its total size
            total_size = sum(len(cluster_points.get(key, set())) for key in component)

            # Only create a global cluster if it has enough points
            if total_size >= min_points:
                # Assign all clusters in this component to the same global ID
                for member in component:
                    global_cluster_map[member] = next_global_id

                next_global_id += 1

        # PHASE 5: Apply global cluster labels to all points
        # -------------------------------------------------
        if progress_callback:
            progress_callback(total_cells + 4, total_steps, "Applying global cluster labels")

        print("Applying global cluster labels...")

        # Initialize final labels array
        final_labels = np.full(len(self.points), -1, dtype=np.int32)

        # Apply the global cluster mapping to all points
        for point_idx, (cell_idx, local_id) in preferred_assignments.items():
            if local_id >= 0:  # Non-noise point
                # Get its global cluster ID
                global_id = global_cluster_map.get((cell_idx, local_id), -1)
                final_labels[point_idx] = global_id

        # Final report
        unique_clusters = np.unique(final_labels)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

        if progress_callback:
            progress_callback(total_steps, total_steps, f"Clustering complete: {num_clusters} clusters found")

        print(f"Clustering complete. Found {num_clusters} global clusters.")

        return final_labels

    # TODO: Add specialized batch processing methods for different analysis types
    # TODO: Add validation and error handling for the processing function


