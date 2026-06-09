# services/batch_processor.py
"""
Spatial grid-based batch processor for efficient point cloud processing.

This module provides functionality to divide large point clouds into smaller batches
for efficient processing, using a spatial grid approach with overlapping regions.
"""

import numpy as np
from typing import Callable, Any, Dict, Tuple, Optional

from config.config import global_variables


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
            overlap_percent: float = 0.1,
            max_batch_size: int = None
    ):
        """
        Initialize the BatchProcessor with point cloud and configuration.

        Args:
            points (np.ndarray): Point cloud data of shape (n, 3).
            batch_size (int, optional): Maximum number of PRIMARY points per leaf cell
                (the "budget"). The cloud is split adaptively so every leaf stays at or
                below this count. Defaults to 100000.
            overlap_percent (float, optional): Percentage of overlap between adjacent cells (0-1).
                Defaults to 0.1 (10%).
            max_batch_size (int, optional): Hard cap on primary + halo points returned
                per tile. When the overlap halo would push a tile past this, the
                farthest halo points are dropped (primaries are always kept), which
                bounds KNN cost and VRAM regardless of how dense a neighbouring tile
                is. None = no cap. Defaults to None.
        """
        self.points = points
        self.batch_size = batch_size
        self.overlap_percent = overlap_percent
        self.max_batch_size = max_batch_size

        # Will be computed in create_spatial_grid
        self.grid_indices = None
        self.grid_dimensions = None
        self.cell_size = None
        self.grid_bounds = None
        self.cell_bounds = {}  # Maps cell_id → (min_bounds, max_bounds) per leaf

        # Initialize the spatial grid
        self.create_spatial_grid()

    def create_spatial_grid(self):
        """
        Partition the cloud into adaptive leaf cells, each holding <= batch_size points.

        Instead of a fixed-size grid, the cloud is split top-down by a balanced
        (median) k-d partition (see _build_adaptive_cells): a cell is cut only while
        it holds more than batch_size points, so sparse regions stay as a few big
        leaves while dense regions keep subdividing. Every leaf is within budget by
        construction, and each leaf stores tight bounds that drive the overlap halo.
        """
        min_bounds = np.min(self.points, axis=0)
        max_bounds = np.max(self.points, axis=0)

        self.grid_bounds = {'min': min_bounds, 'max': max_bounds}

        # cell_size / grid_dimensions are retained only for the legacy
        # non-subdivided fallback in get_batch_for_grid_cell; every adaptive leaf
        # carries its own tight bounds in self.cell_bounds, so that branch is dead
        # in practice.
        self.cell_size = np.maximum(max_bounds - min_bounds, 1e-9)
        self.grid_dimensions = (1, 1, 1)

        # Build the adaptive partition (assigns grid_indices + cell_bounds).
        self._build_adaptive_cells()

    def _build_adaptive_cells(self):
        """
        Partition the cloud into leaf cells each holding <= batch_size points,
        using a balanced (median) k-d split applied top-down.

        Starting from the whole cloud, the longest axis of a cell is cut at the
        median point index — exact 50/50 halves — until every leaf is within
        budget. Sparse regions stay as a few big leaves; dense regions subdivide.
        Each leaf stores its tight bounds in self.cell_bounds so the overlap halo
        (in get_batch_for_grid_cell) scales to local density.

        Works on point indices with an explicit work stack (no recursion, no
        point-array copies), so peak memory stays near one index array of N plus a
        single transient coordinate column — important for 10M+ point clouds.
        """
        n = len(self.points)
        self.grid_indices = np.empty(n, dtype=np.int64)

        stack = [np.arange(n, dtype=np.int64)]
        next_cell_id = 0

        while stack:
            idx = stack.pop()
            lo, hi = self._bounds_of(idx)

            extents = hi - lo
            axis = int(np.argmax(extents))

            # Leaf when within budget, or when the cell can't be split (all points
            # coincide). Coincident oversized leaves are rare; consumers already
            # sub-batch their own work.
            if len(idx) <= self.batch_size or extents[axis] <= 1e-12:
                self.grid_indices[idx] = next_cell_id
                self.cell_bounds[next_cell_id] = (lo, hi)
                next_cell_id += 1
                continue

            # Balanced split: exact median index along the longest axis. Splitting
            # by index (not by value) keeps the halves equal even when many points
            # share the median coordinate, so there is no degenerate empty side.
            coord = self.points[idx, axis]
            half = len(idx) // 2
            order = np.argpartition(coord, half)
            stack.append(idx[order[:half]])
            stack.append(idx[order[half:]])

    def _bounds_of(self, idx):
        """
        Tight (min, max) bounds of the points at the given indices, computed one
        axis at a time so no (m, 3) copy of the subset is materialized.
        """
        lo = np.empty(3, dtype=np.float64)
        hi = np.empty(3, dtype=np.float64)
        for d in range(3):
            col = self.points[idx, d]
            lo[d] = col.min()
            hi[d] = col.max()
        return lo, hi

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
        # Find primary points (those belonging directly to this cell)
        primary_mask = self.grid_indices == cell_idx
        primary_indices = np.where(primary_mask)[0]

        if len(primary_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=bool)

        if cell_idx in self.cell_bounds:
            # Subdivided cell — use stored bounds
            min_bounds, max_bounds = self.cell_bounds[cell_idx]
            overlap_size = (max_bounds - min_bounds) * self.overlap_percent
        else:
            # Original grid cell — compute bounds from 3D grid coordinates
            nx, ny, nz = self.grid_dimensions
            cell_z = cell_idx % nz
            cell_y = (cell_idx // nz) % ny
            cell_x = cell_idx // (ny * nz)
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

        # Cap the batch (primary + halo). An adaptive tile in a sparse region can
        # border a much denser tile; its overlap halo would otherwise scoop up a huge
        # slab of that dense cloud. Keep ALL primary points plus the halo points
        # nearest the cell box (the ones actually neighbouring boundary primaries) and
        # drop the far halo, which neighbours nothing here. Bounds KNN cost and VRAM.
        cap = self.max_batch_size
        if cap is not None and len(batch_indices) > cap and len(batch_indices) > len(primary_indices):
            halo = batch_indices[~primary_mask[batch_indices]]
            keep = cap - len(primary_indices)
            if keep <= 0:
                batch_indices = primary_indices
            elif keep < len(halo):
                hp = self.points[halo]
                # distance² from each halo point to the (tight) cell box
                d = np.maximum(0.0, np.maximum(min_bounds - hp, hp - max_bounds))
                nearest = halo[np.argpartition((d * d).sum(axis=1), keep)[:keep]]
                batch_indices = np.concatenate([primary_indices, nearest])

        # Which batch points are primary to this cell (vectorized lookup).
        is_primary = primary_mask[batch_indices]

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
            if global_variables.global_cancel_event.is_set():
                raise InterruptedError("Cancelled by user")

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

            # Report progress via singleton
            percent = int(((i + 1) / total_cells) * 100)
            global_variables.global_progress = (percent, f"Processing batch {i + 1}/{total_cells}")
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

    def cluster_in_batches(self, clustering_func, min_points=5, eps=0.05, **kwargs):
        """
        Process large point clouds in batches using the provided clustering function.

        This method divides the point cloud into overlapping spatial grid cells,
        clusters each cell independently, and then reconciles the results to maintain
        consistent global cluster labeling, with special handling to avoid
        incorrectly classifying boundary points as noise.

        Progress is reported via global_variables.global_progress singleton.

        Args:
            clustering_func (Callable): Function that performs clustering on a batch of points.
                Should accept (points, eps, min_points, **kwargs) and return cluster labels.
            min_points (int): Minimum number of points required to form a cluster
            eps (float): Maximum distance between two points to be considered neighbors
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
        global_variables.global_progress = (0, "Initializing batch clustering")
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
            if global_variables.global_cancel_event.is_set():
                raise InterruptedError("Cancelled by user")

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

            # Report progress via singleton
            percent = int(((batch_idx + 1) / total_steps) * 100)
            global_variables.global_progress = (percent, f"Clustering cell {batch_idx + 1}/{total_cells}")

            # Count non-noise clusters in this cell for logging
            num_clusters = len(np.unique(batch_labels[batch_labels >= 0]))
            print(f"Processed cell {batch_idx + 1}/{total_cells}, found {num_clusters} local clusters")

        # PHASE 2: Resolve noise vs. cluster conflicts
        # --------------------------------------------
        percent = int(((total_cells + 1) / total_steps) * 100)
        global_variables.global_progress = (percent, "Resolving noise classifications")

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
        percent = int(((total_cells + 2) / total_steps) * 100)
        global_variables.global_progress = (percent, "Building cluster graph")

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
        percent = int(((total_cells + 3) / total_steps) * 100)
        global_variables.global_progress = (percent, "Finding global clusters")

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
        percent = int(((total_cells + 4) / total_steps) * 100)
        global_variables.global_progress = (percent, "Applying global cluster labels")

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

        global_variables.global_progress = (100, f"Clustering complete: {num_clusters} clusters found")

        print(f"Clustering complete. Found {num_clusters} global clusters.")

        return final_labels

    # TODO: Add specialized batch processing methods for different analysis types
    # TODO: Add validation and error handling for the processing function


