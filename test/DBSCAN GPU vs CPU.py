import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.cluster import DBSCAN as SklearnDBSCAN
import open3d as o3d

# Try to import GPU libraries, use flags to track availability
CUPY_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
    print("CuPy is available for GPU acceleration")
except ImportError:
    print("CuPy not available, skipping CuPy-based GPU implementation")

try:
    import cuml
    from cuml.cluster import DBSCAN as CumlDBSCAN

    CUML_AVAILABLE = True
    print("RAPIDS cuML is available for GPU-accelerated DBSCAN")
except ImportError:
    print("RAPIDS cuML not available, skipping cuML-based GPU implementation")


class DBSCANBenchmark:
    """Class for benchmarking DBSCAN implementations on CPU and GPU."""

    def __init__(self):
        self.results = {}

    def generate_test_data(self, num_points, num_clusters=5, noise_ratio=0.1, dimensions=3):
        """
        Generate synthetic point cloud data with clusters for DBSCAN testing.

        Args:
            num_points (int): Total number of points to generate
            num_clusters (int): Number of clusters to generate
            noise_ratio (float): Ratio of points that should be noise (0-1)
            dimensions (int): Number of dimensions (typically 3 for point clouds)

        Returns:
            numpy.ndarray: Generated point cloud data
        """
        # Calculate points per cluster and noise points
        noise_points = int(num_points * noise_ratio)
        points_per_cluster = (num_points - noise_points) // num_clusters

        # Generate cluster centers
        centers = np.random.uniform(-10, 10, size=(num_clusters, dimensions))

        # Generate clusters around centers
        clusters = []
        for i in range(num_clusters):
            # Random standard deviation for each cluster to make them different sizes
            std_dev = np.random.uniform(0.2, 1.0)
            cluster_points = np.random.normal(centers[i], std_dev, size=(points_per_cluster, dimensions))
            clusters.append(cluster_points)

        # Generate noise
        noise = np.random.uniform(-15, 15, size=(noise_points, dimensions))

        # Combine all points
        all_points = np.vstack([*clusters, noise]).astype(np.float32)

        # Shuffle the points
        np.random.shuffle(all_points)

        return all_points

    def run_sklearn_dbscan(self, data, eps, min_samples):
        """Run DBSCAN using scikit-learn (CPU implementation)."""
        start_time = time.time()
        dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(data)
        end_time = time.time()

        execution_time = end_time - start_time
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            'labels': labels,
            'execution_time': execution_time,
            'num_clusters': num_clusters
        }

    def run_open3d_dbscan(self, data, eps, min_samples):
        """Run DBSCAN using Open3D (CPU implementation)."""
        start_time = time.time()

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)

        # Run DBSCAN
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))

        end_time = time.time()

        execution_time = end_time - start_time
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            'labels': labels,
            'execution_time': execution_time,
            'num_clusters': num_clusters
        }

    def run_cupy_dbscan(self, data, eps, min_samples):
        """
        Run DBSCAN using CuPy-based implementation with vectorized operations (GPU).

        This implementation uses GPU-optimized vector operations wherever possible
        and minimizes iterations to maximize GPU utilization.
        """
        if not CUPY_AVAILABLE:
            return {
                'labels': None,
                'execution_time': float('inf'),
                'num_clusters': 0,
                'error': 'CuPy not available'
            }

        try:
            start_time = time.time()

            # Transfer data to GPU
            cp_data = cp.asarray(data, dtype=cp.float32)
            n_points = len(cp_data)

            # Initialize arrays
            labels = cp.full(n_points, -1, dtype=cp.int32)
            visited = cp.zeros(n_points, dtype=cp.bool_)

            # Prepare for metrics
            timing_details = {}

            # Choose strategy based on dataset size
            if n_points <= 10000:  # Only very small datasets can use the distance matrix approach
                # Compute the distance matrix vectorized (much faster than iterations)
                dist_computation_start = time.time()

                # Use broadcasting to compute pairwise distances in a vectorized way
                # This creates a matrix of shape (n_points, n_points)
                diff = cp_data[:, None, :] - cp_data[None, :, :]  # Broadcasting for all pairs
                distances = cp.sqrt(cp.sum(diff ** 2, axis=2))

                timing_details['distance_computation'] = time.time() - dist_computation_start

                # Find neighbors for each point (points within eps distance)
                neighbors_mask = distances <= eps

                # Count neighbors to identify core points (vectorized)
                core_points_start = time.time()
                neighbor_counts = cp.sum(neighbors_mask, axis=1)
                core_points = neighbor_counts >= min_samples
                timing_details['core_points_identification'] = time.time() - core_points_start

                # Perform clustering
                clustering_start = time.time()
                cluster_id = 0

                # Flatten the neighbors_mask to reduce memory usage after computing core points
                # We'll recompute neighbors as needed
                del neighbors_mask

                # Process each point - this part still needs iteration
                # but we minimize work inside the loop and use vectorized operations
                for i in range(n_points):
                    if visited[i] or not core_points[i]:
                        continue

                    # Mark as visited
                    visited[i] = True
                    labels[i] = cluster_id

                    # Get neighbors efficiently using the precomputed distance matrix
                    neighbors = cp.where(distances[i] <= eps)[0]

                    # Use a queue-based approach instead of recursion
                    # Process points in the current cluster
                    queue = neighbors.copy()
                    while len(queue) > 0:
                        # Process a batch of points in the queue for better GPU utilization
                        current_batch = queue
                        queue = cp.empty(0, dtype=cp.int64)

                        # Mark all unvisited points in batch as part of the cluster
                        unvisited_mask = ~visited[current_batch]
                        visited[current_batch[unvisited_mask]] = True
                        labels[current_batch[unvisited_mask]] = cluster_id

                        # Find core points in the batch that need further expansion
                        batch_core_points = current_batch[core_points[current_batch] & unvisited_mask]

                        if len(batch_core_points) > 0:
                            # Get all neighbors of all core points in the batch
                            # Use a vectorized approach to find all neighbors at once
                            batch_neighbors_mask = distances[batch_core_points] <= eps

                            # Get indices of new neighbors not yet visited
                            # This is a bit complex but fully vectorized
                            all_batch_neighbors = cp.where(batch_neighbors_mask.reshape(-1))[0] % n_points
                            unique_neighbors = cp.unique(all_batch_neighbors)

                            # Add unvisited neighbors to the queue
                            new_queue = unique_neighbors[~visited[unique_neighbors]]
                            queue = cp.concatenate([queue, new_queue])

                    cluster_id += 1

                timing_details['clustering'] = time.time() - clustering_start

            else:
                # For large datasets, use a batch-based approach to avoid OOM errors
                # This is more complex but necessary for datasets that would exceed GPU memory

                batch_computation_start = time.time()

                # Use a spatial partitioning approach to avoid full distance matrix
                # Divide space into cells and only compute distances within nearby cells

                # First, find the data range for partitioning
                mins = cp.min(cp_data, axis=0)
                maxs = cp.max(cp_data, axis=0)

                # Determine cell size based on eps
                cell_size = eps

                # Create grid dimensions
                grid_dims = cp.ceil((maxs - mins) / cell_size).astype(cp.int32)

                # Assign points to cells (vectorized)
                cell_indices = cp.floor((cp_data - mins) / cell_size).astype(cp.int32)

                # Create a unique cell ID for each point
                # For 3D data: cell_id = x + y*dim_x + z*dim_x*dim_y
                if cp_data.shape[1] == 3:  # 3D point cloud
                    point_cell_ids = (cell_indices[:, 0] +
                                      cell_indices[:, 1] * grid_dims[0] +
                                      cell_indices[:, 2] * grid_dims[0] * grid_dims[1])
                else:  # Handle 2D point cloud
                    point_cell_ids = cell_indices[:, 0] + cell_indices[:, 1] * grid_dims[0]

                # Get unique cells and points in each cell
                unique_cell_ids = cp.unique(point_cell_ids)

                # Identify core points using the cell structure
                core_points = cp.zeros(n_points, dtype=cp.bool_)

                # Process each cell and its neighbors
                batch_size = min(1000, n_points // 10)  # Adaptive batch size

                # Process in batches for better memory management
                for i in range(0, len(unique_cell_ids), batch_size):
                    batch_cells = unique_cell_ids[i:i + batch_size]

                    # For each cell, get points and neighbors
                    for cell_id in batch_cells:
                        # Get points in this cell
                        cell_points_mask = point_cell_ids == cell_id
                        cell_points = cp.where(cell_points_mask)[0]

                        if len(cell_points) == 0:
                            continue

                        # Get cell coordinates
                        if cp_data.shape[1] == 3:  # 3D
                            cell_z = cell_id // (grid_dims[0] * grid_dims[1])
                            temp = cell_id % (grid_dims[0] * grid_dims[1])
                            cell_y = temp // grid_dims[0]
                            cell_x = temp % grid_dims[0]

                            # Get neighboring cell IDs (27 neighbors in 3D)
                            neighbor_cells = []
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    for dz in [-1, 0, 1]:
                                        nx, ny, nz = cell_x + dx, cell_y + dy, cell_z + dz
                                        if (0 <= nx < grid_dims[0] and
                                                0 <= ny < grid_dims[1] and
                                                0 <= nz < grid_dims[2]):
                                            n_id = nx + ny * grid_dims[0] + nz * grid_dims[0] * grid_dims[1]
                                            neighbor_cells.append(n_id)
                        else:  # 2D
                            cell_y = cell_id // grid_dims[0]
                            cell_x = cell_id % grid_dims[0]

                            # Get neighboring cell IDs (9 neighbors in 2D)
                            neighbor_cells = []
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = cell_x + dx, cell_y + dy
                                    if (0 <= nx < grid_dims[0] and
                                            0 <= ny < grid_dims[1]):
                                        n_id = nx + ny * grid_dims[0]
                                        neighbor_cells.append(n_id)

                        # Convert to CuPy array
                        neighbor_cells = cp.array(neighbor_cells)

                        # Get all points from neighboring cells
                        neighbor_points_mask = cp.isin(point_cell_ids, neighbor_cells)
                        neighbor_points = cp.where(neighbor_points_mask)[0]

                        # For each point in the current cell, compute distances to neighbors
                        for point_idx in cell_points:
                            # Compute distances vectorized
                            dists = cp.sqrt(cp.sum((cp_data[neighbor_points] - cp_data[point_idx]) ** 2, axis=1))

                            # Count points within eps
                            neighbor_count = cp.sum(dists <= eps)

                            # Mark as core point if it meets the criteria
                            if neighbor_count >= min_samples:
                                core_points[point_idx] = True

                timing_details['batch_computation'] = time.time() - batch_computation_start

                # Now perform the clustering using similar approach as before
                clustering_start = time.time()
                cluster_id = 0

                for i in range(n_points):
                    if visited[i] or not core_points[i]:
                        continue

                    # Mark as visited
                    visited[i] = True
                    labels[i] = cluster_id

                    # Build a queue for BFS
                    queue = cp.array([i], dtype=cp.int32)

                    while len(queue) > 0:
                        current_point = queue[0]
                        queue = queue[1:]

                        # Get current point's cell and neighboring cells
                        current_cell = point_cell_ids[current_point]

                        # Calculate cell coordinates
                        if cp_data.shape[1] == 3:  # 3D
                            cell_z = current_cell // (grid_dims[0] * grid_dims[1])
                            temp = current_cell % (grid_dims[0] * grid_dims[1])
                            cell_y = temp // grid_dims[0]
                            cell_x = temp % grid_dims[0]

                            # Get neighboring cell IDs
                            neighbor_cells = []
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    for dz in [-1, 0, 1]:
                                        nx, ny, nz = cell_x + dx, cell_y + dy, cell_z + dz
                                        if (0 <= nx < grid_dims[0] and
                                                0 <= ny < grid_dims[1] and
                                                0 <= nz < grid_dims[2]):
                                            n_id = nx + ny * grid_dims[0] + nz * grid_dims[0] * grid_dims[1]
                                            neighbor_cells.append(n_id)
                        else:  # 2D
                            cell_y = current_cell // grid_dims[0]
                            cell_x = current_cell % grid_dims[0]

                            # Get neighboring cell IDs
                            neighbor_cells = []
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    nx, ny = cell_x + dx, cell_y + dy
                                    if (0 <= nx < grid_dims[0] and
                                            0 <= ny < grid_dims[1]):
                                        n_id = nx + ny * grid_dims[0]
                                        neighbor_cells.append(n_id)

                        # Convert to CuPy array
                        neighbor_cells = cp.array(neighbor_cells)

                        # Get all points from neighboring cells
                        neighbor_points_mask = cp.isin(point_cell_ids, neighbor_cells)
                        potential_neighbors = cp.where(neighbor_points_mask)[0]

                        # Compute distances vectorized to find actual neighbors
                        dists = cp.sqrt(cp.sum((cp_data[potential_neighbors] - cp_data[current_point]) ** 2, axis=1))
                        neighbors = potential_neighbors[dists <= eps]

                        # Find unvisited neighbors
                        unvisited_neighbors = neighbors[~visited[neighbors]]

                        # Mark all unvisited neighbors
                        visited[unvisited_neighbors] = True
                        labels[unvisited_neighbors] = cluster_id

                        # Add core points to the queue
                        core_neighbors = unvisited_neighbors[core_points[unvisited_neighbors]]
                        if len(core_neighbors) > 0:
                            queue = cp.concatenate([queue, core_neighbors])

                    cluster_id += 1

                timing_details['clustering'] = time.time() - clustering_start

            # Transfer results back to CPU
            labels_cpu = cp.asnumpy(labels)

            end_time = time.time()
            execution_time = end_time - start_time

            # Count number of clusters (excluding noise points labeled as -1)
            num_clusters = len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)

            # Print timing details for debugging
            # print(f"CuPy DBSCAN timing details: {timing_details}")

            return {
                'labels': labels_cpu,
                'execution_time': execution_time,
                'num_clusters': num_clusters,
                'timing_details': timing_details
            }

        except Exception as e:
            print(f"Error in CuPy DBSCAN: {e}")
            return {
                'labels': None,
                'execution_time': float('inf'),
                'num_clusters': 0,
                'error': str(e)
            }
        finally:
            # Clean up GPU memory
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

    def run_cuml_dbscan(self, data, eps, min_samples):
        """Run DBSCAN using RAPIDS cuML (GPU implementation)."""
        if not CUML_AVAILABLE:
            return {
                'labels': None,
                'execution_time': float('inf'),
                'num_clusters': 0,
                'error': 'RAPIDS cuML not available'
            }

        try:
            start_time = time.time()

            # Create cuML DBSCAN instance
            dbscan = CumlDBSCAN(eps=eps, min_samples=min_samples)

            # Convert to CuPy array if not already
            if CUPY_AVAILABLE:
                data_gpu = cp.asarray(data)
                labels = dbscan.fit_predict(data_gpu)
                labels_cpu = cp.asnumpy(labels)
            else:
                # cuML can also accept numpy arrays directly
                labels = dbscan.fit_predict(data)
                labels_cpu = labels

            end_time = time.time()

            execution_time = end_time - start_time
            num_clusters = len(set(labels_cpu)) - (1 if -1 in labels_cpu else 0)

            return {
                'labels': labels_cpu,
                'execution_time': execution_time,
                'num_clusters': num_clusters
            }

        except Exception as e:
            print(f"Error in cuML DBSCAN: {e}")
            return {
                'labels': None,
                'execution_time': float('inf'),
                'num_clusters': 0,
                'error': str(e)
            }
        finally:
            # Clean up GPU memory
            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

    def benchmark(self, point_sizes, eps=0.5, min_samples=5, runs=3):
        """
        Run benchmarks for different implementations across multiple point cloud sizes.

        Args:
            point_sizes (list): List of point cloud sizes to test
            eps (float): DBSCAN epsilon parameter
            min_samples (int): DBSCAN min_samples parameter
            runs (int): Number of runs for each test for more reliable results

        Returns:
            dict: Dictionary of results
        """
        results = {
            'sklearn': [],
            'open3d': [],
            'cupy': [],
            'cuml': []
        }

        for size in tqdm(point_sizes, desc="Testing different point cloud sizes"):
            sklearn_times = []
            open3d_times = []
            cupy_times = []
            cuml_times = []

            for run in range(runs):
                # Generate fresh test data for each run
                print(f"Generating {size:,} points for run {run + 1}/{runs}...")
                data = self.generate_test_data(size)

                # Skip very large datasets for CPU implementations
                skip_cpu = size > 1_000_000

                # Run scikit-learn DBSCAN (CPU)
                if not skip_cpu:
                    print(f"Running scikit-learn DBSCAN on {size:,} points...")
                    sklearn_result = self.run_sklearn_dbscan(data, eps, min_samples)
                    sklearn_times.append(sklearn_result['execution_time'])
                    print(
                        f"  Time: {sklearn_result['execution_time']:.4f}s, Clusters: {sklearn_result['num_clusters']}")

                # Run Open3D DBSCAN (CPU)
                if not skip_cpu:
                    print(f"Running Open3D DBSCAN on {size:,} points...")
                    open3d_result = self.run_open3d_dbscan(data, eps, min_samples)
                    open3d_times.append(open3d_result['execution_time'])
                    print(f"  Time: {open3d_result['execution_time']:.4f}s, Clusters: {open3d_result['num_clusters']}")

                # Run CuPy DBSCAN implementation (GPU)
                if CUPY_AVAILABLE:
                    print(f"Running CuPy DBSCAN on {size:,} points...")
                    cupy_result = self.run_cupy_dbscan(data, eps, min_samples)
                    if 'error' not in cupy_result:
                        cupy_times.append(cupy_result['execution_time'])
                        print(f"  Time: {cupy_result['execution_time']:.4f}s, Clusters: {cupy_result['num_clusters']}")
                    else:
                        print(f"  Error: {cupy_result['error']}")

                # Run cuML DBSCAN (GPU)
                if CUML_AVAILABLE:
                    print(f"Running cuML DBSCAN on {size:,} points...")
                    cuml_result = self.run_cuml_dbscan(data, eps, min_samples)
                    if 'error' not in cuml_result:
                        cuml_times.append(cuml_result['execution_time'])
                        print(f"  Time: {cuml_result['execution_time']:.4f}s, Clusters: {cuml_result['num_clusters']}")
                    else:
                        print(f"  Error: {cuml_result['error']}")

                print()  # Empty line between runs

            # Record average times
            if not skip_cpu:
                results['sklearn'].append(np.mean(sklearn_times) if sklearn_times else float('inf'))
                results['open3d'].append(np.mean(open3d_times) if open3d_times else float('inf'))
            else:
                results['sklearn'].append(float('inf'))
                results['open3d'].append(float('inf'))

            results['cupy'].append(np.mean(cupy_times) if cupy_times else float('inf'))
            results['cuml'].append(np.mean(cuml_times) if cuml_times else float('inf'))

        self.results = {
            'point_sizes': point_sizes,
            'times': results
        }

        return self.results

    def plot_results(self, save_path=None):
        """Plot the benchmark results."""
        if not self.results:
            print("No results to plot. Run benchmark() first.")
            return

        point_sizes = self.results['point_sizes']
        times = self.results['times']

        # Convert point sizes to millions for better readability
        sizes_in_millions = [size / 1_000_000 for size in point_sizes]

        plt.figure(figsize=(12, 7))

        # Plot times for each implementation
        if np.any(np.array(times['sklearn']) < float('inf')):
            plt.plot(sizes_in_millions, times['sklearn'], 'o-', label='scikit-learn (CPU)')

        if np.any(np.array(times['open3d']) < float('inf')):
            plt.plot(sizes_in_millions, times['open3d'], 's-', label='Open3D (CPU)')

        if np.any(np.array(times['cupy']) < float('inf')):
            plt.plot(sizes_in_millions, times['cupy'], '^-', label='CuPy (GPU)')

        if np.any(np.array(times['cuml']) < float('inf')):
            plt.plot(sizes_in_millions, times['cuml'], 'd-', label='cuML (GPU)')

        plt.xlabel('Number of Points (millions)')
        plt.ylabel('Time (seconds)')
        plt.title('CPU vs GPU Performance for DBSCAN Clustering')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add log scale plot
        plt.figure(figsize=(12, 7))

        # Plot times for each implementation on log scale
        if np.any(np.array(times['sklearn']) < float('inf')):
            plt.semilogy(sizes_in_millions, times['sklearn'], 'o-', label='scikit-learn (CPU)')

        if np.any(np.array(times['open3d']) < float('inf')):
            plt.semilogy(sizes_in_millions, times['open3d'], 's-', label='Open3D (CPU)')

        if np.any(np.array(times['cupy']) < float('inf')):
            plt.semilogy(sizes_in_millions, times['cupy'], '^-', label='CuPy (GPU)')

        if np.any(np.array(times['cuml']) < float('inf')):
            plt.semilogy(sizes_in_millions, times['cuml'], 'd-', label='cuML (GPU)')

        plt.xlabel('Number of Points (millions)')
        plt.ylabel('Time (seconds) - Log Scale')
        plt.title('CPU vs GPU Performance for DBSCAN Clustering (Log Scale)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Create speedup plot if GPU implementations are available
        if (np.any(np.array(times['cupy']) < float('inf')) or
                np.any(np.array(times['cuml']) < float('inf'))):

            plt.figure(figsize=(12, 7))

            # Use the best CPU time for comparisons
            cpu_times = np.minimum(
                np.array(times['sklearn']),
                np.array(times['open3d'])
            )

            # Calculate speedups
            if np.any(np.array(times['cupy']) < float('inf')):
                cupy_speedups = [
                    cpu / gpu if gpu > 0 and gpu < float('inf') and cpu < float('inf') else 0
                    for cpu, gpu in zip(cpu_times, times['cupy'])
                ]
                plt.plot(sizes_in_millions, cupy_speedups, '^-', label='CuPy Speedup')

            if np.any(np.array(times['cuml']) < float('inf')):
                cuml_speedups = [
                    cpu / gpu if gpu > 0 and gpu < float('inf') and cpu < float('inf') else 0
                    for cpu, gpu in zip(cpu_times, times['cuml'])
                ]
                plt.plot(sizes_in_millions, cuml_speedups, 'd-', label='cuML Speedup')

            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
            plt.xlabel('Number of Points (millions)')
            plt.ylabel('Speedup Factor (CPU time / GPU time)')
            plt.title('GPU Speedup Factor for DBSCAN Clustering')
            plt.grid(True, alpha=0.3)
            plt.legend()

        plt.tight_layout()

        # Save plots if requested
        if save_path:
            plt.savefig(save_path)

        plt.show()

    def print_summary(self):
        """Print a summary of the benchmark results."""
        if not self.results:
            print("No results to summarize. Run benchmark() first.")
            return

        point_sizes = self.results['point_sizes']
        times = self.results['times']

        print("\n==== DBSCAN Benchmark Summary ====")
        print("Point Cloud Sizes (points) | scikit-learn (CPU) | Open3D (CPU) | CuPy (GPU) | cuML (GPU)")
        print("-" * 90)

        for i, size in enumerate(point_sizes):
            sk_time = f"{times['sklearn'][i]:.4f}s" if times['sklearn'][i] < float('inf') else "N/A"
            o3d_time = f"{times['open3d'][i]:.4f}s" if times['open3d'][i] < float('inf') else "N/A"
            cupy_time = f"{times['cupy'][i]:.4f}s" if times['cupy'][i] < float('inf') else "N/A"
            cuml_time = f"{times['cuml'][i]:.4f}s" if times['cuml'][i] < float('inf') else "N/A"

            print(f"{size:20,} | {sk_time:17} | {o3d_time:12} | {cupy_time:10} | {cuml_time:10}")

        print("\n==== Speedup Factors (vs. best CPU implementation) ====")
        print("Point Cloud Sizes (points) | CuPy Speedup | cuML Speedup")
        print("-" * 60)

        for i, size in enumerate(point_sizes):
            # Get best CPU time
            cpu_time = min(times['sklearn'][i], times['open3d'][i])

            # Calculate speedups
            cupy_speedup = cpu_time / times['cupy'][i] if times['cupy'][i] > 0 and times['cupy'][i] < float(
                'inf') and cpu_time < float('inf') else float('nan')
            cuml_speedup = cpu_time / times['cuml'][i] if times['cuml'][i] > 0 and times['cuml'][i] < float(
                'inf') and cpu_time < float('inf') else float('nan')

            cupy_str = f"{cupy_speedup:.2f}x" if not np.isnan(cupy_speedup) else "N/A"
            cuml_str = f"{cuml_speedup:.2f}x" if not np.isnan(cuml_speedup) else "N/A"

            print(f"{size:20,} | {cupy_str:12} | {cuml_str:12}")


def main():
    # Create benchmark instance
    benchmark = DBSCANBenchmark()

    # Define point cloud sizes to test
    # Start small and increase gradually
    point_sizes = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    # Add larger sizes if cuML is available
    if CUML_AVAILABLE:
        point_sizes.extend([2_000_000, 5_000_000, 10_000_000])

    # Run benchmark
    print(f"Running DBSCAN benchmark with point sizes: {[f'{size:,}' for size in point_sizes]}")
    results = benchmark.benchmark(
        point_sizes=point_sizes,
        eps=0.5,  # DBSCAN epsilon parameter
        min_samples=5,  # DBSCAN min_samples parameter
        runs=2  # Number of runs for each test
    )

    # Print summary
    benchmark.print_summary()

    # Plot results
    benchmark.plot_results(save_path="dbscan_benchmark_results.png")


if __name__ == "__main__":
    main()