import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Try to import CuPy, use a flag to track if it's available
USE_GPU = True
try:
    import cupy as cp
except ImportError:
    USE_GPU = False
    print("CuPy not available, only testing CPU implementation.")


def generate_test_data(num_points, num_features=6):
    """Generate synthetic point cloud data with specified number of points."""
    # Create points, colors, normals, and some custom attributes
    points = np.random.rand(num_points, 3).astype(np.float32)
    colors = np.random.rand(num_points, 3).astype(np.float32)
    normals = np.random.rand(num_points, 3).astype(np.float32)
    intensity = np.random.rand(num_points).astype(np.float32)
    dist_to_ground = np.random.rand(num_points).astype(np.float32)

    # Generate random mask (keeping around 50% of points)
    mask = np.random.rand(num_points) > 0.5

    # Create a dictionary simulating a point cloud object
    point_cloud = {
        'points': points,
        'colors': colors,
        'normals': normals,
        'intensity': intensity,
        'distToGround': dist_to_ground,
        'mask': mask,
    }

    return point_cloud


def mask_subset_cpu(point_cloud, mask):
    """Apply mask to point cloud data using CPU (NumPy)."""
    start_time = time.time()

    # Create a copy to simulate the non-inplace behavior
    result = {}

    # Apply mask to each attribute
    for key, value in point_cloud.items():
        if key != 'mask':  # Skip the mask itself
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    result[key] = value[mask]
                else:
                    result[key] = value[mask, ...]

    end_time = time.time()
    return result, end_time - start_time


def mask_subset_gpu(point_cloud, mask):
    """Apply mask to point cloud data using GPU (CuPy)."""
    if not USE_GPU:
        return None, float('inf')

    start_time = time.time()

    try:
        # Create a result dictionary
        result = {}

        # Transfer mask to GPU
        cp_mask = cp.asarray(mask)

        # Transfer and apply mask for each attribute
        for key, value in point_cloud.items():
            if key != 'mask':  # Skip the mask itself
                if isinstance(value, np.ndarray):
                    # Transfer data to GPU
                    cp_value = cp.asarray(value)

                    # Apply mask based on dimensionality
                    if cp_value.ndim == 1:
                        result[key] = cp.asnumpy(cp_value[cp_mask])
                    else:
                        result[key] = cp.asnumpy(cp_value[cp_mask, ...])

        # Memory cleanup (explicitly free GPU memory)
        cp_mask = None
        cp_value = None
        cp.get_default_memory_pool().free_all_blocks()

    except Exception as e:
        print(f"GPU processing error: {e}")
        return None, float('inf')

    end_time = time.time()
    return result, end_time - start_time


def run_benchmark(point_sizes, runs=3):
    """Run benchmark comparing CPU vs GPU for different point cloud sizes."""
    cpu_times = []
    gpu_times = []

    for size in tqdm(point_sizes, desc="Testing different point cloud sizes"):
        cpu_times_for_size = []
        gpu_times_for_size = []

        for run in range(runs):
            # Generate test data
            data = generate_test_data(size)
            mask = data['mask']

            # Run CPU test
            _, cpu_time = mask_subset_cpu(data, mask)
            cpu_times_for_size.append(cpu_time)

            # Run GPU test if available
            if USE_GPU:
                _, gpu_time = mask_subset_gpu(data, mask)
                gpu_times_for_size.append(gpu_time)

        # Take average time across runs
        cpu_times.append(np.mean(cpu_times_for_size))
        if USE_GPU:
            gpu_times.append(np.mean(gpu_times_for_size))

    return cpu_times, gpu_times


def plot_results(sizes, cpu_times, gpu_times):
    """Plot the performance comparison results."""
    plt.figure(figsize=(12, 7))

    # Convert sizes to millions for better readability
    sizes_in_millions = [size / 1_000_000 for size in sizes]

    plt.plot(sizes_in_millions, cpu_times, 'o-', label='CPU (NumPy)')
    if USE_GPU:
        plt.plot(sizes_in_millions, gpu_times, 's-', label='GPU (CuPy)')

    plt.xlabel('Number of Points (millions)')
    plt.ylabel('Time (seconds)')
    plt.title('CPU vs GPU Performance for Point Cloud Masking')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add speedup text if GPU is available
    if USE_GPU:
        plt.figure(figsize=(12, 7))
        speedups = [cpu_time / gpu_time if gpu_time > 0 else 0 for cpu_time, gpu_time in zip(cpu_times, gpu_times)]
        plt.bar(sizes_in_millions, speedups)
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.7)
        plt.xlabel('Number of Points (millions)')
        plt.ylabel('Speedup Factor (CPU time / GPU time)')
        plt.title('GPU Speedup Factor for Point Cloud Masking')
        plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def detailed_analysis(data_size=10_000_000):
    """Perform a detailed analysis of the masking operation with timing of individual steps."""
    print(f"\nDetailed Analysis with {data_size:,} points:")

    # Generate test data
    data = generate_test_data(data_size)
    mask = data['mask']

    # CPU analysis
    print("\nCPU (NumPy) Detailed Timing:")
    cpu_start = time.time()
    cpu_result, _ = mask_subset_cpu(data, mask)
    cpu_total = time.time() - cpu_start
    print(f"- Total CPU time: {cpu_total:.4f} seconds")

    # GPU analysis if available
    if USE_GPU:
        print("\nGPU (CuPy) Detailed Timing:")

        # Measure data transfer to GPU
        transfer_start = time.time()
        cp_mask = cp.asarray(mask)
        cp_data = {k: cp.asarray(v) for k, v in data.items() if k != 'mask'}
        transfer_to_gpu = time.time() - transfer_start

        # Measure computation on GPU
        compute_start = time.time()
        cp_result = {k: v[cp_mask] if v.ndim == 1 else v[cp_mask, ...] for k, v in cp_data.items()}
        compute_time = time.time() - compute_start

        # Measure data transfer back to CPU
        transfer_back_start = time.time()
        result = {k: cp.asnumpy(v) for k, v in cp_result.items()}
        transfer_to_cpu = time.time() - transfer_back_start

        # Total time
        gpu_total = transfer_to_gpu + compute_time + transfer_to_cpu

        print(f"- Transfer to GPU: {transfer_to_gpu:.4f} seconds ({transfer_to_gpu / gpu_total * 100:.1f}%)")
        print(f"- Computation on GPU: {compute_time:.4f} seconds ({compute_time / gpu_total * 100:.1f}%)")
        print(f"- Transfer to CPU: {transfer_to_cpu:.4f} seconds ({transfer_to_cpu / gpu_total * 100:.1f}%)")
        print(f"- Total GPU time: {gpu_total:.4f} seconds")

        print(f"\nSpeedup: {cpu_total / gpu_total:.2f}x")

        # Memory cleanup
        cp_mask = None
        cp_data = None
        cp_result = None
        cp.get_default_memory_pool().free_all_blocks()


def main():
    # Define point cloud sizes to test (increasing exponentially)
    point_sizes = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]

    if not USE_GPU:
        print("CuPy not available. Running CPU tests only.")
        # Use a smaller test set if only testing CPU
        point_sizes = [size for size in point_sizes if size <= 10_000_000]
    else:
        # Check available GPU memory and adjust point sizes accordingly
        try:
            free_memory = cp.cuda.runtime.memGetInfo()[0]
            max_points = int(free_memory / (6 * 4 * 2))  # Rough estimate (6 values, 4 bytes, plus overhead)
            print(f"Estimated maximum points for GPU testing: {max_points:,}")

            # Adjust point sizes based on available memory
            point_sizes = [size for size in point_sizes if size <= max_points]
            print(f"Testing with point sizes: {[f'{size:,}' for size in point_sizes]}")
        except Exception as e:
            print(f"Error checking GPU memory: {e}")

    # Run the benchmark
    print("Running benchmark...")
    cpu_times, gpu_times = run_benchmark(point_sizes)

    # Plot results
    plot_results(point_sizes, cpu_times, gpu_times)

    # Run detailed analysis
    detailed_size = min(10_000_000, max(point_sizes))
    detailed_analysis(detailed_size)

    # Print summary of findings
    print("\nSummary:")
    for i, size in enumerate(point_sizes):
        print(f"{size:,} points - CPU: {cpu_times[i]:.4f}s", end="")
        if USE_GPU:
            speedup = cpu_times[i] / gpu_times[i] if gpu_times[i] > 0 else 0
            print(f", GPU: {gpu_times[i]:.4f}s (Speedup: {speedup:.2f}x)")
        else:
            print("")


if __name__ == "__main__":
    main()