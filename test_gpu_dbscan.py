#!/usr/bin/env python
"""
Simple test script to demonstrate GPU-accelerated DBSCAN using cuML.
This script can run in WSL2 with the rapids environment.
"""

import numpy as np
import time
from core.point_cloud import PointCloud

def generate_test_data(n_points=500000):
    """Generate a test point cloud with clusters."""
    print(f"Generating {n_points:,} test points...")

    # Create 5 clusters
    clusters = []
    for i in range(5):
        center = np.random.rand(3) * 10
        points = np.random.randn(n_points // 5, 3) * 0.3 + center
        clusters.append(points)

    # Add noise
    noise = np.random.rand(n_points // 10, 3) * 15
    clusters.append(noise)

    all_points = np.vstack(clusters)
    np.random.shuffle(all_points)

    return all_points

def test_dbscan(points, eps=0.5, min_samples=10, use_gpu='auto'):
    """Test DBSCAN clustering."""
    print(f"\n{'='*60}")
    print(f"Testing DBSCAN with {len(points):,} points")
    print(f"Parameters: eps={eps}, min_samples={min_samples}, use_gpu={use_gpu}")
    print(f"{'='*60}")

    # Create point cloud
    pc = PointCloud(points=points)

    # Run DBSCAN with timing
    start_time = time.time()
    labels = pc.dbscan(eps=eps, min_points=min_samples, use_gpu=use_gpu)
    end_time = time.time()

    # Results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  - Time: {end_time - start_time:.2f} seconds")
    print(f"  - Clusters found: {n_clusters}")
    print(f"  - Noise points: {n_noise:,} ({100*n_noise/len(points):.1f}%)")
    print(f"{'='*60}\n")

    return labels, end_time - start_time

if __name__ == "__main__":
    # Generate test data
    points = generate_test_data(n_points=500000)

    print("\n" + "="*60)
    print("GPU DBSCAN TEST")
    print("="*60)

    # Test 1: Auto mode (will use GPU if available)
    print("\n[TEST 1] Auto mode - will use GPU if cuML available")
    labels_auto, time_auto = test_dbscan(points, eps=0.5, min_samples=10, use_gpu='auto')

    # Test 2: Force CPU (for comparison)
    print("\n[TEST 2] CPU only mode - forced CPU for comparison")
    labels_cpu, time_cpu = test_dbscan(points, eps=0.5, min_samples=10, use_gpu=False)

    # Compare
    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    print(f"Auto mode time: {time_auto:.2f} seconds")
    print(f"CPU mode time:  {time_cpu:.2f} seconds")
    if time_cpu > time_auto:
        speedup = time_cpu / time_auto
        print(f"Speedup: {speedup:.1f}x faster with auto mode")
    print("="*60)
