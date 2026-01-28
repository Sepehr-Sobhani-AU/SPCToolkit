# plugins/085_Tools/000_dbscan_benchmark_plugin.py
"""
DBSCAN Benchmark Plugin

Compares performance of clustering implementations:
1. cuML DBSCAN (GPU)
2. cuML HDBSCAN (GPU)
3. Open3D DBSCAN (CPU)
"""

from typing import Dict, Any
import time
import numpy as np

from PyQt5.QtWidgets import QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables


class DBSCANBenchmarkPlugin(ActionPlugin):
    """
    Action plugin for benchmarking clustering implementations.

    Compares cuML DBSCAN, cuML HDBSCAN, and Open3D DBSCAN.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "dbscan_benchmark"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define benchmark parameters.

        Returns:
            Parameter schema for clustering benchmark
        """
        return {
            "eps": {
                "type": "float",
                "default": 0.5,
                "min": 0.01,
                "max": 100.0,
                "label": "Epsilon (for DBSCAN)",
                "description": "The maximum distance between two points for them to be considered neighbors"
            },
            "min_samples": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "label": "Minimum Samples",
                "description": "The minimum number of points required to form a dense region"
            },
            "min_cluster_size": {
                "type": "int",
                "default": 50,
                "min": 2,
                "max": 10000,
                "label": "Min Cluster Size (for HDBSCAN)",
                "description": "The minimum size of clusters for HDBSCAN"
            },
            "runs": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "label": "Number of Runs",
                "description": "Number of times to run each implementation for averaging"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the clustering benchmark.

        Args:
            main_window: The main application window
            params: Benchmark parameters
        """
        data_manager = global_variables.global_data_manager

        # Check if a branch is selected
        if not data_manager.selected_branches:
            QMessageBox.warning(
                main_window,
                "Clustering Benchmark",
                "Please select a branch first."
            )
            return

        # Get selected branch and reconstruct to point cloud
        selected_uid = data_manager.selected_branches[0]
        point_cloud = data_manager.reconstruct_branch(selected_uid)

        if point_cloud is None:
            QMessageBox.warning(
                main_window,
                "Clustering Benchmark",
                "Could not reconstruct point cloud from selected branch."
            )
            return

        points = point_cloud.points
        eps = params["eps"]
        min_samples = params["min_samples"]
        min_cluster_size = params["min_cluster_size"]
        num_runs = params["runs"]

        # Calculate point cloud statistics
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        extent = maxs - mins

        print(f"\n{'='*70}")
        print(f"CLUSTERING BENCHMARK (DBSCAN vs HDBSCAN)")
        print(f"{'='*70}")
        print(f"  Points:           {len(points):,}")
        print(f"  Extent:           X={extent[0]:.2f}, Y={extent[1]:.2f}, Z={extent[2]:.2f}")
        print(f"  DBSCAN eps:       {eps}")
        print(f"  min_samples:      {min_samples}")
        print(f"  HDBSCAN min_cluster_size: {min_cluster_size}")
        print(f"  Runs:             {num_runs}")
        print(f"{'='*70}\n")

        results = {}

        # ============================================================
        # 1. Test cuML DBSCAN (GPU)
        # ============================================================
        try:
            from cuml.cluster import DBSCAN as cumlDBSCAN
            import cupy as cp

            print("1. Testing cuML DBSCAN (GPU)...")

            cuml_dbscan_times = []
            cuml_dbscan_clusters = None

            for run in range(num_runs):
                cp.get_default_memory_pool().free_all_blocks()

                start_time = time.time()
                points_gpu = cp.asarray(points, dtype=cp.float32)
                db = cumlDBSCAN(eps=eps, min_samples=min_samples)
                labels_gpu = db.fit_predict(points_gpu)
                labels = cp.asnumpy(labels_gpu)
                elapsed = time.time() - start_time

                cuml_dbscan_times.append(elapsed)
                if cuml_dbscan_clusters is None:
                    cuml_dbscan_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    cuml_dbscan_noise = np.sum(labels == -1)

                del points_gpu, labels_gpu
                cp.get_default_memory_pool().free_all_blocks()

                print(f"   Run {run + 1}: {elapsed:.3f}s")

            results['cuML DBSCAN (GPU)'] = {
                'times': cuml_dbscan_times,
                'clusters': cuml_dbscan_clusters,
                'noise': cuml_dbscan_noise,
                'avg': np.mean(cuml_dbscan_times)
            }

        except ImportError as e:
            print(f"   cuML DBSCAN: not available - {e}")
        except Exception as e:
            print(f"   cuML DBSCAN error: {e}")

        # ============================================================
        # 2. Test cuML HDBSCAN (GPU)
        # ============================================================
        try:
            from cuml.cluster import HDBSCAN as cumlHDBSCAN
            import cupy as cp

            print("\n2. Testing cuML HDBSCAN (GPU)...")

            cuml_hdbscan_times = []
            cuml_hdbscan_clusters = None

            for run in range(num_runs):
                cp.get_default_memory_pool().free_all_blocks()

                start_time = time.time()
                points_gpu = cp.asarray(points, dtype=cp.float32)
                hdb = cumlHDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method='eom'
                )
                labels_gpu = hdb.fit_predict(points_gpu)
                labels = cp.asnumpy(labels_gpu)
                elapsed = time.time() - start_time

                cuml_hdbscan_times.append(elapsed)
                if cuml_hdbscan_clusters is None:
                    cuml_hdbscan_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    cuml_hdbscan_noise = np.sum(labels == -1)

                del points_gpu, labels_gpu
                cp.get_default_memory_pool().free_all_blocks()

                print(f"   Run {run + 1}: {elapsed:.3f}s")

            results['cuML HDBSCAN (GPU)'] = {
                'times': cuml_hdbscan_times,
                'clusters': cuml_hdbscan_clusters,
                'noise': cuml_hdbscan_noise,
                'avg': np.mean(cuml_hdbscan_times)
            }

        except ImportError as e:
            print(f"   cuML HDBSCAN: not available - {e}")
        except Exception as e:
            print(f"   cuML HDBSCAN error: {e}")

        # ============================================================
        # 3. Test Open3D DBSCAN (CPU)
        # ============================================================
        try:
            import open3d as o3d

            print("\n3. Testing Open3D DBSCAN (CPU)...")

            o3d_times = []
            o3d_clusters = None

            for run in range(num_runs):
                start_time = time.time()

                # Convert to Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                # Run DBSCAN
                labels = np.array(pcd.cluster_dbscan(
                    eps=eps,
                    min_points=min_samples,
                    print_progress=False
                ))

                elapsed = time.time() - start_time

                o3d_times.append(elapsed)
                if o3d_clusters is None:
                    o3d_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    o3d_noise = np.sum(labels == -1)

                print(f"   Run {run + 1}: {elapsed:.3f}s")

            results['Open3D DBSCAN (CPU)'] = {
                'times': o3d_times,
                'clusters': o3d_clusters,
                'noise': o3d_noise,
                'avg': np.mean(o3d_times)
            }

        except ImportError as e:
            print(f"   Open3D DBSCAN: not available - {e}")
        except Exception as e:
            print(f"   Open3D DBSCAN error: {e}")

        # ============================================================
        # Build results report
        # ============================================================
        lines = []
        lines.append("=== Clustering Benchmark Results ===")
        lines.append("")
        lines.append(f"Point Cloud: {len(points):,} points")
        lines.append(f"Extent: X={extent[0]:.2f}, Y={extent[1]:.2f}, Z={extent[2]:.2f}")
        lines.append(f"DBSCAN params: eps={eps}, min_samples={min_samples}")
        lines.append(f"HDBSCAN params: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        lines.append(f"Runs: {num_runs}")
        lines.append("")
        lines.append("=== Timing Results ===")
        lines.append("")

        for name, data in results.items():
            times = data['times']
            avg = np.mean(times)
            std = np.std(times)
            noise_pct = 100 * data['noise'] / len(points)
            lines.append(f"{name}:")
            lines.append(f"  Average: {avg:.3f}s (+/- {std:.3f}s)")
            lines.append(f"  Best:    {min(times):.3f}s")
            lines.append(f"  Worst:   {max(times):.3f}s")
            lines.append(f"  Clusters: {data['clusters']}")
            lines.append(f"  Noise:    {data['noise']:,} ({noise_pct:.1f}%)")
            lines.append("")

        # Comparison - sort by average time
        lines.append("=== Ranking (fastest to slowest) ===")
        lines.append("")

        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg'])

        for i, (name, data) in enumerate(sorted_results):
            lines.append(f"  {i+1}. {name}: {data['avg']:.3f}s ({data['clusters']} clusters)")

        if len(sorted_results) >= 2:
            fastest_name, fastest_data = sorted_results[0]
            lines.append("")
            lines.append(f"Speedups (relative to {fastest_name}):")
            for name, data in sorted_results[1:]:
                slowdown = data['avg'] / fastest_data['avg']
                lines.append(f"  {name} is {slowdown:.2f}x slower")

        # Print to console
        result_text = "\n".join(lines)
        print(f"\n{result_text}\n")

        # Show dialog
        msg_box = QMessageBox(main_window)
        msg_box.setWindowTitle("Clustering Benchmark Results")
        msg_box.setText("Benchmark completed. See details below.")
        msg_box.setDetailedText(result_text)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()
