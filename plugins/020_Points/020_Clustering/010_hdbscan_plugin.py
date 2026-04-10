# plugins/analysis/hdbscan_plugin.py

from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.batch_processor import BatchProcessor


class HDBSCANPlugin(Plugin):
    """
    HDBSCAN clustering algorithm plugin with integrated batch processing.

    This plugin implements the Hierarchical Density-Based Spatial Clustering of Applications
    with Noise (HDBSCAN) algorithm for clustering point clouds. It uses a spatial batch
    processing approach to efficiently handle large point clouds by dividing them into
    overlapping spatial regions.

    The batch processing automatically uses a 10% overlap between spatial cells to ensure
    proper cluster continuity across batch boundaries.
    """

    def get_name(self) -> str:
        return "hdbscan"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "min_cluster_size": {
                "type": "int",
                "default": 5,
                "min": 2,
                "max": 1000,
                "label": "Minimum Cluster Size",
                "description": "The minimum size of clusters to be detected"
            },
            "min_samples": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "label": "Minimum Samples",
                "description": "Number of samples in a neighborhood for a point to be considered a core point"
            },
            "cluster_selection_epsilon": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "label": "Cluster Selection Epsilon",
                "description": "Distance threshold for cluster extraction (0.0 for automatic selection)"
            },
            "alpha": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "label": "Alpha",
                "description": "Normalization factor for the mutual reachability distance calculation"
            },
            "target_batch_size": {
                "type": "int",
                "default": 250000,
                "min": 50000,
                "max": 1000000,
                "label": "Target Batch Size",
                "description": "Target number of points per batch for processing (smaller values use less memory)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute HDBSCAN clustering on the point cloud using spatial batch processing.

        Progress is reported throughout the process via global_variables.global_progress.
        """
        from config.config import global_variables

        point_cloud: PointCloud = data_node.data
        points = point_cloud.points

        min_cluster_size = params["min_cluster_size"]
        min_samples = params["min_samples"]
        cluster_selection_epsilon = params["cluster_selection_epsilon"]
        alpha = params["alpha"]
        target_batch_size = params.get("target_batch_size", 250000)

        # Fixed batch overlap at 10%
        BATCH_OVERLAP = 0.1

        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting batch-processed HDBSCAN")
        print(f"{'='*60}")
        print(f"  Total points:     {len(points):,}")
        print(f"  Parameters:       min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        print(f"  Batch size:       {target_batch_size:,}")
        print(f"{'='*60}\n")

        def hdbscan_func(batch_points, eps, min_points, **kwargs):
            """Wrapper for HDBSCAN to use with batch processor."""
            batch_pc = PointCloud(points=batch_points)
            return batch_pc.hdbscan(
                min_cluster_size=kwargs['min_cluster_size'],
                min_samples=min_points,
                cluster_selection_epsilon=kwargs['cluster_selection_epsilon'],
                alpha=kwargs['alpha']
            )

        batch_processor = BatchProcessor(
            points=points,
            batch_size=target_batch_size,
            overlap_percent=BATCH_OVERLAP
        )

        cluster_labels = batch_processor.cluster_in_batches(
            clustering_func=hdbscan_func,
            min_points=min_samples,
            eps=cluster_selection_epsilon,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            alpha=alpha
        )

        clusters = Clusters(cluster_labels)
        clusters.set_random_color()

        elapsed_time = time.time() - start_time
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"  Total points:      {len(points):,}")
        print(f"  Clusters found:    {n_clusters}")
        print(f"  Noise points:      {n_noise:,} ({100*n_noise/len(points):.1f}%)")
        print(f"  Total time:        {elapsed_time:.2f} seconds")
        print(f"  Points/second:     {len(points)/elapsed_time:,.0f}")
        print(f"{'='*60}\n")

        dependencies = [data_node.uid]
        return clusters, "cluster_labels", dependencies
