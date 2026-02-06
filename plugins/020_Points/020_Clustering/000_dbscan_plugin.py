# plugins/analysis/dbscan_plugin.py

from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from services.batch_processor import BatchProcessor


class DBSCANPlugin(Plugin):
    """
    DBSCAN clustering algorithm plugin with integrated batch processing.

    This plugin implements the DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    algorithm for clustering point clouds. It uses a spatial batch processing approach to
    efficiently handle large point clouds by dividing them into overlapping spatial regions.

    The batch processing automatically uses a 10% overlap between spatial cells to ensure
    proper cluster continuity across batch boundaries. This helps maintain consistent
    clustering results regardless of how the point cloud is divided.

    The plugin provides detailed progress reporting during the clustering process, keeping
    the user informed about the current stage and completion percentage.

    The plugin identifies core points, connects them to form clusters, and labels points as
    belonging to clusters or as noise, ensuring consistent cluster labeling across the entire
    point cloud.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "dbscan"
        """
        return "dbscan"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for DBSCAN clustering.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "eps": {
                "type": "float",
                "default": 0.5,
                "min": 0.01,
                "max": 100.0,
                "label": "Epsilon (Neighborhood Size)",
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
        Execute DBSCAN clustering on the point cloud using spatial batch processing.

        This method automatically divides the point cloud into overlapping spatial
        regions, processes each region independently, and then reconciles the results
        to maintain consistent cluster labeling across the entire point cloud.

        Progress is reported throughout the process, showing the current stage and
        completion percentage.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for DBSCAN clustering

        Returns:
            Tuple[Clusters, str, List]:
                - Clusters object with the clustering results
                - Result type identifier "cluster_labels"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud and parameters
        point_cloud: PointCloud = data_node.data
        points = point_cloud.points
        eps = params["eps"]
        min_samples = params["min_samples"]
        target_batch_size = params.get("target_batch_size", 250000)

        # Fixed batch overlap at 10% - this is a programmer decision, not exposed to users
        BATCH_OVERLAP = 0.1

        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting batch-processed DBSCAN")
        print(f"{'='*60}")
        print(f"  Total points:     {len(points):,}")
        print(f"  Parameters:       eps={eps}, min_samples={min_samples}")
        print(f"  Batch size:       {target_batch_size:,}")
        print(f"{'='*60}\n")

        # Define progress callback for reporting progress
        def progress_callback(current, total, stage_name):
            """Report clustering progress to the console"""
            percent = int((current / total) * 100)
            print(f"DBSCAN progress: {percent}% - {stage_name}")

        # Define a wrapper function for DBSCAN that matches the batch processor interface
        def dbscan_func(batch_points, eps, min_points, **kwargs):
            """Wrapper for DBSCAN to use with batch processor"""
            # Create a temporary point cloud for this batch
            batch_pc = PointCloud(points=batch_points)
            # Run DBSCAN on the batch (backend registry auto-selects GPU/CPU)
            return batch_pc.dbscan(eps=eps, min_points=min_points)

        # Create a batch processor with appropriate spatial grid settings
        batch_processor = BatchProcessor(
            points=points,
            batch_size=target_batch_size,
            overlap_percent=BATCH_OVERLAP  # Fixed at 10%
        )

        # Run clustering in batches
        cluster_labels = batch_processor.cluster_in_batches(
            clustering_func=dbscan_func,
            min_points=min_samples,
            eps=eps,
            progress_callback=progress_callback  # Pass the progress callback
        )

        # Create a Clusters object and set random colors
        clusters = Clusters(cluster_labels)
        clusters.set_random_color()

        # Calculate and report elapsed time
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

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return clusters, "cluster_labels", dependencies
