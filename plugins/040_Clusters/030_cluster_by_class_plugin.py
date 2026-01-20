# plugins/040_Clusters/030_cluster_by_class_plugin.py
"""
Cluster by Class Action Plugin.

For pre-classified point clouds (e.g., SemanticKITTI), runs DBSCAN clustering
on each semantic class separately and produces a single Clusters node with
cluster_names mapping each cluster ID to its original class name.
"""

from typing import Dict, Any
import numpy as np
import uuid

from plugins.interfaces import ActionPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.clusters import Clusters
from services.batch_processor import BatchProcessor
from config.config import global_variables
from PyQt5.QtWidgets import QMessageBox


class ClusterByClassPlugin(ActionPlugin):
    """
    Cluster by Class Action Plugin.

    Takes a Clusters node with semantic names (e.g., from SemanticKITTI import),
    runs DBSCAN clustering separately on each class, and produces a single
    Clusters node with globally unique cluster IDs and cluster_names mapping
    each cluster to its original semantic class name.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "cluster_by_class"

    def get_parameters(self) -> Dict[str, Any]:
        """Define parameters for clustering."""
        return {
            "eps": {
                "type": "float",
                "default": 0.5,
                "min": 0.01,
                "max": 100.0,
                "label": "Epsilon (Neighborhood Size)",
                "description": "Maximum distance between points to be considered neighbors"
            },
            "min_samples": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "label": "Minimum Samples",
                "description": "Minimum points required to form a cluster"
            },
            "target_batch_size": {
                "type": "int",
                "default": 250000,
                "min": 50000,
                "max": 1000000,
                "label": "Target Batch Size",
                "description": "Target points per batch for processing (smaller = less memory)"
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Execute clustering by class."""
        # Get managers
        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Check if a branch is selected
        if not data_manager.selected_branches:
            QMessageBox.warning(
                main_window,
                "No Selection",
                "Please select a Clusters branch with semantic names."
            )
            return

        # Get the selected branch
        selected_uid = data_manager.selected_branches[0]
        if isinstance(selected_uid, str):
            selected_uid_uuid = uuid.UUID(selected_uid)
        else:
            selected_uid_uuid = selected_uid

        # Get the DataNode
        selected_node = data_nodes.get_node(selected_uid_uuid)

        # Validate it's a Clusters node with names
        if selected_node.data_type != "cluster_labels":
            QMessageBox.warning(
                main_window,
                "Invalid Selection",
                f"This plugin requires a Clusters node with semantic names.\n"
                f"You selected a '{selected_node.data_type}' node.\n"
                f"Please select a Clusters node (e.g., from SemanticKITTI import)."
            )
            return

        # Get Clusters data
        input_clusters = selected_node.data

        # Check if it has names
        if not input_clusters.has_names():
            QMessageBox.warning(
                main_window,
                "No Semantic Names",
                "This Clusters node doesn't have semantic names.\n"
                "Please select a Clusters node with cluster_names "
                "(e.g., from SemanticKITTI import)."
            )
            return

        # Get parent PointCloud (need to reconstruct to get XYZ coordinates)
        parent_uid = selected_node.parent_uid
        try:
            point_cloud = data_manager.reconstruct_branch(str(parent_uid))
        except Exception as e:
            QMessageBox.critical(
                main_window,
                "Reconstruction Error",
                f"Failed to reconstruct parent PointCloud:\n{str(e)}"
            )
            return

        points = point_cloud.points

        # Validate input
        if len(input_clusters.labels) != len(points):
            QMessageBox.critical(
                main_window,
                "Data Mismatch",
                f"Label array length ({len(input_clusters.labels)}) does not "
                f"match point count ({len(points)})"
            )
            return

        # Extract parameters
        eps = params["eps"]
        min_samples = params["min_samples"]
        target_batch_size = params.get("target_batch_size", 250000)

        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Cluster by Class")
        print(f"{'='*60}")
        print(f"  Total points:     {len(points):,}")
        print(f"  Parameters:       eps={eps}, min_samples={min_samples}")
        print(f"  Batch size:       {target_batch_size:,}")
        print(f"{'='*60}")

        # Get unique classes
        unique_class_ids = np.unique(input_clusters.labels)
        print(f"Found {len(unique_class_ids)} unique classes")

        # Initialize output arrays
        output_cluster_labels = np.full(len(points), -1, dtype=np.int32)
        output_class_labels = np.full(len(points), -1, dtype=np.int32)  # Preserve original class IDs
        next_cluster_id = 0

        # Fixed batch overlap at 10%
        BATCH_OVERLAP = 0.1

        # Process each class with DBSCAN
        print(f"\nClustering each class:")
        for class_id in unique_class_ids:
            # Get class name from mapping
            class_name = input_clusters.cluster_names.get(
                int(class_id),
                f"class_{class_id}"
            )

            # Get mask for points of this class
            class_mask = (input_clusters.labels == class_id)
            class_point_indices = np.where(class_mask)[0]
            class_points = points[class_mask]

            # Skip if too few points
            if len(class_points) < min_samples:
                print(f"  - {class_name}: Skipped ({len(class_points)} points < {min_samples} min_samples)")
                continue

            print(f"  - {class_name}: {len(class_points):,} points...", end=" ", flush=True)

            # Run DBSCAN on this class
            if len(class_points) > target_batch_size:
                # Use batch processor for large point clouds
                batch_processor = BatchProcessor(
                    points=class_points,
                    batch_size=target_batch_size,
                    overlap_percent=BATCH_OVERLAP
                )

                def dbscan_func(batch_points, eps, min_points, **kwargs):
                    """Wrapper for DBSCAN to use with batch processor"""
                    batch_pc = PointCloud(points=batch_points)
                    return batch_pc.dbscan(eps=eps, min_points=min_points)

                local_labels = batch_processor.cluster_in_batches(
                    clustering_func=dbscan_func,
                    min_points=min_samples,
                    eps=eps
                )
            else:
                # Direct DBSCAN for smaller point clouds
                class_pc = PointCloud(points=class_points)
                local_labels = class_pc.dbscan(eps=eps, min_points=min_samples)

            # Map local labels to global cluster IDs
            unique_local_labels = np.unique(local_labels)
            unique_local_labels = unique_local_labels[unique_local_labels >= 0]  # Exclude -1 (noise)

            # Create mapping from local to global IDs
            local_to_global = {}
            for local_id in unique_local_labels:
                local_to_global[local_id] = next_cluster_id
                next_cluster_id += 1

            # Apply global cluster labels AND preserve class labels
            for i, local_label in enumerate(local_labels):
                if local_label >= 0:  # Not noise
                    global_point_idx = class_point_indices[i]
                    output_cluster_labels[global_point_idx] = local_to_global[local_label]
                    output_class_labels[global_point_idx] = class_id  # Preserve original class ID

            n_clusters = len(unique_local_labels)
            print(f"Found {n_clusters} clusters")

        # Check if any clusters were found
        if next_cluster_id == 0:
            QMessageBox.warning(
                main_window,
                "No Clusters Found",
                "DBSCAN found no clusters for any class.\n"
                "Try adjusting eps or min_samples parameters."
            )
            return

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"CLUSTER BY CLASS COMPLETED")
        print(f"{'='*60}")
        print(f"  Total points:      {len(points):,}")
        print(f"  Total instances:   {next_cluster_id}")
        print(f"  Classes processed: {len(unique_class_ids)}")
        print(f"  Processing time:   {elapsed_time:.2f} seconds")
        print(f"  Points/second:     {len(points)/elapsed_time:,.0f}")
        print(f"{'='*60}\n")

        # Build cluster_id -> class_name mapping efficiently using vectorized operations
        cluster_names = {}

        # Filter to non-noise points only
        valid_mask = output_cluster_labels >= 0
        valid_cluster_ids = output_cluster_labels[valid_mask]
        valid_class_ids = output_class_labels[valid_mask]

        # Get unique clusters and the index of their first occurrence (O(n log n) instead of O(n*k))
        unique_cluster_ids, first_indices = np.unique(valid_cluster_ids, return_index=True)

        # Build mapping using first occurrence indices (O(k) loop, no masking)
        for i, cluster_id in enumerate(unique_cluster_ids):
            class_id = valid_class_ids[first_indices[i]]
            class_name = input_clusters.cluster_names.get(int(class_id), f"class_{class_id}")
            cluster_names[int(cluster_id)] = class_name

        # Create single Clusters object with cluster_names and cluster_colors
        clusters = Clusters(
            labels=output_cluster_labels,
            cluster_names=cluster_names,
            cluster_colors=input_clusters.cluster_colors.copy()
        )

        # Create Clusters DataNode (child of input Clusters)
        clusters_node = DataNode(
            data=clusters,
            data_type="cluster_labels",
            parent_uid=selected_uid_uuid,
            depends_on=[selected_uid_uuid],
            tags=["cluster_by_class"]
        )

        # Add Clusters node to tree
        clusters_uid = data_nodes.add_node(clusters_node)
        tree_widget.add_branch(
            str(clusters_uid),
            str(selected_uid_uuid),
            "cluster_labels"
        )

        # Show success message
        QMessageBox.information(
            main_window,
            "Success",
            f"Clustering complete!\n\n"
            f"Created {next_cluster_id} cluster instances across {len(unique_class_ids)} classes.\n\n"
            f"Processing time: {elapsed_time:.2f} seconds\n"
            f"Processing speed: {len(points)/elapsed_time:,.0f} points/sec\n\n"
            f"Clusters node added with semantic names preserved."
        )
