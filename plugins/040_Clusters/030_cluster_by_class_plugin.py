# plugins/040_Clusters/030_cluster_by_class_plugin.py
"""
Cluster by Class Action Plugin.

For pre-classified point clouds (e.g., SemanticKITTI), runs DBSCAN or HDBSCAN
clustering on each semantic class separately and produces a single Clusters node
with cluster_names mapping each cluster ID to its original class name.
"""

from typing import Dict, Any
import time
import numpy as np
import uuid

from plugins.interfaces import ActionPlugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.batch_processor import BatchProcessor
from config.config import global_variables
from PyQt5.QtWidgets import QMessageBox, QApplication


def run_clustering_direct(points: np.ndarray, algorithm: str, eps: float, min_samples: int,
                          min_cluster_size: int) -> np.ndarray:
    """
    Run clustering using the specified algorithm (without batching).

    Args:
        points: Point cloud array (n, 3)
        algorithm: "DBSCAN" or "HDBSCAN"
        eps: Epsilon for DBSCAN
        min_samples: Minimum samples for both algorithms
        min_cluster_size: Minimum cluster size (clusters smaller than this become noise)

    Returns:
        Cluster labels array
    """
    if algorithm == "HDBSCAN":
        try:
            from cuml.cluster import HDBSCAN as cumlHDBSCAN
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            points_gpu = cp.asarray(points, dtype=cp.float32)

            hdb = cumlHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method='eom'
            )
            labels_gpu = hdb.fit_predict(points_gpu)
            labels = cp.asnumpy(labels_gpu)

            del points_gpu, labels_gpu
            cp.get_default_memory_pool().free_all_blocks()

            return labels

        except ImportError:
            # Fallback to CPU hdbscan if cuML not available
            try:
                from hdbscan import HDBSCAN

                hdb = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method='eom',
                    core_dist_n_jobs=-1
                )
                return hdb.fit_predict(points)
            except ImportError:
                raise ImportError("Neither cuML nor hdbscan package available for HDBSCAN")

    else:  # DBSCAN
        pc = PointCloud(points=points)
        labels = pc.dbscan(eps=eps, min_points=min_samples)

        # Apply min_cluster_size filter - mark small clusters as noise
        labels = filter_small_clusters(labels, min_cluster_size)

        return labels


def filter_small_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """
    Filter out clusters smaller than min_cluster_size by marking them as noise (-1).

    Args:
        labels: Cluster labels array
        min_cluster_size: Minimum cluster size

    Returns:
        Filtered labels array with small clusters marked as noise
    """
    if min_cluster_size <= 1:
        return labels

    # Count points per cluster
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

    # Find clusters that are too small
    small_clusters = unique_labels[counts < min_cluster_size]

    if len(small_clusters) > 0:
        # Create mask for points in small clusters
        small_mask = np.isin(labels, small_clusters)
        # Mark them as noise
        labels = labels.copy()
        labels[small_mask] = -1

    return labels


class ClusterByClassPlugin(ActionPlugin):
    """
    Cluster by Class Action Plugin.

    Takes any branch derived from a Clusters node with semantic names
    (e.g., from SemanticKITTI import, or a filtered Masks branch from one),
    runs DBSCAN or HDBSCAN clustering separately on each class, and produces
    a single Clusters node with globally unique cluster IDs and cluster_names
    mapping each cluster to its original semantic class name.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "cluster_by_class"

    def get_parameters(self) -> Dict[str, Any]:
        """Define parameters for clustering."""
        return {
            "algorithm": {
                "type": "dropdown",
                "options": {"DBSCAN": "DBSCAN (requires eps)", "HDBSCAN": "HDBSCAN (auto-detects density)"},
                "default": "DBSCAN",
                "label": "Clustering Algorithm",
                "description": "DBSCAN requires eps parameter, HDBSCAN auto-detects cluster density"
            },
            "eps": {
                "type": "float",
                "default": 0.5,
                "min": 0.01,
                "max": 100.0,
                "label": "Epsilon (DBSCAN only)",
                "description": "Maximum distance between points to be considered neighbors (ignored for HDBSCAN)"
            },
            "min_samples": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 1000,
                "label": "Minimum Samples",
                "description": "Minimum points required to form a dense region"
            },
            "min_cluster_size": {
                "type": "int",
                "default": 50,
                "min": 2,
                "max": 10000,
                "label": "Min Cluster Size",
                "description": "Minimum size of clusters (smaller clusters become noise)"
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
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Check if a branch is selected
        if not controller.selected_branches:
            QMessageBox.warning(
                main_window,
                "No Selection",
                "Please select a branch with cluster metadata (e.g., Clusters node or filtered Masks)."
            )
            return

        # Get the selected branch
        selected_uid = controller.selected_branches[0]
        if isinstance(selected_uid, str):
            selected_uid_uuid = uuid.UUID(selected_uid)
        else:
            selected_uid_uuid = selected_uid

        # Reconstruct the selected branch
        try:
            point_cloud = controller.reconstruct(str(selected_uid_uuid))
        except Exception as e:
            QMessageBox.critical(
                main_window,
                "Reconstruction Error",
                f"Failed to reconstruct branch:\n{str(e)}"
            )
            return

        # Get cluster metadata from PointCloud attributes
        cluster_names = point_cloud.get_attribute("_cluster_names")
        cluster_colors = point_cloud.get_attribute("_cluster_colors")
        cluster_labels = point_cloud.get_attribute("cluster_labels")

        # Validate that cluster metadata exists
        if cluster_names is None or cluster_labels is None:
            QMessageBox.warning(
                main_window,
                "No Cluster Metadata",
                "Selected branch has no cluster metadata.\n"
                "Please select a branch derived from a Clusters node with semantic names\n"
                "(e.g., from SemanticKITTI import or a filtered branch from one)."
            )
            return

        if len(cluster_names) == 0:
            QMessageBox.warning(
                main_window,
                "No Semantic Names",
                "Selected branch has cluster labels but no semantic names.\n"
                "Please select a branch with cluster_names "
                "(e.g., from SemanticKITTI import)."
            )
            return

        points = point_cloud.points

        # Validate input
        if len(cluster_labels) != len(points):
            QMessageBox.critical(
                main_window,
                "Data Mismatch",
                f"Label array length ({len(cluster_labels)}) does not "
                f"match point count ({len(points)})"
            )
            return

        # Extract parameters
        algorithm = params.get("algorithm", "DBSCAN")
        eps = params["eps"]
        min_samples = params["min_samples"]
        min_cluster_size = params.get("min_cluster_size", 50)
        target_batch_size = params.get("target_batch_size", 250000)

        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Cluster by Class ({algorithm})")
        print(f"{'='*60}")
        print(f"  Algorithm:        {algorithm}")
        print(f"  Total points:     {len(points):,}")
        if algorithm == "DBSCAN":
            print(f"  Parameters:       eps={eps}, min_samples={min_samples}, min_cluster_size={min_cluster_size}")
        else:
            print(f"  Parameters:       min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        print(f"  Batch size:       {target_batch_size:,}")
        print(f"{'='*60}")

        # Get unique classes
        unique_class_ids = np.unique(cluster_labels)
        print(f"Found {len(unique_class_ids)} unique classes")

        # Show processing overlay
        main_window.tree_overlay.position_over(tree_widget)
        main_window.tree_overlay.show_processing(
            f"Cluster by Class ({algorithm}) — {len(unique_class_ids)} classes, {len(points):,} points..."
        )
        main_window.disable_menus()
        main_window.disable_tree()
        QApplication.processEvents()

        try:
            self._do_clustering(
                main_window, tree_widget, data_nodes,
                points, cluster_labels, unique_class_ids, cluster_names, cluster_colors,
                algorithm, eps, min_samples, min_cluster_size, target_batch_size,
                selected_uid_uuid, start_time
            )
        finally:
            main_window.tree_overlay.hide_processing()
            main_window.enable_menus()
            main_window.enable_tree()

    def _do_clustering(self, main_window, tree_widget, data_nodes,
                       points, cluster_labels, unique_class_ids, cluster_names, cluster_colors,
                       algorithm, eps, min_samples, min_cluster_size, target_batch_size,
                       selected_uid_uuid, start_time):
        """Run the actual clustering loop (separated for try/finally cleanup)."""
        # Initialize output arrays
        output_cluster_labels = np.full(len(points), -1, dtype=np.int32)
        output_class_labels = np.full(len(points), -1, dtype=np.int32)
        next_cluster_id = 0

        # Fixed batch overlap at 10%
        BATCH_OVERLAP = 0.1

        n_classes = len(unique_class_ids)

        # Process each class with clustering
        print(f"\nClustering each class:")
        for class_idx, class_id in enumerate(unique_class_ids):
            # Get class name from mapping
            class_name = cluster_names.get(
                int(class_id),
                f"class_{class_id}"
            )

            # Get mask for points of this class
            class_mask = (cluster_labels == class_id)
            class_point_indices = np.where(class_mask)[0]
            class_points = points[class_mask]

            # Skip if too few points
            if len(class_points) < min_cluster_size:
                print(f"  - {class_name}: Skipped ({len(class_points)} points < {min_cluster_size} min_cluster_size)")
                continue

            print(f"  - {class_name}: {len(class_points):,} points...", end=" ", flush=True)
            main_window.tree_overlay.show_processing(
                f"Clustering class {class_idx + 1}/{n_classes}: {class_name} ({len(class_points):,} pts)..."
            )
            QApplication.processEvents()

            # Run clustering on this class
            if len(class_points) > target_batch_size:
                # Use batch processor for large point clouds
                batch_processor = BatchProcessor(
                    points=class_points,
                    batch_size=target_batch_size,
                    overlap_percent=BATCH_OVERLAP
                )

                if algorithm == "HDBSCAN":
                    def hdbscan_func(batch_points, min_cluster_size, min_samples, **kwargs):
                        return run_clustering_direct(
                            batch_points, "HDBSCAN", 0, min_samples, min_cluster_size
                        )

                    local_labels = batch_processor.cluster_in_batches(
                        clustering_func=hdbscan_func,
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples
                    )
                else:
                    def dbscan_func(batch_points, eps, min_points, **kwargs):
                        batch_pc = PointCloud(points=batch_points)
                        return batch_pc.dbscan(eps=eps, min_points=min_points)

                    local_labels = batch_processor.cluster_in_batches(
                        clustering_func=dbscan_func,
                        min_points=min_samples,
                        eps=eps
                    )
                    # Apply min_cluster_size filter after batching
                    local_labels = filter_small_clusters(local_labels, min_cluster_size)
            else:
                local_labels = run_clustering_direct(
                    class_points, algorithm, eps, min_samples, min_cluster_size
                )

            # Map local labels to global cluster IDs
            unique_local_labels = np.unique(local_labels)
            unique_local_labels = unique_local_labels[unique_local_labels >= 0]

            # Create mapping from local to global IDs
            local_to_global = {}
            for local_id in unique_local_labels:
                local_to_global[local_id] = next_cluster_id
                next_cluster_id += 1

            # Apply global cluster labels AND preserve class labels
            for i, local_label in enumerate(local_labels):
                if local_label >= 0:
                    global_point_idx = class_point_indices[i]
                    output_cluster_labels[global_point_idx] = local_to_global[local_label]
                    output_class_labels[global_point_idx] = class_id

            n_clusters = len(unique_local_labels)
            print(f"Found {n_clusters} clusters")

        # Check if any clusters were found
        if next_cluster_id == 0:
            QMessageBox.warning(
                main_window,
                "No Clusters Found",
                f"{algorithm} found no clusters for any class.\n"
                "Try adjusting parameters."
            )
            return

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"CLUSTER BY CLASS COMPLETED ({algorithm})")
        print(f"{'='*60}")
        print(f"  Total points:      {len(points):,}")
        print(f"  Total instances:   {next_cluster_id}")
        print(f"  Classes processed: {len(unique_class_ids)}")
        print(f"  Processing time:   {elapsed_time:.2f} seconds")
        print(f"  Points/second:     {len(points)/elapsed_time:,.0f}")
        print(f"{'='*60}\n")

        # Build cluster_id -> class_name mapping
        output_cluster_names = {}

        valid_mask = output_cluster_labels >= 0
        valid_cluster_ids = output_cluster_labels[valid_mask]
        valid_class_ids = output_class_labels[valid_mask]

        unique_cluster_ids, first_indices = np.unique(valid_cluster_ids, return_index=True)

        for i, cluster_id in enumerate(unique_cluster_ids):
            class_id = valid_class_ids[first_indices[i]]
            class_name = cluster_names.get(int(class_id), f"class_{class_id}")
            output_cluster_names[int(cluster_id)] = class_name

        # Create Clusters object
        output_cluster_colors = cluster_colors.copy() if cluster_colors else {}
        clusters = Clusters(
            labels=output_cluster_labels,
            cluster_names=output_cluster_names,
            cluster_colors=output_cluster_colors
        )

        # Create Clusters DataNode
        clusters_node = DataNode(
            data=clusters,
            data_type="cluster_labels",
            parent_uid=selected_uid_uuid,
            depends_on=[selected_uid_uuid],
            tags=["cluster_by_class", algorithm.lower()]
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
            f"Algorithm: {algorithm}\n"
            f"Created {next_cluster_id} cluster instances across {len(unique_class_ids)} classes.\n\n"
            f"Processing time: {elapsed_time:.2f} seconds\n"
            f"Processing speed: {len(points)/elapsed_time:,.0f} points/sec\n\n"
            f"Clusters node added with semantic names preserved."
        )
