# plugins/040_Clusters/040_cluster_by_value_plugin.py
"""
Cluster by Value Action Plugin.

For point clouds with numeric attributes (e.g., intensity, classification),
runs DBSCAN or HDBSCAN clustering on each unique value separately and produces
a single Clusters node with cluster_names mapping each cluster ID to its source value.
"""

from typing import Dict, Any
import numpy as np
import uuid

from plugins.interfaces import ActionPlugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.batch_processor import BatchProcessor
from config.config import global_variables
from PyQt5.QtWidgets import QMessageBox


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


class ClusterByValuePlugin(ActionPlugin):
    """
    Cluster by Value Action Plugin.

    Takes a PointCloud node with numeric attributes,
    runs DBSCAN or HDBSCAN clustering separately on each unique value, and produces
    a single Clusters node with globally unique cluster IDs and cluster_names
    mapping each cluster to its source attribute value.
    """

    def get_name(self) -> str:
        """Return the plugin name."""
        return "cluster_by_value"

    def get_parameters(self) -> Dict[str, Any]:
        """Define parameters for clustering."""
        # Get available attributes from selected point cloud
        attribute_options = self._get_available_attributes()

        return {
            "attribute": {
                "type": "dropdown",
                "options": attribute_options,
                "default": next(iter(attribute_options)) if attribute_options else "",
                "label": "Attribute to Cluster By",
                "description": "Select the attribute whose unique values will define separate clustering groups"
            },
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

    def _get_available_attributes(self) -> Dict[str, str]:
        """
        Get available categorical attributes from the selected point cloud.

        Returns:
            Dict mapping attribute keys to display names
        """
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes

        if not controller or not controller.selected_branches:
            return {}

        try:
            selected_uid = controller.selected_branches[0]
            if isinstance(selected_uid, str):
                selected_uid = uuid.UUID(selected_uid)

            selected_node = data_nodes.get_node(selected_uid)

            # Check if it's a point cloud or can be reconstructed
            if selected_node.data_type == "point_cloud":
                point_cloud = selected_node.data
            else:
                point_cloud = controller.reconstruct(str(selected_uid))

            # Collect available attributes
            attribute_options = {}

            if hasattr(point_cloud, 'attributes') and point_cloud.attributes:
                for attr_name, attr_values in point_cloud.attributes.items():
                    if isinstance(attr_values, np.ndarray):
                        unique_count = len(np.unique(attr_values))

                        if np.issubdtype(attr_values.dtype, np.integer):
                            attribute_options[attr_name] = f"{attr_name} ({unique_count} unique values)"
                        elif attr_values.dtype.kind in ('U', 'S', 'O'):
                            attribute_options[attr_name] = f"{attr_name} ({unique_count} unique values)"
                        elif np.issubdtype(attr_values.dtype, np.floating) and unique_count <= 256:
                            attribute_options[attr_name] = f"{attr_name} ({unique_count} unique values)"
                    elif isinstance(attr_values, (list, tuple)) and len(attr_values) > 0:
                        unique_count = len(set(attr_values))
                        if unique_count <= 1000:
                            attribute_options[attr_name] = f"{attr_name} ({unique_count} unique values)"

            return attribute_options

        except Exception:
            return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Execute clustering by value."""
        # Get managers
        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Check if a branch is selected
        if not controller.selected_branches:
            QMessageBox.warning(
                main_window,
                "No Selection",
                "Please select a PointCloud branch."
            )
            return

        # Get the selected branch
        selected_uid = controller.selected_branches[0]
        if isinstance(selected_uid, str):
            selected_uid_uuid = uuid.UUID(selected_uid)
        else:
            selected_uid_uuid = selected_uid

        # Get the DataNode
        selected_node = data_nodes.get_node(selected_uid_uuid)

        # Reconstruct the point cloud
        try:
            point_cloud = controller.reconstruct(str(selected_uid_uuid))
        except Exception as e:
            QMessageBox.critical(
                main_window,
                "Reconstruction Error",
                f"Failed to reconstruct PointCloud:\n{str(e)}"
            )
            return

        points = point_cloud.points

        # Get the selected attribute
        attribute_name = params.get("attribute", "")
        if not attribute_name:
            QMessageBox.warning(
                main_window,
                "No Attribute Selected",
                "Please select an attribute to cluster by."
            )
            return

        # Get attribute values
        if not hasattr(point_cloud, 'attributes') or attribute_name not in point_cloud.attributes:
            QMessageBox.warning(
                main_window,
                "Attribute Not Found",
                f"Attribute '{attribute_name}' not found in point cloud.\n"
                f"Available attributes: {list(point_cloud.attributes.keys()) if point_cloud.attributes else 'None'}"
            )
            return

        attribute_values = point_cloud.attributes[attribute_name]

        # Validate attribute length
        if len(attribute_values) != len(points):
            QMessageBox.critical(
                main_window,
                "Data Mismatch",
                f"Attribute array length ({len(attribute_values)}) does not "
                f"match point count ({len(points)})"
            )
            return

        # Extract parameters
        algorithm = params.get("algorithm", "DBSCAN")
        eps = params["eps"]
        min_samples = params["min_samples"]
        min_cluster_size = params.get("min_cluster_size", 50)
        target_batch_size = params.get("target_batch_size", 250000)

        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Cluster by Value ({algorithm})")
        print(f"{'='*60}")
        print(f"  Algorithm:        {algorithm}")
        print(f"  Attribute:        {attribute_name}")
        print(f"  Total points:     {len(points):,}")
        if algorithm == "DBSCAN":
            print(f"  Parameters:       eps={eps}, min_samples={min_samples}, min_cluster_size={min_cluster_size}")
        else:
            print(f"  Parameters:       min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        print(f"  Batch size:       {target_batch_size:,}")
        print(f"{'='*60}")

        # Get unique values
        unique_values = np.unique(attribute_values)
        print(f"Found {len(unique_values)} unique values")

        # Initialize output arrays
        output_cluster_labels = np.full(len(points), -1, dtype=np.int32)
        output_value_labels = np.full(len(points), -1, dtype=np.int32)
        next_cluster_id = 0

        # Create value-to-id mapping and name mapping
        value_to_class_id = {}
        value_names = {}
        cluster_colors = {}

        # Check if attribute contains string values
        is_string_attr = (
            isinstance(attribute_values, np.ndarray) and
            attribute_values.dtype.kind in ('U', 'S', 'O')
        ) or (
            isinstance(attribute_values, (list, tuple)) and
            len(attribute_values) > 0 and
            isinstance(attribute_values[0], str)
        )

        for i, value in enumerate(unique_values):
            value_to_class_id[value] = i
            if is_string_attr:
                value_names[i] = str(value)
            elif isinstance(value, (np.floating, float)):
                value_names[i] = f"{attribute_name}={value:.2f}"
            else:
                value_names[i] = f"{attribute_name}={value}"
            np.random.seed(int(hash(str(value))) % (2**31))
            cluster_colors[value_names[i]] = np.random.rand(3).astype(np.float32)

        # Fixed batch overlap at 10%
        BATCH_OVERLAP = 0.1

        # Process each unique value with clustering
        print(f"\nClustering each value:")
        for value in unique_values:
            class_id = value_to_class_id[value]
            class_name = value_names[class_id]

            # Get mask for points with this value
            value_mask = (attribute_values == value)
            value_point_indices = np.where(value_mask)[0]
            value_points = points[value_mask]

            # Skip if too few points
            if len(value_points) < min_cluster_size:
                print(f"  - {class_name}: Skipped ({len(value_points)} points < {min_cluster_size} min_cluster_size)")
                continue

            print(f"  - {class_name}: {len(value_points):,} points...", end=" ", flush=True)

            # Run clustering on this value group
            if len(value_points) > target_batch_size:
                # Use batch processor for large point clouds
                batch_processor = BatchProcessor(
                    points=value_points,
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
                    value_points, algorithm, eps, min_samples, min_cluster_size
                )

            # Map local labels to global cluster IDs
            unique_local_labels = np.unique(local_labels)
            unique_local_labels = unique_local_labels[unique_local_labels >= 0]

            # Create mapping from local to global IDs
            local_to_global = {}
            for local_id in unique_local_labels:
                local_to_global[local_id] = next_cluster_id
                next_cluster_id += 1

            # Apply global cluster labels AND preserve value labels
            for i, local_label in enumerate(local_labels):
                if local_label >= 0:
                    global_point_idx = value_point_indices[i]
                    output_cluster_labels[global_point_idx] = local_to_global[local_label]
                    output_value_labels[global_point_idx] = class_id

            n_clusters = len(unique_local_labels)
            print(f"Found {n_clusters} clusters")

        # Check if any clusters were found
        if next_cluster_id == 0:
            QMessageBox.warning(
                main_window,
                "No Clusters Found",
                f"{algorithm} found no clusters for any value.\n"
                "Try adjusting parameters."
            )
            return

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"CLUSTER BY VALUE COMPLETED ({algorithm})")
        print(f"{'='*60}")
        print(f"  Attribute:         {attribute_name}")
        print(f"  Total points:      {len(points):,}")
        print(f"  Total instances:   {next_cluster_id}")
        print(f"  Values processed:  {len(unique_values)}")
        print(f"  Processing time:   {elapsed_time:.2f} seconds")
        print(f"  Points/second:     {len(points)/elapsed_time:,.0f}")
        print(f"{'='*60}\n")

        # Build cluster_id -> value_name mapping
        cluster_names = {}

        valid_mask = output_cluster_labels >= 0
        valid_cluster_ids = output_cluster_labels[valid_mask]
        valid_value_ids = output_value_labels[valid_mask]

        unique_cluster_ids, first_indices = np.unique(valid_cluster_ids, return_index=True)

        for i, cluster_id in enumerate(unique_cluster_ids):
            value_class_id = valid_value_ids[first_indices[i]]
            value_name = value_names.get(int(value_class_id), f"value_{value_class_id}")
            cluster_names[int(cluster_id)] = value_name

        # Create Clusters object
        clusters = Clusters(
            labels=output_cluster_labels,
            cluster_names=cluster_names,
            cluster_colors=cluster_colors
        )

        # Create Clusters DataNode
        clusters_node = DataNode(
            data=clusters,
            data_type="cluster_labels",
            parent_uid=selected_uid_uuid,
            depends_on=[selected_uid_uuid],
            tags=["cluster_by_value", attribute_name, algorithm.lower()]
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
            f"Attribute: {attribute_name}\n"
            f"Created {next_cluster_id} cluster instances across {len(unique_values)} unique values.\n\n"
            f"Processing time: {elapsed_time:.2f} seconds\n"
            f"Processing speed: {len(points)/elapsed_time:,.0f} points/sec\n\n"
            f"Clusters node added with value names preserved."
        )
