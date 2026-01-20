# plugins/040_Clusters/040_cluster_by_value_plugin.py
"""
Cluster by Value Action Plugin.

For point clouds with numeric attributes (e.g., intensity, classification),
runs DBSCAN clustering on each unique value separately and produces a single
Clusters node with cluster_names mapping each cluster ID to its source value.
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


class ClusterByValuePlugin(ActionPlugin):
    """
    Cluster by Value Action Plugin.

    Takes a PointCloud node with numeric attributes,
    runs DBSCAN clustering separately on each unique value, and produces
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

    def _get_available_attributes(self) -> Dict[str, str]:
        """
        Get available categorical attributes from the selected point cloud.

        Includes both numeric (integer) and string attributes that have
        a reasonable number of unique values for clustering.

        Returns:
            Dict mapping attribute keys to display names
        """
        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes

        if not data_manager or not data_manager.selected_branches:
            return {}

        try:
            selected_uid = data_manager.selected_branches[0]
            if isinstance(selected_uid, str):
                selected_uid = uuid.UUID(selected_uid)

            selected_node = data_nodes.get_node(selected_uid)

            # Check if it's a point cloud or can be reconstructed
            if selected_node.data_type == "point_cloud":
                point_cloud = selected_node.data
            else:
                # Try to reconstruct to get attributes
                point_cloud = data_manager.reconstruct_branch(str(selected_uid))

            # Collect available attributes
            attribute_options = {}

            # Check the attributes dictionary
            if hasattr(point_cloud, 'attributes') and point_cloud.attributes:
                for attr_name, attr_values in point_cloud.attributes.items():
                    if isinstance(attr_values, np.ndarray):
                        unique_count = len(np.unique(attr_values))

                        # Include integer attributes
                        if np.issubdtype(attr_values.dtype, np.integer):
                            attribute_options[attr_name] = f"{attr_name} ({unique_count} unique values)"
                        # Include string/object attributes (e.g., "Car", "Tree")
                        elif attr_values.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
                            attribute_options[attr_name] = f"{attr_name} ({unique_count} unique values)"
                        # Include float attributes with small number of unique values (likely categorical)
                        elif np.issubdtype(attr_values.dtype, np.floating) and unique_count <= 256:
                            attribute_options[attr_name] = f"{attr_name} ({unique_count} unique values)"
                    # Handle list of strings or other sequences
                    elif isinstance(attr_values, (list, tuple)) and len(attr_values) > 0:
                        unique_count = len(set(attr_values))
                        if unique_count <= 1000:  # Reasonable limit for categorical
                            attribute_options[attr_name] = f"{attr_name} ({unique_count} unique values)"

            return attribute_options

        except Exception:
            return {}

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """Execute clustering by value."""
        # Get managers
        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Check if a branch is selected
        if not data_manager.selected_branches:
            QMessageBox.warning(
                main_window,
                "No Selection",
                "Please select a PointCloud branch."
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

        # Reconstruct the point cloud
        try:
            point_cloud = data_manager.reconstruct_branch(str(selected_uid_uuid))
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
        eps = params["eps"]
        min_samples = params["min_samples"]
        target_batch_size = params.get("target_batch_size", 250000)

        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Cluster by Value")
        print(f"{'='*60}")
        print(f"  Attribute:        {attribute_name}")
        print(f"  Total points:     {len(points):,}")
        print(f"  Parameters:       eps={eps}, min_samples={min_samples}")
        print(f"  Batch size:       {target_batch_size:,}")
        print(f"{'='*60}")

        # Get unique values
        unique_values = np.unique(attribute_values)
        print(f"Found {len(unique_values)} unique values")

        # Initialize output arrays
        output_cluster_labels = np.full(len(points), -1, dtype=np.int32)
        output_value_labels = np.full(len(points), -1, dtype=np.int32)
        next_cluster_id = 0

        # Create value-to-id mapping and name mapping for Clusters
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
            # Format value nicely for display
            # For string attributes, use the value directly (e.g., "Car", "Tree")
            # For numeric attributes, show "attribute=value"
            if is_string_attr:
                value_names[i] = str(value)
            elif isinstance(value, (np.floating, float)):
                value_names[i] = f"{attribute_name}={value:.2f}"
            else:
                value_names[i] = f"{attribute_name}={value}"
            # Generate a color for this value
            np.random.seed(int(hash(str(value))) % (2**31))
            cluster_colors[value_names[i]] = np.random.rand(3).astype(np.float32)

        # Fixed batch overlap at 10%
        BATCH_OVERLAP = 0.1

        # Process each unique value with DBSCAN
        print(f"\nClustering each value:")
        for value in unique_values:
            class_id = value_to_class_id[value]
            class_name = value_names[class_id]

            # Get mask for points with this value
            value_mask = (attribute_values == value)
            value_point_indices = np.where(value_mask)[0]
            value_points = points[value_mask]

            # Skip if too few points
            if len(value_points) < min_samples:
                print(f"  - {class_name}: Skipped ({len(value_points)} points < {min_samples} min_samples)")
                continue

            print(f"  - {class_name}: {len(value_points):,} points...", end=" ", flush=True)

            # Run DBSCAN on this value group
            if len(value_points) > target_batch_size:
                # Use batch processor for large point clouds
                batch_processor = BatchProcessor(
                    points=value_points,
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
                value_pc = PointCloud(points=value_points)
                local_labels = value_pc.dbscan(eps=eps, min_points=min_samples)

            # Map local labels to global cluster IDs
            unique_local_labels = np.unique(local_labels)
            unique_local_labels = unique_local_labels[unique_local_labels >= 0]  # Exclude -1 (noise)

            # Create mapping from local to global IDs
            local_to_global = {}
            for local_id in unique_local_labels:
                local_to_global[local_id] = next_cluster_id
                next_cluster_id += 1

            # Apply global cluster labels AND preserve value labels
            for i, local_label in enumerate(local_labels):
                if local_label >= 0:  # Not noise
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
                "DBSCAN found no clusters for any value.\n"
                "Try adjusting eps or min_samples parameters."
            )
            return

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"CLUSTER BY VALUE COMPLETED")
        print(f"{'='*60}")
        print(f"  Attribute:         {attribute_name}")
        print(f"  Total points:      {len(points):,}")
        print(f"  Total instances:   {next_cluster_id}")
        print(f"  Values processed:  {len(unique_values)}")
        print(f"  Processing time:   {elapsed_time:.2f} seconds")
        print(f"  Points/second:     {len(points)/elapsed_time:,.0f}")
        print(f"{'='*60}\n")

        # Build cluster_id -> value_name mapping efficiently using vectorized operations
        cluster_names = {}

        # Filter to non-noise points only
        valid_mask = output_cluster_labels >= 0
        valid_cluster_ids = output_cluster_labels[valid_mask]
        valid_value_ids = output_value_labels[valid_mask]

        # Get unique clusters and the index of their first occurrence (O(n log n) instead of O(n*k))
        unique_cluster_ids, first_indices = np.unique(valid_cluster_ids, return_index=True)

        # Build mapping using first occurrence indices (O(k) loop, no masking)
        for i, cluster_id in enumerate(unique_cluster_ids):
            value_class_id = valid_value_ids[first_indices[i]]
            value_name = value_names.get(int(value_class_id), f"value_{value_class_id}")
            cluster_names[int(cluster_id)] = value_name

        # Create single Clusters object with cluster_names and cluster_colors
        clusters = Clusters(
            labels=output_cluster_labels,
            cluster_names=cluster_names,
            cluster_colors=cluster_colors
        )

        # Create Clusters DataNode (child of selected node)
        clusters_node = DataNode(
            data=clusters,
            data_type="cluster_labels",
            parent_uid=selected_uid_uuid,
            depends_on=[selected_uid_uuid],
            tags=["cluster_by_value", attribute_name]
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
            f"Attribute: {attribute_name}\n"
            f"Created {next_cluster_id} cluster instances across {len(unique_values)} unique values.\n\n"
            f"Processing time: {elapsed_time:.2f} seconds\n"
            f"Processing speed: {len(points)/elapsed_time:,.0f} points/sec\n\n"
            f"Clusters node added with value names preserved."
        )
