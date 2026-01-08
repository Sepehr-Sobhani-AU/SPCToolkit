# plugins/040_Clusters/030_cluster_by_class_plugin.py
"""
Cluster by Class Action Plugin.

For pre-classified point clouds (e.g., SemanticKITTI), runs DBSCAN clustering
on each semantic class separately and produces:
1. Clusters node (merged cluster IDs from all classes)
2. FeatureClasses child node (maps each cluster ID to its class name)
"""

from typing import Dict, Any
import numpy as np
import uuid

from plugins.interfaces import ActionPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.feature_classes import FeatureClasses
from core.clusters import Clusters
from services.batch_processor import BatchProcessor
from config.config import global_variables
from PyQt5.QtWidgets import QMessageBox


class ClusterByClassPlugin(ActionPlugin):
    """
    Cluster by Class Action Plugin.

    Takes a FeatureClasses node where every point has a class label,
    runs DBSCAN clustering separately on each class, and produces:
    1. Clusters node with globally unique cluster IDs
    2. FeatureClasses child that maps cluster IDs to class names

    This follows the program architecture:
    - Clusters stores the cluster IDs
    - FeatureClasses (child of Clusters) classifies those clusters
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
            },
            "use_gpu": {
                "type": "choice",
                "options": ["Auto", "Force GPU", "CPU Only"],
                "default": "Auto",
                "label": "GPU Acceleration",
                "description": "Auto: Use GPU if cuML available, Force GPU: Require GPU, CPU Only: Disable GPU"
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
                "Please select a FeatureClasses branch."
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

        # Validate it's a FeatureClasses node
        if selected_node.data_type != "feature_classes":
            QMessageBox.warning(
                main_window,
                "Invalid Selection",
                f"This plugin only works on FeatureClasses nodes.\n"
                f"You selected a '{selected_node.data_type}' node.\n"
                f"Please select a FeatureClasses node (pre-classified point cloud)."
            )
            return

        # Get FeatureClasses data
        input_feature_classes = selected_node.data

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
        if len(input_feature_classes.labels) != len(points):
            QMessageBox.critical(
                main_window,
                "Data Mismatch",
                f"Label array length ({len(input_feature_classes.labels)}) does not "
                f"match point count ({len(points)})"
            )
            return

        # Extract parameters
        eps = params["eps"]
        min_samples = params["min_samples"]
        target_batch_size = params.get("target_batch_size", 250000)

        # Convert GPU mode string to parameter value
        gpu_mode = params.get("use_gpu", "Auto")
        if gpu_mode == "Force GPU":
            use_gpu = True
        elif gpu_mode == "CPU Only":
            use_gpu = False
        else:  # "Auto"
            use_gpu = 'auto'

        import time
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Starting Cluster by Class")
        print(f"{'='*60}")
        print(f"  Total points:     {len(points):,}")
        print(f"  Parameters:       eps={eps}, min_samples={min_samples}")
        print(f"  Batch size:       {target_batch_size:,}")
        print(f"  GPU mode:         {gpu_mode}")
        print(f"{'='*60}")

        # Get unique classes
        unique_class_ids = np.unique(input_feature_classes.labels)
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
            class_name = input_feature_classes.class_mapping.get(
                int(class_id),
                f"class_{class_id}"
            )

            # Get mask for points of this class
            class_mask = (input_feature_classes.labels == class_id)
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
                    return batch_pc.dbscan(eps=eps, min_points=min_points, use_gpu=use_gpu)

                local_labels = batch_processor.cluster_in_batches(
                    clustering_func=dbscan_func,
                    min_points=min_samples,
                    eps=eps
                )
            else:
                # Direct DBSCAN for smaller point clouds
                class_pc = PointCloud(points=class_points)
                local_labels = class_pc.dbscan(eps=eps, min_points=min_samples, use_gpu=use_gpu)

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

        # Create Clusters object
        clusters = Clusters(labels=output_cluster_labels)
        clusters.set_random_color()

        # Create Clusters DataNode (child of input FeatureClasses)
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

        # Create FeatureClasses object (preserves original class structure)
        # FeatureClasses.labels contains CLASS IDs (same as input FeatureClasses)
        # FeatureClasses.class_mapping maps class IDs to class names (same as input)
        feature_classes = FeatureClasses(
            labels=output_class_labels,
            class_mapping=input_feature_classes.class_mapping.copy(),
            class_colors=input_feature_classes.class_colors.copy()
        )

        # Create FeatureClasses DataNode as child of Clusters
        feature_classes_node = DataNode(
            data=feature_classes,
            data_type="feature_classes",
            parent_uid=clusters_uid,
            depends_on=[clusters_uid],
            tags=["cluster_by_class", "auto_classified"]
        )

        # Add FeatureClasses node to tree
        feature_classes_uid = data_nodes.add_node(feature_classes_node)
        tree_widget.add_branch(
            str(feature_classes_uid),
            str(clusters_uid),
            "feature_classes"
        )

        # Show success message
        QMessageBox.information(
            main_window,
            "Success",
            f"Clustering complete!\n\n"
            f"Created {next_cluster_id} cluster instances across {len(unique_class_ids)} classes.\n\n"
            f"Processing time: {elapsed_time:.2f} seconds\n"
            f"Processing speed: {len(points)/elapsed_time:,.0f} points/sec\n\n"
            f"- Clusters node added with merged cluster IDs\n"
            f"- FeatureClasses child node added with automatic classification"
        )
