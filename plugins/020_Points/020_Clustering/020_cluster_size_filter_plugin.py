# plugins/analysis/cluster_size_filter_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import Plugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.masks import Masks


class ClusterSizeFilterPlugin(Plugin):
    """
    Plugin for filtering clusters based on the number of points they contain.

    This plugin analyzes a point cloud that has cluster labels assigned and creates a mask
    identifying points that belong to clusters with at least the specified
    minimum number of points. It can also optionally include or exclude noise
    points (typically labeled as -1 in clustering algorithms like DBSCAN).

    The plugin is useful for filtering out small clusters that might represent
    noise or insignificant features, allowing users to focus on more prominent
    structures in the point cloud data.

    This plugin is designed to be robust and will not interrupt the program flow
    if errors occur. Instead, it will report errors and return a fallback result
    that allows the process to continue.
    """

    def _create_fallback_mask(self, point_cloud):
        """
        Create a fallback mask that includes all points.

        This method is used when an error occurs during processing to ensure
        the program can continue running.

        Args:
            point_cloud (PointCloud or None): The point cloud to create a mask for

        Returns:
            Masks: A Masks object with all points included
        """
        try:
            if point_cloud is not None and hasattr(point_cloud, 'size'):
                size = point_cloud.size
                all_points_mask = np.ones(size, dtype=bool)
            else:
                # If we can't determine the size, create a small dummy mask
                all_points_mask = np.ones(1, dtype=bool)

            return Masks(all_points_mask)
        except Exception as e:
            print(f"[ClusterSizeFilter] ERROR: Error creating fallback mask: {str(e)}")
            # Last resort fallback
            return Masks(np.ones(1, dtype=bool))

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The name "cluster_size_filter"
        """
        return "cluster_size_filter"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for cluster size filtering.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        return {
            "min_points": {
                "type": "int",
                "default": 1024,
                "min": 1,
                "max": 100000,
                "label": "Minimum Points Per Cluster",
                "description": "Minimum number of points required for a cluster to be included"
            },
            "include_noise": {
                "type": "bool",
                "default": False,
                "label": "Include Noise Points",
                "description": "Whether to include points labeled as noise (typically -1)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the cluster size filtering on the reconstructed point cloud with cluster labels.

        Args:
            data_node (DataNode): The data node containing the point cloud with cluster labels
            params (Dict[str, Any]): Parameters for filtering (min_points, include_noise)

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the filtering mask
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency

        Note:
            This plugin handles errors gracefully. If any issues occur during processing,
            it will print error messages but won't raise exceptions that would interrupt
            the program flow. In error cases, it returns a mask that includes all points.
        """
        try:
            # Extract parameters with defaults in case they're missing
            min_points = params.get("min_points", 100)
            include_noise = params.get("include_noise", False)

            # Validate the data node
            if data_node is None or not hasattr(data_node, 'data'):
                print("[ClusterSizeFilter] ERROR: Invalid data node.")
                return self._create_fallback_mask(None), "masks", []

            # Get the point cloud
            point_cloud = data_node.data
            if point_cloud is None:
                print("[ClusterSizeFilter] ERROR: Data node contains no data.")
                return self._create_fallback_mask(None), "masks", [data_node.uid]

            # Print debug information
            print(f"[ClusterSizeFilter] DataNode data_type: {data_node.data_type}")
            print(f"[ClusterSizeFilter] DataNode data class: {point_cloud.__class__.__name__}")

            # Validate cluster labels
            cluster_labels = point_cloud.get_attribute("cluster_labels")
            print(f"[ClusterSizeFilter] Has cluster_labels: {cluster_labels is not None}")

            if cluster_labels is None:
                print("[ClusterSizeFilter] ERROR: The point cloud does not have cluster labels.")
                return self._create_fallback_mask(point_cloud), "masks", [data_node.uid]

            if not hasattr(cluster_labels, '__len__'):
                print("[ClusterSizeFilter] ERROR: Cluster labels attribute is not a collection.")
                return self._create_fallback_mask(point_cloud), "masks", [data_node.uid]

            if len(cluster_labels) == 0:
                print("[ClusterSizeFilter] ERROR: Cluster labels array is empty.")
                return self._create_fallback_mask(point_cloud), "masks", [data_node.uid]

            # Main processing block
            try:
                # Get the cluster labels
                labels = cluster_labels

                # Count the number of points in each cluster
                unique_labels, point_counts = np.unique(labels, return_counts=True)

                print(f"[ClusterSizeFilter] Found {len(unique_labels)} unique cluster labels")
                print(f"[ClusterSizeFilter] Unique labels: {unique_labels}")
                print(f"[ClusterSizeFilter] Counts: {point_counts}")

                # Create a boolean array where True means the cluster meets the size criteria
                valid_clusters = point_counts >= min_points

                # Handle noise points (labeled as -1) separately based on include_noise parameter
                noise_mask = unique_labels == -1
                if np.any(noise_mask):
                    valid_clusters[noise_mask] = include_noise

                # Create a mapping from cluster labels to validity status
                valid_label_dict = dict(zip(unique_labels, valid_clusters))

                # Use numpy vectorize to apply the mapping to all points' labels
                vectorized_lookup = np.vectorize(lambda x: valid_label_dict.get(x, False))

                # Apply the function to all labels to get the final mask
                mask = vectorized_lookup(labels)

                # Count statistics for reporting
                kept_points = np.sum(mask)
                kept_clusters = np.sum(valid_clusters & (unique_labels != -1))

                # Log information about the filtering
                print(f"[ClusterSizeFilter] Filtering clusters with less than {min_points} points")
                print(
                    f"[ClusterSizeFilter] Original clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
                print(f"[ClusterSizeFilter] Clusters meeting criteria: {kept_clusters}")
                print(f"[ClusterSizeFilter] Points kept: {kept_points} out of {len(labels)} "
                      f"({kept_points / len(labels) * 100:.1f}%)")

                # Create a Masks object with the result
                result_mask = Masks(mask)

                # Return results
                return result_mask, "masks", [data_node.uid]

            except Exception as e:
                print(f"[ClusterSizeFilter] ERROR during processing: {str(e)}")
                return self._create_fallback_mask(point_cloud), "masks", [data_node.uid]

        except Exception as e:
            print(f"[ClusterSizeFilter] CRITICAL ERROR: {str(e)}")
            try:
                return self._create_fallback_mask(None), "masks", [data_node.uid if data_node else None]
            except:
                return Masks(np.ones(1, dtype=bool)), "masks", []
