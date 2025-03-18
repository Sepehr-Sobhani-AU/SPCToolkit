# plugins/analysis/dist_to_ground_plugin.py
"""
Plugin for calculating vertical distances from points to nearest ground points.

This module provides functionality to compute the vertical (Z-axis) distance
between points in a data node and their nearest neighbors in the XY plane
from a reference "ground" data node.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.spatial import KDTree

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.dist_to_ground import DistToGround


class DistToGroundPlugin(AnalysisPlugin):
    """
    Plugin for calculating vertical distances from points to nearest ground points.

    This plugin finds the nearest point in XY plane from a reference point cloud
    (typically ground) and calculates the vertical (Z-axis) distance between each
    source point and its corresponding nearest neighbor.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The unique name "dist_to_ground"
        """
        return "dist_to_ground"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for vertical distance calculation.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        # Get the data nodes manager from global variables to build dropdown
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes

        # Get all node UUIDs and names for the dropdown
        node_options = {}
        for node_uid, node in data_nodes.data_nodes.items():
            node_options[str(node_uid)] = node.params

        # Set default if options exist
        default_value = ""
        if node_options:
            default_value = next(iter(node_options))

        return {
            "ground_node": {
                "type": "dropdown",
                "options": node_options,
                "default": default_value,
                "label": "Ground Reference",
                "description": "Branch to use as ground reference"
            },
            "max_xy_distance": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 100.0,
                "label": "Maximum XY Distance",
                "description": "Maximum horizontal distance to search for ground points"
            },
            "batch_size": {
                "type": "int",
                "default": 100000,
                "min": 1000,
                "max": 1000000,
                "label": "Batch Size",
                "description": "Number of points to process in each batch"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Calculate vertical distances from source points to ground points.

        Args:
            data_node (DataNode): The data node containing source points
            params (Dict[str, Any]): Parameters for the calculation

        Returns:
            Tuple[DistToGround, str, List]:
                - DistToGround object with vertical distances
                - Result type identifier "dist_to_ground"
                - List containing data_node UIDs as dependencies
        """
        # 1. Set up batch processing
        batch_size = params["batch_size"]

        # 2. Get source and ground point clouds
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes
        data_manager = global_variables.global_data_manager

        # Extract source points
        source_pc = data_node.data
        source_points = source_pc.points

        # Get ground point cloud
        try:
            import uuid
            ground_uid = uuid.UUID(params["ground_node"])
            ground_node = data_nodes.get_node(ground_uid)

            if ground_node is None:
                raise ValueError(f"Ground reference branch with UUID {ground_uid} not found")

            # Reconstruct ground point cloud
            ground_pc = data_manager.reconstruct_branch(ground_uid)
            ground_points = ground_pc.points

            print(f"Source points: {len(source_points)}, Ground points: {len(ground_points)}")

            # Set maximum search distance
            max_xy_distance = params["max_xy_distance"]

            # 3. Calculate vertical distances in batches
            distances = self._calculate_vertical_distances(
                source_points,
                ground_points,
                max_xy_distance,
                batch_size
            )

            # 4. Create DistToGround object
            dist_to_ground = DistToGround(distances)

            # Return results and dependencies
            dependencies = [data_node.uid, ground_uid]
            return dist_to_ground, "dist_to_ground", dependencies

        except Exception as e:
            raise ValueError(f"Error calculating vertical distances: {str(e)}")

    def _calculate_vertical_distances(self, source_points, ground_points, max_xy_distance, batch_size):
        """
        Calculate vertical distances from source points to nearest ground points.

        Args:
            source_points (np.ndarray): Source point cloud points
            ground_points (np.ndarray): Ground point cloud points
            max_xy_distance (float): Maximum horizontal distance to search
            batch_size (int): Number of points to process in each batch

        Returns:
            np.ndarray: Array of vertical distances
        """
        total_points = len(source_points)

        # Create array to store results
        distances = np.full(total_points, np.nan, dtype=np.float32)

        # Process in batches to handle large point clouds
        num_batches = (total_points + batch_size - 1) // batch_size

        # Build KD-Tree for ground points (in XY plane only)
        ground_points_xy = ground_points[:, :2]
        kdtree = KDTree(ground_points_xy)

        print(f"Processing {total_points} points in {num_batches} batches...")

        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_points)

            # Get batch points
            batch_points = source_points[start_idx:end_idx]
            batch_points_xy = batch_points[:, :2]

            # Find nearest neighbors in XY plane
            dists, indices = kdtree.query(batch_points_xy, k=1, distance_upper_bound=max_xy_distance)

            # Handle points with no neighbors within the threshold
            valid_mask = np.isfinite(dists)

            # Get Z coordinates
            batch_z = batch_points[:, 2]

            # Initialize batch distances with NaN (for points with no neighbors)
            batch_distances = np.full(len(batch_points), np.nan, dtype=np.float32)

            # Calculate Z-differences for valid matches
            if np.any(valid_mask):
                ground_z = ground_points[indices[valid_mask], 2]
                batch_distances[valid_mask] = batch_z[valid_mask] - ground_z

            # Store batch results
            distances[start_idx:end_idx] = batch_distances

            print(f"Processed batch {batch_idx + 1}/{num_batches}, found matches for "
                  f"{np.sum(valid_mask)}/{len(batch_points)} points")

        # Replace remaining NaN values with large sentinel value
        nan_mask = np.isnan(distances)
        if np.any(nan_mask):
            print(f"Warning: {np.sum(nan_mask)} points had no ground points within "
                  f"{max_xy_distance} units in XY plane")
            distances[nan_mask] = 9999.0  # Sentinel value

        return distances
