# plugins/analysis/eigenvalue_visualization_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.colors import Colors
from services.eigenvalue_utils import EigenvalueUtils


class EigenvalueVisualizationPlugin(AnalysisPlugin):
    """
    Plugin for visualizing eigenvalues by converting them to colors.

    Takes a PointCloud with eigenvalues and creates colors derived
    from those eigenvalues. The colors are returned as a Colors object.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "eigenvalue_visualization"
        """
        return "eigenvalue_visualization"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for eigenvalue visualization.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        return {
            "visualization_mode": {
                "type": "choice",
                "options": ["raw_eigenvalues", "geometric_features", "sparsity"],
                "default": "raw_eigenvalues",
                "label": "Visualization Mode",
                "description": "How to visualize eigenvalues (raw, geometric features, or sparsity)"
            },
            "feature_type": {
                "type": "choice",
                "options": ["planarity", "linearity", "sphericity", "anisotropy", "total_variance"],
                "default": "planarity",
                "label": "Feature Type",
                "description": "Geometric feature to visualize (used when mode is geometric_features)"
            },
            "color_scale": {
                "type": "choice",
                "options": ["blue_to_red", "viridis", "grayscale"],
                "default": "blue_to_red",
                "label": "Color Scale",
                "description": "Color scale to use for visualization"
            },
            "sparsity_threshold": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 1.0,
                "label": "Sparsity Threshold",
                "description": "Threshold for identifying sparse regions (lower values = more sensitive)"
            },
            "k_neighbors": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "label": "Number of Neighbors",
                "description": "Number of neighbors to consider when computing sparsity"
            },
            "smooth": {
                "type": "bool",
                "default": True,
                "label": "Smooth Colors",
                "description": "Whether to smooth colors across neighbors"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute eigenvalue visualization on the PointCloud.

        Args:
            data_node (DataNode): The data node containing the PointCloud
            params (Dict[str, Any]): Parameters for visualization

        Returns:
            Tuple[Colors, str, List]:
                - Colors object containing RGB values derived from eigenvalues
                - Result type identifier "colors"
                - List containing the data_node's UID as a dependency
        """
        # Ensure input data is a PointCloud
        if not isinstance(data_node.data, PointCloud):
            raise ValueError("Input data must be a PointCloud object")

        point_cloud = data_node.data

        # Check if eigenvalues were previously computed
        eigenvalue_utils = EigenvalueUtils()

        # Get parameters
        visualization_mode = params.get("visualization_mode", "raw_eigenvalues")
        feature_type = params.get("feature_type", "planarity")
        color_scale = params.get("color_scale", "blue_to_red")
        smooth_colors = params.get("smooth", True)
        sparsity_threshold = params.get("sparsity_threshold", 0.5)
        k_neighbors = params.get("k_neighbors", 20)

        # If eigenvalues weren't previously computed or we need them for a new k value (for sparsity)
        compute_new_eigenvalues = eigenvalue_utils._last_computed_eigenvalues is None

        # Force recomputing eigenvalues for sparsity mode with specific k
        if visualization_mode == "sparsity":
            compute_new_eigenvalues = True

        if compute_new_eigenvalues:
            # Compute eigenvalues with the specified parameters
            eigenvalues = eigenvalue_utils.get_eigenvalues(point_cloud.points, k=k_neighbors, smooth=smooth_colors)
        else:
            # Use the last computed eigenvalues
            eigenvalues = eigenvalue_utils._last_computed_eigenvalues

        # Generate colors based on parameters
        if visualization_mode == "raw_eigenvalues":
            # Use the default eigenvalues_to_colors method
            color_values = eigenvalue_utils.eigenvalues_to_colors(eigenvalues)

        elif visualization_mode == "sparsity":
            # Compute sparsity based on eigenvalues and k-nearest neighbors
            if eigenvalue_utils._last_tree is None or eigenvalue_utils._last_indices is None:
                # If we don't have the KD-tree from the eigenvalue computation, create it
                from scipy.spatial import KDTree
                tree = KDTree(point_cloud.points)
                distances, indices = tree.query(point_cloud.points, k=k_neighbors)
            else:
                # Use the existing KD-tree and indices from eigenvalue computation
                distances = None
                indices = eigenvalue_utils._last_indices
                tree = eigenvalue_utils._last_tree

            # Calculate mean distances to k-nearest neighbors if needed
            if distances is None:
                # Compute distances manually using the indices
                neighbor_points = point_cloud.points[indices]
                center_points = np.expand_dims(point_cloud.points, axis=1)
                distances = np.sqrt(np.sum((neighbor_points - center_points) ** 2, axis=2))

            # Get the mean distance to k-nearest neighbors
            mean_distances = np.mean(distances, axis=1)

            # Normalize the mean distances to [0,1] range
            max_dist = np.max(mean_distances)
            min_dist = np.min(mean_distances)
            if max_dist > min_dist:
                normalized_distances = (mean_distances - min_dist) / (max_dist - min_dist)
            else:
                normalized_distances = np.zeros_like(mean_distances)

            # Adjust with the sparsity threshold
            # Higher threshold = fewer points considered sparse
            sparsity_score = normalized_distances / sparsity_threshold
            sparsity_score = np.clip(sparsity_score, 0, 1)

            # Create color values based on sparsity score and selected color scale
            if color_scale == "blue_to_red":
                # Blue (0,0,1) to Red (1,0,0) - red indicates higher sparsity
                color_values = np.zeros((len(sparsity_score), 3), dtype=np.float32)
                color_values[:, 0] = sparsity_score  # Red channel increases with sparsity
                color_values[:, 2] = 1.0 - sparsity_score  # Blue channel decreases with sparsity
            elif color_scale == "viridis":
                # Approximate viridis colormap for sparsity
                color_values = np.zeros((len(sparsity_score), 3), dtype=np.float32)
                color_values[:, 0] = 0.8 * sparsity_score  # Red increases with sparsity
                color_values[:, 1] = 0.8 * np.sin(np.pi * sparsity_score)  # Green peaks in middle
                color_values[:, 2] = 0.8 * (1.0 - sparsity_score)  # Blue decreases with sparsity
            else:  # grayscale
                # Simple grayscale - brighter values indicate higher sparsity
                color_values = np.zeros((len(sparsity_score), 3), dtype=np.float32)
                # Use inverted score so sparse points are brighter
                gray_value = sparsity_score
                color_values[:, 0] = gray_value  # Red
                color_values[:, 1] = gray_value  # Green
                color_values[:, 2] = gray_value  # Blue

        else:  # geometric_features
            # Compute geometric features and use the specified feature for coloring
            features = eigenvalue_utils.compute_geometric_features(eigenvalues)

            # Get the selected feature
            feature_values = features[feature_type]

            # Normalize feature values to [0,1] range
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            normalized_values = (feature_values - min_val) / (
                        max_val - min_val) if max_val > min_val else np.zeros_like(feature_values)

            # Apply the selected color scale
            if color_scale == "blue_to_red":
                # Blue (0,0,1) to Red (1,0,0)
                color_values = np.zeros((len(normalized_values), 3), dtype=np.float32)
                color_values[:, 0] = normalized_values  # Red channel increases with value
                color_values[:, 2] = 1.0 - normalized_values  # Blue channel decreases with value
            elif color_scale == "viridis":
                # Approximate viridis colormap (simplified version)
                color_values = np.zeros((len(normalized_values), 3), dtype=np.float32)
                color_values[:, 0] = 0.8 * normalized_values  # Red increases
                color_values[:, 1] = 0.8 * np.sin(np.pi * normalized_values)  # Green peaks in middle
                color_values[:, 2] = 0.8 * (1.0 - normalized_values)  # Blue decreases
            else:  # grayscale
                # Simple grayscale
                color_values = np.zeros((len(normalized_values), 3), dtype=np.float32)
                color_values[:, 0] = normalized_values  # Red
                color_values[:, 1] = normalized_values  # Green
                color_values[:, 2] = normalized_values  # Blue

        # Create a Colors object
        colors = Colors(color_values)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return colors, "colors", dependencies