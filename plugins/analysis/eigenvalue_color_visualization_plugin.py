# plugins/analysis/eigenvalue_color_visualization_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.eigenvalues import Eigenvalues
from services.eigenvalue_utils import EigenvalueAnalyser


class EigenvalueColorVisualizationPlugin(AnalysisPlugin):
    """
    Plugin for visualizing eigenvalues using color mapping.

    This plugin takes eigenvalues and applies different visualization techniques
    to create an intuitive representation of geometric features through colors.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "eigenvalue_color_visualization"
        """
        return "eigenvalue_color_visualization"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for eigenvalue visualization.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "visualization_type": {
                "type": "choice",
                "default": "eigenvalue_diff",
                "choices": [
                    "eigenvalue_diff",
                    "planarity",
                    "linearity",
                    "sphericity",
                    "anisotropy"
                ],
                "label": "Visualization Type",
                "description": "Property to visualize"
            },
            "enhance_contrast": {
                "type": "bool",
                "default": False,
                "label": "Enhance Contrast",
                "description": "Apply contrast enhancement to make differences more visible"
            },
            "apply_gamma": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 3.0,
                "label": "Gamma Correction",
                "description": "Apply gamma correction to the colors (values > 1 darken, values < 1 brighten)"
            },
            "calculate_eigenvalues": {
                "type": "bool",
                "default": False,
                "label": "Calculate Eigenvalues",
                "description": "Calculate eigenvalues if not already available (for point clouds)"
            },
            "k_neighbors": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "label": "Number of Neighbours",
                "description": "Number of nearest neighbours for eigenvalue computation (if calculating eigenvalues)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute eigenvalue visualization on point cloud data.

        Args:
            data_node (DataNode): The data node to visualize
            params (Dict[str, Any]): Parameters for visualization

        Returns:
            Tuple[PointCloud, str, List]:
                - Point cloud with colors applied
                - Result type identifier "point_cloud"
                - List containing dependencies
        """
        # Extract parameters
        visualization_type = params["visualization_type"]
        enhance_contrast = params.get("enhance_contrast", False)
        gamma = params.get("apply_gamma", 1.0)
        calculate_eigenvalues = params.get("calculate_eigenvalues", False)
        k_neighbors = params.get("k_neighbors", 20)

        # Track dependencies
        dependencies = []

        # CASE 1: Direct visualization of eigenvalue data node
        if data_node.data_type == "eigenvalues":
            print("Found eigenvalue data, using it directly for visualization")
            eigenvalue_obj = data_node.data
            eigenvalues = eigenvalue_obj.eigenvalues  # Get the numpy array from Eigenvalues object
            dependencies.append(data_node.uid)

            # Get dependencies (to find the original point cloud)
            if not data_node.depends_on or len(data_node.depends_on) == 0:
                raise ValueError("Eigenvalue data node must have a dependency on a point cloud")

            # Get the parent data node (the point cloud)
            parent_uid = data_node.depends_on[0]
            dependencies.append(parent_uid)

            # Access the global data nodes to get the parent
            from config.config import global_variables
            parent_data_node = global_variables.global_data_nodes.get_node(parent_uid)

            if not parent_data_node or not hasattr(parent_data_node.data, "points"):
                raise ValueError("Cannot find the original point cloud. Please ensure proper dependencies.")

            # Get the original point cloud
            point_cloud_data = parent_data_node.data

        # CASE 2: Working directly with a point cloud
        elif hasattr(data_node.data, "points"):
            point_cloud_data = data_node.data
            dependencies.append(data_node.uid)

            # Check if the point cloud already has eigenvalues as an attribute
            if hasattr(point_cloud_data, "attributes") and "eigenvalues" in point_cloud_data.attributes:
                print("Found eigenvalues stored as attributes in the point cloud")
                eigenvalues = point_cloud_data.attributes["eigenvalues"]
            # Or try to calculate eigenvalues if requested
            elif calculate_eigenvalues:
                print(f"Calculating eigenvalues with k={k_neighbors}...")
                analyser = EigenvalueAnalyser(use_cpu=True)
                eigenvalues = analyser.compute_eigenvalues(
                    point_cloud_data.points,
                    k=k_neighbors,
                    smooth=True,
                    batch_size=1000
                )
            else:
                raise ValueError(
                    "This point cloud doesn't have eigenvalues. Either:\n"
                    "1. Run 'Calculate Eigenvalues' first and select that result, or\n"
                    "2. Check 'Calculate Eigenvalues' in the plugin parameters."
                )
        # CASE 3: Unsupported data type
        else:
            print(f"Data node type: {data_node.data_type}, Data type: {type(data_node.data)}")
            raise ValueError(
                "This plugin requires eigenvalues or a point cloud. "
                "The selected data node doesn't appear to be either."
            )

        # Create a copy of the point cloud
        point_cloud = point_cloud_data.get_subset(np.ones(point_cloud_data.size(), dtype=bool))

        print(f"Processing eigenvalues for visualization...")

        # Verify that we have the right number of eigenvalues
        if len(eigenvalues) != point_cloud.size():
            raise ValueError(f"Eigenvalue count ({len(eigenvalues)}) does not match point count ({point_cloud.size()})")

        # Generate colors based on the visualization type
        if visualization_type == "eigenvalue_diff":
            print("Applying eigenvalue difference coloring...")
            colors = self._eigenvalues_to_colors(eigenvalues)
        else:
            # For other visualization types, get the specific feature
            print(f"Visualizing {visualization_type}...")

            if visualization_type == "planarity":
                feature_values = self._compute_planarity(eigenvalues)
            elif visualization_type == "linearity":
                feature_values = self._compute_linearity(eigenvalues)
            elif visualization_type == "sphericity":
                feature_values = self._compute_sphericity(eigenvalues)
            elif visualization_type == "anisotropy":
                feature_values = self._compute_anisotropy(eigenvalues)
            else:
                raise ValueError(f"Unknown visualization type: {visualization_type}")

            # Normalize feature values to [0, 1] for visualization
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)

            if max_val > min_val:
                normalized_values = (feature_values - min_val) / (max_val - min_val)
            else:
                normalized_values = np.zeros_like(feature_values)

            # Create a colormap (blue to red)
            # Blue for low values, green for medium values, red for high values
            colors = np.zeros((len(normalized_values), 3), dtype=np.float32)

            # Apply colormap
            for i, val in enumerate(normalized_values):
                if val < 0.5:
                    # Blue to green (0.0 - 0.5)
                    t = val * 2
                    colors[i, 0] = 0.0  # R
                    colors[i, 1] = t  # G
                    colors[i, 2] = 1.0 - t  # B
                else:
                    # Green to red (0.5 - 1.0)
                    t = (val - 0.5) * 2
                    colors[i, 0] = t  # R
                    colors[i, 1] = 1.0 - t  # G
                    colors[i, 2] = 0.0  # B

            print(f"Feature range: [{min_val:.4f}, {max_val:.4f}]")

        # Apply optional contrast enhancement
        if enhance_contrast:
            print("Enhancing contrast...")
            # Calculate percentiles for each channel to clip outliers
            percentile_low = 5
            percentile_high = 95

            for channel in range(3):
                channel_values = colors[:, channel]
                p_low = np.percentile(channel_values, percentile_low)
                p_high = np.percentile(channel_values, percentile_high)

                # Clip and rescale to enhance contrast
                channel_values = np.clip(channel_values, p_low, p_high)
                channel_min = np.min(channel_values)
                channel_max = np.max(channel_values)

                if channel_max > channel_min:
                    colors[:, channel] = (channel_values - channel_min) / (channel_max - channel_min)

        # Apply gamma correction if needed
        if gamma != 1.0:
            print(f"Applying gamma correction with gamma={gamma}...")
            colors = np.power(colors, 1.0 / gamma)

        # Ensure all values are in [0, 1] range
        colors = np.clip(colors, 0, 1)

        # Apply colors to the point cloud
        point_cloud.colors = colors.astype(np.float32)

        print("Eigenvalue visualization complete")

        # Return the point cloud with colors applied
        return point_cloud, "point_cloud", dependencies

    def _eigenvalues_to_colors(self, eigenvalues):
        """
        Convert eigenvalues to RGB colors based on eigenvalue differences.

        Args:
            eigenvalues (np.ndarray): Array of eigenvalues with shape (n_points, 3)

        Returns:
            np.ndarray: RGB color values with shape (n_points, 3)
        """
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10

        # Normalize the eigenvalues row-wise
        eigenvalue_sums = np.sum(eigenvalues, axis=1, keepdims=True)
        # Replace zero sums with epsilon to avoid division by zero
        eigenvalue_sums = np.maximum(eigenvalue_sums, epsilon)
        normalized_eigenvalues = eigenvalues / eigenvalue_sums

        # Calculate the differences
        differences = np.zeros_like(normalized_eigenvalues)
        differences[:, 0] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 1]  # First minus Second
        differences[:, 1] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 2]  # First minus Third
        differences[:, 2] = normalized_eigenvalues[:, 2] - normalized_eigenvalues[:, 1]  # Third minus Second

        # Normalize the differences to [0, 1]
        min_vals = np.min(differences, axis=0)
        max_vals = np.max(differences, axis=0)

        # Avoid division by zero when min and max are equal
        divisors = np.maximum(max_vals - min_vals, epsilon)
        scaled_differences = (differences - min_vals) / divisors

        # Ensure all values are in [0, 1] range
        return np.clip(scaled_differences, 0, 1)

    def _compute_planarity(self, eigenvalues):
        """
        Calculate planarity (λ₂ - λ₁)/λ₃ for each point.

        Planarity measures how flat the local surface is. Higher values indicate
        more planar (flat) regions.

        Args:
            eigenvalues (np.ndarray): Array of eigenvalues with shape (n_points, 3)

        Returns:
            np.ndarray: Planarity values for each point
        """
        epsilon = 1e-10  # To prevent division by zero
        lambda1 = eigenvalues[:, 0]  # Smallest eigenvalue
        lambda2 = eigenvalues[:, 1]  # Middle eigenvalue
        lambda3 = eigenvalues[:, 2]  # Largest eigenvalue

        planarity = (lambda2 - lambda1) / np.maximum(lambda3, epsilon)
        return np.nan_to_num(planarity)

    def _compute_linearity(self, eigenvalues):
        """
        Calculate linearity (λ₃ - λ₂)/λ₃ for each point.

        Linearity measures how much the local neighborhood resembles a line.
        Higher values indicate more linear structures like edges or thin features.

        Args:
            eigenvalues (np.ndarray): Array of eigenvalues with shape (n_points, 3)

        Returns:
            np.ndarray: Linearity values for each point
        """
        epsilon = 1e-10
        lambda2 = eigenvalues[:, 1]
        lambda3 = eigenvalues[:, 2]

        linearity = (lambda3 - lambda2) / np.maximum(lambda3, epsilon)
        return np.nan_to_num(linearity)

    def _compute_sphericity(self, eigenvalues):
        """
        Calculate sphericity λ₁/λ₃ for each point.

        Sphericity measures how spherical or corner-like the local neighborhood is.
        Higher values indicate corners or point-like features.

        Args:
            eigenvalues (np.ndarray): Array of eigenvalues with shape (n_points, 3)

        Returns:
            np.ndarray: Sphericity values for each point
        """
        epsilon = 1e-10
        lambda1 = eigenvalues[:, 0]
        lambda3 = eigenvalues[:, 2]

        sphericity = lambda1 / np.maximum(lambda3, epsilon)
        return np.nan_to_num(sphericity)

    def _compute_anisotropy(self, eigenvalues):
        """
        Calculate anisotropy (λ₃ - λ₁)/λ₃ for each point.

        Anisotropy measures the directional dependence of the local structure.
        Higher values indicate stronger directional structure.

        Args:
            eigenvalues (np.ndarray): Array of eigenvalues with shape (n_points, 3)

        Returns:
            np.ndarray: Anisotropy values for each point
        """
        epsilon = 1e-10
        lambda1 = eigenvalues[:, 0]
        lambda3 = eigenvalues[:, 2]

        anisotropy = (lambda3 - lambda1) / np.maximum(lambda3, epsilon)
        return np.nan_to_num(anisotropy)