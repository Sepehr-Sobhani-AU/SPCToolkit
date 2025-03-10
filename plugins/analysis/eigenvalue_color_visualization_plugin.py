# plugins/analysis/eigenvalue_color_visualization_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from services.eigenvalue_utils import EigenvalueUtils


class EigenvalueColorVisualizationPlugin(AnalysisPlugin):
    """
    Plugin for visualizing eigenvalues as colors on a point cloud.

    This plugin takes eigenvalues and applies different color mapping techniques
    to create intuitive visualizations of geometric features in point clouds.
    It can work with existing eigenvalue data or calculate eigenvalues on-the-fly.
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
        Define the parameters needed for eigenvalue color visualization.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "visualization_mode": {
                "type": "choice",
                "default": "eigenvalue_diff",
                "choices": [
                    "eigenvalue_diff",
                    "rgb_eigenvalues",
                    "geometric_features",
                    "single_feature"
                ],
                "label": "Visualization Mode",
                "description": "Method for mapping eigenvalues to colors"
            },
            "feature_type": {
                "type": "choice",
                "default": "planarity",
                "choices": [
                    "planarity",
                    "linearity",
                    "sphericity",
                    "anisotropy",
                    "smallest_eigenvalue",
                    "middle_eigenvalue",
                    "largest_eigenvalue"
                ],
                "label": "Feature Type",
                "description": "Geometric feature to visualize (when using single_feature mode)"
            },
            "colormap": {
                "type": "choice",
                "default": "viridis",
                "choices": [
                    "viridis",
                    "plasma",
                    "inferno",
                    "jet",
                    "turbo",
                    "rainbow",
                    "coolwarm",
                    "blues_to_red"
                ],
                "label": "Colormap",
                "description": "Color scheme to use for feature visualization"
            },
            "enhance_contrast": {
                "type": "bool",
                "default": False,
                "label": "Enhance Contrast",
                "description": "Apply contrast enhancement to make differences more visible"
            },
            "gamma": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 3.0,
                "label": "Gamma Correction",
                "description": "Apply gamma correction to colors (values > 1 darken, values < 1 brighten)"
            },
            "min_percentile": {
                "type": "float",
                "default": 5.0,
                "min": 0.0,
                "max": 49.0,
                "label": "Min Percentile",
                "description": "Percentile for color range minimum (removes outliers)"
            },
            "max_percentile": {
                "type": "float",
                "default": 95.0,
                "min": 51.0,
                "max": 100.0,
                "label": "Max Percentile",
                "description": "Percentile for color range maximum (removes outliers)"
            },
            "calculate_eigenvalues": {
                "type": "bool",
                "default": True,
                "label": "Calculate Eigenvalues if Needed",
                "description": "Calculate eigenvalues on-the-fly if the selected node doesn't have them"
            },
            "k_neighbors": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "label": "Number of Neighbours",
                "description": "Number of nearest neighbours for eigenvalue computation (if calculating eigenvalues)"
            },
            "smooth_eigenvalues": {
                "type": "bool",
                "default": True,
                "label": "Smooth Eigenvalues",
                "description": "Apply smoothing when calculating eigenvalues (if calculating eigenvalues)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute eigenvalue color visualization on the point cloud.

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
        visualization_mode = params["visualization_mode"]
        enhance_contrast = params["enhance_contrast"]
        gamma = params["gamma"]
        min_percentile = params["min_percentile"]
        max_percentile = params["max_percentile"]
        colormap = params["colormap"]
        feature_type = params["feature_type"]

        # Get point cloud and eigenvalues
        point_cloud, eigenvalues, dependencies = self._get_data_sources(data_node, params)

        print(f"Visualizing eigenvalues with mode: {visualization_mode}")
        print(f"Point cloud has {point_cloud.size()} points")
        print(f"Eigenvalue array shape: {eigenvalues.shape}")

        # Verify eigenvalues and point cloud match
        if len(eigenvalues) != point_cloud.size():
            raise ValueError(f"Eigenvalues count ({len(eigenvalues)}) doesn't match point count ({point_cloud.size()})")

        # Get utils for eigenvalue operations
        utils = EigenvalueUtils()

        # Apply the appropriate visualization mode
        if visualization_mode == "eigenvalue_diff":
            # Use the eigenvalue difference method from EigenvalueUtils
            colors = utils.eigenvalues_to_colors(eigenvalues)
            print("Applied eigenvalue difference coloring")

        elif visualization_mode == "rgb_eigenvalues":
            # Directly map eigenvalues to RGB channels
            # Normalize eigenvalues by row sums first
            eigenvalue_sums = np.sum(eigenvalues, axis=1, keepdims=True)
            eigenvalue_sums = np.maximum(eigenvalue_sums, 1e-10)  # Avoid division by zero
            colors = eigenvalues / eigenvalue_sums
            print("Applied direct RGB eigenvalue mapping")

        elif visualization_mode == "geometric_features":
            # Calculate geometric features
            features = utils.compute_geometric_features(eigenvalues)

            # Use planarity for red, linearity for green, sphericity for blue
            planarity = features['planarity']
            linearity = features['linearity']
            sphericity = features['sphericity']

            # Stack features into an RGB array
            colors = np.column_stack((planarity, linearity, sphericity))

            # Process each channel separately for better contrast
            for i in range(3):
                p_min = np.percentile(colors[:, i], min_percentile)
                p_max = np.percentile(colors[:, i], max_percentile)
                # Normalize to [0, 1] range
                colors[:, i] = np.clip((colors[:, i] - p_min) / max(p_max - p_min, 1e-10), 0, 1)

            print("Applied geometric features RGB mapping")

        elif visualization_mode == "single_feature":
            # Get features
            features = utils.compute_geometric_features(eigenvalues)

            # Select the specified feature
            if feature_type == "smallest_eigenvalue":
                feature_values = eigenvalues[:, 0]
            elif feature_type == "middle_eigenvalue":
                feature_values = eigenvalues[:, 1]
            elif feature_type == "largest_eigenvalue":
                feature_values = eigenvalues[:, 2]
            elif feature_type in features:
                feature_values = features[feature_type]
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            # Apply percentile clipping to remove outliers
            p_min = np.percentile(feature_values, min_percentile)
            p_max = np.percentile(feature_values, max_percentile)
            # Normalize to [0, 1] range
            normalized_values = np.clip((feature_values - p_min) / max(p_max - p_min, 1e-10), 0, 1)

            # Apply the selected colormap
            colors = self._apply_colormap(normalized_values, colormap)

            print(f"Applied {feature_type} visualization with {colormap} colormap")
            print(f"Feature range after percentile clipping: [{p_min:.4f}, {p_max:.4f}]")

        else:
            raise ValueError(f"Unknown visualization mode: {visualization_mode}")

        # Apply contrast enhancement if requested
        if enhance_contrast and visualization_mode != "single_feature":  # Already done for single_feature
            print("Enhancing contrast...")
            for channel in range(3):
                p_min = np.percentile(colors[:, channel], min_percentile)
                p_max = np.percentile(colors[:, channel], max_percentile)
                # Normalize to [0, 1] range
                colors[:, channel] = np.clip((colors[:, channel] - p_min) / max(p_max - p_min, 1e-10), 0, 1)

        # Apply gamma correction if requested
        if gamma != 1.0:
            print(f"Applying gamma correction with gamma={gamma}...")
            colors = np.power(colors, 1.0 / gamma)

        # Ensure all values are in [0, 1] range
        colors = np.clip(colors, 0, 1).astype(np.float32)

        # Create a copy of the point cloud with colors applied
        result_point_cloud = point_cloud.get_subset(np.ones(point_cloud.size(), dtype=bool))
        result_point_cloud.colors = colors

        print("Eigenvalue color visualization complete")

        return result_point_cloud, "point_cloud", dependencies

    def _get_data_sources(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[PointCloud, np.ndarray, List]:
        """
        Get the point cloud and eigenvalues needed for visualization.

        Handles different data node types and can calculate eigenvalues on-the-fly if needed.

        Args:
            data_node (DataNode): The data node to process
            params (Dict[str, Any]): Parameters with calculation options

        Returns:
            Tuple[PointCloud, np.ndarray, List]:
                - The point cloud for visualization
                - Eigenvalue array with shape (n_points, 3)
                - List of data node dependencies
        """
        dependencies = [data_node.uid]
        calculate_eigenvalues = params["calculate_eigenvalues"]
        k_neighbors = params["k_neighbors"]
        smooth = params["smooth_eigenvalues"]

        # Case 1: Direct eigenvalue data node
        if data_node.data_type == "eigenvalues":
            print("Processing eigenvalue data node")

            # We need to find the parent point cloud that these eigenvalues belong to
            if not data_node.depends_on or len(data_node.depends_on) == 0:
                raise ValueError("Eigenvalue data node must have a dependency on a point cloud")

            parent_uid = data_node.depends_on[0]
            dependencies.append(parent_uid)

            # Access the global data nodes to get the parent
            from config.config import global_variables
            parent_data_node = global_variables.global_data_nodes.get_node(parent_uid)

            if not parent_data_node or not hasattr(parent_data_node.data, "points"):
                raise ValueError("Cannot find the original point cloud. Please ensure proper dependencies.")

            # Get point cloud and eigenvalues
            point_cloud = parent_data_node.data
            eigenvalues = data_node.data.eigenvalues

        # Case 2: Point cloud with eigenvalues in attributes
        elif hasattr(data_node.data, "points") and hasattr(data_node.data,
                                                           "attributes") and "eigenvalues" in data_node.data.attributes:
            print("Processing point cloud with eigenvalues in attributes")
            point_cloud = data_node.data
            eigenvalues = data_node.data.attributes["eigenvalues"]

        # Case 3: Regular point cloud, calculate eigenvalues if allowed
        elif hasattr(data_node.data, "points") and calculate_eigenvalues:
            print(f"Processing point cloud and calculating eigenvalues (k={k_neighbors}, smooth={smooth})")
            point_cloud = data_node.data

            # Calculate eigenvalues
            utils = EigenvalueUtils(use_cpu=True)
            eigenvalues = utils.compute_eigenvalues(
                point_cloud.points,
                k=k_neighbors,
                smooth=smooth,
                batch_size=1000
            )

        # Case 4: Unsupported node type
        else:
            data_type = data_node.data_type
            print(f"Unknown data node type: {data_type}")
            if hasattr(data_node.data, "points"):
                print("Node has points but no eigenvalues and calculation is disabled")
            raise ValueError("This plugin requires eigenvalue data or a point cloud with calculate_eigenvalues=True")

        return point_cloud, eigenvalues, dependencies

    def _apply_colormap(self, values: np.ndarray, colormap_name: str) -> np.ndarray:
        """
        Apply a colormap to a 1D array of values.

        Args:
            values (np.ndarray): 1D array of values in range [0, 1]
            colormap_name (str): Name of the colormap to use

        Returns:
            np.ndarray: RGB colors with shape (n_values, 3)
        """
        n_values = len(values)
        colors = np.zeros((n_values, 3), dtype=np.float32)

        if colormap_name == "blues_to_red":
            # Custom blue to red colormap (through white in the middle)
            for i, val in enumerate(values):
                if val < 0.5:
                    # Blue to white (0-0.5)
                    t = val * 2
                    colors[i, 0] = t  # Red
                    colors[i, 1] = t  # Green
                    colors[i, 2] = 1.0  # Blue
                else:
                    # White to red (0.5-1.0)
                    t = (val - 0.5) * 2
                    colors[i, 0] = 1.0  # Red
                    colors[i, 1] = 1.0 - t  # Green
                    colors[i, 2] = 1.0 - t  # Blue

        elif colormap_name == "rainbow":
            # Simple rainbow colormap
            for i, val in enumerate(values):
                if val < 0.25:
                    # Blue to cyan (0-0.25)
                    t = val * 4
                    colors[i] = [0, t, 1]
                elif val < 0.5:
                    # Cyan to green (0.25-0.5)
                    t = (val - 0.25) * 4
                    colors[i] = [0, 1, 1 - t]
                elif val < 0.75:
                    # Green to yellow (0.5-0.75)
                    t = (val - 0.5) * 4
                    colors[i] = [t, 1, 0]
                else:
                    # Yellow to red (0.75-1.0)
                    t = (val - 0.75) * 4
                    colors[i] = [1, 1 - t, 0]

        elif colormap_name == "jet":
            # Jet colormap (blue-cyan-yellow-red)
            for i, val in enumerate(values):
                if val < 0.125:
                    # Dark blue to blue
                    t = val * 8
                    colors[i] = [0, 0, 0.5 + t * 0.5]
                elif val < 0.375:
                    # Blue to cyan
                    t = (val - 0.125) * 4
                    colors[i] = [0, t, 1]
                elif val < 0.625:
                    # Cyan to yellow
                    t = (val - 0.375) * 4
                    colors[i] = [t, 1, 1 - t]
                elif val < 0.875:
                    # Yellow to red
                    t = (val - 0.625) * 4
                    colors[i] = [1, 1 - t, 0]
                else:
                    # Red to dark red
                    t = (val - 0.875) * 8
                    colors[i] = [1 - t * 0.5, 0, 0]

        elif colormap_name == "viridis":
            # Approximate viridis colormap (dark blue -> green -> yellow)
            for i, val in enumerate(values):
                if val < 0.5:
                    # Dark blue to green
                    t = val * 2
                    colors[i, 0] = 0.0  # R: 0 -> 0
                    colors[i, 1] = t * 0.8  # G: 0 -> 0.8
                    colors[i, 2] = 0.4 + (0.6 - t * 0.6)  # B: 1.0 -> 0.4
                else:
                    # Green to yellow
                    t = (val - 0.5) * 2
                    colors[i, 0] = t  # R: 0 -> 1.0
                    colors[i, 1] = 0.8 + t * 0.2  # G: 0.8 -> 1.0
                    colors[i, 2] = 0.4 - t * 0.4  # B: 0.4 -> 0

        elif colormap_name == "plasma":
            # Approximate plasma colormap (dark purple -> red -> yellow)
            for i, val in enumerate(values):
                if val < 0.5:
                    # Dark purple to red
                    t = val * 2
                    colors[i, 0] = 0.05 + t * 0.95  # R: 0.05 -> 1.0
                    colors[i, 1] = 0.0 + t * 0.4  # G: 0 -> 0.4
                    colors[i, 2] = 0.5 - t * 0.3  # B: 0.5 -> 0.2
                else:
                    # Red to yellow
                    t = (val - 0.5) * 2
                    colors[i, 0] = 1.0  # R: 1.0 -> 1.0
                    colors[i, 1] = 0.4 + t * 0.6  # G: 0.4 -> 1.0
                    colors[i, 2] = 0.2 - t * 0.2  # B: 0.2 -> 0

        elif colormap_name == "inferno":
            # Approximate inferno colormap (black -> purple -> yellow)
            for i, val in enumerate(values):
                if val < 0.33:
                    # Black to purple
                    t = val * 3
                    colors[i, 0] = t * 0.6  # R: 0 -> 0.6
                    colors[i, 1] = 0.0  # G: 0 -> 0
                    colors[i, 2] = t * 0.6  # B: 0 -> 0.6
                elif val < 0.66:
                    # Purple to red
                    t = (val - 0.33) * 3
                    colors[i, 0] = 0.6 + t * 0.4  # R: 0.6 -> 1.0
                    colors[i, 1] = 0.0 + t * 0.3  # G: 0 -> 0.3
                    colors[i, 2] = 0.6 - t * 0.4  # B: 0.6 -> 0.2
                else:
                    # Red to yellow
                    t = (val - 0.66) * 3
                    colors[i, 0] = 1.0  # R: 1.0 -> 1.0
                    colors[i, 1] = 0.3 + t * 0.7  # G: 0.3 -> 1.0
                    colors[i, 2] = 0.2 - t * 0.2  # B: 0.2 -> 0

        elif colormap_name == "turbo":
            # Approximate turbo colormap (blue -> cyan -> green -> yellow -> red)
            for i, val in enumerate(values):
                if val < 0.25:
                    # Blue to cyan
                    t = val * 4
                    colors[i] = [0, t, 1]
                elif val < 0.5:
                    # Cyan to green
                    t = (val - 0.25) * 4
                    colors[i] = [0, 1, 1 - t]
                elif val < 0.75:
                    # Green to yellow
                    t = (val - 0.5) * 4
                    colors[i] = [t, 1, 0]
                else:
                    # Yellow to red
                    t = (val - 0.75) * 4
                    colors[i] = [1, 1 - t, 0]

        elif colormap_name == "coolwarm":
            # Coolwarm colormap (blue -> white -> red)
            for i, val in enumerate(values):
                if val < 0.5:
                    # Blue to white (0-0.5)
                    t = val * 2
                    colors[i, 0] = t  # Red
                    colors[i, 1] = t  # Green
                    colors[i, 2] = 1.0  # Blue
                else:
                    # White to red (0.5-1.0)
                    t = (val - 0.5) * 2
                    colors[i, 0] = 1.0  # Red
                    colors[i, 1] = 1.0 - t  # Green
                    colors[i, 2] = 1.0 - t  # Blue

        else:
            # Default to viridis-like if unknown
            print(f"Unknown colormap '{colormap_name}', defaulting to viridis")
            return self._apply_colormap(values, "viridis")

        return colors