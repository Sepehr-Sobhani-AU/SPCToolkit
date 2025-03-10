# plugins/analysis/eigenvalue_filtering_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.eigenvalues import Eigenvalues
from core.masks import Masks
from services.eigenvalue_utils import EigenvalueUtils


class EigenvalueFilteringPlugin(AnalysisPlugin):
    """
    Plugin for filtering eigenvalues based on custom thresholds or predefined criteria.

    This plugin allows filtering points based on their eigenvalues or derived geometric
    features, enabling the identification of specific structures like planes, edges,
    and corners in point cloud data.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "eigenvalue_filtering"
        """
        return "eigenvalue_filtering"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for eigenvalue filtering.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "filter_type": {
                "type": "choice",
                "default": "predefined",
                "choices": [
                    "predefined",
                    "manual",
                    "geometric_features"
                ],
                "label": "Filter Type",
                "description": "Method for filtering eigenvalues"
            },
            # Parameters for predefined filters
            "predefined_filter": {
                "type": "choice",
                "default": "planes",
                "choices": [
                    "planes",
                    "edges",
                    "corners",
                    "linear_features",
                    "noise"
                ],
                "label": "Predefined Filter",
                "description": "Predefined geometric structure to extract"
            },
            # Parameters for manual eigenvalue filtering
            "lambda1_min": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "label": "λ₁ Minimum (Smallest Eigenvalue)",
                "description": "Minimum value for the smallest eigenvalue λ₁"
            },
            "lambda1_max": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "label": "λ₁ Maximum",
                "description": "Maximum value for the smallest eigenvalue λ₁"
            },
            "lambda2_min": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "label": "λ₂ Minimum (Middle Eigenvalue)",
                "description": "Minimum value for the middle eigenvalue λ₂"
            },
            "lambda2_max": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "label": "λ₂ Maximum",
                "description": "Maximum value for the middle eigenvalue λ₂"
            },
            "lambda3_min": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "label": "λ₃ Minimum (Largest Eigenvalue)",
                "description": "Minimum value for the largest eigenvalue λ₃"
            },
            "lambda3_max": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "label": "λ₃ Maximum",
                "description": "Maximum value for the largest eigenvalue λ₃"
            },
            # Parameters for geometric feature filtering
            "planarity_min": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "label": "Minimum Planarity",
                "description": "Minimum value for planarity (λ₂ - λ₁)/λ₃"
            },
            "planarity_max": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "label": "Maximum Planarity",
                "description": "Maximum value for planarity"
            },
            "linearity_min": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "label": "Minimum Linearity",
                "description": "Minimum value for linearity (λ₃ - λ₂)/λ₃"
            },
            "linearity_max": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "label": "Maximum Linearity",
                "description": "Maximum value for linearity"
            },
            "sphericity_min": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "label": "Minimum Sphericity",
                "description": "Minimum value for sphericity λ₁/λ₃"
            },
            "sphericity_max": {
                "type": "float",
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "label": "Maximum Sphericity",
                "description": "Maximum value for sphericity"
            },
            # Common parameters
            "normalize_eigenvalues": {
                "type": "bool",
                "default": True,
                "label": "Normalize Eigenvalues",
                "description": "Normalize eigenvalues by their sum before filtering"
            },
            "combination_logic": {
                "type": "choice",
                "default": "AND",
                "choices": ["AND", "OR"],
                "label": "Combination Logic",
                "description": "How to combine multiple conditions (AND: all must be true, OR: at least one must be true)"
            },
            "invert_filter": {
                "type": "bool",
                "default": False,
                "label": "Invert Filter",
                "description": "Invert the filter results (select points that DON'T match criteria)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute eigenvalue filtering based on the selected criteria.

        Args:
            data_node (DataNode): The data node containing eigenvalues
            params (Dict[str, Any]): Parameters for eigenvalue filtering

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the filtering results
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        # Get eigenvalues from the data node
        eigenvalue_array = self._get_eigenvalues_from_node(data_node)

        # Extract parameters
        filter_type = params["filter_type"]
        normalize = params["normalize_eigenvalues"]
        combination_logic = params["combination_logic"]
        invert_filter = params["invert_filter"]

        # Normalize eigenvalues if requested
        if normalize:
            # Add a small epsilon to avoid division by zero
            eigenvalue_sums = np.sum(eigenvalue_array, axis=1, keepdims=True)
            eigenvalue_sums = np.maximum(eigenvalue_sums, 1e-10)
            normalized_eigenvalues = eigenvalue_array / eigenvalue_sums
        else:
            normalized_eigenvalues = eigenvalue_array

        # Apply the appropriate filter based on filter_type
        if filter_type == "predefined":
            predefined_filter = params["predefined_filter"]
            mask = self._apply_predefined_filter(normalized_eigenvalues, predefined_filter)

        elif filter_type == "manual":
            # Extract manual threshold parameters
            lambda1_min = params["lambda1_min"]
            lambda1_max = params["lambda1_max"]
            lambda2_min = params["lambda2_min"]
            lambda2_max = params["lambda2_max"]
            lambda3_min = params["lambda3_min"]
            lambda3_max = params["lambda3_max"]

            mask = self._apply_manual_filter(
                normalized_eigenvalues,
                lambda1_min, lambda1_max,
                lambda2_min, lambda2_max,
                lambda3_min, lambda3_max,
                combination_logic
            )

        elif filter_type == "geometric_features":
            # Extract geometric feature parameters
            planarity_min = params["planarity_min"]
            planarity_max = params["planarity_max"]
            linearity_min = params["linearity_min"]
            linearity_max = params["linearity_max"]
            sphericity_min = params["sphericity_min"]
            sphericity_max = params["sphericity_max"]

            mask = self._apply_geometric_feature_filter(
                normalized_eigenvalues,
                planarity_min, planarity_max,
                linearity_min, linearity_max,
                sphericity_min, sphericity_max,
                combination_logic
            )

        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        # Invert the mask if requested
        if invert_filter:
            mask = ~mask

        # Count the number of points that match the filter
        match_count = np.sum(mask)
        total_count = len(mask)
        match_percentage = (match_count / total_count) * 100 if total_count > 0 else 0

        print(f"Filter results: {match_count} of {total_count} points match criteria ({match_percentage:.2f}%)")

        # Create a Masks object with the result
        masks_result = Masks(mask)

        # Return the masks, result type, and dependencies
        dependencies = [data_node.uid]
        return masks_result, "masks", dependencies

    def _get_eigenvalues_from_node(self, data_node: DataNode) -> np.ndarray:
        """
        Extract eigenvalue data from a data node.

        Args:
            data_node (DataNode): The data node to extract eigenvalues from

        Returns:
            np.ndarray: The eigenvalue array with shape (n_points, 3)

        Raises:
            ValueError: If eigenvalues cannot be found in the data node
        """
        # Print diagnostic information
        print(f"Extracting eigenvalues from node. Type: {data_node.data_type}")

        # Try multiple approaches to get eigenvalues
        eigenvalue_array = None

        # Check for eigenvalues attribute
        if hasattr(data_node.data, 'eigenvalues') and isinstance(data_node.data.eigenvalues, np.ndarray):
            eigenvalue_array = data_node.data.eigenvalues
            print(f"Found eigenvalues through object attribute")

        # Check data type string
        elif data_node.data_type == "eigenvalues":
            eigenvalue_array = data_node.data.eigenvalues
            print(f"Found eigenvalues through data type")

        # Check attributes dictionary
        elif hasattr(data_node.data, 'attributes') and 'eigenvalues' in data_node.data.attributes:
            eigenvalue_array = data_node.data.attributes['eigenvalues']
            print(f"Found eigenvalues in node attributes")

        # Check if this is already an array with the right shape
        elif isinstance(data_node.data, np.ndarray) and len(data_node.data.shape) == 2 and data_node.data.shape[1] == 3:
            eigenvalue_array = data_node.data
            print(f"Found data that looks like eigenvalues (shape {data_node.data.shape})")

        else:
            # Include helpful diagnostic info in the error
            print(f"Failed to find eigenvalues. Node type: {data_node.data_type}")
            print(f"Data object type: {type(data_node.data)}")
            if hasattr(data_node.data, '__dict__'):
                print(f"Data attributes: {list(data_node.data.__dict__.keys())}")
            raise ValueError("This plugin requires eigenvalue data. Please select an eigenvalue data node.")

        # Validate the eigenvalue array
        if eigenvalue_array is None or not isinstance(eigenvalue_array, np.ndarray):
            raise ValueError("Could not extract valid eigenvalue array from data node")

        if len(eigenvalue_array.shape) != 2 or eigenvalue_array.shape[1] != 3:
            raise ValueError(f"Eigenvalue array has incorrect shape: {eigenvalue_array.shape}, expected (n_points, 3)")

        return eigenvalue_array

    def _apply_predefined_filter(self, eigenvalues: np.ndarray, filter_name: str) -> np.ndarray:
        """
        Apply a predefined filter to eigenvalues.

        Args:
            eigenvalues (np.ndarray): Eigenvalue array with shape (n_points, 3)
            filter_name (str): Name of the predefined filter to apply

        Returns:
            np.ndarray: Boolean mask of matching points
        """
        # Calculate geometric features
        utils = EigenvalueUtils()
        features = utils.compute_geometric_features(eigenvalues)

        planarity = features['planarity']
        linearity = features['linearity']
        sphericity = features['sphericity']

        if filter_name == "planes":
            # Planar surfaces: high planarity, low linearity, low sphericity
            mask = (planarity > 0.6) & (linearity < 0.3) & (sphericity < 0.3)
            print(f"Applied planes filter: {np.sum(mask)} points match")

        elif filter_name == "edges":
            # Edges: high linearity, low planarity, low sphericity
            mask = (linearity > 0.5) & (planarity < 0.3) & (sphericity < 0.3)
            print(f"Applied edges filter: {np.sum(mask)} points match")

        elif filter_name == "corners":
            # Corners/intersections: high sphericity, low planarity, low linearity
            mask = (sphericity > 0.2) & (planarity < 0.4) & (linearity < 0.4)
            print(f"Applied corners filter: {np.sum(mask)} points match")

        elif filter_name == "linear_features":
            # Linear features (pipes, wires): very high linearity
            mask = (linearity > 0.7)
            print(f"Applied linear_features filter: {np.sum(mask)} points match")

        elif filter_name == "noise":
            # Noise points: roughly equal eigenvalues (all features low)
            ratios = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 2], 1e-10)
            mask = (ratios > 0.5) & (planarity < 0.3) & (linearity < 0.3)
            print(f"Applied noise filter: {np.sum(mask)} points match")

        else:
            raise ValueError(f"Unknown predefined filter: {filter_name}")

        return mask

    def _apply_manual_filter(
            self,
            eigenvalues: np.ndarray,
            lambda1_min: float, lambda1_max: float,
            lambda2_min: float, lambda2_max: float,
            lambda3_min: float, lambda3_max: float,
            combination_logic: str
    ) -> np.ndarray:
        """
        Apply manual thresholds to filter eigenvalues.

        Args:
            eigenvalues (np.ndarray): Eigenvalue array with shape (n_points, 3)
            lambda1_min (float): Minimum value for λ₁
            lambda1_max (float): Maximum value for λ₁
            lambda2_min (float): Minimum value for λ₂
            lambda2_max (float): Maximum value for λ₂
            lambda3_min (float): Minimum value for λ₃
            lambda3_max (float): Maximum value for λ₃
            combination_logic (str): "AND" or "OR" for combining conditions

        Returns:
            np.ndarray: Boolean mask of matching points
        """
        # Create individual masks for each eigenvalue
        lambda1_mask = (eigenvalues[:, 0] >= lambda1_min) & (eigenvalues[:, 0] <= lambda1_max)
        lambda2_mask = (eigenvalues[:, 1] >= lambda2_min) & (eigenvalues[:, 1] <= lambda2_max)
        lambda3_mask = (eigenvalues[:, 2] >= lambda3_min) & (eigenvalues[:, 2] <= lambda3_max)

        # Combine masks according to the specified logic
        if combination_logic == "AND":
            final_mask = lambda1_mask & lambda2_mask & lambda3_mask
        else:  # OR
            final_mask = lambda1_mask | lambda2_mask | lambda3_mask

        # Print filter results for each eigenvalue
        print(f"λ₁ filter: {np.sum(lambda1_mask)} points match range [{lambda1_min}, {lambda1_max}]")
        print(f"λ₂ filter: {np.sum(lambda2_mask)} points match range [{lambda2_min}, {lambda2_max}]")
        print(f"λ₃ filter: {np.sum(lambda3_mask)} points match range [{lambda3_min}, {lambda3_max}]")
        print(f"Combined with {combination_logic} logic: {np.sum(final_mask)} points match")

        return final_mask

    def _apply_geometric_feature_filter(
            self,
            eigenvalues: np.ndarray,
            planarity_min: float, planarity_max: float,
            linearity_min: float, linearity_max: float,
            sphericity_min: float, sphericity_max: float,
            combination_logic: str
    ) -> np.ndarray:
        """
        Apply filters based on geometric features derived from eigenvalues.

        Args:
            eigenvalues (np.ndarray): Eigenvalue array with shape (n_points, 3)
            planarity_min (float): Minimum value for planarity
            planarity_max (float): Maximum value for planarity
            linearity_min (float): Minimum value for linearity
            linearity_max (float): Maximum value for linearity
            sphericity_min (float): Minimum value for sphericity
            sphericity_max (float): Maximum value for sphericity
            combination_logic (str): "AND" or "OR" for combining conditions

        Returns:
            np.ndarray: Boolean mask of matching points
        """
        # Calculate geometric features
        utils = EigenvalueUtils()
        features = utils.compute_geometric_features(eigenvalues)

        # Create individual masks for each feature
        planarity_mask = (features['planarity'] >= planarity_min) & (features['planarity'] <= planarity_max)
        linearity_mask = (features['linearity'] >= linearity_min) & (features['linearity'] <= linearity_max)
        sphericity_mask = (features['sphericity'] >= sphericity_min) & (features['sphericity'] <= sphericity_max)

        # Combine masks according to the specified logic
        if combination_logic == "AND":
            final_mask = planarity_mask & linearity_mask & sphericity_mask
        else:  # OR
            final_mask = planarity_mask | linearity_mask | sphericity_mask

        # Print filter results for each feature
        print(f"Planarity filter: {np.sum(planarity_mask)} points match range [{planarity_min}, {planarity_max}]")
        print(f"Linearity filter: {np.sum(linearity_mask)} points match range [{linearity_min}, {linearity_max}]")
        print(f"Sphericity filter: {np.sum(sphericity_mask)} points match range [{sphericity_min}, {sphericity_max}]")
        print(f"Combined with {combination_logic} logic: {np.sum(final_mask)} points match")

        return final_mask