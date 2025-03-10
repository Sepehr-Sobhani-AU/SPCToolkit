# plugins/analysis/eigenvalue_analysis_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.eigenvalues import Eigenvalues
from core.masks import Masks


class EigenvalueAnalysisPlugin(AnalysisPlugin):
    """
    Plugin for analyzing eigenvalues and identifying geometric features.

    This plugin takes previously calculated eigenvalues and performs geometric
    analysis to identify specific features like planar regions, edges, and corners.
    It can generate masks for these features as well as provide distributions
    of various geometric properties.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "eigenvalue_analysis"
        """
        return "eigenvalue_analysis"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for eigenvalue analysis.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "analysis_type": {
                "type": "choice",
                "default": "feature_segmentation",
                "choices": [
                    "feature_segmentation",
                    "planarity_threshold",
                    "linearity_threshold",
                    "sphericity_threshold",
                    "anisotropy_threshold"
                ],
                "label": "Analysis Type",
                "description": "Type of analysis to perform on eigenvalues"
            },
            "threshold": {
                "type": "float",
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "label": "Threshold",
                "description": "Threshold value for feature detection (0.0-1.0)"
            },
            "planar_threshold": {
                "type": "float",
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "label": "Planar Threshold",
                "description": "Threshold for detecting planar regions (used in feature segmentation)"
            },
            "linear_threshold": {
                "type": "float",
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "label": "Linear Threshold",
                "description": "Threshold for detecting linear features (used in feature segmentation)"
            },
            "spherical_threshold": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "label": "Spherical Threshold",
                "description": "Threshold for detecting corners/spherical regions (used in feature segmentation)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute eigenvalue analysis on the eigenvalue data.

        Args:
            data_node (DataNode): The data node containing eigenvalues
            params (Dict[str, Any]): Parameters for eigenvalue analysis

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing results of the analysis
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        # Check that we have eigenvalue data
        if data_node.data_type != "eigenvalues":
            print(f"Data type: {data_node.data_type}")
            raise ValueError("This plugin requires eigenvalue data. Please select an eigenvalue data node.")

        # Extract parameters
        analysis_type = params["analysis_type"]
        threshold = params["threshold"]
        planar_threshold = params["planar_threshold"]
        linear_threshold = params["linear_threshold"]
        spherical_threshold = params["spherical_threshold"]

        # Get eigenvalues from data node
        eigenvalue_obj = data_node.data
        eigenvalues = eigenvalue_obj.eigenvalues

        print(f"Analyzing {len(eigenvalues)} eigenvalue sets...")

        # Calculate geometric features
        planarity = self._compute_planarity(eigenvalues)
        linearity = self._compute_linearity(eigenvalues)
        sphericity = self._compute_sphericity(eigenvalues)
        anisotropy = self._compute_anisotropy(eigenvalues)

        # Print statistics on features
        print(f"Planarity: min={np.min(planarity):.3f}, max={np.max(planarity):.3f}, mean={np.mean(planarity):.3f}")
        print(f"Linearity: min={np.min(linearity):.3f}, max={np.max(linearity):.3f}, mean={np.mean(linearity):.3f}")
        print(f"Sphericity: min={np.min(sphericity):.3f}, max={np.max(sphericity):.3f}, mean={np.mean(sphericity):.3f}")
        print(f"Anisotropy: min={np.min(anisotropy):.3f}, max={np.max(anisotropy):.3f}, mean={np.mean(anisotropy):.3f}")

        # Calculate percentages of different feature types
        high_planarity = np.sum(planarity > 0.7) / len(planarity) * 100
        high_linearity = np.sum(linearity > 0.7) / len(linearity) * 100
        high_sphericity = np.sum(sphericity > 0.3) / len(sphericity) * 100

        print(f"Points with high planarity (>0.7): {high_planarity:.1f}%")
        print(f"Points with high linearity (>0.7): {high_linearity:.1f}%")
        print(f"Points with high sphericity (>0.3): {high_sphericity:.1f}%")

        # Create masks based on the selected analysis type
        if analysis_type == "feature_segmentation":
            # Multi-class segmentation based on dominant feature
            is_planar = np.logical_and(planarity > planar_threshold,
                                       planarity > linearity,
                                       planarity > sphericity * 2)

            is_linear = np.logical_and(linearity > linear_threshold,
                                       linearity > planarity,
                                       linearity > sphericity * 2)

            is_spherical = np.logical_and(sphericity > spherical_threshold,
                                          sphericity * 2 > planarity,
                                          sphericity * 2 > linearity)

            # Create a composite mask (1=planar, 2=linear, 3=spherical, 0=unclassified)
            result_mask = np.zeros(len(eigenvalues), dtype=np.bool)

            # We use the classification with the highest confidence
            classification = np.zeros(len(eigenvalues), dtype=np.int32)
            classification[is_planar] = 1
            classification[is_linear] = 2
            classification[is_spherical] = 3

            # Count points in each category
            planar_count = np.sum(classification == 1)
            linear_count = np.sum(classification == 2)
            spherical_count = np.sum(classification == 3)
            unclassified_count = np.sum(classification == 0)

            print(f"Classification results:")
            print(f"  Planar regions: {planar_count} points ({planar_count / len(eigenvalues) * 100:.1f}%)")
            print(f"  Linear features: {linear_count} points ({linear_count / len(eigenvalues) * 100:.1f}%)")
            print(
                f"  Spherical/corner regions: {spherical_count} points ({spherical_count / len(eigenvalues) * 100:.1f}%)")
            print(f"  Unclassified: {unclassified_count} points ({unclassified_count / len(eigenvalues) * 100:.1f}%)")

            # For the mask, we'll use one of the classifications (planar by default)
            # The visualization can be changed later
            result_mask = (classification == 1)

        elif analysis_type == "planarity_threshold":
            result_mask = planarity > threshold
            count = np.sum(result_mask)
            print(f"Found {count} points ({count / len(eigenvalues) * 100:.1f}%) with planarity > {threshold}")

        elif analysis_type == "linearity_threshold":
            result_mask = linearity > threshold
            count = np.sum(result_mask)
            print(f"Found {count} points ({count / len(eigenvalues) * 100:.1f}%) with linearity > {threshold}")

        elif analysis_type == "sphericity_threshold":
            result_mask = sphericity > threshold
            count = np.sum(result_mask)
            print(f"Found {count} points ({count / len(eigenvalues) * 100:.1f}%) with sphericity > {threshold}")

        elif analysis_type == "anisotropy_threshold":
            result_mask = anisotropy > threshold
            count = np.sum(result_mask)
            print(f"Found {count} points ({count / len(eigenvalues) * 100:.1f}%) with anisotropy > {threshold}")

        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        # Create a Masks object with the result
        masks = Masks(result_mask)

        # Return the results
        dependencies = [data_node.uid]
        return masks, "masks", dependencies

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