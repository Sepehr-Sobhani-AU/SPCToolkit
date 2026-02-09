# plugins/Points/Analysis/compute_eigenvalues_plugin.py
"""
Plugin for computing eigenvalues of point clouds for geometric feature analysis.

This plugin computes eigenvalues from local covariance matrices at each point,
which describe the geometric structure of the local neighborhood. Eigenvalues
are useful for:
- Distinguishing between planar, linear, and volumetric structures
- Feature extraction for machine learning
- Geometric classification of points
"""

from typing import Dict, Any, List, Tuple

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.eigenvalues import Eigenvalues


class ComputeEigenvaluesPlugin(Plugin):
    """
    Plugin for computing eigenvalues from local point neighborhoods.

    Eigenvalues are computed from the covariance matrix of k-nearest neighbors
    around each point. The three eigenvalues (λ1 >= λ2 >= λ3) describe the
    geometric structure:
    - Large λ1, small λ2, λ3: Linear structure (edges, cables)
    - Large λ1, λ2, small λ3: Planar structure (walls, ground)
    - Similar λ1, λ2, λ3: Volumetric structure (vegetation)
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "compute_eigenvalues"
        """
        return "compute_eigenvalues"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for eigenvalue computation.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "k": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "label": "Number of Neighbors (k)",
                "description": "Number of nearest neighbors to use for local covariance calculation"
            },
            "smooth": {
                "type": "bool",
                "default": True,
                "label": "Smooth Eigenvalues",
                "description": "Apply neighborhood averaging to reduce noise in eigenvalue estimates"
            },
            "target_batch_size": {
                "type": "int",
                "default": 250000,
                "min": 10000,
                "max": 1000000,
                "label": "Batch Size",
                "description": "Number of points per spatial batch. Smaller values use less memory."
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute eigenvalue computation on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for eigenvalue computation (k and smooth)

        Returns:
            Tuple[Eigenvalues, str, List]:
                - Eigenvalues object containing (n_points, 3) array
                - Result type identifier "eigenvalues"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Extract parameters
        k = params["k"]
        smooth = params["smooth"]
        batch_size = params.get("target_batch_size", 250000)

        # Compute eigenvalues using the point cloud's built-in method
        # This method leverages EigenvalueUtils for efficient computation
        eigenvalues_array = point_cloud.get_eigenvalues(k=k, smooth=smooth, batch_size=batch_size)

        # Wrap in Eigenvalues object
        eigenvalues = Eigenvalues(eigenvalues_array)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return eigenvalues, "eigenvalues", dependencies
