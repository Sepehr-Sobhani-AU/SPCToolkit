# plugins/analysis/eigenvalue_calculation_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np
import time

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud
from core.eigenvalues import Eigenvalues
from services.eigenvalue_utils import EigenvalueUtils


class EigenvalueCalculationPlugin(AnalysisPlugin):
    """
    Plugin for calculating eigenvalues of local neighborhoods in a point cloud.

    This plugin computes eigenvalues of the covariance matrix for k-nearest neighbors
    of each point in a point cloud and returns an Eigenvalues object.
    """

    def get_name(self) -> str:
        """
        Return the name of this plugin.

        Returns:
            str: The unique name "eigenvalue_calculation"
        """
        return "eigenvalue_calculation"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters needed for eigenvalue calculation.

        Returns:
            Dict[str, Any]: Parameter schema with types, defaults, and UI hints
        """
        return {
            "k_neighbors": {
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 100,
                "label": "Number of Neighbours",
                "description": "Number of nearest neighbours to use for eigenvalue computation"
            },
            "smooth": {
                "type": "bool",
                "default": False,
                "label": "Smooth Results",
                "description": "Whether to smooth eigenvalues by averaging over neighbours"
            },
            "batch_size": {
                "type": "int",
                "default": 100000,
                "min": 100000,
                "max": 10000000,
                "label": "Batch Size",
                "description": "Number of points to process at once (lower values use less memory)"
            },
            "force_cpu": {
                "type": "bool",
                "default": False,
                "label": "Force CPU Usage",
                "description": "Use CPU instead of GPU to avoid memory issues with large point clouds"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute eigenvalue calculation on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for eigenvalue calculation

        Returns:
            Tuple[Eigenvalues, str, List]:
                - Eigenvalues object containing the computed eigenvalues
                - Result type identifier "eigenvalues"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        # Extract parameters
        k_neighbors = params.get("k_neighbors", 20)
        smooth = params.get("smooth", False)
        batch_size = params.get("batch_size", 10000)
        force_cpu = params.get("force_cpu", False)

        # For very small point clouds, don't bother with batching
        if point_cloud.size < batch_size:
            batch_size = point_cloud.size

        # Define a progress callback
        def progress_callback(current, total):
            percent = (current / total) * 100
            print(f"Computing eigenvalues: {percent:.1f}% complete ({current}/{total} batches)")

        # Create an instance of EigenvalueAnalyser with the specified device preference
        eigenvalue_utils = EigenvalueUtils()

        print(f"Computing eigenvalues for {point_cloud.size} points with k={k_neighbors}...")
        start_time = time.time()

        # Compute eigenvalues using the eigenvalue_utils
        try:
            # Get eigenvalues as a numpy array
            eigenvalue_array = eigenvalue_utils.get_eigenvalues(point_cloud.points, k=k_neighbors, smooth=smooth, batch_size=batch_size)

            # Create an Eigenvalues object
            eigenvalues = Eigenvalues(eigenvalue_array)

            elapsed_time = time.time() - start_time
            print(f"Eigenvalue calculation complete for {point_cloud.size} points in {elapsed_time:.2f} seconds")
            print(f"Eigenvalue array shape: {eigenvalue_array.shape}")

            # Return results, type, and dependencies
            dependencies = [data_node.uid]
            return eigenvalues, "eigenvalues", dependencies

        except Exception as e:
            print(f"Error during eigenvalue calculation: {e}")
            # If memory error occurs, suggest reducing batch size
            if "OOM" in str(e) or "memory" in str(e).lower():
                print("Memory error detected. Try reducing the batch size or enabling 'Force CPU Usage'")
            raise