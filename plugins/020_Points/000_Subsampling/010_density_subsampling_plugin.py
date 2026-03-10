# plugins/analysis/density_subsampling_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.masks import Masks


class DensitySubsamplingPlugin(Plugin):
    """
    Plugin for density-based subsampling of point clouds using Open3D.

    This plugin implements voxel downsampling, which divides the space into
    voxels (small cubes) and replaces all points within each voxel with their centroid,
    resulting in a more uniform point density throughout the point cloud while preserving
    the overall shape and features.
    """

    def get_name(self) -> str:
        """
        Return the unique name for this plugin.

        Returns:
            str: The name "density_subsampling"
        """
        return "density_subsampling"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Define the parameters for density-based subsampling.

        Returns:
            Dict[str, Any]: Parameter schema for the dialog box
        """
        return {
            "voxel_size": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 10.0,
                "label": "Voxel Size",
                "description": "Size of voxels for downsampling (smaller values preserve more detail)"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute density-based subsampling on the point cloud.

        Args:
            data_node (DataNode): The data node containing the point cloud
            params (Dict[str, Any]): Parameters for subsampling (voxel_size)

        Returns:
            Tuple[Masks, str, List]:
                - Masks object containing the subsampling mask
                - Result type identifier "masks"
                - List containing the data_node's UID as a dependency
        """
        # Extract the point cloud from the data node
        point_cloud: PointCloud = data_node.data

        from config.config import global_variables

        # Get the voxel size parameter
        voxel_size = params["voxel_size"]

        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D is not installed. Please install it with: pip install open3d")

        # Create an Open3D point cloud from the NumPy points
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud.points)

        # Perform voxel downsampling
        global_variables.global_progress = (None, f"Voxel downsampling ({len(point_cloud.points):,} points, voxel={voxel_size})...")

        downsampled_point_cloud = o3d_point_cloud.voxel_down_sample(voxel_size=voxel_size)
        downsampled_points = np.asarray(downsampled_point_cloud.points)

        # Now we need to create a mask that identifies which original points are kept
        # Since voxel downsampling creates new points (centroids), we need to find the closest
        # original point to each centroid
        from scipy.spatial import cKDTree

        global_variables.global_progress = (60, f"Mapping {len(downsampled_points):,} downsampled points to originals...")

        # Build a KD-tree from the original points
        tree = cKDTree(point_cloud.points)

        # For each downsampled point, find the closest original point
        _, indices = tree.query(downsampled_points)

        # Create a boolean mask where True indicates a point to keep
        mask = np.zeros(len(point_cloud.points), dtype=bool)
        mask[indices] = True

        # Create a Masks object with the result
        result_mask = Masks(mask)

        # Return results, type, and dependencies
        dependencies = [data_node.uid]
        return result_mask, "masks", dependencies