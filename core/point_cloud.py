# Standard library imports
import copy
import time
import math
import os
import subprocess
import sys
import warnings
# from importlib.metadata import version, PackageNotFoundError

# Third-party imports
import numpy as np
import pandas as pd
import open3d as o3d
import tensorflow as tf
from scipy.stats import norm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# Local application imports
from services.eigenvalue_utils import EigenvalueUtils


class PointCloud:
    """
    Initialize a Clusters object.

    Required Parameters:
    - points (np.ndarray): A numpy array (n, 3) of points in the format [[x1, y1, z1],
                                                                         [x2, y2, z2],
                                                                               .
                                                                               .
                                                                               .
                                                                         [xn, yn, zn]]

    Optional Keyword Arguments:
    - parent_uuid (str): The uid of the parent cluster.
    - child_uuid (str): The uid of the child cluster.
    - color (np.ndarray): An array (n, 3) of color values (R, G, B) for each point.
    - normal (np.ndarray): An array (n, 3) of normal unit vector values (Nx, Ny, Nz) for each point.
    - intensity (np.ndarray): An array (n) of intensity values (0-255) for each point.
    - distToGround (np.ndarray): An array (n) of distance to ground values (float) for each point.
    - feature (list): A list containing [predicted_feature (int or str),
                                         prediction_probability (0-1),
                                         model_path (str)].
    - metadata (dict): Any additional metadata related to the cluster.
                       Defaults to an empty dictionary if not provided.

    """

    # In point_cloud.py, update the __init__ method (around line 50-90)

    def __init__(self, points, colors=None, normals=None, **kwargs):

        # Validation points
        if not isinstance(points, np.ndarray):
            raise ValueError("Points must be a numpy array with shape (n, 3)")

        self.points = points
        self.colors = colors
        self.normals = normals
        self.translation = np.array([0, 0, 0])
        self.name = kwargs.get('params', None)
        self.uuid = kwargs.get('uid', None)
        self.parent_uuid = kwargs.get('parent_uuid', None)
        self.child_uuid = kwargs.get('child_uuid', None)

        self.cluster_labels = []
        self.prediction = ''
        self.probability = 0
        self.model_weight = ''
        self.center = [0, 0, 0]
        self.length = 0
        self.width = 0
        self.height = 0

        # If cluster has no point in it, ignore calculating obb
        if self.size > 3:
            # Check if points are coplanar (all have the same Z coordinate)
            # If so, skip OBB calculation as it will fail
            if not self._are_points_coplanar():
                self._update_obb_dim()
            else:
                # For coplanar points, calculate 2D bounding box manually
                self._update_2d_bbox()
        else:
            return

        # ... rest of the __init__ code ...

        # Validation for color, intensity, normal and distToGround
        for attr_name in ['intensity', 'distToGround']:
            attr_value = kwargs.get(attr_name, np.array([]))
            if not isinstance(attr_value, np.ndarray) or (len(attr_value) != 0 and len(attr_value) != len(points)):
                raise ValueError(f"{attr_name} must be a numpy array of the same length as points")
            setattr(self, attr_name, attr_value)

        self.feature = kwargs.get('feature', [])
        self.metadata = kwargs.get('metadata', {})

        # Add a dictionary to store arbitrary attributes
        self.attributes = {}

    @property
    def size(self):
        """
        Returns the number of points in the point cloud.

        This property provides access to the total number of points contained in the
        point cloud, which is equivalent to the length of the points array.

        Returns:
            int: The total number of points in the point cloud.
        """
        return len(self.points)

    # This code should replace the existing add_attribute method in the PointCloud class
    def add_attribute(self, name, values):
        """Add or update a per-point attribute.

        Args:
            name (str): Name of the attribute
            values (np.ndarray): Array of values associated with points.
                The first dimension must match the number of points.
                For example:
                - 1D array: One scalar value per point
                - 2D array: A vector of values for each point
                - 3D array: A matrix of values for each point

        Raises:
            ValueError: If the first dimension of values doesn't match the number of points
            TypeError: If values is not a numpy array
        """
        import numpy as np

        # Convert to numpy array if possible
        if not isinstance(values, np.ndarray):
            try:
                values = np.array(values)
            except:
                raise TypeError(f"Attribute '{name}' must be a numpy array or convertible to one")

        # Handle multi-dimensional arrays
        if values.ndim == 1:
            # 1D array case - must have length equal to number of points
            if len(values) != len(self.points):
                raise ValueError(
                    f"Attribute '{name}' has length {len(values)} but point cloud has {len(self.points)} points"
                )
        else:
            # Multi-dimensional case - first dimension must match number of points
            if values.shape[0] != len(self.points):
                raise ValueError(
                    f"Attribute '{name}' has {values.shape[0]} elements in first dimension but point cloud has {len(self.points)} points"
                )

        # Store the attribute
        self.attributes[name] = values

        print(f"Added attribute '{name}' with shape {values.shape} to point cloud")

    def get_attribute(self, name):
        """Get a per-point attribute by name."""
        return self.attributes.get(name)

    def translate(self, translation):
        """
        Translate the cluster by a given translation vector.

        Parameters:
        - translation (np.ndarray): A 1D numpy array representing the translation vector [dx, dy, dz].
        """
        self.points += translation
        self.translation += translation

    def augment_points(self, scale=0.005, augmentation_factor=2):
        """
        Augment the points of the cluster in place by duplicating and jittering them.
        This method also updates the color, intensity, and distToGround arrays.

        Parameters:
        - scale (float): The scale of the noise to add during jittering. Default is 0.005.
        - augmentation_factor (int): The factor by which to augment the points. Default is 2.
        """
        if len(self.points) == 0:
            raise ValueError("The cluster has no points to augment.")

        # Generating augmented points
        augmented_points = np.repeat(self.points, augmentation_factor, axis=0)

        # Jitter the augmented points
        noise = np.random.normal(0, scale, augmented_points.shape)
        augmented_points += noise

        # Update the cluster's points
        self.points = augmented_points

        # Update corresponding attributes: colors, intensity, and distToGround
        if len(self.colors) > 0:
            self.colors = np.repeat(self.colors, augmentation_factor, axis=0)

        if len(self.intensity) > 0:
            self.intensity = np.repeat(self.intensity, augmentation_factor)

        if len(self.distToGround) > 0:
            # Repeat and add noise to the Z-axis component
            augmented_distToGround = np.repeat(self.distToGround, augmentation_factor)
            augmented_distToGround += noise[:, 2]
            self.distToGround = augmented_distToGround

    def dbscan(self, eps=0.05, min_points=10, return_clusters_object=False, use_sklearn=False):
        """
        Apply DBSCAN clustering to the points in the cluster.

        Parameters:
        - eps (float): The maximum distance between two samples for one to be considered
                       as in the neighborhood of the other. Default is 0.05.
        - min_points (int): The number of samples in a neighborhood for a point to be considered
                            as a core point. Default is 10.
        - return_clusters_object (bool): If True, returns a Clusters object containing the DBSCAN result.
                                         Default is False.
        - use_sklearn (bool): If True, uses scikit-learn's DBSCAN implementation, which may be
                             more efficient for large point clouds. Default is False.

        Returns:
        - labels (np.ndarray) or (np.ndarray, Clusters): An array of cluster labels, and optionally a Clusters object.
        """
        if len(self.points) == 0:
            raise ValueError("The cluster has no points for DBSCAN clustering.")

        if use_sklearn:
            try:
                from sklearn.cluster import DBSCAN
                import sklearn

                print(f"Using scikit-learn DBSCAN (version {sklearn.__version__})")
                print(f"Processing {len(self.points)} points with eps={eps}, min_samples={min_points}")

                # Create and run the DBSCAN algorithm
                db = DBSCAN(eps=eps, min_samples=min_points, n_jobs=-1)  # n_jobs=-1 uses all available cores
                labels = db.fit_predict(self.points)

                print(f"DBSCAN completed. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
            except ImportError:
                print("scikit-learn not found. Falling back to Open3D implementation.")
                return self._dbscan_open3d(eps, min_points, return_clusters_object)
        else:
            # Use the original Open3D implementation
            labels = self._dbscan_open3d(eps, min_points, return_clusters_object)

        # Store the DBSCAN labels
        self.cluster_labels = labels

        if return_clusters_object:
            # Create and return Clusters object
            from core.clusters import Clusters
            clusters = Clusters(labels)
            # Optionally set random colors for visualization
            clusters.set_random_color()
            return clusters
        else:
            return labels

    def _dbscan_open3d(self, eps, min_points, return_clusters_object=False):
        """
        Internal method to perform DBSCAN using Open3D implementation.

        Parameters:
        - eps (float): Epsilon parameter for DBSCAN.
        - min_points (int): Minimum points parameter for DBSCAN.
        - return_clusters_object (bool): Not used in this method, kept for API consistency.

        Returns:
        - labels (np.ndarray): An array of cluster labels.
        """
        print(f"Using Open3D DBSCAN")
        print(f"Processing {len(self.points)} points with eps={eps}, min_points={min_points}")

        # Convert points to Open3D point cloud and perform DBSCAN
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # Use VerbosityContextManager to control console output
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

        print(f"DBSCAN completed. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return labels

    def density_downsample(self, voxel_size):
        """
        A density based downsampling of the cluster. It preserves
        additional attributes (like colors or normals).

        Parameters:
        - voxel_size (float): The voxel size to determine the new point cloud density.

        Returns:
        - A Clusters object created from the downsampled point cloud with original attributes of the point.
        """

        # Create a point cloud object from the cluster's points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # Voxel downsampling with trace
        _, trace, _ = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
        trace = trace.flatten()

        # Initialize the mask array
        mask = np.full((len(pcd.points)), False, dtype=bool)
        mask[trace] = True

        return self.get_subset(mask)

    # def get_clusters(self, min_points=200):
    #     """
    #     Generate a Clusters object containing individual cluster_labels from DBSCAN labels,
    #     excluding cluster_labels with fewer than min_points.
    #
    #     Args:
    #         min_points (int, optional): Minimum number of points for a cluster to be included. Default is 200.
    #
    #     Returns:
    #         Clusters: An object containing all the valid cluster_labels (excluding noise and smaller cluster_labels).
    #     """
    #
    #     if not hasattr(self, 'cluster_labels'):
    #         raise ValueError("DBSCAN must be applied before generating 'Clusters' object.")
    #
    #     # Filter out noise points (usually labeled as -1)
    #     unique_labels = set(self.cluster_labels)
    #     unique_labels.discard(-1)
    #
    #     # Create a Clusters object to hold the cluster_labels
    #     cluster_labels = Clusters()
    #
    #     # Filter labels to include only those with enough points
    #     valid_labels = [labels for labels in unique_labels if np.sum(self.cluster_labels == labels) >= min_points]
    #
    #     for labels in valid_labels:
    #         # Create mask for points belonging to the current labels
    #         mask = self.cluster_labels == labels
    #
    #         # Create a copy of cluster
    #         child_cluster = copy.deepcopy(self)
    #         child_cluster.points = self.points[mask]
    #         # Add the new cluster to cluster_labels instance
    #         cluster_labels.add(child_cluster)
    #
    #     return cluster_labels

    def get_eigenvalues(self, k, smooth=True, batch_size=None):
        """
        Calculate eigenvalues using the EigenvalueAnalyser.

        This method maintains backward compatibility while leveraging
        the improved implementation in EigenvalueAnalyser.

        Args:
            k (int): Number of neighbors
            smooth (bool): Whether to smooth eigenvalues
            batch_size (int, optional): Batch size for processing large clouds

        Returns:
            np.ndarray: Eigenvalues for each point
        """

        # Create analyser instance (using CPU to avoid memory issues)
        analyser = EigenvalueUtils()

        # Use the analyser to compute eigenvalues
        return analyser.get_eigenvalues(self.points, k=k, smooth=smooth, batch_size=batch_size)

    # def get_subset(self, mask, inplace=False):
    #     """
    #     Extracts a subset based on a mask, either by modifying the current cluster or returning a new one.
    #
    #     Parameters:
    #     - mask (np.ndarray): A boolean array where True values indicate points to include in the subset.
    #     - inplace (bool): If True, modifies the current cluster in-place. Default is False.
    #
    #     Returns:
    #     - Clusters: A new Clusters instance containing the subset if inplace is False.
    #     """
    #
    #     if not np.count_nonzero(mask) >= 4:
    #         # Handle the case where the mask filters out all points
    #         print("Not enough points found for subset.")
    #         if inplace:
    #             self.points = np.array([])
    #             self.parent = 0
    #             self.cluster_labels = []
    #             self.prediction = ''
    #             self.probability = 0
    #             self.model_weight = ''
    #             self.length = 0
    #             self.width = 0
    #             self.height = 0
    #             self.feature = []
    #             self.metadata = {}
    #         else:
    #             # Return an empty Clusters instance with (1, 3) dimension as
    #             # Clusters constructor needs (n, 3) ndarray for points
    #             return PointCloud(np.empty((1, 3)))
    #     else:
    #         if inplace:
    #             # Modify the current cluster's points and attributes in-place
    #             # Apply the mask to attributes of the cluster
    #             self.points = self.points[mask]
    #             self._update_obb_dim()
    #
    #             if hasattr(self, 'colors') and self.colors.shape[0] > 0:
    #                 self.colors = self.colors[mask]
    #             if hasattr(self, 'intensity') and self.intensity.shape[0] > 0:
    #                 self.intensity = self.intensity[mask]
    #             if hasattr(self, 'distToGround') and self.distToGround.shape[0] > 0:
    #                 self.distToGround = self.distToGround[mask]
    #         else:
    #             # Create a deep copy of the cluster and apply the mask
    #             subset = copy.deepcopy(self)
    #
    #             # Apply the mask to attributes of the cluster
    #             subset.points = subset.points[mask]
    #             subset._update_obb_dim()
    #
    #             if subset.colors is not None:
    #                 if subset.colors.shape[0] > 0:
    #                     subset.colors = subset.colors[mask]
    #             if subset.normals is not None:
    #                 if subset.normals.shape[0] > 0:
    #                     subset.normals = subset.normals[mask]
    #             if hasattr(subset, 'intensity'):
    #                 if subset.intensity.shape[0] > 0:
    #                     subset.intensity = subset.intensity[mask]
    #             if hasattr(subset, 'distToGround'):
    #                 if subset.distToGround.shape[0] > 0:
    #                     subset.distToGround = subset.distToGround[mask]
    #
    #             return subset
    def get_subset(self, mask, inplace=False):
        """
        Extracts a subset based on a mask, either by modifying the current point cloud or returning a new one.
        Uses GPU acceleration when CuPy is available for improved performance.

        Parameters:
        - mask (np.ndarray): A boolean array where True values indicate points to include in the subset.
        - inplace (bool): If True, modifies the current point cloud in-place. Default is False.

        Returns:
        - PointCloud: A new PointCloud instance containing the subset if inplace is False.
        """
        # Check if mask selects enough points (at least 4)
        if not np.count_nonzero(mask) >= 4:
            # Handle the case where the mask filters out all points
            print("Not enough points found for subset.")
            if inplace:
                self.points = np.array([])
                self.parent = 0
                self.cluster_labels = []
                self.prediction = ''
                self.probability = 0
                self.model_weight = ''
                self.length = 0
                self.width = 0
                self.height = 0
                self.feature = []
                self.metadata = {}
            else:
                # Return an empty PointCloud instance with (1, 3) dimension as
                # PointCloud constructor needs (n, 3) ndarray for points
                return PointCloud(np.empty((1, 3)))
        else:
            try:
                # Try to use GPU acceleration with CuPy
                import cupy as cp

                if inplace:
                    # Transfer data to GPU
                    cp_mask = cp.asarray(mask)

                    # Apply mask on GPU
                    if hasattr(self, 'points') and len(self.points) > 0:
                        cp_points = cp.asarray(self.points)
                        self.points = cp.asnumpy(cp_points[cp_mask])

                    # Update OBB dimensions after modifying points
                    self._update_obb_dim()

                    # Apply mask to other attributes if they exist
                    if hasattr(self, 'colors') and self.colors is not None and self.colors.shape[0] > 0:
                        cp_colors = cp.asarray(self.colors)
                        self.colors = cp.asnumpy(cp_colors[cp_mask])

                    if hasattr(self, 'normals') and self.normals is not None and self.normals.shape[0] > 0:
                        cp_normals = cp.asarray(self.normals)
                        self.normals = cp.asnumpy(cp_normals[cp_mask])

                    if hasattr(self, 'intensity') and hasattr(self.intensity, 'shape') and self.intensity.shape[0] > 0:
                        cp_intensity = cp.asarray(self.intensity)
                        self.intensity = cp.asnumpy(cp_intensity[cp_mask])

                    if hasattr(self, 'distToGround') and hasattr(self.distToGround, 'shape') and \
                            self.distToGround.shape[0] > 0:
                        cp_dist = cp.asarray(self.distToGround)
                        self.distToGround = cp.asnumpy(cp_dist[cp_mask])

                    # Process any custom attributes in the attributes dictionary
                    for attr_name in list(self.attributes.keys()):
                        attr_value = self.attributes[attr_name]
                        if isinstance(attr_value, np.ndarray):
                            if attr_value.shape[0] == len(mask):
                                cp_attr = cp.asarray(attr_value)
                                if cp_attr.ndim == 1:
                                    self.attributes[attr_name] = cp.asnumpy(cp_attr[cp_mask])
                                else:
                                    self.attributes[attr_name] = cp.asnumpy(cp_attr[cp_mask, ...])
                else:
                    # Create a deep copy of the point cloud and apply the mask
                    subset = copy.deepcopy(self)

                    # Transfer data to GPU and apply mask
                    cp_mask = cp.asarray(mask)

                    if subset.points is not None and len(subset.points) > 0:
                        cp_points = cp.asarray(subset.points)
                        subset.points = cp.asnumpy(cp_points[cp_mask])

                    # Update OBB dimensions
                    subset._update_obb_dim()

                    if subset.colors is not None and subset.colors.shape[0] > 0:
                        cp_colors = cp.asarray(subset.colors)
                        subset.colors = cp.asnumpy(cp_colors[cp_mask])

                    if subset.normals is not None and subset.normals.shape[0] > 0:
                        cp_normals = cp.asarray(subset.normals)
                        subset.normals = cp.asnumpy(cp_normals[cp_mask])

                    if hasattr(subset, 'intensity') and hasattr(subset.intensity, 'shape') and subset.intensity.shape[
                        0] > 0:
                        cp_intensity = cp.asarray(subset.intensity)
                        subset.intensity = cp.asnumpy(cp_intensity[cp_mask])

                    if hasattr(subset, 'distToGround') and hasattr(subset.distToGround, 'shape') and \
                            subset.distToGround.shape[0] > 0:
                        cp_dist = cp.asarray(subset.distToGround)
                        subset.distToGround = cp.asnumpy(cp_dist[cp_mask])

                    # Process any custom attributes in the attributes dictionary
                    for attr_name in list(subset.attributes.keys()):
                        attr_value = subset.attributes[attr_name]
                        if isinstance(attr_value, np.ndarray):
                            if attr_value.shape[0] == len(mask):
                                cp_attr = cp.asarray(attr_value)
                                if cp_attr.ndim == 1:
                                    subset.attributes[attr_name] = cp.asnumpy(cp_attr[cp_mask])
                                else:
                                    subset.attributes[attr_name] = cp.asnumpy(cp_attr[cp_mask, ...])

                    return subset

            except (ImportError, ModuleNotFoundError):
                # Fallback to CPU implementation if CuPy is not available
                if inplace:
                    # Modify the current point cloud's points and attributes in-place
                    # Apply the mask to attributes of the point cloud
                    self.points = self.points[mask]
                    self._update_obb_dim()

                    if hasattr(self, 'colors') and self.colors is not None and self.colors.shape[0] > 0:
                        self.colors = self.colors[mask]

                    if hasattr(self, 'normals') and self.normals is not None and self.normals.shape[0] > 0:
                        self.normals = self.normals[mask]

                    if hasattr(self, 'intensity') and hasattr(self.intensity, 'shape') and self.intensity.shape[0] > 0:
                        self.intensity = self.intensity[mask]

                    if hasattr(self, 'distToGround') and hasattr(self.distToGround, 'shape') and \
                            self.distToGround.shape[0] > 0:
                        self.distToGround = self.distToGround[mask]

                    # Process any custom attributes in the attributes dictionary
                    for attr_name in list(self.attributes.keys()):
                        attr_value = self.attributes[attr_name]
                        if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == len(mask):
                            if attr_value.ndim == 1:
                                self.attributes[attr_name] = attr_value[mask]
                            else:
                                self.attributes[attr_name] = attr_value[mask, ...]
                else:
                    # Create a deep copy of the point cloud and apply the mask
                    subset = copy.deepcopy(self)

                    # Apply the mask to attributes of the point cloud
                    subset.points = subset.points[mask]
                    subset._update_obb_dim()

                    if subset.colors is not None and subset.colors.shape[0] > 0:
                        subset.colors = subset.colors[mask]

                    if subset.normals is not None and subset.normals.shape[0] > 0:
                        subset.normals = subset.normals[mask]

                    if hasattr(subset, 'intensity') and hasattr(subset.intensity, 'shape') and subset.intensity.shape[
                        0] > 0:
                        subset.intensity = subset.intensity[mask]

                    if hasattr(subset, 'distToGround') and hasattr(subset.distToGround, 'shape') and \
                            subset.distToGround.shape[0] > 0:
                        subset.distToGround = subset.distToGround[mask]

                    # Process any custom attributes in the attributes dictionary
                    for attr_name in list(subset.attributes.keys()):
                        attr_value = subset.attributes[attr_name]
                        if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == len(mask):
                            if attr_value.ndim == 1:
                                subset.attributes[attr_name] = attr_value[mask]
                            else:
                                subset.attributes[attr_name] = attr_value[mask, ...]

                    return subset
    def get_obb(self):
        """
        Calculates the 3D Oriented Bounding Box (OBB) for the cluster, aligned with the Z-axis.

        This method computes the OBB by first finding the 2D OBB on the XY plane using PCA
        and then extruding it along the Z-axis.

        Returns:
            o3d.geometry.OrientedBoundingBox: The 3D OBB of the cluster, aligned with the Z-axis.
        """
        if len(self.points) == 0:
            raise ValueError("Cannot compute OBB for a cluster with no points.")

        # Project the points onto the XY plane
        points_xy = self.points[:, :2]

        # Performing PCA on the XY coordinates
        mean_xy = np.mean(points_xy, axis=0)
        centered_points_xy = points_xy - mean_xy
        covariance_matrix = np.cov(centered_points_xy.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Check for numerical stability
        if np.iscomplex(eigenvalues).any() or np.iscomplex(eigenvectors).any():
            raise ValueError("Numerical instability in PCA computation.")

        # Project points onto the PCA axes to find the extents
        transformed_points = np.dot(centered_points_xy, eigenvectors)

        min_extent = np.min(transformed_points, axis=0)
        max_extent = np.max(transformed_points, axis=0)

        # Generate the corner points of the OBB
        corner_points = np.array([
            min_extent,
            [max_extent[0], min_extent[1]],
            max_extent,
            [min_extent[0], max_extent[1]],
            min_extent  # Close the loop
        ])

        # Transform the corners back to the original space
        obb_corners = np.dot(corner_points, eigenvectors.T) + mean_xy

        # ------------------------- -------------------------

        # Extrude the 2D OBB corners to the minimum and maximum Z extent, parallel to Z axis
        min_z = np.min(self.points[:, 2])  # Min Z extent of the point cloud
        max_z = np.max(self.points[:, 2])  # Max Z extent of the point cloud
        bottom_corners = np.hstack((obb_corners, np.full((5, 1), min_z)))
        top_corners = np.hstack((obb_corners, np.full((5, 1), max_z)))
        corners_3d = np.vstack((bottom_corners, top_corners))

        # Create a 3D OBB from the extruded corners
        obb_z = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners_3d))
        return obb_z

    def get_obb_orientation(self):
        """
        Calculates the angle of the OBB's orientation on the XY plane relative to the global X-axis.

        Returns:
            float: The angle of orientation in Radian.
        """
        # Extract the rotation matrix of the OBB
        obb = self.get_obb()
        r = obb.R

        # Use the x-axis of the OBB (first column of the rotation matrix)
        obb_x_axis = r[:, 0]

        # Calculate the angle of this vector on the XY plane
        angle_rad = np.arctan2(obb_x_axis[1], obb_x_axis[0])

        return angle_rad

    def jitter_points(self, scale=0.005):
        """
        Apply jittering to the points of the cluster in place. This method adds
        a small random displacement to the points and updates distToGround if it exists.

        Parameters:
        - scale (float): The scale of the noise to add. Default is 0.005.
        """
        if len(self.points) == 0:
            raise ValueError("The cluster has no points to jitter.")

        # Generating random noise
        noise = np.random.normal(0, scale, self.points.shape)

        # Adding noise to points
        self.points += noise

        # Update distToGround if it exists
        if len(self.distToGround) > 0:
            self.distToGround += noise[:, 2]  # Update only the Z-axis component

    def KNN(self, k=2):
        """
        Perform a k-nearest neighbors (KNN) search on the cluster's points.

        Parameters:
        - k (int): The number of nearest neighbors to find for each point in the cluster.

        Returns:
        - distances (numpy.ndarray): A 2D array where each row contains the distances
                                     to the nearest neighbors for each point.
        - indices (numpy.ndarray): A 2D array where each row contains the indices
                                   of the nearest neighbors for each point.
        """
        if len(self.points) == 0:
            raise ValueError("The cluster has no points to perform KNN.")

        # Create a k-d tree for the point cloud
        tree = KDTree(self.points)

        # Query the k-d tree for KNN
        distances, indices = tree.query(self.points, k=k)

        return distances, indices

    def normalise(self, apply_scaling=True, apply_centering=True, rotation_axes=(True, True, True)):
        """
        Normalize the points of the cluster by centering, scaling, and optionally applying random rotations.

        Args:
        - apply_scaling (bool): If True, scales the points to fit within a unit sphere.
        - apply_centering (bool): If True, centers the points around the origin.
        - rotation_axes (tuple): A tuple of booleans indicating random rotation around 'x', 'y', and 'z' axes.

        Returns:
        - None: Updates the points of the cluster in place.
        """

        # Centering the points
        if apply_centering:
            centroid = np.mean(self.points, axis=0)
            self.points -= centroid

        # Scaling the points
        if apply_scaling:
            max_distance = np.max(np.sqrt(np.sum(self.points ** 2, axis=1)))
            if max_distance > 0:
                self.points /= max_distance

        # Generate random rotation for each axis
        base_seed = int(time.time())
        np.random.seed(base_seed)
        alpha = np.random.uniform(0, 2 * np.pi) if rotation_axes[0] else 0

        np.random.seed(base_seed + 1)
        beta = np.random.uniform(0, 2 * np.pi) if rotation_axes[1] else 0

        np.random.seed(base_seed + 2)
        gamma = np.random.uniform(0, 2 * np.pi) if rotation_axes[2] else 0

        # Rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma), np.cos(gamma), 0],
                       [0, 0, 1]])

        # Combine the rotations
        R = np.dot(Rz, np.dot(Ry, Rx))

        # Apply the rotation to points
        self.points = np.matmul(R, self.points.T).T

    def save(self, file_name):
        """
        Saves the Clusters instance to a file. The format is determined by the file extension.

        Parameters:
        file_name (str): The params of the file to save the instance to.

        Supported Formats:
        .ply - Saves the instance as a PLY file.

        Raises:
        NotImplementedError: If the file extension is not supported.
        """
        extension = file_name.split('.')[-1]

        if extension == 'ply':
            self._save_as_ply(file_name)
        else:
            raise NotImplementedError(f"File extension '.{extension}' is not supported.")

    def save_property(self, property_name, file_name):
        """
        Saves a specified property of the Clusters to a .npy file.

        Parameters:
        property_name (str): The params of the property to save.
        file_name (str): The params of the .npy file to save the property to.

        Raises:
        AttributeError: If the specified property does not exist.
        """

        if hasattr(self, property_name):
            property_data = getattr(self, property_name)
            np.save(file_name, property_data)
        else:
            raise AttributeError(f"Property '{property_name}' not found in Clusters.")

            # o3d.io.write_point_cloud(groundFile, pcd_ground)

    def show(self, pick_point=False, show_obb=False):
        """
        Visualizes the cluster using Open3D. If colors are set for the cluster points, they will be
        used in the visualization. The method also supports visualizing the oriented bounding box (OBB)
        of the cluster and picking a point from the visualized point cloud.

        Parameters:
        - pick_point (bool): If True, enables the point picking feature which allows selecting a point
          from the point cloud by clicking on it. The method will return the index of the picked point.
          Default is False.
        - show_obb (bool): If True, visualizes the oriented bounding box (OBB) around the cluster points.
          The OBB is displayed in red. Cannot be used together with pick_point. Default is False.

        Returns:
        - None if pick_point is False.
        - List of picked point indices if pick_point is True and a point is successfully picked.

        Remarks:
        - The method checks if the cluster contains any points before proceeding with visualization.
        - It is not possible to use pick_point and show_obb simultaneously due to visualization constraints.
        - Colors for the points are normalized to the range [0, 1] if they exceed this range, assuming
          the initial range is [0, 255].
        - When pick_point is enabled, the method utilizes Open3D's VisualizerWithEditing to facilitate
          point selection. Point picking is done using Shift + Left Click.
        - If colors are provided for the points, they will be applied to the visualization. Otherwise,
          the default color scheme of Open3D is used.
        """
        if len(self.points) == 0:
            print("The cluster has no points to show.")
            return None
        elif pick_point and show_obb is True:
            raise ValueError(
                "Bounding box cannot be visible when picking point is enabled. Both pick_point and show_obb"
                " parameters cannot be set to True simultaneously.")

        # Convert points to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(self.colors[:, :3]) if len(self.colors) > 0 else None

        geometries_to_draw = [pcd]

        # Visualize OBB if requested
        if show_obb:
            obb = self.get_obb()
            obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            # Set OBB color to red
            obb_lines.colors = o3d.utility.Vector3dVector(
                [1, 0, 0] * np.ones((len(obb_lines.lines), 3))
            )
            geometries_to_draw.append(obb_lines)

        if not pick_point:
            # Visualize the point cloud (and OBB if included)
            o3d.visualization.draw_geometries(geometries_to_draw, window_name="Clusters Visualization")
        else:
            # Visualize the point cloud to be able to select a point
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window()
            for geometry in geometries_to_draw:
                vis.add_geometry(geometry)
            vis.run()

            # Select point using Shift + Left Click
            picked_point_index = vis.get_picked_points()
            vis.destroy_window()

            return picked_point_index, self.points[picked_point_index]

    def shuffle_points(self):
        """
        Shuffle the points of the cluster in place while maintaining the relationship
        with corresponding attributes like color, intensity, and distance to the ground.
        """
        if len(self.points) == 0:
            raise ValueError("The cluster has no points to shuffle.")

        # Generating a random permutation of indices
        shuffle_indices = np.random.permutation(len(self.points))

        # Shuffling points and related attributes
        self.points = self.points[shuffle_indices]
        # TODO: What if there is no color/intensity/distToGround provided?
        if len(self.colors) != 0:
            self.colors = self.colors[shuffle_indices]
        if len(self.intensity) != 0:
            self.intensity = self.intensity[shuffle_indices]
        if len(self.distToGround) != 0:
            self.distToGround = self.distToGround[shuffle_indices]

    def slice_by_dist_to_ground(self, bottom_dist, top_dist, inplace=False):
        """
        Slice the cluster based on distances to the ground.
        Includes points whose distance to the ground is within the specified range.

        Parameters:
        - bottom_dist (float): The lower bound of the distance range (inclusive).
        - top_dist (float): The upper bound of the distance range (inclusive).
        - inplace (bool): If True, modifies the current Clusters object in place by removing points outside the specified distance range. If False, returns a new Clusters object containing only the points within the specified range, leaving the original Clusters unchanged.

        Returns:
        - Clusters: A new Clusters object containing the sliced points and corresponding attributes, unless inplace is True, in which case the original Clusters is modified and nothing is returned.

        Raises:
        - ValueError: If `distToGround` property is not available for slicing.
        """

        if len(self.distToGround) == 0:
            raise ValueError("distToGround property is not available for slicing.")

        # Create a boolean mask for points within the specified distance range
        mask = (self.distToGround >= bottom_dist) & (self.distToGround <= top_dist)

        if inplace:
            self.get_subset(mask, inplace)
        else:
            return self.get_subset(mask, inplace)

    def slice_by_dist_to_point(self, picked_point, max_dist, min_dist=0, inplace=False):
        # Finding the distance of each point to a reference point and return a distance array

        # Create connection vectors
        dist_pt = picked_point - self.points

        # Calculate l2 Norm which is equal to the magnitude (length) of the vector
        dists = np.linalg.norm(dist_pt, axis=1)

        # Create a boolean mask for points within the specified distance range
        mask = (dists >= min_dist) & (dists <= max_dist)

        if inplace:
            self.get_subset(mask, inplace=True)
        else:
            return self.get_subset(mask, inplace=False)

    def SOR(self, nb_neighbors=20, std_ratio=2.0):
        """
        Apply Statistical Outlier Removal (SOR) to filter out noise from the point cloud.

        Args:
        - nb_neighbors (int): Number of nearest neighbors to use for mean distance calculation.
        - std_ratio (float): Standard deviation ratio. Points with a distance larger than
                             (mean distance + std_ratio * standard deviation) are considered outliers.

        Returns:
        - mask (np.ndarray): A boolean mask array where True indicates inliers.
        """
        if len(self.points) == 0:
            raise ValueError("The cluster has no points to apply SOR filter.")

        # Convert points to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # Apply Statistical Outlier Removal
        _, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

        # Create a mask for inliers
        mask = np.zeros(len(self.points), dtype=bool)
        mask[np.asarray(ind)] = True

        return mask

    def subsample(self, rate, boolean=False):
        """
        Subsample the points of the cluster using a random mask.

        Vars:
        - rate (float): The subsampling rate in the range (0, 1].
        - boolean : Return a mask array

        Returns:
        - PointCloud: A new PointCloud instance containing the subsampled points.
        - np.ndarray: A boolean mask array where True values indicate points to include in the subset.
        """
        if rate <= 0 or rate > 1:
            raise ValueError("Subsampling rate must be in the range (0, 1].")

        # Generate a random mask for subsampling
        mask = np.random.rand(len(self.points)) < rate

        if not boolean:
            return self.get_subset(mask, inplace=False)
        else:
            return mask

    def _update_obb_dim(self):
        """
        Calculate the dimensions (length, width, height) of the cluster based on its Oriented Bounding Box (OBB).
        Assumes that the height of the OBB is almost parallel to the Z-axis.

        Returns:
        - np.ndarray: An array containing the dimensions [length, width, height] of the cluster.

        CAUTION: This method is accurate for cluster_labels with OBB height almost parallel to the Z-axis.
        """
        # Compute the OBB of the cluster
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, :3])
        obb = pcd.get_oriented_bounding_box()

        # Compute the OBB's three main axes
        axis1 = obb.R[:, 0]
        axis2 = obb.R[:, 1]
        axis3 = obb.R[:, 2]

        # Project the axes onto world's Z-axis
        proj1 = np.abs(np.dot(axis1, [0, 0, 1]))
        proj2 = np.abs(np.dot(axis2, [0, 0, 1]))
        proj3 = np.abs(np.dot(axis3, [0, 0, 1]))

        # Determine the height along the axis with the largest projection onto Z-axis
        if proj1 >= proj2 and proj1 >= proj3:
            i = 0
        elif proj2 >= proj1 and proj2 >= proj3:
            i = 1
        else:
            i = 2

        height = obb.extent[i]
        length = max(np.delete(obb.extent, i))
        width = min(np.delete(obb.extent, i))

        self.length = length
        self.width = width
        self.height = height

    def _save_as_ply(self, file_name):
        """
        Saves the PointCloud instance as a PLY file.

        This method exports the point cloud to a PLY file format, including
        points (required), colors (if available), and normals (if available).

        Parameters:
            file_name (str): The name of the PLY file to save the instance to.
        """
        if hasattr(self, 'points'):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)

            # Save colors if available
            if hasattr(self, 'colors') and self.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(self.colors)

            # Save normals if available (fixed: check for 'normals' not 'normal')
            if hasattr(self, 'normals') and self.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(self.normals)

            o3d.io.write_point_cloud(file_name, pcd)
        else:
            raise AttributeError("PointCloud instance does not have 'points' attribute.")

    def _are_points_coplanar(self, tolerance=1e-6):
        """
        Check if all points are coplanar (lie on the same plane).

        This method checks if all Z coordinates are the same within a tolerance,
        which would indicate the points are coplanar in the XY plane.

        Args:
            tolerance (float): The tolerance for comparing Z coordinates.
                              Default is 1e-6.

        Returns:
            bool: True if points are coplanar, False otherwise.
        """
        if len(self.points) < 4:
            # Less than 4 points are always coplanar
            return True

        # Check if all Z coordinates are the same (within tolerance)
        z_coords = self.points[:, 2]
        z_range = np.max(z_coords) - np.min(z_coords)

        return z_range < tolerance

    def _update_2d_bbox(self):
        """
        Calculate 2D bounding box dimensions for coplanar points.

        This method is used when points are coplanar (e.g., after draping),
        and 3D OBB calculation would fail. It calculates the bounding box
        in the XY plane and sets the height to the Z-axis range (usually ~0).
        """
        if len(self.points) == 0:
            return

        # Calculate bounding box in XY plane
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)

        # Set dimensions
        self.length = max_coords[0] - min_coords[0]
        self.width = max_coords[1] - min_coords[1]
        self.height = max_coords[2] - min_coords[2]  # Usually ~0 for coplanar points

        # Set center
        self.center = [
            (min_coords[0] + max_coords[0]) / 2,
            (min_coords[1] + max_coords[1]) / 2,
            (min_coords[2] + max_coords[2]) / 2
        ]

        print(
            f"Calculated 2D bounding box for coplanar points: L={self.length:.2f}, W={self.width:.2f}, H={self.height:.6f}")

    def estimate_normals(self, k: int = 30):
        """
        Estimate normals for the point cloud using KNN.

        Uses Open3D's normal estimation based on k-nearest neighbors.
        The normals are stored in self.normals as a (n, 3) numpy array.

        Args:
            k: Number of nearest neighbors to use for normal estimation

        Raises:
            ImportError: If Open3D is not available
            ValueError: If point cloud has no points
        """
        if not hasattr(self, 'points') or self.points is None or len(self.points) == 0:
            raise ValueError("Point cloud has no points")

        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D is required for normal estimation. Install with: pip install open3d")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        # Estimate normals using KNN
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )

        # Extract normals back to numpy array
        self.normals = np.asarray(pcd.normals, dtype=np.float32)
