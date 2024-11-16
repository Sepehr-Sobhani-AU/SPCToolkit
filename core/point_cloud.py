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
# import cupy as cp
import pandas as pd
import open3d as o3d
import tensorflow as tf
from scipy.stats import norm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


class PointCloud:
    """
    Initialize a Cluster object.

    Required Parameters:
    - points (np.ndarray): A numpy array (n, 3) of points in the format [[x1, y1, z1],
                                                                         [x2, y2, z2],
                                                                               .
                                                                               .
                                                                               .
                                                                         [xn, yn, zn]]

    Optional Keyword Arguments:
    - parent_uuid (str): The uuid of the parent cluster.
    - child_uuid (str): The uuid of the child cluster.
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
    def __init__(self, points, colors=None, normals=None, **kwargs):


        # Validation points
        if not isinstance(points, np.ndarray):  # or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a numpy array with shape (n, 3)")

        self.points = points
        self.colors = colors
        self.normals = normals
        self.name = kwargs.get('name', None)
        self.uuid = kwargs.get('uuid', None)
        self.parent_uuid = kwargs.get('parent_uuid', None)
        self.child_uuid = kwargs.get('child_uuid', None)

        self.clusters = []
        self.prediction = ''
        self.probability = 0
        self.model_weight = ''
        self.center = [0, 0, 0]
        self.length = 0
        self.width = 0
        self.height = 0

        # If cluster has no point in it, ignore calculating obb
        if self.size() > 1:
            self._update_obb_dim()

        # Validation for color, intensity, normal and distToGround
        for attr_name in ['color', 'intensity', 'distToGround', 'normal']:
            attr_value = kwargs.get(attr_name, np.array([]))
            if not isinstance(attr_value, np.ndarray) or (len(attr_value) != 0 and len(attr_value) != len(points)):
                raise ValueError(f"{attr_name} must be a numpy array of the same length as points")
            setattr(self, attr_name, attr_value)

        self.feature = kwargs.get('feature', [])
        self.metadata = kwargs.get('metadata', {})

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

        # Update corresponding attributes: color, intensity, and distToGround
        if len(self.color) > 0:
            self.color = np.repeat(self.color, augmentation_factor, axis=0)

        if len(self.intensity) > 0:
            self.intensity = np.repeat(self.intensity, augmentation_factor)

        if len(self.distToGround) > 0:
            # Repeat and add noise to the Z-axis component
            augmented_distToGround = np.repeat(self.distToGround, augmentation_factor)
            augmented_distToGround += noise[:, 2]
            self.distToGround = augmented_distToGround

    # def dbscan(self, eps=0.05, min_points=10, return_clusters_object=False):
    #     """
    #     Apply DBSCAN clustering to the points in the cluster using Open3D.
    #
    #     Parameters:
    #     - eps (float): The maximum distance between two samples for one to be considered
    #                    as in the neighborhood of the other. Default is 0.05.
    #     - min_points (int): The number of samples in a neighborhood for a point to be considered
    #                         as a core point. Default is 10.
    #     - return_clusters_object (bool): If True, returns a Clusters object containing the DBSCAN result.
    #                                      Default is False.
    #
    #     Returns:
    #     - labels (np.ndarray) or (np.ndarray, Clusters): An array of cluster labels, and optionally a Clusters object.
    #     """
    #     if len(self.points) == 0:
    #         raise ValueError("The cluster has no points for DBSCAN clustering.")
    #
    #     # Convert points to Open3D point cloud and perform DBSCAN
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(self.points)
    #     with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #         labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    #
    #     # Store the DBSCAN labels
    #     self.clusters = labels
    #
    #     if return_clusters_object:
    #         # Create and return Clusters object
    #         clusters = Clusters()
    #         clusters.points = self.points
    #         clusters.labels = labels
    #         clusters.colors = self.color
    #         clusters.normals = self.normal
    #         return clusters
    #     else:
    #         return labels

    def density_downsample(self, voxel_size):
        """
        A density based downsampling of the cluster. It preserves
        additional attributes (like colors or normals).

        Parameters:
        - voxel_size (float): The voxel size to determine the new point cloud density.

        Returns:
        - A Cluster object created from the downsampled point cloud with original attributes of the point.
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

        return self.get_subcluster(mask)

    # def get_clusters(self, min_points=200):
    #     """
    #     Generate a Clusters object containing individual clusters from DBSCAN labels,
    #     excluding clusters with fewer than min_points.
    #
    #     Args:
    #         min_points (int, optional): Minimum number of points for a cluster to be included. Default is 200.
    #
    #     Returns:
    #         Clusters: An object containing all the valid clusters (excluding noise and smaller clusters).
    #     """
    #
    #     if not hasattr(self, 'clusters'):
    #         raise ValueError("DBSCAN must be applied before generating 'Clusters' object.")
    #
    #     # Filter out noise points (usually labeled as -1)
    #     unique_labels = set(self.clusters)
    #     unique_labels.discard(-1)
    #
    #     # Create a Clusters object to hold the clusters
    #     clusters = Clusters()
    #
    #     # Filter labels to include only those with enough points
    #     valid_labels = [label for label in unique_labels if np.sum(self.clusters == label) >= min_points]
    #
    #     for label in valid_labels:
    #         # Create mask for points belonging to the current label
    #         mask = self.clusters == label
    #
    #         # Create a copy of cluster
    #         child_cluster = copy.deepcopy(self)
    #         child_cluster.points = self.points[mask]
    #         # Add the new cluster to clusters instance
    #         clusters.add(child_cluster)
    #
    #     return clusters

    def get_eigenvalues(self, k, smooth=True):
        """
        Calculate the eigenvalues of the covariance matrix of k-nearest neighbor points for each point in the cluster.

        This method leverages a k-d tree to efficiently find the k-nearest neighbors (KNN) for each point
        in the cluster, computes the covariance matrix for these points, and then derives the eigenvalues.
        Optionally, it can smooth the eigenvalues by averaging them across the neighbors of each point.

        Parameters:
        - k (int): The number of nearest neighbors to consider for each point.
        - smooth (bool, optional): If True, the eigenvalues are smoothed by averaging over each point's neighbors.
                                   If False, the raw eigenvalues for each point's neighborhood are returned.

        Returns:
        - np.ndarray: An array of eigenvalues.

        Raises:
        - ValueError: If 'k' is less than or equal to the number of points in the cluster or if 'k' is non-positive.

        Examples:
        --------
        # Assuming 'points' is a numpy array representing points in the cluster
        cluster_instance = Cluster(points=np.random.rand(100, 3))
        eigenvalues = cluster_instance.get_eigenvalues(k=5, smooth=True)
        print("Averaged Eigenvalues:", eigenvalues)

        Notes:
        -----
        - The method uses TensorFlow to perform batch operations for computing eigenvalues, which requires
          the installation of TensorFlow alongside NumPy and SciPy.
        - This function is computationally intensive for large 'k' or very large point sets due to the
          computation of eigenvalues for each point's KNN.
        """
        point_cloud = self.points

        # Create a k-d tree
        tree = KDTree(point_cloud)

        # Query the k-d tree for KNN
        distances, indices = tree.query(point_cloud,
                                        k=k)  # 'indices' now contains the indices of the k-nearest neighbors for each point

        # Gather KNN points
        knn_points = point_cloud[indices]  # Shape: [num_points, k, 3]

        # Compute means and center the points
        mean_knn_points = np.mean(knn_points, axis=1, keepdims=True)
        centered_knn_points = knn_points - mean_knn_points

        # Reshape for batch matrix multiplication
        centered_knn_points_reshaped = centered_knn_points.transpose(0, 2, 1)

        # Batch covariance matrix computation
        cov_matrices = np.matmul(centered_knn_points_reshaped, centered_knn_points) / (
                k - 1)  # cov_matrices.shape is [num_points, 3, 3]

        # Convert the NumPy array to a TensorFlow tensor
        cov_matrices_tf = tf.convert_to_tensor(cov_matrices, dtype=tf.float32)

        # Compute the eigenvalues and eigenvectors in batch
        eigenvalues_tf, eigenvectors_tf = tf.linalg.eigh(cov_matrices_tf)

        # Convert the eigenvalues back to a NumPy array if needed
        eigenvalues = eigenvalues_tf.numpy()

        if smooth:
            # Use advanced indexing to compute the mean eigenvalues across neighbors
            neighbor_eigenvalues = eigenvalues[indices]  # Use indices to gather neighbor eigenvalues
            avg_eigenvalues = np.mean(neighbor_eigenvalues, axis=1)  # Average over the neighbor axis

            self.eigenvalues = avg_eigenvalues
            return avg_eigenvalues
        else:
            self.eigenvalues = eigenvalues
            return eigenvalues

    # def get_subcluster(self, mask, inplace=False):
    #     """
    #     Extracts a subcluster based on a mask, either by modifying the current cluster or returning a new one.
    #
    #     Parameters:
    #     - mask (np.ndarray): A boolean array where True values indicate points to include in the subcluster.
    #     - inplace (bool): If True, modifies the current cluster in-place. Default is False.
    #
    #     Returns:
    #     - Cluster: A new Cluster instance containing the subcluster if inplace is False.
    #     """
    #
    #     if not np.count_nonzero(mask) >= 4:
    #         # Handle the case where the mask filters out all points
    #         print("Not enough points found for subcluster.")
    #         if inplace:
    #             self.points = np.array([])
    #             self.label = 0
    #             self.parent = 0
    #             self.clusters = np.array([])
    #             self.prediction = ''
    #             self.probability = 0
    #             self.model_weight = ''
    #             self.length = 0
    #             self.width = 0
    #             self.height = 0
    #             self.feature = []
    #             self.metadata = {}
    #         else:
    #             # Return an empty Cluster instance with (1, 3) dimension as
    #             # Cluster constructor needs (n, 3) ndarray for points
    #             return Cluster(np.empty((1, 3)))
    #     else:
    #         if inplace:
    #             # Modify the current cluster's points and attributes in-place
    #             # Apply the mask to attributes of the cluster
    #             self.points = self.points[mask]
    #             self._update_obb_dim()
    #
    #             if hasattr(self, 'color') and self.color.shape[0] > 0:
    #                 self.color = self.color[mask]
    #             if hasattr(self, 'intensity') and self.intensity.shape[0] > 0:
    #                 self.intensity = self.intensity[mask]
    #             if hasattr(self, 'distToGround') and self.distToGround.shape[0] > 0:
    #                 self.distToGround = self.distToGround[mask]
    #         else:
    #             # Create a deep copy of the cluster and apply the mask
    #             subcluster = copy.deepcopy(self)
    #
    #             # Apply the mask to attributes of the cluster
    #             subcluster.points = subcluster.points[mask]
    #             subcluster._update_obb_dim()
    #
    #             if hasattr(subcluster, 'color') and subcluster.color.shape[0] > 0:
    #                 subcluster.color = subcluster.color[mask]
    #             if hasattr(subcluster, 'intensity') and subcluster.intensity.shape[0] > 0:
    #                 subcluster.intensity = subcluster.intensity[mask]
    #             if hasattr(subcluster, 'distToGround') and subcluster.distToGround.shape[0] > 0:
    #                 subcluster.distToGround = subcluster.distToGround[mask]
    #
    #             return subcluster

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
        Saves the Cluster instance to a file. The format is determined by the file extension.

        Parameters:
        file_name (str): The name of the file to save the instance to.

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
        Saves a specified property of the Cluster to a .npy file.

        Parameters:
        property_name (str): The name of the property to save.
        file_name (str): The name of the .npy file to save the property to.

        Raises:
        AttributeError: If the specified property does not exist.
        """

        if hasattr(self, property_name):
            property_data = getattr(self, property_name)
            np.save(file_name, property_data)
        else:
            raise AttributeError(f"Property '{property_name}' not found in Cluster.")

            # o3d.io.write_point_cloud(groundFile, pcd_ground)

    def show(self, pick_point=False, show_obb=False, random_color_attr=None):
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
        elif pick_point and show_obb == True:
            raise ValueError(
                "Bounding box cannot be visible when picking point is enabled. Both pick_point and show_obb"
                " parameters cannot be set to True simultaneously.")

        # Convert points to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[:, :3])

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
            o3d.visualization.draw_geometries(geometries_to_draw, window_name="Cluster Visualization")
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
        if len(self.color) != 0:
            self.color = self.color[shuffle_indices]
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
        - inplace (bool): If True, modifies the current Cluster object in place by removing points outside the specified distance range. If False, returns a new Cluster object containing only the points within the specified range, leaving the original Cluster unchanged.

        Returns:
        - Cluster: A new Cluster object containing the sliced points and corresponding attributes, unless inplace is True, in which case the original Cluster is modified and nothing is returned.

        Raises:
        - ValueError: If `distToGround` property is not available for slicing.
        """

        if len(self.distToGround) == 0:
            raise ValueError("distToGround property is not available for slicing.")

        # Create a boolean mask for points within the specified distance range
        mask = (self.distToGround >= bottom_dist) & (self.distToGround <= top_dist)

        if inplace:
            self.get_subcluster(mask, inplace)
        else:
            return self.get_subcluster(mask, inplace)

    def slice_by_dist_to_point(self, picked_point, max_dist, min_dist=0, inplace=False):
        # Finding the distance of each point to a reference point and return a distance array

        # Create connection vectors
        dist_pt = picked_point - self.points

        # Calculate l2 Norm which is equal to the magnitude (length) of the vector
        dists = np.linalg.norm(dist_pt, axis=1)

        # Create a boolean mask for points within the specified distance range
        mask = (dists >= min_dist) & (dists <= max_dist)

        if inplace:
            self.get_subcluster(mask, inplace=True)
        else:
            return self.get_subcluster(mask, inplace=False)

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

        return self.get_subcluster(mask, inplace=False)

    def size(self):
        """
        Return the number of points in the cluster.

        Returns:
        int: The total number of points in the cluster.
        """
        # TODO: Why not to set size as property?!
        return len(self.points)

    def _update_obb_dim(self):
        """
        Calculate the dimensions (length, width, height) of the cluster based on its Oriented Bounding Box (OBB).
        Assumes that the height of the OBB is almost parallel to the Z-axis.

        Returns:
        - np.ndarray: An array containing the dimensions [length, width, height] of the cluster.

        CAUTION: This method is accurate for clusters with OBB height almost parallel to the Z-axis.
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
        Saves the Cluster instance as a PLY file.

        Parameters:
        file_name (str): The name of the PLY file to save the instance to.
        """
        # Assuming the Cluster instance has a point cloud attribute named 'points'
        if hasattr(self, 'points'):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)

            if hasattr(self, 'color'):
                pcd.colors = o3d.utility.Vector3dVector(self.color)

            #            if hasattr(self, 'normal'):
            #                pcd.normals = o3d.utility.Vector3dVector(self.normal)

            o3d.io.write_point_cloud(file_name, pcd)
        else:
            raise AttributeError("Cluster instance does not have 'points' attribute.")