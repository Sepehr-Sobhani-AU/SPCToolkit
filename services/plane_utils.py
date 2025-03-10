"""
Plane Utilities Module

This module provides a Plane class for handling operations related to a plane in 3D space,
such as fitting planes to point clouds, calculating distances from points to a plane,
projecting points onto a plane, and identifying points that lie on or near a plane.

The implementation is based on the equation of a plane: Ax + By + Cz + D = 0
where [A, B, C] represents the normal vector of the plane.
"""

import numpy as np
import pyransac3d as pyrsc


class Plane:
    """
    A class for handling operations related to planes in 3D space.

    This class provides methods for fitting planes to point clouds,
    calculating distances from points to planes, projecting points onto planes,
    and identifying points that lie on or near planes.

    Attributes:
        params (numpy.ndarray): Parameters of the plane in the form [A, B, C, D]
                               for the equation Ax + By + Cz + D = 0.
    """

    def __init__(self, params=None):
        """
        Initialize a Plane object.

        Parameters:
            params (numpy.ndarray, optional): Parameters of the plane in the form
                                             [A, B, C, D] for the equation Ax + By + Cz + D = 0.
                                             If None, no plane is initialized.
        """
        self.params = params

    def fit(self, points, ransac_threshold=0.001, ransac_iterations=1000):
        """
        Fit a plane to the given points using RANSAC algorithm.

        Parameters:
            points (numpy.ndarray): Array of 3D points of shape (n, 3).
            ransac_threshold (float, optional): Threshold for RANSAC algorithm. Default is 0.001.
            ransac_iterations (int, optional): Number of iterations for RANSAC algorithm. Default is 1000.

        Returns:
            tuple: A tuple containing:
                - self: The Plane object with updated params.
                - inliers (numpy.ndarray): Indices of points that are inliers to the fitted plane.
        """
        # Create Plane object from pyransac3d
        obj_plane = pyrsc.Plane()

        # Fit plane using RANSAC and returns the inliers and plane parameters A, B, C & D (Ax+By+Cz+D=0)
        self.params, inliers = obj_plane.fit(points, thresh=ransac_threshold, maxIteration=ransac_iterations)

        return self, inliers

    def get_unit_normal_vector(self):
        """
        Calculate and return the unit normal vector of the plane.

        Returns:
            numpy.ndarray: Unit normal vector of the plane.

        Raises:
            ValueError: If the plane parameters are not set.
        """
        if self.params is None:
            raise ValueError("Plane parameters are not set. Fit a plane first.")

        # Finding 2 vectors on the plane
        x0, y0 = [0, 0]
        x1, y1 = [1, 0]
        x2, y2 = [0, 1]

        A, B, C, D = self.params[0], self.params[1], self.params[2], self.params[3]

        # Calculate z-coordinates using the plane equation: Ax + By + Cz + D = 0
        # => z = (-Ax - By - D) / C
        z0 = (- A * x0 - B * y0 - D) / C
        z1 = (- A * x1 - B * y1 - D) / C
        z2 = (- A * x2 - B * y2 - D) / C

        # Create three points on the plane
        p0 = np.array([x0, y0, z0])
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])

        # Create two vectors on the plane
        v1 = p1 - p0
        v2 = p2 - p0

        # Finding Normal vector of the plane using cross product of 2 vectors on the plane
        n = np.cross(v1, v2)  # Normal Vector

        # Unit Normal Vector (Magnitude = 1)
        # Unit Normal Vector = Normal vector / Normal vector's Magnitude
        # Normal vector's Magnitude = sqrt(n·n) where n·n is the dot product
        n_hat = n / np.sqrt(np.dot(n, n))

        return n_hat

    def calculate_distances(self, points):
        """
        Calculate the signed distances from points to the plane.

        The distance is positive if the point is on the same side of the plane as
        the normal vector, and negative otherwise.

        Parameters:
            points (numpy.ndarray): Array of 3D points of shape (n, 3).

        Returns:
            numpy.ndarray: Array of signed distances from each point to the plane.

        Raises:
            ValueError: If the plane parameters are not set.
        """
        if self.params is None:
            raise ValueError("Plane parameters are not set. Fit a plane first.")

        # Finding a point on the plane to use as reference
        x0, y0 = [0, 0]
        A, B, C, D = self.params[0], self.params[1], self.params[2], self.params[3]
        z0 = (- A * x0 - B * y0 - D) / C
        p0 = np.array([x0, y0, z0])

        # Get unit normal vector
        n_hat = self.get_unit_normal_vector()

        # Calculate distances as the dot product of the vector (point - p0) with the normal vector
        distances = np.dot(points - p0, n_hat)

        return distances

    def project_points(self, points):
        """
        Project points onto the plane.

        The projection is done by subtracting the distance vector from each point.
        The distance vector is parallel to the normal vector of the plane.

        Parameters:
            points (numpy.ndarray): Array of 3D points of shape (n, 3).

        Returns:
            numpy.ndarray: Array of projected points on the plane.

        Raises:
            ValueError: If the plane parameters are not set.
        """
        if self.params is None:
            raise ValueError("Plane parameters are not set. Fit a plane first.")

        # Get unit normal vector
        n_hat = self.get_unit_normal_vector()

        # Calculate distances to the plane
        distances = self.calculate_distances(points)

        # Reshape distances for broadcasting
        distances_reshaped = distances.reshape((distances.shape[0], 1))

        # Calculate distance vectors (parallel to normal vector)
        distance_vectors = np.multiply(distances_reshaped, n_hat)

        # Project points onto the plane by subtracting the distance vector
        projected_points = points - distance_vectors

        return projected_points

    def find_points_on_plane(self, points, dist_to_plane=0.01):
        """
        Find points that lie on or near the plane within a specified distance threshold.

        Parameters:
            points (numpy.ndarray): Array of 3D points of shape (n, 3).
            dist_to_plane (float, optional): Maximum distance to consider a point as being on the plane.
                                           Default is 0.01.

        Returns:
            numpy.ndarray: Array of points that lie on or near the plane.

        Raises:
            ValueError: If the plane parameters are not set.
        """
        if self.params is None:
            raise ValueError("Plane parameters are not set. Fit a plane first.")

        # Calculate distances to the plane
        distances = self.calculate_distances(points)

        # Select points where absolute distance is less than the threshold
        plane_points_indices = np.where(np.abs(distances) <= dist_to_plane)[0]

        # Return selected points
        return points[plane_points_indices]

    def refine_with_points(self, points, dist_to_plane=0.01, ransac_threshold=0.001,
                           ransac_iterations=1000, fine_tuning_iterations=5):
        """
        Refine the plane parameters using points that lie near the current plane.

        This method iteratively:
        1. Finds points close to the current plane
        2. Fits a new plane to these points
        3. Repeats the process to refine the plane estimate

        Parameters:
            points (numpy.ndarray): Array of 3D points of shape (n, 3).
            dist_to_plane (float, optional): Maximum distance to consider a point as being on the plane.
                                           Default is 0.01.
            ransac_threshold (float, optional): Threshold for RANSAC algorithm. Default is 0.001.
            ransac_iterations (int, optional): Number of iterations for RANSAC algorithm. Default is 1000.
            fine_tuning_iterations (int, optional): Number of refinement iterations. Default is 5.

        Returns:
            tuple: A tuple containing:
                - self: The Plane object with updated params.
                - plane_points (numpy.ndarray): Points that lie on the refined plane.

        Raises:
            ValueError: If the plane parameters are not set.
        """
        if self.params is None:
            raise ValueError("Plane parameters are not set. Fit a plane first.")

        # Distance will be positive or negative depends on which side of the plane it is
        distances = self.calculate_distances(points)

        # Select indexes where distance is less than the threshold
        plane_points_indices = np.where(np.abs(distances) <= dist_to_plane)[0]

        # Select points using their index
        plane_points = points[plane_points_indices]

        # Points to fit the plane to in the first iteration
        points_to_fit_plane = plane_points

        for i in range(fine_tuning_iterations):
            # If there are enough points on the plane
            if points_to_fit_plane.shape[0] > 50:
                # Fit a plane to the selected points using RANSAC
                self.fit(points_to_fit_plane, ransac_threshold, ransac_iterations)

                # Calculate distances again with the refined plane
                distances = self.calculate_distances(points)

                # Select indexes where distance is less than the threshold
                plane_points_indices = np.where(np.abs(distances) <= dist_to_plane)[0]

                # Select points using their index
                plane_points = points[plane_points_indices]

                # Update points to fit the plane for the next iteration
                points_to_fit_plane = plane_points

        return self, plane_points

    @classmethod
    def fit_from_point(cls, points, picked_point, selection_radius, dist_to_plane=0.01,
                       ransac_threshold=0.001, ransac_iterations=1000, fine_tuning_iterations=5):
        """
        Fit a plane to points near a selected point and refine it.

        This method:
        1. Selects points within a radius of the picked point
        2. Fits an initial plane to these points
        3. Refines the plane with all provided points

        Parameters:
            points (numpy.ndarray): Array of 3D points of shape (n, 3).
            picked_point (numpy.ndarray): A 3D point around which to find plane points.
            selection_radius (float): Radius around picked_point to select initial points.
            dist_to_plane (float, optional): Maximum distance to consider a point as being on the plane.
                                           Default is 0.01.
            ransac_threshold (float, optional): Threshold for RANSAC algorithm. Default is 0.001.
            ransac_iterations (int, optional): Number of iterations for RANSAC algorithm. Default is 1000.
            fine_tuning_iterations (int, optional): Number of refinement iterations. Default is 5.

        Returns:
            tuple: A tuple containing:
                - plane (Plane): A new Plane object fitted to the points.
                - plane_points (numpy.ndarray): Points that lie on the fitted plane.
        """
        # Calculate distances to the picked point
        dists_to_point = cls.calculate_point_distances(points, picked_point)

        # Select points within the selection radius
        selected_points_indices = np.where(dists_to_point <= selection_radius)[0]
        points_to_fit_plane = points[selected_points_indices]

        # Create a new plane object
        plane = cls()

        # Fit a plane to the selected points and get inliers
        plane.fit(points_to_fit_plane, ransac_threshold, ransac_iterations)

        # Refine the plane with all points
        plane, plane_points = plane.refine_with_points(
            points, dist_to_plane, ransac_threshold, ransac_iterations, fine_tuning_iterations
        )

        return plane, plane_points

    @staticmethod
    def calculate_point_distances(points, reference_point):
        """
        Calculate the Euclidean distances from points to a reference point.

        Parameters:
            points (numpy.ndarray): Array of 3D points of shape (n, 3).
            reference_point (numpy.ndarray): A 3D point to calculate distances from.

        Returns:
            numpy.ndarray: Array of distances from each point to the reference point.
        """
        # Calculate vectors from reference point to each point
        vectors = points - reference_point

        # Calculate Euclidean distances (L2 norm)
        distances = np.linalg.norm(vectors, axis=1)

        return distances