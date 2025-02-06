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


def downsample_array(array, axis=0, ratio=0.5):
    """
    Randomly downsample an array along a specified axis by a given ratio.

    Parameters:
        array (np.ndarray): The array to be downsampled.
        axis (int): The axis along which to downsample the array.
        ratio (float): The fraction of the elements along the specified axis to retain.

    Returns:
        np.ndarray: A new array which is a downsampled version of the input array along the specified axis.

    Raises:
        ValueError: If the specified ratio is not between 0 and 1.
        IndexError: If the specified axis is not within the range of the array dimensions.

    Example usage:
        # Create a sample 3D array
        array_3d = np.random.rand(10, 10, 10)  # A 10x10x10 array of random floats
        # Downsample the array along the first axis (axis=0)
        downsampled_array = downsample_array(array_3d, axis=0, ratio=0.3)
        print("Original shape:", array_3d.shape)
        print("Downsampled shape:", downsampled_array.shape)

        Original shape: (10, 10, 10)
        Downsampled shape: (3, 10, 10)
    """
    if not (0 <= ratio <= 1):
        raise ValueError("Ratio must be between 0 and 1.")
    if not (-array.ndim <= axis < array.ndim):
        raise IndexError("Axis out of range for array dimensions.")

    # Normalize negative axis
    axis = axis if axis >= 0 else array.ndim + axis

    # Calculate the number of elements to keep
    num_elements_to_keep = int(array.shape[axis] * ratio)
    if num_elements_to_keep < 1:
        raise ValueError("The ratio is too small to keep any elements along the specified axis.")

    # Generate random indices along the specified axis
    indices = np.random.permutation(array.shape[axis])[:num_elements_to_keep]

    # Create a tuple of slices to index the array
    slice_tuple = tuple(
        indices if i == axis else slice(None)
        for i in range(array.ndim)
    )

    # Return the downsampled array
    return array[slice_tuple]


def pick_point(pcd):
    import open3d as o3d

    # Create Open3D object with VisualizerWithEditing to be able to select a point
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()

    # Select point using Shift + Left Click
    pickedPointIndex = vis.get_picked_points()
    vis.destroy_window()

    return pickedPointIndex


def order_points(points):
    """
    Order points in a counter-clockwise manner around their centroid.
    """
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def eigenvalues_to_colors(eigenvalues):
    # Normalize the eigenvalues row-wise
    normalized_eigenvalues = eigenvalues / np.sum(eigenvalues, axis=1, keepdims=True)

    # Calculate the differences
    differences = np.zeros_like(normalized_eigenvalues)
    differences[:, 0] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 1]  # First minus Second
    differences[:, 1] = normalized_eigenvalues[:, 0] - normalized_eigenvalues[:, 2]  # First minus Third
    differences[:, 2] = normalized_eigenvalues[:, 2] - normalized_eigenvalues[:, 1]  # Third minus Second

    # Normalize the differences to [0, 1]
    min_vals = np.min(differences, axis=0)
    max_vals = np.max(differences, axis=0)
    scaled_differences = (differences - min_vals) / (max_vals - min_vals)
    return scaled_differences


def get_eigenvalues(data, k, smooth=True):
    """
    Compute the eigenvalues of the covariance matrix of k-nearest neighbor points for each point.

    This function supports input data in two forms:
    - A Clusters object, where it uses `data.points` for the points.
    - A numpy array directly representing 3D points.

    Parameters:
    - data (Clusters or np.ndarray): A Clusters object or a numpy array representing points.
    - k (int): The number of nearest neighbors to consider for each point.
    - smooth (bool, optional): If True, averages the eigenvalues across neighbors. Defaults to True.

    Returns:
    - np.ndarray: A 2D array of eigenvalues for each point's neighborhood.
    """
    # Extract points based on input type
    if isinstance(data, np.ndarray):
        point_cloud = data
    elif hasattr(data, 'points'):
        point_cloud = data.points
    else:
        raise ValueError("The first parameter must be a Clusters object or a numpy array.")

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

    # Use CPU for TensorFlow operations
    with tf.device('/CPU:0'):
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

        return avg_eigenvalues
    else:
        return eigenvalues


def get_obb_top_face(obb):
    """
    Extract and order the top face points from the given OBB.
    """
    # Extract the corner points from the OBB
    box_points = np.array(obb.get_box_points())
    # Sort the points based on Z-values and take the top 4 points
    top_points = sorted(box_points, key=lambda p: p[2], reverse=True)[:4]
    return order_points(np.array(top_points))


def get_obb_bottom_face(obb):
    """
    Extract and order the bottom face points from the given OBB.
    """
    # Extract the corner points from the OBB
    box_points = np.array(obb.get_box_points())
    # Sort the points based on Z-values and take the bottom 4 points
    bottom_points = sorted(box_points, key=lambda p: p[2])[:4]
    return order_points(np.array(bottom_points))


# Return the Oriented Bounding Box's height
# obb.extent returns the oriented bounding box size (width, Depth, Height), but in different order. So this function calculates
# the length of the projection of each side on the Z axis and assumes the longest projection on Z axis is hight.
# CAUTION: THIS METHOD JUST WORKS FOR CLUSTERS WITH OBB HEIGHT ALMOST PARRALEL TO Z AXIS.
def get_obb_dim(obb):
    # Compute the OBB's three main axes
    axis1 = obb.R[:, 0]
    axis2 = obb.R[:, 1]
    axis3 = obb.R[:, 2]

    # Project the axes onto world's Z-axis
    proj1 = np.abs(np.dot(axis1, [0, 0, 1]))
    proj2 = np.abs(np.dot(axis2, [0, 0, 1]))
    proj3 = np.abs(np.dot(axis3, [0, 0, 1]))

    # The height will correspond to the extent along the axis with the largest projection onto Z-axis
    if proj1 >= proj2 and proj1 >= proj3:
        i = 0
        height = obb.extent[i]
    elif proj2 >= proj1 and proj2 >= proj3:
        i = 1
        height = obb.extent[i]
    else:
        i = 2
        height = obb.extent[i]

    length = max(np.delete(obb.extent, i))
    width = min(np.delete(obb.extent, i))

    dim = np.array([length, width, height])

    return dim


# Return the order of Length, Width & Height of an Oriented Bounding Box
# obb.extent returns the oriented bounding box size (width, Depth, Height), but in different order. So this function calculates
# the length of the projection of each side on the Z axis and assumes the longest projection on Z axis is hight.
# CAUTION: THIS METHOD JUST WORKS FOR CLUSTERS WITH OBB HEIGHT ALMOST PARRALEL TO Z AXIS.
def get_obb_LWH_order(obb):
    # Compute the OBB's three main axes
    axis1 = obb.R[:, 0]
    axis2 = obb.R[:, 1]
    axis3 = obb.R[:, 2]

    # Project the axes onto world's Z-axis
    proj1 = np.abs(np.dot(axis1, [0, 0, 1]))
    proj2 = np.abs(np.dot(axis2, [0, 0, 1]))
    proj3 = np.abs(np.dot(axis3, [0, 0, 1]))

    # The height will correspond to the extent along the axis with the largest projection onto Z-axis
    if proj1 >= proj2 and proj1 >= proj3:
        l = np.argmax(np.delete(obb.extent, 0)) + 1
        w = np.argmin(np.delete(obb.extent, 0)) + 1
        h = 0
    elif proj2 >= proj1 and proj2 >= proj3:
        if np.argmax(np.delete(obb.extent, 1)) == 0:
            l = 0
            w = 2
            h = 1
        else:
            l = 2
            w = 0
            h = 1
    else:
        l = np.argmax(np.delete(obb.extent, 2))
        w = np.argmin(np.delete(obb.extent, 2))
        h = 2

    LWHOrder = np.array([l, w, h])

    return LWHOrder


def get_obb_euler_angles(obb):
    # Extract rotation matrix from OBB
    R = obb.R

    # Compute Euler angles based on the XYZ rotation sequence
    alpha = np.arctan2(R[2, 1], R[2, 2])
    beta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    gamma = np.arctan2(R[1, 0], R[0, 0])

    return alpha, beta, gamma


def add_rectangle_from_obb_to_DXF(doc, obb, layer, color, global_width, offset=(0, 0, 0)):
    """
    Add a rectangle to a given DXF document based on the bottom face of an OBB, with a specified global width.

    Args:
    - doc:   The DXF document.
    - obb:   The oriented bounding box.
    - layer: The layer.
    - color: The color.
    - global_width: The global width to set for the polyline.
    - offset: An optional XYZ offset for the rectangle.
    """
    # Create the "Seat" layer only if it doesn't exist.
    if layer not in doc.layers:
        doc.layers.new(name=layer, dxfattribs={'color': color})

    # Get the MSP (ModelSpace) which is the space where we draw our entities.
    msp = doc.modelspace()

    topPoints = get_obb_top_face(obb)
    bottomPoints = get_obb_bottom_face(obb)

    p1 = topPoints[0] + np.array(offset)
    p2 = topPoints[1] + np.array(offset)
    p3 = topPoints[2] + np.array(offset)
    p4 = topPoints[3] + np.array(offset)

    z_elevation = bottomPoints[0][2]

    # Create a 2D polyline from the points with the specified global width.
    polyline = msp.add_lwpolyline(
        [p1, p2, p3, p4],
        dxfattribs={
            'layer': layer,
            'elevation': z_elevation,
            'const_width': global_width  # Set the global width here
        }
    )
    polyline.closed = True

    return polyline  # It's often a good practice to return the created entity, in case further manipulation is required outside this function.


def add_circle_from_obb_to_DXF(doc, obb, layer, color, offset=(0, 0, 0)):
    """
    Add a circle to a given DXF document based on the bottom face of an OBB.

    Args:
    - doc:   The DXF document.
    - obb:   The oriented bounding box.
    - layer: The layer.
    - color: The color.
    """
    # Create the "Seat" layer only if it doesn't exist.
    if layer not in doc.layers:
        doc.layers.new(name=layer, dxfattribs={'color': color})

    # Get the MSP (ModelSpace) which is the space where we draw our entities.
    msp = doc.modelspace()

    # Extract obb information
    obbDim = get_obb_dim(obb)
    R = obb.R
    center = np.array(obb.center)
    extent = np.array(obb.extent)

    # Compute the center and sides lengths of the bottom face
    center_bottom = center - [0, 0, obbDim[2] / 2] + np.array(offset)
    side_lengths = obbDim[:2]  # The lengths of the sides are the x and y extents of the OBB
    radius = side_lengths.mean() / 2

    # In a 2D CAD drawing, we just consider x, y coordinates
    msp.add_circle(center=(center_bottom[0], center_bottom[1], center_bottom[2]), radius=radius,
                   dxfattribs={'layer': layer})


def add_euler_angles_to_dxf(doc, obb):
    """
    Add the Euler angles of an OBB to a DXF document.

    Parameters:
    - doc: The DXF document to add the text entities to.
    - obb: The oriented bounding box.

    Returns:
    - The modified DXF document.
    """
    # Retrieve the euler angles
    angles = get_obb_euler_angles(obb)

    # Names of the axes
    axes_names = ['X', 'Y', 'Z']

    # Modelspace is the "canvas" where the drawing will be added
    msp = doc.modelspace()

    # Center of the OBB
    center = obb.center

    # Add each angle as a text entity to the DXF
    for i, (angle, axis_name) in enumerate(zip(angles, axes_names)):
        msp.add_text(
            f"{axis_name}-axis: {angle:.2f}°",
            dxfattribs={
                'insert': (center[0], center[1] - i * 0.25, center[2]),  # Move down by 1.5 units for each new text
                'height': 0.2  # Text height, adjust as needed
            }
        )

    return doc


# Jitter the points in a point cloud
def jitter(ClusterPoints, num_required_points, tolerance):
    num_points = ClusterPoints.shape[0]

    # calculate how many additional points we need
    num_additional_points = num_required_points - num_points

    if num_additional_points > 0:
        # randomly choose indices from your points
        random_indices = np.random.choice(num_points, num_additional_points)

        # duplicate these points
        additional_points = ClusterPoints[random_indices, :]

        # add a small random displacement (jitter)
        jitter = np.random.uniform(-tolerance, tolerance, size=additional_points.shape)
        additional_points += jitter

        # concatenate the original points with the new jittered points
        final_points = np.concatenate([ClusterPoints, additional_points], axis=0)
    else:
        # if there are enough points, just return original points
        final_points = ClusterPoints

    return final_points


def add_block_ref_from_obb_to_DXF(doc, obb, block_name, layer, color, offset=(0, 0, 0), same_ratio=True):
    """
    Insert a block into a given DXF document based on the bottom face of an OBB.
    The block will be scaled and aligned to match the OBB's dimensions and orientation.

    Args:
    - doc:       The DXF document.
    - obb:       The oriented bounding box.
    - block_name: The name of the block to insert.
    - layer:     The layer.
    - color:     The color to assign to the layer.
    - offset:    An optional XYZ offset for the block insertion.
    - same_ratio: Boolian, True for same X & Y scale
    """

    # Create the "Seat" layer only if it doesn't exist.
    if layer not in doc.layers:
        doc.layers.new(name=layer, dxfattribs={'color': color})

    # Get the MSP (ModelSpace) which is the space where we draw our entities.
    msp = doc.modelspace()

    # Extract obb information
    obbDim = get_obb_dim(obb)
    R = obb.R
    center = np.array(obb.center)
    extent = np.array(obb.extent)

    # Compute the center of the bottom face
    center_bottom = center - [0, 0, obbDim[2] / 2]

    # Apply the offset
    center_with_offset = center_bottom + np.array(offset)

    # Get the block's original dimensions
    block = doc.blocks.get(block_name)

    # TO DO: Get the size of block to be able to insert with the same size of OBB
    block_dim = [1, 1, 1]

    # Compute the scaling factors
    scale_x = (obbDim[0] / block_dim[0]) / 2
    scale_y = (obbDim[1] / block_dim[1]) / 2
    scale_z = (obbDim[2] / block_dim[2]) / 2

    if same_ratio:
        scale_x = (scale_x + scale_y) / 2
        scale_y = scale_x

    # Find the longest axis of the OBB
    longest_axis_idx = np.argmax(extent)
    longest_axis = R[:, longest_axis_idx]

    # Project the longest axis onto the XY plane
    longest_axis_xy = np.array([longest_axis[0], longest_axis[1], 0])

    if not np.all(longest_axis_xy == 0):
        # Normalize the projected axis
        longest_axis_xy = longest_axis_xy / np.linalg.norm(longest_axis_xy)

        # Compute the angle between the projected axis and the X-axis
        dot_product = np.dot(longest_axis_xy, [1, 0, 0])
        magnitude_product = np.linalg.norm(longest_axis_xy) * np.linalg.norm([1, 0, 0])
        cos_theta = dot_product / magnitude_product

        # Compute the angle in radians and convert it to degrees
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        theta_deg = np.degrees(theta_rad)

        # Check the direction of rotation and adjust the angle accordingly
        if np.cross(longest_axis_xy, [1, 0, 0])[2] < 0:
            theta_deg = -theta_deg
    else:
        theta_deg = 0

    # Insert the block with scaling and alignment
    blockRef = msp.add_blockref(block_name,
                                insert=(center_with_offset[0], center_with_offset[1], center_with_offset[2]),
                                dxfattribs={
                                    'xscale': scale_x,
                                    'yscale': scale_y,
                                    'zscale': scale_z,
                                    'rotation': -theta_deg,
                                    'layer': layer
                                })
    return blockRef


def add_xdata_to_entity(entity, app_name, data_dict):
    """
    Add XData to a DXF entity.

    Args:
    - entity: The DXF entity to which XData will be added.
    - app_name: The application name registering the XData (must be unique).
    - data_dict: A dictionary containing key-value pairs of data.
    """
    # Ensure the application name is registered in the DXF document
    doc = entity.doc  # Get the document to which the entity belongs
    if app_name not in doc.appids:
        doc.appids.new(app_name)

    # Start adding XData with the application name
    xdata = [(1001, app_name)]

    # Add key-value pairs from the data dictionary
    for key, value in data_dict.items():
        # XData strings are represented by group code 1000
        xdata.append((1000, f"{key}: {value}"))

    # Add the XData to the entity
    entity.set_xdata(app_name, xdata)

    return entity


def change_layer_color(doc, layer_name, new_color):
    """
    Change the color of a specified layer in a given DXF document.

    Args:
    - doc:        The DXF document.
    - layer_name: Name of the layer whose color you want to change.
    - new_color:  The new color value (as an integer). It can be a value between 1 and 255 inclusive,
                  representing the AutoCAD color index (ACI).
    """
    # Check if the layer exists
    if layer_name in doc.layers:
        layer = doc.layers.get(layer_name)
        layer.color = new_color
    else:
        print(f"Layer '{layer_name}' not found in the document.")


def get_random_color(cluster_labels: np.ndarray, noise_color: np.ndarray = [0.2, 0.2, 0.2]):
    """
    Generates a random RGB color for each unique labels in cluster_labels.

    Args:
        cluster_labels (np.array): An array of cluster labels.
        noise_color (np.array): The color to assign to noise points (default is dark gray).

    Returns:
        np.array: An array of RGB colors corresponding to the cluster labels.
    """
    # Get unique labels and their indexes
    unique_labels, label_indexes = np.unique(cluster_labels, return_inverse=True)

    # Generate random colors for all labels (including -1)
    np.random.seed(42)  # Optional: Set a seed for reproducibility
    cluster_colors = np.random.rand(len(unique_labels), 3)

    # Assign a color for noise (-1 labels)
    noise_labels = unique_labels == -1
    cluster_colors[noise_labels] = noise_color

    # Assign colors to each point based on their label_indexes
    point_colors = cluster_colors[label_indexes]

    return point_colors


def normalize_point_cloud(point_cloud, scale=True, random_rotation=False):
    """
    Normalize the input point cloud.

    :param point_cloud: np.ndarray
        The input point cloud of shape (n, 3), where n is the number of points.
    :param scale: bool
        Whether to scale the point cloud to the unit sphere.
    :param random_rotation: bool
        Whether to apply a random rotation around the z-axis to the point cloud.

    :return: np.ndarray
        The normalized point cloud of shape (n, 3).
    """

    # Center the point cloud
    centroid = np.mean(point_cloud, axis=0)
    point_cloud_centered = point_cloud - centroid

    if scale:
        # Scale the point cloud
        max_distance = np.max(np.sqrt(np.sum(point_cloud_centered ** 2, axis=1)))
        point_cloud_centered /= max_distance

    if random_rotation:
        # Set the seed using the current timestamp
        np.random.seed(int(time.time()))

        # Generate a random angle between 0 and 2*pi
        angle = np.random.uniform(0, 2 * np.pi)

        # Define the rotation matrix around the z-axis
        rotation_matrix = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])

        # Apply the rotation to the point cloud
        point_cloud_centered = np.matmul(rotation_matrix, point_cloud_centered.T).T

    return point_cloud_centered


def read_excel_to_nested_dict(file_path):
    """
    Read an Excel file and convert each sheet to a nested dictionary.
    - Main Key: Feature name.
    - Sub Key: Worksheet name, pointing to a dictionary of properties for that feature.
    """
    # Create an ExcelFile object to read multiple sheets
    xls = pd.ExcelFile(file_path)

    # Initialize the main dictionary
    main_dict = {}

    # Iterate over each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = xls.parse(sheet_name, index_col=0)

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # If the feature already exists, update the dictionary directly
            if index in main_dict:
                main_dict[index][sheet_name] = row.to_dict()
            else:
                # Otherwise, create a new sub-dictionary with the worksheet name as the key
                main_dict[index] = {sheet_name: row.to_dict()}

    return main_dict


def size_check(pcdCluster, predictionLabel, dic):
    """
    This function checks whether the size of the provided point cloud cluster
    is within the predefined range specified in the dictionary for a given labels.

    :param pcdCluster: The point cloud cluster for which the size is to be checked.
                       Expected to be an instance of open3d.geometry.PointCloud or equivalent.
    :param predictionLabel: The labels predicted by the model for the given point cloud cluster.
                            Expected to be a key in the provided dictionary.
    :param dic: A dictionary containing the acceptable size ranges for different labels.
                The keys are expected to be the possible values of predictionLabel.
                Each entry should be another dictionary with keys 'Min_Length', 'Max_Length',
                'Min_Width', 'Max_Width', 'Min_Height', 'Max_Height', each corresponding to
                the minimum or maximum acceptable size along a specific axis.

    :return: A boolean value, True if the size of the point cloud cluster is within the acceptable
             range for the given labels, False otherwise.
    """

    sizeCheck = False

    # Compute oriented bounding box for the given point cloud cluster.
    obb = pcdCluster.get_minimal_oriented_bounding_box(robust=True)

    # Compute and retrieve the dimensions of the oriented bounding box.
    obbDim = get_obb_dim(obb)

    # Retrieve the length, width, and height of the oriented bounding box.
    L = obbDim[0]
    W = obbDim[1]
    H = obbDim[2]

    # Retrieve the acceptable size range for the given prediction labels from the dictionary.
    morph = dic[predictionLabel]["Morphology"]
    minL = morph["Min_Length"]
    maxL = morph["Max_Length"]
    minW = morph["Min_Width"]
    maxW = morph["Max_Width"]
    minH = morph["Min_Height"]
    maxH = morph["Max_Height"]

    # Check whether the size of the point cloud cluster is within the acceptable range.
    if minL < L < maxL and minW < W < maxW and minH < H < maxH:
        sizeCheck = True

    return sizeCheck


def get_obb_corners(obb):
    """
    Calculate the rotated corners of an Oriented Bounding Box (OBB) from Open3D.

    :param obb: open3d.geometry.OrientedBoundingBox object.
    :return: np.ndarray representing the 3D coordinates of the 8 corners.
    """
    # Extract the center, extents, and rotation matrix from the OBB
    center = np.asarray(obb.center)
    extents = np.asarray(obb.extent) / 2  # half the size of the OBB in each dimension
    R = np.asarray(obb.R)

    # Create an array of size 8x3 (for 8 corners, each with 3 coordinates)
    corners = np.zeros((8, 3))
    for i in range(8):
        sign = [-1 if i & 1 else 1, -1 if i & 2 else 1, -1 if i & 4 else 1]
        corner = center + np.dot(R, np.multiply(extents, sign))  # rotate the signed extent and then add to center
        corners[i] = corner

    return corners


def obb_size_check(obb, predictionLabel, dic):
    """
    This function checks whether the size of the provided oriented bounding box
    is within the predefined range specified in the dictionary for a given labels.

    :param obb: The point cloud cluster oriented bounding box for which the size is to be checked.
                       Expected to be an instance of open3d oriented bounding box or equivalent.
    :param predictionLabel: The labels predicted by the model for the given point cloud cluster.
                            Expected to be a key in the provided dictionary.
    :param dic: A dictionary containing the acceptable size ranges for different labels.
                The keys are expected to be the possible values of predictionLabel.
                Each entry should be another dictionary with keys 'Min_Length', 'Max_Length',
                'Min_Width', 'Max_Width', 'Min_Height', 'Max_Height', each corresponding to
                the minimum or maximum acceptable size along a specific axis.

    :return: A boolean value, True if the size of the point cloud cluster is within the acceptable
             range for the given labels, False otherwise.
    """

    sizeCheck = False

    # Compute and retrieve the dimensions of the oriented bounding box.
    obbDim = get_obb_dim(obb)

    # Retrieve the length, width, and height of the oriented bounding box.
    L = obbDim[0]
    W = obbDim[1]
    H = obbDim[2]

    # Retrieve the acceptable size range for the given prediction labels from the dictionary.
    morph = dic[predictionLabel]["Morphology"]
    minL = morph["Min_Length"]
    maxL = morph["Max_Length"]
    minW = morph["Min_Width"]
    maxW = morph["Max_Width"]
    minH = morph["Min_Height"]
    maxH = morph["Max_Height"]

    # Check whether the size of the point cloud cluster is within the acceptable range.
    if minL < L < maxL and minW < W < maxW and minH < H < maxH:
        sizeCheck = True

    return sizeCheck


def elev_check(pcdCluster, pcdGround, predictionLabel, dic):
    """
    This function checks whether the elevation of the provided point cloud cluster
    above the ground is within the predefined range specified in the dictionary for
    a given labels.

    :param pcdCluster: The point cloud cluster for which the elevation is to be checked.
                       Expected to be an instance of open3d.geometry.PointCloud or equivalent.
    :param pcdGround: The point cloud representing the ground.
                      Expected to be an instance of open3d.geometry.PointCloud or equivalent.
    :param predictionLabel: The labels predicted by the model for the given point cloud cluster.
                            Expected to be a key in the provided dictionary.
    :param dic: A dictionary containing the acceptable elevation ranges for different labels.
                The keys are expected to be the possible values of predictionLabel.
                Each entry should be another dictionary with keys 'Min_Elev' and 'Max_Elev',
                corresponding to the minimum and maximum acceptable elevation above the ground.

    :return: A boolean value, True if the elevation of the point cloud cluster is within the
             acceptable range for the given labels, False otherwise.
    """

    elevCheck = False

    # Compute the minimum distance from the point cloud cluster to the ground,
    # which serves as the elevation of the cluster above the ground.
    distToGround = min(pcdCluster.compute_point_cloud_distance(pcdGround))

    # Retrieve the acceptable elevation range for the given prediction labels from the dictionary.
    morph = dic[predictionLabel]["Morphology"]
    minElev = morph["Min_Elev"]
    maxElev = morph["Max_Elev"]

    # Check whether the elevation of the point cloud cluster is within the acceptable range.
    if minElev < distToGround < maxElev:
        elevCheck = True

    return elevCheck


def obb_elev_check(obb, pcdGround, predictionLabel, dic):
    """
    This function checks whether the elevation of the provided point cloud cluster
    above the ground is within the predefined range specified in the dictionary for
    a given labels.

    :param pcdCluster: The point cloud cluster for which the elevation is to be checked.
                       Expected to be an instance of open3d.geometry.PointCloud or equivalent.
    :param pcdGround: The point cloud representing the ground.
                      Expected to be an instance of open3d.geometry.PointCloud or equivalent.
    :param predictionLabel: The labels predicted by the model for the given point cloud cluster.
                            Expected to be a key in the provided dictionary.
    :param dic: A dictionary containing the acceptable elevation ranges for different labels.
                The keys are expected to be the possible values of predictionLabel.
                Each entry should be another dictionary with keys 'Min_Elev' and 'Max_Elev',
                corresponding to the minimum and maximum acceptable elevation above the ground.

    :return: A boolean value, True if the elevation of the point cloud cluster is within the
             acceptable range for the given labels, False otherwise.
    """

    elevCheck = False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(get_obb_corners(obb))

    # Compute the minimum distance from the point cloud cluster to the ground,
    # which serves as the elevation of the cluster above the ground.
    distToGround = min(pcd.compute_point_cloud_distance(pcdGround))

    # Retrieve the acceptable elevation range for the given prediction labels from the dictionary.
    morph = dic[predictionLabel]["Morphology"]
    minElev = morph["Min_Elev"]
    maxElev = morph["Max_Elev"]

    # Check whether the elevation of the point cloud cluster is within the acceptable range.
    if minElev < distToGround < maxElev:
        elevCheck = True

    return elevCheck


def compute_projection_angle_to_x_axis(pcd):
    """
    Compute the angle between the projection of the longest axis of the
    point cloud's oriented bounding box (OBB) onto the XY-plane and the
    positive direction of the X-axis.

    Parameters:
        pcd (o3d.geometry.PointCloud): The input point cloud.

    Returns:
        float: The angle in degrees between the projected longest axis
               and the X-axis. The angle is positive if the rotation is
               counter-clockwise when viewed from above the XY-plane;
               otherwise, it is negative.

    Note:
        Ensure the input point cloud (pcd) is non-empty and valid.
    """
    # Compute the oriented bounding box of the point cloud
    obb = pcd.get_oriented_bounding_box()

    # Extracting the axes of the OBB and identifying the longest axis
    axes = obb.R
    longest_axis_idx = np.argmax(obb.extent)
    longest_axis = axes[:, longest_axis_idx]

    # Projecting the longest axis onto the XY-plane by setting the z-component to 0
    longest_axis_projected = np.array([longest_axis[0], longest_axis[1], 0])

    # Normalizing the projected axis to have a unit length
    longest_axis_projected = longest_axis_projected / np.linalg.norm(longest_axis_projected)

    # Calculating the angle between the projected axis and X-axis
    dot_product = np.dot(longest_axis_projected, [1, 0, 0])
    magnitude_product = np.linalg.norm(longest_axis_projected) * np.linalg.norm([1, 0, 0])
    cos_theta = dot_product / magnitude_product

    # Calculating the angle in radians and converting it to degrees
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # Determining the direction of rotation
    cross_product = np.cross(longest_axis_projected, [1, 0, 0])
    rotation_direction = np.sign(cross_product[2])

    # Adjusting the angle based on the rotation direction
    angle *= rotation_direction

    return angle


def get_prediction(clusterPoints, model, NUM_POINTS, MIN_NUM_POINTS):
    """
    This function predicts the labels for a single cluster of points using a specified model.

    Args:
    clusterPoints (ndarray): The points in the cluster.
    model (Model): The machine learning model to use for prediction.
    NUM_POINTS (int): The number of points to use for prediction.
    MIN_NUM_POINTS (int): The minimum number of points required for a valid prediction.

    Returns:
    tuple: A tuple containing the predicted labels and the maximum probability.
    """
    # Shuffle the points in the cluster
    np.random.shuffle(clusterPoints)

    # Get the number of points in the cluster
    num_points = clusterPoints.shape[0]

    # Check if the number of points in the cluster is within the specified range
    if MIN_NUM_POINTS < num_points < NUM_POINTS:
        # Augment the training set if the number of points is less than required
        for i in range(int(NUM_POINTS / clusterPoints.shape[0]) + 1):
            # Add noise to the cluster points
            noise = np.random.normal(0, 0.005, clusterPoints.shape)
            noisyClusterPoints = clusterPoints + noise
            np.random.shuffle(noisyClusterPoints)
            # Combine original and noisy points
            augmentedClusterPoints = np.vstack((clusterPoints, noisyClusterPoints))
            clusterPoints = augmentedClusterPoints
    elif clusterPoints.shape[0] < MIN_NUM_POINTS:
        # If the cluster is too small, return None
        print("------------------- TOO SMALL CLUSTER ---------------------")
        return None, None

    # Shuffle the points again
    np.random.shuffle(clusterPoints)

    # Normalize the cluster points by moving the center to the origin
    clusterPoints = normalize_point_cloud(clusterPoints, False, False)

    num_points = clusterPoints.shape[0]

    # Randomly select NUM_POINTS indices from the cluster points
    random_indices = np.random.choice(num_points, NUM_POINTS, replace=False)

    # Retrieve the points corresponding to the selected indices
    points = clusterPoints[random_indices, :]

    # Reshape the points and convert them to a tensor
    points = points.reshape(1, NUM_POINTS, 3)
    points = tf.convert_to_tensor(points, dtype=tf.float64)

    # Make a prediction using the model
    probabilities = model.predict(points)
    probability = np.max(probabilities)

    # Retrieve the predicted labels with the highest probability
    prediction = tf.math.argmax(probabilities, -1)

    # Return the prediction and the maximum probability
    return prediction, probability


def get_predictions(clustersPoints, clusterLabels, model, NUM_POINTS, MIN_NUM_POINTS):
    """
    This function predicts the labels for clusters of points using a specified model.

    Args:
    clustersPoints (ndarray): The points in each cluster.
    clusterLabels (ndarray): The labels for each cluster.
    model (Model): The machine learning model to use for prediction.
    NUM_POINTS (int): The number of points to use for prediction.
    MIN_NUM_POINTS (int): The minimum number of points required for a valid prediction.

    Returns:
    list: A list of predictions, each prediction includes:
                                            cluster labels,
                                            predicted labels,
                                            probability,
                                            number of points in the cluster,
                                            oriented bounding box,
                                            oob rotation matrix
    """
    # Initialize the list to store predictions for each cluster.
    dicClustersPrediction = {}

    # Identify unique cluster labels.
    unique_labels = np.unique(clusterLabels)

    # Iterate through each unique cluster labels.
    for clusterLabel in unique_labels:
        # Gather points corresponding to the current cluster labels.
        clusterPoints = clustersPoints[(clusterLabels == clusterLabel)]

        # Predict the labels for the current cluster of points.
        # Invalid clusters return 'None' for prediction and probability
        prediction, probability = get_prediction(clusterPoints, model, NUM_POINTS, MIN_NUM_POINTS)

        # Proceed only if a valid prediction is returned.
        if prediction is not None:
            # Create a point cloud from the cluster points.
            pcdCluster = o3d.geometry.PointCloud()
            pcdCluster.points = o3d.utility.Vector3dVector(clusterPoints)

            # Calculate the minimal oriented bounding box for the point cloud.
            obb = pcdCluster.get_minimal_oriented_bounding_box(robust=True)

            # Retrieve the dimensions of the oriented bounding box.
            obbDim = get_obb_dim(obb)

            # Prepare the prediction data for the current cluster.
            prediction_data = [prediction, probability, clusterPoints.shape[0], obb, obb.R]

            # Append the prediction data to the list of predictions.
            dicClustersPrediction[clusterLabel] = prediction_data
    # Return the list of predictions for all clusters.
    return dicClustersPrediction


def add_feature_to_dxf(doc, obb, CADFeature, clusterMetada):
    """
    This function adds a specific feature to a DXF document based on its type (Block or Rectangle).

    Args:
    doc (Document): The DXF document to which the feature will be added.
    obb (OrientedBoundingBox): The oriented bounding box related to the feature.
    CADFeature (dict): A dictionary containing feature details.
    clusterMetada (dict): Metadata related to the cluster.

    Returns:
    None: The function doesn't return anything but modifies the 'doc' by adding the feature to it.
    """

    # Determine the type of the CAD feature by getting the first key in the CADFeature dictionary
    entity_type = list(CADFeature.keys())[0]

    # Check if the feature is a Block
    if entity_type == "Block":
        # Get the Block reference details from the CADFeature dictionary
        block_ref = CADFeature["Block"]
        # Add the Block to the DXF document with specific properties and orientation based on the OBB
        entity = add_block_ref_from_obb_to_DXF(doc,
                                               obb,
                                               block_ref["Block_Name"],
                                               block_ref["Layer"],
                                               block_ref["Color"],
                                               (0, 0, 0),
                                               # This seems to be a default offset; ensure it's what you want
                                               block_ref["Same_Ratio"]
                                               )

        # Add extra data (XDATA) to the Block entity for storing metadata
        add_xdata_to_entity(entity, "Cluster_Metadata", clusterMetada)

        # Check if the feature is a Rectangle
    elif entity_type == "Rectangle":
        # Get the Rectangle details from the CADFeature dictionary
        rectangle = CADFeature["Rectangle"]
        # Add the Rectangle to the DXF document with specific properties and orientation based on the OBB
        entity = add_rectangle_from_obb_to_DXF(doc,
                                               obb,
                                               rectangle["Layer"],
                                               rectangle["Color"],
                                               rectangle["Global_Width"],
                                               (0, 0, 0)  # This seems to be a default offset; ensure it's what you want
                                               )

        # Add extra data (XDATA) to the Rectangle entity for storing metadata
        add_xdata_to_entity(entity, "Cluster_Metadata", clusterMetada)

    # The function ends here and doesn't return anything
    return


def get_slice(clustersPoints, z_diffs, bottomDistanceToGround, topDistanceToGround):
    """
    This function filters points in a cluster based on their Z-axis differences (heights).

    Args:
    clustersPoints (array-like): The points in the clusters, presumably in a 3D space.
    z_diffs (array-like): The distance to the ground for each point in the clusters.
    bottomDistanceToGround (float): The minimum distance from the ground to include a point.
    topDistanceToGround (float): The maximum distance from the ground to include a point.

    Returns:
    array-like: A subset of clustersPoints, including only the points that meet the height criteria.
    """

    # Create a boolean mask: True if a point's Z-axis difference is within the specified range, False otherwise.
    # The '&' operator is used here for element-wise logical AND operation, comparing each element in 'z_diffs'
    # with 'bottomDistanceToGround' and 'topDistanceToGround'.
    z_diffs_mask = (z_diffs >= bottomDistanceToGround) & (z_diffs <= topDistanceToGround)

    # Return the points in the clusters that meet the criteria, using the boolean mask to filter them.
    return clustersPoints[z_diffs_mask]


def o3d_dbscan(clustersPoints, eps, min_points):
    """
    Perform DBSCAN clustering on points using the Open3D library.

    Args:
    clustersPoints (np.ndarray): Points to cluster, should be an Nx3 array where N is the number of points.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_points (int): The number of points in a neighborhood for a point to be considered as a core point. This includes the point itself.

    Returns:
    np.ndarray: An array where the ith entry is the cluster labels of the ith input point. Noise points are labeled as -1.
    """

    # Create an Open3D PointCloud object
    pcd_Sliced = o3d.geometry.PointCloud()
    # Assign points to the PointCloud object
    pcd_Sliced.points = o3d.utility.Vector3dVector(clustersPoints)

    # Set the verbosity level to debug to output detailed information during DBSCAN
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # Perform DBSCAN clustering from the point cloud, this might take some time depending on the number of points
        # 'eps' is the maximum distance between two points to be considered in the same cluster
        # 'min_points' is the minimum number of points to form a valid cluster
        # 'print_progress' shows a progress bar, useful for large datasets
        labels_Sliced = np.array(pcd_Sliced.cluster_dbscan(eps, min_points, print_progress=False))

    # Return the labels of the points, each labels represents the cluster index each point belongs to
    return labels_Sliced


def open_file_dialog(initialDir=''):
    import tkinter as tk
    from tkinter import filedialog

    dialog = tk.Tk()
    dialog.withdraw()

    dialog.attributes('-topmost', True)

    file_paths = tk.filedialog.askopenfilenames(initialdir=initialDir)

    return file_paths


def save_file_dialog(initialDir=''):
    import tkinter as tk
    from tkinter import filedialog

    dialog = tk.Tk()
    dialog.withdraw()

    dialog.attributes('-topmost', True)

    file_paths = tk.filedialog.asksaveasfilename(initialdir=initialDir)

    return file_paths


def get_user_input(userInput='User input: '):
    import tkinter as tk
    from tkinter import simpledialog

    dialog = tk.Tk()
    dialog.attributes('-topmost', True)
    dialog.withdraw()

    answer = simpledialog.askstring("Input", userInput, parent=dialog)

    return answer


def voxelise(point_clouds, voxel_size):
    """
    Divides the bounding box of one or multiple point clouds by user-specified voxel sizes and returns the spatial index
    of each point in each point cloud.

    Parameters:
    - point_clouds (np.ndarray or list of np.ndarray): A single point cloud array or a list of point cloud arrays.
    - voxel_size (float or tuple/list of floats): Voxel dimensions. If a single float is provided, it's applied to all axes.
      Otherwise, use a 3-tuple or list for individual dimensions.

    Returns:
    - list of np.ndarray or np.ndarray: A list containing the voxel indices for each point cloud, or a single numpy array if only one point cloud is provided.
    """
    # Ensure voxel_size is in the form of (x, y, z)
    if isinstance(voxel_size, (float, int)):
        voxel_size = [voxel_size] * 3
    elif isinstance(voxel_size, (list, tuple)) and len(voxel_size) != 3:
        raise ValueError(f"Expected 3 voxel dimensions, but got {len(voxel_size)}")

    # Convert voxel_size to a numpy array
    voxel_size = np.array(voxel_size)

    # If a single numpy array is provided, convert to a list for consistent processing
    is_single = isinstance(point_clouds, np.ndarray)
    if is_single:
        point_clouds = [point_clouds]

    # Initialize variables to track the global bounding box
    bb_min = np.full(3, np.inf)
    bb_max = np.full(3, -np.inf)

    # Iterate through each point cloud to find the global bounding box
    for points in point_clouds:
        bb_min = np.minimum(bb_min, np.min(points, axis=0))
        bb_max = np.maximum(bb_max, np.max(points, axis=0))

    bb_size = bb_max - bb_min

    # Determine the number of voxels in each axis
    voxels = (bb_size / voxel_size).astype(int) + 1

    # Calculate the scaling factor
    scale_factor = 1 / voxel_size

    # Initialize a list to store the voxel indices for each point cloud
    indices_list = []

    # Compute the voxel indices for each point cloud
    for points in point_clouds:
        index = ((points - bb_min) * scale_factor).astype(int)
        indices_list.append(index)

    # Return a list of indices if multiple point clouds were provided, otherwise return a single numpy array
    return indices_list if not is_single else indices_list[0]


def voxelise1(points, voxel_size):
    """

    Divides the bounding box of 'points' by 'voxel_size' cubes and returns a 2D Numpy array with the same shape as
    'points', presenting the spatial index of each point in 'points'.

    Parameters:
    points (numpy array): Point cloud
    voxel_size (float): voxel dimension size

    Returns:
    numpy array (points.shape): Voxel index for each point in 'points'

    """

    # Determine bounding box (bb) of the point cloud
    bb_min = np.min(points, axis=0)
    bb_max = np.max(points, axis=0)
    bb_size = bb_max - bb_min

    # Determine number of voxels in each axis
    voxels = (bb_size / voxel_size).astype(int) + 1

    scale_factor = 1 / bb_size

    # Shift the points to origin (0)
    # Scale each dimension to 0 to 1
    # Multiply to number of voxels in each axis
    # Get the integer of the array
    index = ((points - bb_min) * (scale_factor * voxels)).astype(int)

    return index


def voxel_based_distance_to_ground(non_ground_points, non_ground_voxels_index, ground_points, ground_voxels_index):
    """
    Computes the elevation of non-ground points to the nearest ground point in each voxel.

    Parameters:
    - non_ground_points (np.ndarray): The 2D array of non-ground points.
    - non_ground_voxels (np.ndarray): The voxel indices for non-ground points.
    - ground_points (np.ndarray): The 2D array of ground points.
    - ground_voxels (np.ndarray): The voxel indices for ground points.

    Returns:
    - np.ndarray: An array of the same length as `non_ground_points`, containing the distances to the ground.
    """
    # Initialize results for non-ground points' distances
    results = np.full(len(non_ground_points), np.nan)

    # Loop through each unique voxel shared between ground and non-ground points
    common_voxels = np.unique(
        np.vstack((np.unique(ground_voxels_index, axis=0), np.unique(non_ground_voxels_index, axis=0))), axis=0)

    for voxel in common_voxels:
        print("Voxel: ", voxel)

        # Masks for current voxel

        ground_mask = np.all(ground_voxels_index == voxel, axis=1)
        non_ground_mask = np.all(non_ground_voxels_index == voxel, axis=1)

        # Get ground and non-ground points in this voxel
        ground_voxel_points = ground_points[ground_mask]
        non_ground_voxel_points = non_ground_points[non_ground_mask]
        print("Ground points count: ", len(ground_voxel_points))
        print("Nonground points count: ", len(non_ground_voxel_points))
        print(" ")
        # Check if there are ground points available
        if len(ground_voxel_points) > 0:
            # Build a KDTree for the ground points
            tree = KDTree(ground_voxel_points)

            # Query nearest neighbors for the non-ground points
            distances, _ = tree.query(non_ground_voxel_points, k=1)

            # Assign distances to the corresponding non-ground points
            results[non_ground_mask] = distances

    return results


def voxel_based_process(points, voxels, func):
    """
    Processes points in each voxel using a specified function and assigns results back to each point.

    Parameters:
    - points (np.ndarray): A 2D array of shape (num_points, 3) containing coordinates or properties.
    - voxels (np.ndarray): A 1D array of length `num_points` with voxel indices for each point.
    - func (callable): A function that accepts a 2D array (subset of points) and returns an appropriate
                                  result for each voxel (e.g., a 1D array or scalar to assign back).

    Returns:
    - np.ndarray: A 2D array of shape (num_points, 3) with processed values corresponding to each point.

    Example:
    --------
    Assume you want to compute the mean coordinates for each voxel:

    >>> def mean_coordinates(points, offset):
    ...     return np.mean(points, axis=0) + offset

    >>> np.random.seed(0)
    >>> points = np.random.rand(10, 3)
    >>> voxels = np.random.randint(0, 4, size=10)

    >>> def wrapped_func(points):
    ...     return mean_coordinates(points, np.array([0.1, 0.1, 0.1]))

    >>> processed = voxel_based_process(points, voxels, wrapped_func)
    >>> print("Processed:\\n", processed)
    """

    # Initialize the results array with the same shape as points
    results = np.zeros(len(points))

    unique_voxels_index = np.unique(voxels, axis=0)

    # Loop through each unique voxel
    for voxel in unique_voxels_index:
        # Create a boolean mask for the current voxel
        mask = np.all(voxels == voxel, axis=1)

        # Filter points based on this mask
        voxel_points = points[mask]

        # Apply the processing function to the points in this voxel
        result = func(voxel_points)

        # Assign the processed value back to the appropriate indices
        results[mask] = result

    return results


def fit_plane_to_points(points, ransacThreshold, ransacIteration):
    import pyransac3d as pyrsc

    # Create Plane object
    objPlane = pyrsc.Plane()

    # .fit randomly takes 3 points of point-cloud to verify inliers based on a threshold and returns:
    # Parameters (planeParam) of the Plane using Ax+By+Cy+D as a np.array (1, 4) and
    # inlier points

    planeParams, inliers = objPlane.fit(points, thresh=ransacThreshold, maxIteration=ransacIteration)
    return planeParams, inliers


# Finding the distance of each point to a reference point and return a distance array
def points_distance_to_point(picked_point, points):
    import numpy as np

    # Create connection vectors
    dist_pt = picked_point - points

    # Calculate l2 Norm which is equal to the magnitude (length) of the vector
    dist = np.linalg.norm(dist_pt, axis=1)

    return dist


# Finding the distance of each point to the plane and return a distance array
# Finding distance of a point to a plane is possible by finding the projection of the connection vector of the point
# to the plane (from the point to a random point on the plane) on the normal vector of the plane
# For finding the normal vector of a plane we need the cross product of 2 vectors on the plan
def points_distance_to_plane(planeParams, points):
    import numpy as np

    # Finding 2 vectors on the plan
    x0, y0 = [0, 0]
    x1, y1 = [1, 0]
    x2, y2 = [0, 1]

    A, B, C, D = planeParams[0], planeParams[1], planeParams[2], planeParams[3]
    z0 = (- A * x0 - B * y0 - D) / C
    z1 = (- A * x1 - B * y1 - D) / C
    z2 = (- A * x2 - B * y2 - D) / C

    p0 = np.array([x0, y0, z0])
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])

    v1 = p1 - p0
    v2 = p2 - p0

    # Finding Normal vector of the plane using cross product of 2 vectors on the plane
    n = np.cross(v1, v2)  # Normal Vector

    # Unit Normal Vector (Magnitude = 1)
    # Unit Normal Vector = Normal vector / Normal vector's Magnitude
    # Normal vector's Magnitude = n.n (dot product) is equal to projection of n on n, multiply by n magnitude, which is
    # equal to n^2
    nHat = n / np.sqrt(np.dot(n, n))

    distance = np.dot(points - p0, nHat)
    return distance


# Finding the Unit Normal Vector of a plane
# For finding the normal vector of a plane we need the cross product of 2 vectors on the plan
def unit_normal_vector_of_plane(planeParams):
    import numpy as np

    # Finding 2 vectors on the plan
    x0, y0 = [0, 0]
    x1, y1 = [1, 0]
    x2, y2 = [0, 1]

    A, B, C, D = planeParams[0], planeParams[1], planeParams[2], planeParams[3]
    z0 = (- A * x0 - B * y0 - D) / C
    z1 = (- A * x1 - B * y1 - D) / C
    z2 = (- A * x2 - B * y2 - D) / C

    p0 = np.array([x0, y0, z0])
    p1 = np.array([x1, y1, z1])
    p2 = np.array([x2, y2, z2])

    v1 = p1 - p0
    v2 = p2 - p0

    # Finding Normal vector of the plane using cross product of 2 vectors on the plane
    n = np.cross(v1, v2)  # Normal Vector

    # Unit Normal Vector (Magnitude = 1)
    # Unit Normal Vector = Normal vector / Normal vector's Magnitude
    # Normal vector's Magnitude = n.n (dot product) is equal to projection of n on n, multiply by n magnitude, which is
    # equal to n^2
    nHat = n / np.sqrt(np.dot(n, n))

    return nHat


# Returns an array of the projection of points on a Plane
# The projection point on the plane = point vector - distance to the plane vector
# distance to the plan vector (parallel to normal vector) = unit normal vector * distance to the plane
def points_projection_on_plane(points, planeParams):
    import numpy as np

    nHat = unit_normal_vector_of_plane(planeParams)
    pointsDistanceToPlane = points_distance_to_plane(planeParams, points)
    distanceReshaped = pointsDistanceToPlane.reshape((pointsDistanceToPlane.shape[0], 1))

    distanceVector = np.multiply(distanceReshaped, nHat)
    return points - distanceVector


def points_on_plane_by_point(points, pickedPoint, selectionRadius, distToPlane, ransacThreshold=0.001,
                             ransacIterations=1000, fineTuningIterations=5):
    import numpy as np

    planeParams = []
    planePoints = []

    distsToPoint = points_distance_to_point(pickedPoint, points)

    # Select indexes where distance is larger than the threshold
    selectedPoints = np.where(np.abs(distsToPoint) <= selectionRadius)[0]

    # Select points using their index to fit a plane to them
    pointsToFitPlane = points[selectedPoints]

    for i in range(fineTuningIterations):
        # Fit a plane to the selected points using RANSAC and returns the inliers and plane
        # parameters A, B, C & D (Ax+By+Cz+D=0)
        planeParams, inliers = fit_plane_to_points(pointsToFitPlane, ransacThreshold, ransacIterations)

        # Distance will be positive or negative depends on which side of the plan it is
        distance = points_distance_to_plane(planeParams, points)

        # Select indexes where distance is less than the threshold
        planePointsIndex = np.where(np.abs(distance) <= distToPlane)[0]

        # Select points using their index
        planePoints = points[planePointsIndex]

        # Replace 'Points to fit the plane to' by inlier points for the next iteration to fine tune the plane parameters
        pointsToFitPlane = planePoints

    return planeParams, planePoints


def points_on_plane_by_plane(points, planeParams, distToPlane, ransacThreshold=0.001, ransacIterations=1000,
                             fineTuningIterations=5):
    import numpy as np

    # Distance will be positive or negative depends on which side of the plan it is
    distance = points_distance_to_plane(planeParams, points)

    # Select indexes where distance is less than the threshold
    planePointsIndex = np.where(np.abs(distance) <= distToPlane)[0]

    # Select points using their index
    planePoints = points[planePointsIndex]

    # Replace 'Points to fit the plane to' by inlier points for the next iteration to fine tune the plane parameters
    pointsToFitPlane = planePoints

    for i in range(fineTuningIterations):
        # If there enough points on the plane
        if pointsToFitPlane.shape[0] > 50:
            # Fit a plane to the selected points using RANSAC and returns the inliers and plane
            # parameters A, B, C & D (Ax+By+Cz+D=0)
            planeParams, inliers = fit_plane_to_points(pointsToFitPlane, ransacThreshold, ransacIterations)

            # Distance will be positive or negative depends on which side of the plan it is
            distance = points_distance_to_plane(planeParams, points)

            # Select indexes where distance is less than the threshold
            planePointsIndex = np.where(np.abs(distance) <= distToPlane)[0]

            # Select points using their index
            planePoints = points[planePointsIndex]

            # Replace 'Points to fit the plane to' by inlier points for the next iteration to fine tune the plane parameters
            pointsToFitPlane = planePoints

    return planeParams, planePoints


def histogram(values, bins, log_show=False, display=False, title="Histogram", xlabel="Value", ylabel="Frequency"):
    # Create histogram
    hist, bin_edges = np.histogram(values, bins=bins, density=True)  # Ensure density=True for normalization

    # Find the bin with the highest frequency (peak of the bell curve)
    peak_index = np.argmax(hist)
    peak_value = bin_edges[peak_index]
    peak_range = (bin_edges[peak_index], bin_edges[peak_index + 1])

    if display:
        # Plot the histogram to visualize, with density=True to match the scale of the PDF
        plt.hist(values, bins=bins, density=True, alpha=0.6, log=log_show)
        plt.grid(True, which="both", ls="-")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Fit a normal distribution to the data: Calculate the mean and standard deviation
        mu, std = norm.fit(values)

        # Plot the PDF (Probability Density Function)
        xmin, xmax = np.min(values), np.max(values)
        x = np.linspace(xmin, xmax, 500)

        if not log_show:
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label="Normal Distribution")  # Plot the bell curve

        bin_width = (xmax - xmin) / bins
        peak_range_label = f'{peak_range[0]:.4f}-{peak_range[1]:.4f}'
        plt.axvline(peak_value + (bin_width / 2), color='r', linestyle='dashed', linewidth=2, label=peak_range_label)

        # Draw lines for the mean and 1st, 2nd, and 3rd standard deviations
        colors = ['blue', 'green', 'red']
        for i in range(1, 4):
            pos_label = f'{i}std (σ = {mu + i * std:.3f})'
            plt.axvline(mu + i * std, color=colors[i - 1], linestyle='dashed', linewidth=1, label=pos_label)

        # Remove the dead space (both vertical and horizontal)
        # that Matplotlib adds by default
        axes = plt.gca()
        axes.set_xlim([xmin, xmax])
        ymin, ymax = axes.get_ylim()
        axes.set_ylim([0, ymax])

        plt.legend()
        plt.show()

    return peak_value


def execute_python_file(file_path):
    try:
        os.system(f'python {file_path}')
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")

'''
def check_and_install_package(package, required_version):
    """
    Checks if a package with a specific version is installed, and if not, attempts to install it.

    Args:
    package (str): The name of the package to check.
    required_version (str): The version of the package to ensure is installed.

    Example:
    # List of packages to check and potentially install with their required versions
    packages_versions = [("numpy", "1.19.2"), ("pandas", "1.1.5"), ("matplotlib", "3.3.4")]
    for pkg, ver in packages_versions:
        check_and_install_package(pkg, ver)
    """
    try:
        # Check the currently installed version of the package
        installed_version = version(package)
        if installed_version == required_version:
            print(f"{package} version {required_version} is already installed.")
        else:
            print(
                f"{package} version {installed_version} is installed but not the required version {required_version}. Updating...")
            raise PackageNotFoundError
    except PackageNotFoundError:
        # Package is not installed or the wrong version, proceed to install or update
        print(f"{package} not found or wrong version. Installing/updating to {required_version}...")
        try:
            # Use subprocess to install the specific version of the package
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={required_version}"])
            print(f"{package} has been installed/updated to {required_version}.")
        except subprocess.CalledProcessError:
            # Handle potential installation errors
            print(
                f"Failed to install/update {package} to version {required_version}. Please check your permissions or internet connection.")
'''

class Clusters:
    def __init__(self, points, **kwargs):
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
        - labels (int): A unique identifier for the cluster.
        - parent (int): The labels of the parent cluster.
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

        # Validation points
        if not isinstance(points, np.ndarray):  # or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a numpy array with shape (n, 3)")

        self.points = points
        self.label = kwargs.get('labels', None)
        self.parent = kwargs.get('parent', None)
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

    def classify(self, attributes, classifier):
        """
        Classifies the attributes using the provided classifier and stores the classification results as 'classes' attribute.

        This method applies a machine learning classifier to an array of attributes of the claster to determine
        the class for each attribute set. The results are stored in the instance for later use.

        Parameters:
        - attributes (np.ndarray): A numpy array containing the features to classify. This should
                                   be in the format expected by the classifier (typically 2D: samples x features).
        - classifier (object): A trained classifier instance that supports the 'predict' method. This could
                               be any classifier from popular libraries like scikit-learn.

        Returns:
        - np.ndarray: An array of class labels as determined by the classifier.

        Raises:
        - ValueError: If 'attributes' is not in the expected format (e.g., not a 2D numpy array).
        - AttributeError: If the 'classifier' does not have a 'predict' method.

        Example:
        --------
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> classifier = RandomForestClassifier()
        >>> data = np.random.rand(10, 5)  # Random data with 10 samples and 5 features each
        >>> my_object = MyClass()
        >>> classes = my_object.classify(data, classifier)
        >>> print(classes)
        """
        # Validate the input attributes are in the correct format (e.g., a 2D numpy array)
        if not (isinstance(attributes, np.ndarray) and attributes.ndim == 2):
            raise ValueError("Attributes must be a 2D numpy array.")

        # Check that the classifier has a predict method
        if not hasattr(classifier, 'predict'):
            raise AttributeError("Classifier must have a 'predict' method.")

        # Perform classification
        self.classes = classifier.predict(attributes)

        # Return the classification results
        return self.classes

    def dbscan(self, eps=0.05, min_points=10, return_clusters_object=False):
        """
        Apply DBSCAN clustering to the points in the cluster using Open3D.

        Parameters:
        - eps (float): The maximum distance between two samples for one to be considered
                       as in the neighborhood of the other. Default is 0.05.
        - min_points (int): The number of samples in a neighborhood for a point to be considered
                            as a core point. Default is 10.
        - return_clusters_object (bool): If True, returns a Clusters object containing the DBSCAN result.
                                         Default is False.

        Returns:
        - labels (np.ndarray) or (np.ndarray, Clusters): An array of cluster labels, and optionally a Clusters object.
        """
        if len(self.points) == 0:
            raise ValueError("The cluster has no points for DBSCAN clustering.")

        # Convert points to Open3D point cloud and perform DBSCAN
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

        # Store the DBSCAN labels
        self.clusters = labels

        if return_clusters_object:
            # Create and return Clusters object
            clusters = Clusters()
            clusters.points = self.points
            clusters.labels = labels
            clusters.colors = self.color
            clusters.normals = self.normal
            return clusters
        else:
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

        return self.get_subcluster(mask)

    def get_clusters(self, min_points=200):
        """
        Generate a Clusters object containing individual clusters from DBSCAN labels,
        excluding clusters with fewer than min_points.

        Args:
            min_points (int, optional): Minimum number of points for a cluster to be included. Default is 200.

        Returns:
            Clusters: An object containing all the valid clusters (excluding noise and smaller clusters).
        """

        if not hasattr(self, 'clusters'):
            raise ValueError("DBSCAN must be applied before generating 'Clusters' object.")

        # Filter out noise points (usually labeled as -1)
        unique_labels = set(self.clusters)
        unique_labels.discard(-1)

        # Create a Clusters object to hold the clusters
        clusters = Clusters()

        # Filter labels to include only those with enough points
        valid_labels = [label for label in unique_labels if np.sum(self.clusters == label) >= min_points]

        for label in valid_labels:
            # Create mask for points belonging to the current labels
            mask = self.clusters == label

            # Create a copy of cluster
            child_cluster = copy.deepcopy(self)
            child_cluster.points = self.points[mask]
            # Add the new cluster to clusters instance
            clusters.add(child_cluster)

        return clusters

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
        cluster_instance = Clusters(points=np.random.rand(100, 3))
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

    def get_subcluster(self, mask, inplace=False):
        """
        Extracts a subcluster based on a mask, either by modifying the current cluster or returning a new one.

        Parameters:
        - mask (np.ndarray): A boolean array where True values indicate points to include in the subcluster.
        - inplace (bool): If True, modifies the current cluster in-place. Default is False.

        Returns:
        - Clusters: A new Clusters instance containing the subcluster if inplace is False.
        """

        if not np.count_nonzero(mask) >= 4:
            # Handle the case where the mask filters out all points
            print("Not enough points found for subcluster.")
            if inplace:
                self.points = np.array([])
                self.label = 0
                self.parent = 0
                self.clusters = np.array([])
                self.prediction = ''
                self.probability = 0
                self.model_weight = ''
                self.length = 0
                self.width = 0
                self.height = 0
                self.feature = []
                self.metadata = {}
            else:
                # Return an empty Clusters instance with (1, 3) dimension as
                # Clusters constructor needs (n, 3) ndarray for points
                return Cluster(np.empty((1, 3)))
        else:
            if inplace:
                # Modify the current cluster's points and attributes in-place
                # Apply the mask to attributes of the cluster
                self.points = self.points[mask]
                self._update_obb_dim()

                if hasattr(self, 'color') and self.color.shape[0] > 0:
                    self.color = self.color[mask]
                if hasattr(self, 'intensity') and self.intensity.shape[0] > 0:
                    self.intensity = self.intensity[mask]
                if hasattr(self, 'distToGround') and self.distToGround.shape[0] > 0:
                    self.distToGround = self.distToGround[mask]
            else:
                # Create a deep copy of the cluster and apply the mask
                subcluster = copy.deepcopy(self)

                # Apply the mask to attributes of the cluster
                subcluster.points = subcluster.points[mask]
                subcluster._update_obb_dim()

                if hasattr(subcluster, 'color') and subcluster.color.shape[0] > 0:
                    subcluster.color = subcluster.color[mask]
                if hasattr(subcluster, 'intensity') and subcluster.intensity.shape[0] > 0:
                    subcluster.intensity = subcluster.intensity[mask]
                if hasattr(subcluster, 'distToGround') and subcluster.distToGround.shape[0] > 0:
                    subcluster.distToGround = subcluster.distToGround[mask]

                return subcluster

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

    def normalize(self, apply_scaling=True, apply_centering=True, rotation_axes=(True, True, True)):
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

    def predict(self, model, class_map, weight_path):
        """
        Predicts the feature labels for the cluster using the provided PointNet model.

        This method processes the cluster's points and uses the given model to predict the most likely class labels.
        It also computes the probability of the prediction. The prediction, its probability, and the model's weight path
        are stored as attributes of the cluster.

        Parameters:
        - model (tf.keras.Model): The trained PointNet model used for prediction.
        - class_map (dict): A dictionary mapping model output indices to class labels.
        - weight_path (str): The path to the model's weights.

        Returns:
        tuple: A tuple containing the predicted class labels and the corresponding probability.

        Note:
        The prediction will be 'Noise' if the preprocessed points are not suitable for prediction.
        """

        # Preprocess points for PointNet
        points_for_pointnet = self._preprocess_points_for_prediction(2048, 512)
        if points_for_pointnet is None:
            # Storing 'Noise' for prediction
            self.prediction = "Noise"
            self.probability = 0
            self.model_weight = ""

            return "Noise", 0

        # Making prediction using the model
        probabilities = model.predict(points_for_pointnet)
        prediction = tf.math.argmax(probabilities, -1)

        predicted_label = class_map[prediction[0].numpy()]
        probability = np.max(probabilities)

        # Storing prediction information in the cluster
        self.prediction = predicted_label
        self.probability = probability
        self.model_weight = weight_path

        return predicted_label, probability

    def save(self, file_name):
        """
        Saves the Clusters instance to a file. The format is determined by the file extension.

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
        Saves a specified property of the Clusters to a .npy file.

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
            raise AttributeError(f"Property '{property_name}' not found in Clusters.")

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

        if random_color_attr is None:
            # Check if colors are set and apply them
            if len(self.color) != 0 and self.color.shape[1] == 3:
                # Normalize color values if necessary
                if self.color.max() > 1:
                    normalized_colors = self.color / 255.0
                else:
                    normalized_colors = self.color
                pcd.colors = o3d.utility.Vector3dVector(normalized_colors)
        elif hasattr(self, random_color_attr):
            labels = getattr(self, random_color_attr)
            random_color = get_random_color(labels)
            pcd.colors = o3d.utility.Vector3dVector(random_color)
        else:
            raise ValueError(f"There is no {random_color_attr} attribute for cluster instance!")

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

    def _preprocess_points_for_prediction(self, NUM_POINTS, MIN_NUM_POINTS):
        """
        Prepare the cluster points for prediction using a specified model,
        working on a copy of the points to preserve the original data.

        Args:
        NUM_POINTS (int): The number of points to use for prediction.
        MIN_NUM_POINTS (int): The minimum number of points required for a valid prediction.

        Returns:
        np.ndarray or None: A processed numpy array suitable for prediction, or None if the cluster is too small.
        """
        # Copy the points to avoid modifying the original points
        points_copy = np.copy(self.points)
        clusterLWHs = points_copy[:NUM_POINTS, 3:]
        points_copy = points_copy[:, :3]
        num_points = len(points_copy)

        # Check for sufficient points
        if num_points < MIN_NUM_POINTS:
            print("------------- TOO SMALL CLUSTER -------------")
            return None

        # Shuffle and jitter the points
        shuffle_indices = np.random.permutation(num_points)
        points_copy = points_copy[shuffle_indices]

        noise = np.random.normal(0, 0.005, points_copy.shape)
        points_copy += noise

        # Augment the points if necessary
        if num_points < NUM_POINTS:
            required_augmentation_factor = int(np.ceil(NUM_POINTS / num_points))
            points_copy = np.repeat(points_copy, required_augmentation_factor, axis=0)

        # Normalize the points
        centroid = np.mean(points_copy, axis=0)
        points_copy -= centroid
        max_distance = np.max(np.sqrt(np.sum(points_copy ** 2, axis=1)))
        if max_distance > 0:
            points_copy /= max_distance

        # Ensure we have exactly NUM_POINTS for prediction
        if len(points_copy) > NUM_POINTS:
            random_indices = np.random.choice(len(points_copy), NUM_POINTS, replace=False)
            points_copy = points_copy[random_indices, :]

        # Reshape and prepare for prediction (assuming TensorFlow usage)
        # To Do
        # points_copy = points_copy.reshape(1, NUM_POINTS, XXX)

        points_copy = np.hstack((points_copy, clusterLWHs))
        points_copy = points_copy.reshape(1, NUM_POINTS, 3)

        points_copy = tf.convert_to_tensor(points_copy, dtype=tf.float32)  # Ensure dtype consistency
        return points_copy

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
        Saves the Clusters instance as a PLY file.

        Parameters:
        file_name (str): The name of the PLY file to save the instance to.
        """
        # Assuming the Clusters instance has a point cloud attribute named 'points'
        if hasattr(self, 'points'):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)

            if hasattr(self, 'color'):
                pcd.colors = o3d.utility.Vector3dVector(self.color)

            #            if hasattr(self, 'normal'):
            #                pcd.normals = o3d.utility.Vector3dVector(self.normal)

            o3d.io.write_point_cloud(file_name, pcd)
        else:
            raise AttributeError("Clusters instance does not have 'points' attribute.")


class Clusters1:
    def __init__(self):
        """
        Initialize a Clusters container.

        Attributes:
        - clusters (list): A list to store Clusters instances.
        """
        self.clusters = []
        self.labels = []

    def add_clusters_from_points(self, points, labels, distToGround=None, color=None, intensity=None, normal=None):
        """
        Create Clusters objects from points and labels and add them to the Clusters object.

        Parameters:
        - points (np.ndarray): A numpy array of points.
        - labels (np.ndarray): An array of labels corresponding to each point in 'points'.
        - distToGround (np.ndarray, optional): An array of distances to the ground for each point.
        - color (np.ndarray, optional): An array of color values for each point.
        - intensity (np.ndarray, optional): An array of intensity values for each point.
        """
        unique_labels = set(labels)
        self.labels = list(unique_labels)

        for label in unique_labels:
            if label == -1:  # Skipping noise points
                continue

            mask = labels == label
            cluster_points = points[mask]

            # Optional attributes
            cluster_distToGround = distToGround[mask] if distToGround is not None else np.array([])
            cluster_color = color[mask] if color is not None else np.array([])
            cluster_intensity = intensity[mask] if intensity is not None else np.array([])
            cluster_normal = normal[mask] if normal is not None else np.array([])

            # Creating the Clusters object
            cluster = Cluster(
                points=cluster_points,
                distToGround=cluster_distToGround,
                color=cluster_color,
                intensity=cluster_intensity,
                label=label,
                normal=cluster_normal
            )

            # Adding the cluster to the Clusters container
            self.clusters.append(cluster)

    def add(self, cluster):
        """
        Add a new Clusters instance to the container.

        Parameters:
        - cluster (Clusters): The Clusters instance to add.
        """
        if not isinstance(cluster, Cluster):
            raise TypeError("The added object must be an instance of Clusters.")
        self.clusters.append(cluster)

        # keep labels unique
        if not cluster.label in self.labels:
            self.labels.append(cluster.label)

    def remove(self, cluster):
        """
        Remove a Clusters instance from the container.

        Parameters:
        - cluster (Clusters): The Clusters instance to remove.
        """
        if not isinstance(cluster, Cluster):
            raise TypeError("The removed object must be an instance of Clusters.")
        self.clusters.remove(cluster)
        self.labels.remove(cluster.label)

    def get(self, label):
        """
        Retrieve a Clusters instance by its labels.

        Parameters:
        - labels (int): The labels of the Clusters to retrieve.

        Returns:
        Clusters or None: The Clusters instance with the given labels, or None if not found.
        """
        for cluster in self.clusters:
            if cluster.labels == label:
                return cluster
        return None

    def count(self):
        """
        Get the count of Clusters instances in the container.

        Returns:
        int: The number of Clusters instances.
        """
        return len(self.clusters)

    def apply_to_all(self, function, *args, **kwargs):
        """
        Apply a given function to all Clusters instances in the container.

        Parameters:
        - function (callable): The function to apply. It should take a Clusters instance as its first argument.
        - args, kwargs: Additional arguments and keyword arguments to pass to the function.

        Returns:
        - list: A list of results from applying the function to each Clusters.

        Example:
        --------
        Assume we have a container of Clusters instances named `cluster_container`, and we define the following function to count the number of points in each Clusters:

            def count_points(cluster):
                return len(cluster.points)

        We can apply this function to all Clusters instances in the container like this:

            results = cluster_container.apply_to_all(count_points)

        `results` will be a list containing the number of points in each Clusters within the container.
        """
        return [function(cluster, *args, **kwargs) for cluster in self.clusters]

    def filter(self, condition):
        """
        Filter Clusters instances based on a provided condition.

        Parameters:
        - condition (callable): A function that takes a Clusters instance and returns a boolean.

        Returns:
        Clusters: A Clusters instance containing Clusters instances that satisfy the condition.
        """

        filtered_clusters = Clusters()
        for cluster in self.clusters:
            if condition(cluster):
                filtered_clusters.add(cluster)

        return filtered_clusters

    def merge(self, labels_to_merge):
        """
        Merges multiple clusters into a single cluster.

        Parameters:
        - labels_to_merge (list): A list of labels indicating which clusters to merge.
        - remove_originals (bool): If True, removes the original clusters from the container
                                   after merging. Default is True.

        Returns:
        - merged_cluster (Clusters): The new Clusters instance resulting from the merge.
        """

        first_cluster = self.get(labels_to_merge[0])
        merged_points = first_cluster.points

        if hasattr(first_cluster, 'distToGround') and len(first_cluster.distToGround) > 0:
            merged_dists = first_cluster.distToGround
        if hasattr(first_cluster, 'color') and len(first_cluster.color) > 0:
            merged_colors = first_cluster.color
        if hasattr(first_cluster, 'intensity') and len(first_cluster.intensity) > 0:
            merged_intensities = first_cluster.intensity
        if hasattr(first_cluster, 'normal') and len(first_cluster.normal) > 0:
            merged_normals = first_cluster.normals

        for label in labels_to_merge[1:]:
            cluster = self.get(label)
            if cluster is not None:
                merged_points = np.vstack([merged_points, cluster.points])

                if hasattr(first_cluster, 'distToGround') and len(first_cluster.distToGround) > 0:
                    merged_dists = np.concatenate([merged_dists, cluster.distToGround])

                if hasattr(first_cluster, 'color') and hasattr(cluster, 'color') and len(first_cluster.color) > 0:
                    cluster_colors = cluster.color
                    cluster_colors[:] = first_cluster.color[0]
                    merged_colors = np.vstack([merged_colors, cluster_colors])

                if hasattr(first_cluster, 'intensity') and hasattr(cluster, 'intensity') and len(cluster.intensity) > 0:
                    merged_intensities = np.concatenate([merged_intensities, cluster.intensity])

                if hasattr(first_cluster, 'normal') and hasattr(cluster, 'normal') and len(cluster.normal) > 0:
                    merged_normals = np.vstack([merged_normals, cluster.normal])

                self.remove(cluster)

        # Create a new Clusters instance with the merged attributes
        first_cluster.points = merged_points

        if hasattr(first_cluster, 'distToGround') and len(first_cluster.distToGround) > 0:
            first_cluster.distToGround = merged_dists
        if hasattr(first_cluster, 'color') and len(first_cluster.color) > 0:
            first_cluster.color = merged_colors
        if hasattr(first_cluster, 'intensity') and len(first_cluster.intensity) > 0:
            first_cluster.intensity = merged_intensities
        if hasattr(first_cluster, 'normal') and len(first_cluster.normal) > 0:
            first_cluster.normals = merged_normals

        return first_cluster

    def predict(self, model, class_map, weight_path, selected_labels=None):
        """
        Predicts the features for selected or all clusters in the container using the provided PointNet model.

        Parameters:
        - model (tf.keras.Model): The trained PointNet model used for prediction.
        - class_map (dict): A dictionary mapping model output indices to class labels.
        - weight_path (str): The path to the model's weights.
        - selected_labels (list, optional): A list of cluster labels to predict. If None, predicts for all clusters.
        """
        for cluster in self.clusters:
            if selected_labels is None or cluster.labels in selected_labels:
                try:
                    cluster.predict(model, class_map, weight_path)
                except Exception as e:
                    print(f"Prediction failed for cluster {cluster.labels}: {e}")

    def pick_clusters(self):
        """
        Interactively select clusters using Open3D visualization. This method allows
        the user to visually pick clusters and returns the labels of the selected clusters.

        Returns:
            list: A list of labels of the selected clusters.
        """

        # Create an empty point cloud to merge all clusters
        merged_pcd = o3d.geometry.PointCloud()

        # Dictionary to map the color to cluster labels
        color_label_map = {}

        for cluster in self.clusters:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster.points)

            color = np.random.rand(3)
            pcd.paint_uniform_color(color)  # Assign random color

            # Merge with the main point cloud
            merged_pcd += pcd

            # Update color-labels map
            key = color[0] + color[1] + color[2]
            color_label_map[key] = cluster.labels

        # Visualize the merged point cloud and allow user to pick points
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(merged_pcd)
        vis.run()
        picked_point_indices = vis.get_picked_points()
        vis.destroy_window()

        # Identify the clusters corresponding to picked points
        picked_labels = []
        for index in picked_point_indices:
            color = merged_pcd.colors[index]
            key = color[0] + color[1] + color[2]
            label = color_label_map[key]
            picked_labels.append(label)

        return picked_labels

    def show(self, selected_labels=None, random_color=True):
        """
        Visualize the clusters using Open3D.

        Parameters:
        - selected_labels (list, optional): A list of cluster labels to be visualized.
                                            If None or empty, all clusters are visualized.
        """
        # Setting up Open3D visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add clusters to the visualization
        for cluster in self.clusters:
            if selected_labels is None or cluster.labels in selected_labels:
                # Prepare the points for visualization
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cluster.points)

                # Set colors if available

                if not random_color:
                    pcd.colors = o3d.utility.Vector3dVector(cluster.color)
                else:
                    color = [np.random.random(), np.random.random(), np.random.random()]
                    pcd.paint_uniform_color(color)

                vis.add_geometry(pcd)

        # Visualize
        vis.run()
        vis.destroy_window()

    def save(self, filepath):
        """
        Save the clusters' points and labels to .npy files.

        Parameters:
        - filepath (str): The base filepath without extension where the points and labels will be saved.
                          Two files will be generated: filepath_points.npy and filepath_labels.npy
        """
        # Concatenate all points from all clusters into a single array
        all_points = np.concatenate([cluster.points for cluster in self.clusters], axis=0)
        print(all_points.shape)
        # Concatenate all points from all clusters into a single array
        all_dists = np.concatenate([cluster.distToGround if hasattr(cluster,
                                                                    'distToGround') and cluster.distToGround.size > 0 else np.full(
            cluster.points.shape[0], np.nan) for cluster in self.clusters])
        print(all_dists.shape)
        # Create an array of labels with the same length as all_points
        # For each cluster, repeat its labels for each of its points
        all_labels = np.concatenate([np.full(cluster.points.shape[0], cluster.labels) for cluster in self.clusters])
        print(all_labels.shape)
        # Save points, labels and distance to grounds as .npy files
        np.save(f"{filepath}_points.npy", all_points)
        np.save(f"{filepath}_labels.npy", all_labels)
        np.save(f"{filepath}_Dist_to_Ground.npy", all_dists)

    def __iter__(self):
        """
        Make the Clusters class iterable over its Clusters instances.
        """
        return iter(self.clusters)

    def __len__(self):
        """
        Return the number of Clusters instances in the container.
        """
        return len(self.clusters)

    def __getitem__(self, index):
        """
        Allow indexed access to Clusters instances.
        """
        return self.clusters[index]

    def __repr__(self):
        """
        Return a string representation of the Clusters container.
        """
        return f"Clusters({len(self)})"


class Clusters:
    def __init__(self):
        """
        Initialize the Clusters1 container.

        Attributes:
            points (np.ndarray): Stores points for all clusters.
            labels (np.ndarray): Stores the labels for each point.
            colors (np.ndarray): Stores colors for each point, NaN if absent.
            normals (np.ndarray): Stores normals for each point, NaN if absent.
            distToGround (np.ndarray): Stores distance to the ground for each point, NaN if absent.
            intensity (np.ndarray): Stores intensity for each point, NaN if absent.
        """
        self.points = np.array([], dtype=np.float32).reshape(0, 3)  # 3D points
        self.labels = np.array([], dtype=np.int32)
        self.colors = np.array([], dtype=np.float32).reshape(0, 3)  # RGB
        self.normals = np.array([], dtype=np.float32).reshape(0, 3)  # 3D normals
        self.eigenvalues = np.array([], dtype=np.float32).reshape(0, 3)  # Eigenvalues
        # TODO: distsToGround, intensities representing a list not just a single value
        self.distToGround = np.array([], dtype=np.float32)
        self.intensity = np.array([], dtype=np.float32)
        self.pcd = o3d.geometry.PointCloud()

    def add(self, cluster, next_label=True):
        """
        Add a new cluster's data to the Clusters1 container.

        Parameters:
        - cluster (Clusters): The cluster instance to add. It is expected to have attributes like points, labels, and optionally colors, normals, distToGround, intensity.
        """
        if not isinstance(cluster, Cluster):
            raise TypeError("The added object must be an instance of Clusters.")

        if next_label:
            if len(self.labels) == 0:
                label = 0
            else:
                label = np.max(self.labels)
        else:
            if cluster.label is None:
                raise ValueError("Clusters labels cannot be None.")
            label = cluster.label

        num_points = len(cluster.points)
        label_array = np.full(num_points, label, dtype=np.int32)

        self.points = np.vstack([self.points, cluster.points])
        self.labels = np.concatenate([self.labels, label_array])

        # Check if optional attributes exist and handle them appropriately
        if hasattr(cluster, 'color') and cluster.color is not None:
            self.colors = np.vstack([self.colors, cluster.color])
        else:
            self.colors = np.vstack([self.colors, np.full((num_points, 3), np.nan, dtype=np.float32)])

        if hasattr(cluster, 'normal') and cluster.normal is not None and len(cluster.normal) > 0:
            self.normals = np.vstack([self.normals, cluster.normal])
        else:
            pass
            # self.normals = np.vstack([self.normals, np.full((num_points, 3), np.nan, dtype=np.float32)])

        if hasattr(cluster, 'distToGround') and cluster.distToGround is not None:
            self.distToGround = np.concatenate([self.distToGround, cluster.distToGround])
        else:
            self.distToGround = np.concatenate([self.distToGround, np.full(num_points, np.nan, dtype=np.float32)])

        if hasattr(cluster, 'intensity') and cluster.intensity is not None:
            self.intensity = np.concatenate([self.intensity, cluster.intensity])
        else:
            self.intensity = np.concatenate([self.intensity, np.full(num_points, np.nan, dtype=np.float32)])

    def add_clusters_from_points(self, points, labels, noise=False, distToGround=None, color=None, intensity=None,
                                 normal=None, eigenvalues=None):
        """
        Add clusters from points and associated attributes to the Clusters object using parallel arrays.

        Parameters:
        - points (np.ndarray): A numpy array of points.
        - labels (np.ndarray): An array of labels corresponding to each point in 'points'.
        - noise (Boolean): Include noise points if True
        - distToGround (np.ndarray, optional): An array of distances to the ground for each point.
        - color (np.ndarray, optional): An array of color values for each point.
        - intensity (np.ndarray, optional): An array of intensity values for each point.
        - normal (np.ndarray, optional): An array of normal vectors for each point.
        """

        if not noise:
            # Skip noise points with labels -1
            valid_mask = labels != -1
        else:
            # Include noise points with labels -1
            valid_mask = np.full(len(points), True, dtype=bool)

        valid_points = points[valid_mask]
        valid_labels = labels[valid_mask]

        # Handle each attribute with parallel arrays
        self.points = np.vstack([self.points, valid_points])
        self.labels = np.concatenate([self.labels, valid_labels])

        self.pcd.points = o3d.utility.Vector3dVector(self.points)

        # For optional attributes, append if provided, else append NaNs
        if color is not None:
            valid_colors = color[valid_mask]
            self.colors = np.vstack([self.colors, valid_colors])
            self.pcd.colors = o3d.utility.Vector3dVector(valid_colors)
        else:
            self.colors = np.vstack([self.colors, np.full((len(valid_points), 3), np.nan, dtype=np.float32)])

        if normal is not None:
            valid_normals = normal[valid_mask]
            self.normals = np.vstack([self.normals, valid_normals])
            self.pcd.normals = o3d.utility.Vector3dVector(valid_normals)
        else:
            self.normals = np.vstack([self.normals, np.full((len(valid_points), 3), np.nan, dtype=np.float32)])

        if distToGround is not None:
            valid_dists = distToGround[valid_mask]
            self.distToGround = np.concatenate([self.distToGround, valid_dists])
        else:
            self.distToGround = np.concatenate(
                [self.distToGround, np.full(len(valid_points), np.nan, dtype=np.float32)])

        if intensity is not None:
            valid_intensities = intensity[valid_mask]
            self.intensity = np.concatenate([self.intensity, valid_intensities])
        else:
            self.intensity = np.concatenate([self.intensity, np.full(len(valid_points), np.nan, dtype=np.float32)])

        if eigenvalues is not None:
            valid_eigenvalues = eigenvalues[valid_mask]
            self.eigenvalues = np.vstack([self.eigenvalues, valid_eigenvalues])
        else:
            self.eigenvalues = np.vstack([self.eigenvalues, np.full((len(valid_points), 3), np.nan, dtype=np.float32)])

    def get(self, label):
        """
        Retrieve a Clusters instance by its labels. This method searches through the labels
        stored in the Clusters container and constructs a new Clusters object with attributes
        corresponding to the specified labels.

        Parameters:
        - labels (int): The labels of the Clusters to retrieve.

        Returns:
        - Clusters or None: The Clusters instance with the given labels, or None if the labels
          is not found in the dataset.

        Raises:
        - ValueError: If the labels type is not an integer.

        Example:
        --------
        >>> cluster = clusters.get(1)
        """
        if not isinstance(label, int):
            raise ValueError("Label must be an integer")

        # Find the indices where the labels matches
        mask = self.labels == label
        if not np.any(mask):
            return None  # No points with the given labels

        # Use filtered attributes or create empty numpy arrays if attributes are not present
        filtered_points = self.points[mask]
        filtered_colors = self.colors[mask] if self.colors.size > 0 else np.empty((0, 3), dtype=np.float32)
        filtered_normals = self.normals[mask] if self.normals.size > 0 else np.empty((0, 3), dtype=np.float32)
        filtered_distToGround = self.distToGround[mask] if self.distToGround.size > 0 else np.empty(0, dtype=np.float32)
        filtered_intensity = self.intensity[mask] if self.intensity.size > 0 else np.empty(0, dtype=np.float32)

        return Cluster(
            points=filtered_points,
            label=label,
            color=filtered_colors,
            normals=filtered_normals,
            distToGround=filtered_distToGround,
            intensity=filtered_intensity
        )

    def filter(self, condition):
        """
        Filter data based on a provided condition function that evaluates attributes of this Clusters instance.

        Parameters:
        - condition (callable): A function that takes a Clusters instance and returns a boolean array.

        Returns:
        Clusters: A new Clusters instance containing data that satisfies the condition.

        Example:
        --------
        # Filter to include only clusters with labels greater than 10
        filtered_clusters = clusters.filter(lambda cls: cls.labels > 10)
        """
        mask = condition(self)  # Pass the entire Clusters instance to the condition
        new_clusters = Clusters()
        new_clusters.points = self.points[mask]
        new_clusters.labels = self.labels[mask]
        new_clusters.colors = self.colors[mask]
        new_clusters.normals = self.normals[mask]
        new_clusters.distToGrounds = self.distToGrounds[mask]
        new_clusters.intensity = self.intensity[mask]

        return new_clusters

    def pick_clusters(self, selected_labels=None, random_color=True):
        """
        Interactively select clusters using Open3D visualization and return the selected cluster labels.

        Parameters:
        - selected_labels (list of int, optional): Labels of the clusters to allow picking from.
                                                  If None, all clusters are available for picking.
        - random_color (bool, optional): If True, assign random colors to each cluster.
                                         If False, use stored colors if available.

        Returns:
        - list of int: List of labels of the picked clusters.
        """
        # Initialize Open3D Visualizer
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()

        # Prepare the point cloud
        pcd = o3d.geometry.PointCloud()

        # Determine which points to display
        if selected_labels is not None:
            mask = np.isin(self.labels, selected_labels)
            points_to_show = self.points[mask]
            labels_to_use = self.labels[mask]
        else:
            points_to_show = self.points
            labels_to_use = self.labels

        # Assign points to the point cloud
        pcd.points = o3d.utility.Vector3dVector(points_to_show)

        # Handle coloring of points
        if random_color:
            colors = get_random_color(labels_to_use)
        else:
            colors = self.colors[mask] if selected_labels is not None else self.colors

        # Assign colors to the point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Add geometry to the visualizer and enable picking
        vis.add_geometry(pcd)
        vis.run()  # User can pick points in the GUI
        vis.destroy_window()

        # Process picked points to find their labels
        picked_indices = vis.get_picked_points()
        picked_labels = [labels_to_use[idx] for idx in picked_indices if idx < len(labels_to_use)]

        return picked_labels

    def show(self, selected_labels=None, random_color=False, downsample=1):
        """
        Visualize the clusters using Open3D. This method supports optional filtering by labels,
        optional random coloring, and downsampling.

        Parameters:
        - selected_labels (list of int, optional): A list of labels specifying which clusters to visualize.
                                                   If None, all clusters are visualized.
        - random_color (bool, optional): If True, assigns random colors to each cluster.
                                         If False, uses stored colors if available.
        - downsample (float, optional): A fraction (0 < downsample <= 1) specifying the downsampling ratio.
                                        Points will be randomly removed to achieve this ratio.

        Notes:
        - Normals will only be assigned if their count matches the count of points.
        - Warnings are issued if the number of normals does not match the number of points.
        - Error messages are displayed if the downsampling parameter is out of expected range.
        """
        # Initialize Open3D Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        pcd = self.pcd

        # Determine which points to display
        if selected_labels is not None:
            # Prepare the point cloud
            pcd = o3d.geometry.PointCloud()

            # Filter points based on selected labels
            mask = np.isin(self.labels, selected_labels)
            points_to_show = self.points[mask]
            labels_to_use = self.labels[mask]
            colors_to_use = self.colors[mask] if not random_color else get_random_color(selected_labels)
            normals_to_use = self.normals[mask] if self.normals.size == self.points.size else None
        else:
            points_to_show = self.points
            labels_to_use = self.labels
            colors_to_use = self.colors if not random_color else get_random_color(np.unique(self.labels))
            normals_to_use = self.normals if self.normals.size == self.points.size else None

        # Assign points to the point cloud
        pcd.points = o3d.utility.Vector3dVector(points_to_show)

        # Assign normals, if exists, to the point cloud
        if len(self.normals) == len(self.points):
            pcd.normals = o3d.utility.Vector3dVector(normals_to_use)
        elif 0 < len(self.normals) < len(self.points):
            raise ValueError("The number of normals must match the number of points or be zero.")

        # Handle coloring of points
        if random_color:
            # Generate random colors based on labels
            colors_to_use = get_random_color(labels_to_use)

        # Assign colors to the point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors_to_use)

        # Downsampling for large point clouds to make it manageable
        if 0 < downsample < 1:
            pcd = pcd.random_down_sample(downsample)
        elif downsample <= 0 or downsample > 1:
            raise ValueError("Downsample value must be between 0 and 1 (exclusive).")

        # Add geometry to the visualizer and display it
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

    def __repr__(self):
        return f"Clusters1({len(np.unique(self.labels))} unique clusters, {len(self.points)} total points)"
