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
    - block_name: The params of the block to insert.
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
    - app_name: The application params registering the XData (must be unique).
    - data_dict: A dictionary containing key-value pairs of data.
    """
    # Ensure the application params is registered in the DXF document
    doc = entity.doc  # Get the document to which the entity belongs
    if app_name not in doc.appids:
        doc.appids.new(app_name)

    # Start adding XData with the application params
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
    - Main Key: Feature params.
    - Sub Key: Worksheet params, pointing to a dictionary of properties for that feature.
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
                # Otherwise, create a new sub-dictionary with the worksheet params as the key
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
    This function predicts the labels for cluster_labels of points using a specified model.

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
        # Invalid cluster_labels return 'None' for prediction and probability
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
    # Return the list of predictions for all cluster_labels.
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
    clustersPoints (array-like): The points in the cluster_labels, presumably in a 3D space.
    z_diffs (array-like): The distance to the ground for each point in the cluster_labels.
    bottomDistanceToGround (float): The minimum distance from the ground to include a point.
    topDistanceToGround (float): The maximum distance from the ground to include a point.

    Returns:
    array-like: A subset of clustersPoints, including only the points that meet the height criteria.
    """

    # Create a boolean mask: True if a point's Z-axis difference is within the specified range, False otherwise.
    # The '&' operator is used here for element-wise logical AND operation, comparing each element in 'z_diffs'
    # with 'bottomDistanceToGround' and 'topDistanceToGround'.
    z_diffs_mask = (z_diffs >= bottomDistanceToGround) & (z_diffs <= topDistanceToGround)

    # Return the points in the cluster_labels that meet the criteria, using the boolean mask to filter them.
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
    package (str): The params of the package to check.
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
