# plugins/analysis/tf_mls_densification_plugin.py
from typing import Dict, Any, List, Tuple
import numpy as np

from plugins.interfaces import AnalysisPlugin
from core.data_node import DataNode
from core.point_cloud import PointCloud


class MLSAugmentationPlugin(AnalysisPlugin):
    """
    Plugin for point cloud densification using TensorFlow and MLS.

    This plugin identifies low-density regions in a point cloud and adds points
    to those regions using Moving Least Squares surface fitting, creating a more
    uniform point density throughout the point cloud.
    """

    def get_name(self) -> str:
        """Return the unique name for this plugin."""
        return "mls_densification"

    def get_parameters(self) -> Dict[str, Any]:
        """Define the parameters for MLS densification."""
        return {
            "voxel_size": {
                "type": "float",
                "default": 0.05,
                "min": 0.001,
                "max": 1.0,
                "label": "Voxel Size",
                "description": "Size of voxels for density analysis (smaller values give finer control)"
            },
            "target_density": {
                "type": "int",
                "default": 15,
                "min": 1,
                "max": 100,
                "label": "Target Density",
                "description": "Target number of points per voxel (regions with fewer points will be densified)"
            },
            "mls_radius": {
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "label": "MLS Radius",
                "description": "Radius for MLS surface fitting (larger values produce smoother surfaces)"
            },
            "polynomial_degree": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 3,
                "label": "Polynomial Degree",
                "description": "Degree of polynomial for MLS fitting (higher values can capture more detail)"
            },
            "use_gpu": {
                "type": "dropdown",
                "options": {"true": "Yes (if available)", "false": "No (use CPU)"},
                "default": "true",
                "label": "Use GPU Acceleration",
                "description": "Use GPU for faster processing if available"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute MLS densification on the point cloud using TensorFlow.

        This method identifies sparse regions in the point cloud and adds points in
        those regions using MLS surface fitting, creating a more uniform density.
        """
        # Extract parameters
        voxel_size = params["voxel_size"]
        target_density = params["target_density"]
        mls_radius = params["mls_radius"]
        polynomial_degree = params["polynomial_degree"]
        use_gpu = params["use_gpu"] == "true"

        # Extract point cloud data
        input_pc: PointCloud = data_node.data
        points = input_pc.points
        has_colors = hasattr(input_pc, 'colors') and input_pc.colors is not None
        has_normals = hasattr(input_pc, 'normals') and input_pc.normals is not None

        print(f"[TF-MLS] Starting densification on point cloud with {len(points)} points")
        print(f"[TF-MLS] Target density: {target_density} points per voxel of size {voxel_size}")

        try:
            import tensorflow as tf

            # Configure TensorFlow to use GPU or CPU
            if not use_gpu:
                print("[TF-MLS] Forcing CPU execution as requested")
                tf.config.set_visible_devices([], 'GPU')
            else:
                # Check if GPU is available
                gpus = tf.config.list_physical_devices('GPU')
                if len(gpus) > 0:
                    print(f"[TF-MLS] Using GPU acceleration: {gpus[0].name}")
                    # Set memory growth to avoid allocating all GPU memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    print("[TF-MLS] No GPU available, falling back to CPU")

            # Convert points to TensorFlow tensor
            tf_points = tf.convert_to_tensor(points, dtype=tf.float32)

            # STEP 1: Voxelize the space and count points per voxel
            print("[TF-MLS] Analyzing point density using voxel grid")

            # Scale points to voxel coordinates and convert to integer indices
            voxel_indices = tf.cast(tf.floor(tf_points / voxel_size), tf.int32)

            # Create a unique string key for each voxel for grouping
            voxel_keys = tf.strings.reduce_join(
                tf.strings.as_string(voxel_indices),
                axis=1,
                separator=','
            )

            # Count points per voxel and identify unique voxels
            unique_voxels, unique_idx, voxel_counts = tf.unique_with_counts(voxel_keys)

            # STEP 2: Identify low-density voxels
            sparse_mask = voxel_counts < target_density
            sparse_voxels = tf.boolean_mask(unique_voxels, sparse_mask)
            sparse_counts = tf.boolean_mask(voxel_counts, sparse_mask)

            # Find all point indices belonging to sparse voxels
            is_sparse = tf.reduce_any(
                tf.equal(
                    tf.expand_dims(voxel_keys, 1),
                    tf.expand_dims(sparse_voxels, 0)
                ),
                axis=1
            )

            # Get points in sparse regions
            sparse_points_indices = tf.where(is_sparse)[:, 0]
            sparse_points = tf.gather(tf_points, sparse_points_indices)

            # Get points in dense regions (these stay unchanged)
            dense_points_indices = tf.where(tf.logical_not(is_sparse))[:, 0]
            dense_points = tf.gather(tf_points, dense_points_indices)

            print(f"[TF-MLS] Found {len(sparse_voxels)} low-density voxels out of {len(unique_voxels)}")
            print(f"[TF-MLS] Sparse regions contain {len(sparse_points)} points")

            # STEP 3: For each sparse voxel, determine how many points to add
            # and create a distribution for point generation

            # Convert sparse_voxels to actual voxel indices
            sparse_voxel_keys = [key.numpy().decode('utf-8') for key in sparse_voxels]
            sparse_voxel_indices = np.array([
                list(map(int, key.split(','))) for key in sparse_voxel_keys
            ])

            # Calculate voxel centers
            voxel_centers = (sparse_voxel_indices + 0.5) * voxel_size

            # Calculate points to add for each voxel
            points_to_add_per_voxel = target_density - sparse_counts.numpy()

            print(f"[TF-MLS] Need to add approximately {np.sum(points_to_add_per_voxel)} new points")

            # STEP 4: Build a KDTree for neighbor searches using TensorFlow
            # (Use CPU for this part as TF doesn't have a direct GPU KDTree)
            # For large point clouds, we'll use batching to process voxels

            # Convert back to numpy for efficient neighbor search
            points_np = tf_points.numpy()

            # We'll use scikit-learn for the KDTree which is faster for this specific task
            from sklearn.neighbors import KDTree
            tree = KDTree(points_np)

            # STEP 5: Apply MLS augmentation for each sparse voxel in batches
            batch_size = 100  # Process this many voxels at once
            num_voxels = len(voxel_centers)
            num_batches = (num_voxels + batch_size - 1) // batch_size

            # Prepare a list to collect all new points
            all_new_points = []
            all_new_colors = [] if has_colors else None
            all_new_normals = [] if has_normals else None

            print(f"[TF-MLS] Processing {num_voxels} sparse voxels in {num_batches} batches")

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_voxels)

                if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                    print(f"[TF-MLS] Processing batch {batch_idx + 1}/{num_batches}")

                # Get voxel centers for this batch
                batch_centers = voxel_centers[start_idx:end_idx]
                batch_points_to_add = points_to_add_per_voxel[start_idx:end_idx]

                # Find all points within MLS radius of each voxel center
                batch_neighbors_indices = tree.query_radius(
                    batch_centers,
                    r=mls_radius,
                    return_distance=False
                )

                # Process each voxel in the batch
                for i, (center, neighbor_indices, points_to_add) in enumerate(
                        zip(batch_centers, batch_neighbors_indices, batch_points_to_add)
                ):
                    # Skip if not enough neighbors for fitting
                    min_neighbors = (polynomial_degree + 1) * (polynomial_degree + 2) // 2
                    if len(neighbor_indices) < min_neighbors:
                        continue

                    # Get neighboring points for MLS fitting
                    neighbor_points = points_np[neighbor_indices]

                    # Convert to TensorFlow tensors for GPU processing
                    tf_center = tf.convert_to_tensor(center, dtype=tf.float32)
                    tf_neighbors = tf.convert_to_tensor(neighbor_points, dtype=tf.float32)

                    # Shift to local coordinate system centered at voxel center
                    centered_neighbors = tf_neighbors - tf_center

                    # Construct polynomial features for MLS
                    # For a polynomial of degree 2 in 3D, we need features:
                    # [1, x, y, z, x², y², z², xy, xz, yz]

                    # Start with basic coordinates
                    x = centered_neighbors[:, 0]
                    y = centered_neighbors[:, 1]
                    z = centered_neighbors[:, 2]

                    # Build feature matrix based on polynomial degree
                    if polynomial_degree == 1:
                        # Linear features: [1, x, y, z]
                        features = tf.stack([
                            tf.ones_like(x),
                            x, y, z
                        ], axis=1)
                    elif polynomial_degree == 2:
                        # Quadratic features: [1, x, y, z, x², y², z², xy, xz, yz]
                        features = tf.stack([
                            tf.ones_like(x),
                            x, y, z,
                            x * x, y * y, z * z,
                            x * y, x * z, y * z
                        ], axis=1)
                    else:  # polynomial_degree == 3
                        # Cubic features (adding x³, y³, z³, x²y, x²z, xy², xz², y²z, yz²)
                        features = tf.stack([
                            tf.ones_like(x),
                            x, y, z,
                            x * x, y * y, z * z,
                            x * y, x * z, y * z,
                            x * x * x, y * y * y, z * z * z,
                            x * x * y, x * x * z, x * y * y, x * z * z, y * y * z, y * z * z
                        ], axis=1)

                    # Compute weights based on distance (Gaussian kernel)
                    squared_dists = tf.reduce_sum(tf.square(centered_neighbors), axis=1)
                    weights = tf.exp(-squared_dists / (2 * (mls_radius / 3) ** 2))

                    # Apply weighted least squares to fit the polynomial
                    # We'll solve the normal equations: (X^T W X)β = X^T W y
                    weighted_features = features * tf.expand_dims(weights, 1)
                    normal_matrix = tf.matmul(
                        tf.transpose(weighted_features),
                        features
                    )

                    # Right-hand sides for x, y, z coordinates
                    rhs_x = tf.matmul(
                        tf.transpose(weighted_features),
                        tf.expand_dims(x, 1)
                    )
                    rhs_y = tf.matmul(
                        tf.transpose(weighted_features),
                        tf.expand_dims(y, 1)
                    )
                    rhs_z = tf.matmul(
                        tf.transpose(weighted_features),
                        tf.expand_dims(z, 1)
                    )

                    # Solve the systems to get coefficients for x, y, z
                    # Adding a small regularization term for numerical stability
                    reg = 1e-10 * tf.eye(tf.shape(normal_matrix)[0], dtype=tf.float32)
                    normal_matrix_reg = normal_matrix + reg

                    # Solve the three systems of equations
                    try:
                        coeffs_x = tf.linalg.solve(normal_matrix_reg, rhs_x)
                        coeffs_y = tf.linalg.solve(normal_matrix_reg, rhs_y)
                        coeffs_z = tf.linalg.solve(normal_matrix_reg, rhs_z)
                    except tf.errors.InvalidArgumentError:
                        # If matrix is not invertible, skip this voxel
                        continue

                    # Now we have the MLS surface fitted as a polynomial
                    # Generate new points within the voxel using this surface

                    # Generate uniform 3D grid of points within the voxel
                    num_points = int(points_to_add)
                    if num_points <= 0:
                        continue

                    # Determine number of points per dimension based on total
                    points_per_dim = max(1, int(np.ceil(num_points ** (1 / 3))))

                    # Create evenly spaced points within the voxel
                    grid_positions = tf.linspace(0.1, 0.9, points_per_dim)

                    # Create a 3D grid of points
                    x_grid, y_grid, z_grid = tf.meshgrid(
                        grid_positions,
                        grid_positions,
                        grid_positions
                    )

                    # Flatten and combine to get grid points
                    grid_points = tf.stack([
                        tf.reshape(x_grid, [-1]),
                        tf.reshape(y_grid, [-1]),
                        tf.reshape(z_grid, [-1])
                    ], axis=1)

                    # Scale and shift to fit within the voxel
                    voxel_min = sparse_voxel_indices[start_idx + i] * voxel_size
                    grid_points = grid_points * voxel_size + voxel_min

                    # Only take as many points as we need
                    grid_points = grid_points[:num_points]

                    # Project these points onto the MLS surface
                    # First center them relative to the voxel center
                    centered_grid = grid_points - tf_center

                    # For each grid point, compute the feature vector
                    grid_x = centered_grid[:, 0]
                    grid_y = centered_grid[:, 1]
                    grid_z = centered_grid[:, 2]

                    # Build feature matrix for grid points
                    if polynomial_degree == 1:
                        grid_features = tf.stack([
                            tf.ones_like(grid_x),
                            grid_x, grid_y, grid_z
                        ], axis=1)
                    elif polynomial_degree == 2:
                        grid_features = tf.stack([
                            tf.ones_like(grid_x),
                            grid_x, grid_y, grid_z,
                            grid_x * grid_x, grid_y * grid_y, grid_z * grid_z,
                            grid_x * grid_y, grid_x * grid_z, grid_y * grid_z
                        ], axis=1)
                    else:  # polynomial_degree == 3
                        grid_features = tf.stack([
                            tf.ones_like(grid_x),
                            grid_x, grid_y, grid_z,
                            grid_x * grid_x, grid_y * grid_y, grid_z * grid_z,
                            grid_x * grid_y, grid_x * grid_z, grid_y * grid_z,
                            grid_x * grid_x * grid_x, grid_y * grid_y * grid_y, grid_z * grid_z * grid_z,
                            grid_x * grid_x * grid_y, grid_x * grid_x * grid_z,
                            grid_x * grid_y * grid_y, grid_x * grid_z * grid_z,
                            grid_y * grid_y * grid_z, grid_y * grid_z * grid_z
                        ], axis=1)

                    # Compute projected points
                    projected_x = tf.matmul(grid_features, coeffs_x)
                    projected_y = tf.matmul(grid_features, coeffs_y)
                    projected_z = tf.matmul(grid_features, coeffs_z)

                    # Combine and shift back to world coordinates
                    projected_points = tf.concat([projected_x, projected_y, projected_z], axis=1)
                    new_points = projected_points + tf.expand_dims(tf_center, 0)

                    # Add these new points to our collection
                    all_new_points.append(new_points.numpy())

                    # Interpolate colors for new points if needed
                    if has_colors:
                        # Use inverse distance weighting to interpolate colors
                        neighbor_colors = input_pc.colors[neighbor_indices]

                        # Convert to tensors
                        tf_neighbor_points = tf.convert_to_tensor(neighbor_points, dtype=tf.float32)
                        tf_neighbor_colors = tf.convert_to_tensor(neighbor_colors, dtype=tf.float32)

                        # For each new point, compute weighted average of colors
                        new_colors = []
                        for point in new_points:
                            # Compute distances to neighbors
                            dists = tf.sqrt(tf.reduce_sum(
                                tf.square(tf_neighbor_points - point), axis=1
                            ))

                            # Compute weights (inverse distance)
                            # Add small epsilon to avoid division by zero
                            color_weights = 1.0 / (dists + 1e-10)
                            color_weights = color_weights / tf.reduce_sum(color_weights)

                            # Compute weighted average color
                            avg_color = tf.reduce_sum(
                                tf.expand_dims(color_weights, 1) * tf_neighbor_colors,
                                axis=0
                            )

                            new_colors.append(avg_color.numpy())

                        all_new_colors.append(np.array(new_colors))

                    # Compute normals for new points if needed
                    if has_normals:
                        # For simplicity, we'll compute the normal from the MLS surface
                        # For a fitted polynomial surface, the normal can be derived from the partial derivatives

                        # We won't implement this here since it's complex and specific to the polynomial form
                        # Instead, we'll estimate normals from the neighbor points
                        neighbor_normals = input_pc.normals[neighbor_indices]

                        # For each new point, assign the normal of the closest original point
                        new_normals = []
                        for point in new_points:
                            # Find closest neighbor
                            dists = tf.sqrt(tf.reduce_sum(
                                tf.square(tf_neighbor_points - point), axis=1
                            ))
                            closest_idx = tf.argmin(dists)

                            # Assign its normal
                            new_normals.append(neighbor_normals[closest_idx])

                        all_new_normals.append(np.array(new_normals))

            # STEP 6: Combine original points with new points
            if all_new_points:
                # Convert lists of arrays to single arrays
                new_points = np.vstack(all_new_points)

                if has_colors:
                    new_colors = np.vstack(all_new_colors)

                if has_normals:
                    new_normals = np.vstack(all_new_normals)

                print(f"[TF-MLS] Generated {len(new_points)} new points")

                # Combine with original points
                combined_points = np.vstack([points, new_points])

                if has_colors:
                    combined_colors = np.vstack([input_pc.colors, new_colors])

                if has_normals:
                    combined_normals = np.vstack([input_pc.normals, new_normals])

                # Create final point cloud
                result_pc = PointCloud(
                    "Densified_" + input_pc.name,
                    points=combined_points.astype(np.float32)
                )

                if has_colors:
                    result_pc.colors = combined_colors.astype(np.float32)

                if has_normals:
                    result_pc.normals = combined_normals.astype(np.float32)

                print(f"[TF-MLS] Final point cloud has {len(combined_points)} points")
            else:
                print("[TF-MLS] No new points were generated, returning original point cloud")
                result_pc = input_pc

            # Return the result
            dependencies = [data_node.uid]
            return result_pc, "point_cloud", dependencies

        except ImportError as e:
            print(f"[TF-MLS] Import error: {str(e)}")
            print("[TF-MLS] Please install TensorFlow and scikit-learn: pip install tensorflow scikit-learn")
            raise ImportError("Required libraries not installed")

        except Exception as e:
            print(f"[TF-MLS] Error during processing: {str(e)}")
            raise