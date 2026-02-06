# Standard library imports
import time
import math
import os
import subprocess
import sys
import warnings
import logging
import traceback
# from importlib.metadata import version, PackageNotFoundError

# Third-party imports
import numpy as np
import pandas as pd
import open3d as o3d
import torch  # Used by eigenvalue_utils
from scipy.stats import norm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# Local application imports
from services.eigenvalue_utils import EigenvalueUtils

# Get logger for this module
logger = logging.getLogger(__name__)

# Singleton for EigenvalueUtils - preserves KD-tree cache across calls
_eigenvalue_utils_instance = None


def _get_eigenvalue_utils():
    """Get or create singleton EigenvalueUtils instance."""
    global _eigenvalue_utils_instance
    if _eigenvalue_utils_instance is None:
        _eigenvalue_utils_instance = EigenvalueUtils()
    return _eigenvalue_utils_instance


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

        logger.debug(f"PointCloud.__init__() called")

        # Validation points
        if not isinstance(points, np.ndarray):
            raise ValueError("Points must be a numpy array with shape (n, 3)")

        logger.debug(f"  Points shape: {points.shape}")
        logger.debug(f"  Points dtype: {points.dtype}")
        logger.debug(f"  Points memory: {points.nbytes / 1024 / 1024:.2f} MB")

        self.points = points
        self.colors = colors
        self.normals = normals
        self.translation = np.array([0, 0, 0])
        self.name = kwargs.get('params', None)
        self.uuid = kwargs.get('uid', None)
        self.parent_uuid = kwargs.get('parent_uuid', None)
        self.child_uuid = kwargs.get('child_uuid', None)

        if colors is not None:
            logger.debug(f"  Colors shape: {colors.shape}, memory: {colors.nbytes / 1024 / 1024:.2f} MB")
        if normals is not None:
            logger.debug(f"  Normals shape: {normals.shape}, memory: {normals.nbytes / 1024 / 1024:.2f} MB")

        self.prediction = ''
        self.probability = 0
        self.model_weight = ''

        # OBB dimensions - lazy evaluation (calculated on first access)
        self._center = [0, 0, 0]
        self._length = 0
        self._width = 0
        self._height = 0
        self._obb_calculated = False

        logger.debug(f"  PointCloud created: {self.size} points, name={self.name}")

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

    # --- OBB Properties (Lazy Evaluation) ---
    # These properties calculate OBB dimensions only when first accessed,
    # avoiding expensive computation for operations that don't need them.

    @property
    def length(self):
        """Get OBB length (lazy calculation on first access)."""
        if not self._obb_calculated:
            self._calculate_obb_lazy()
        return self._length

    @length.setter
    def length(self, value):
        """Set length directly (used by _update_obb_dim)."""
        self._length = value

    @property
    def width(self):
        """Get OBB width (lazy calculation on first access)."""
        if not self._obb_calculated:
            self._calculate_obb_lazy()
        return self._width

    @width.setter
    def width(self, value):
        """Set width directly (used by _update_obb_dim)."""
        self._width = value

    @property
    def height(self):
        """Get OBB height (lazy calculation on first access)."""
        if not self._obb_calculated:
            self._calculate_obb_lazy()
        return self._height

    @height.setter
    def height(self, value):
        """Set height directly (used by _update_obb_dim)."""
        self._height = value

    @property
    def center(self):
        """Get OBB center (lazy calculation on first access)."""
        if not self._obb_calculated:
            self._calculate_obb_lazy()
        return self._center

    @center.setter
    def center(self, value):
        """Set center directly (used by _update_2d_bbox)."""
        self._center = value

    def _calculate_obb_lazy(self):
        """
        Calculate OBB dimensions on first access (lazy evaluation).

        This method is called automatically when length, width, height, or center
        are accessed for the first time. The result is cached for subsequent accesses.
        """
        if self._obb_calculated:
            return

        logger.debug(f"_calculate_obb_lazy() called for {self.size:,} points")

        if self.size <= 3:
            logger.debug("  Too few points for OBB")
        elif self.size > 10_000_000:
            logger.debug("  Using 2D bbox for large point cloud")
            self._update_2d_bbox()
        elif self._are_points_coplanar():
            logger.debug("  Using 2D bbox for coplanar points")
            self._update_2d_bbox()
        else:
            logger.debug("  Calculating full OBB")
            self._update_obb_dim()

        self._obb_calculated = True
        logger.debug(f"  OBB calculated: L={self._length:.2f}, W={self._width:.2f}, H={self._height:.2f}")

    @classmethod
    def merge(cls, point_clouds: list, name: str = "Merged") -> 'PointCloud':
        """
        Merge multiple point clouds into a single PointCloud.

        Memory-optimized: pre-allocates arrays and fills directly instead of
        using np.concatenate which creates intermediate copies.

        Args:
            point_clouds: List of PointCloud instances to merge
            name: Name for the merged point cloud

        Returns:
            PointCloud: Merged point cloud with all points, colors, normals, and attributes

        Example:
            merged = PointCloud.merge([pc1, pc2, pc3], name="Combined")
        """
        if not point_clouds:
            raise ValueError("Cannot merge empty list of point clouds")

        if len(point_clouds) == 1:
            # Single point cloud - return a shallow copy
            pc = point_clouds[0]
            result = cls(
                points=pc.points,
                colors=pc.colors,
                normals=pc.normals,
                params=name
            )
            result.attributes = pc.attributes.copy()
            return result

        logger.debug(f"PointCloud.merge(): merging {len(point_clouds)} clouds")

        # Calculate total size for pre-allocation
        total_points = sum(pc.size for pc in point_clouds)
        logger.debug(f"  Total points: {total_points:,}")

        # Pre-allocate and fill points
        merged_points = np.empty((total_points, 3), dtype=np.float32)
        offset = 0
        for pc in point_clouds:
            n = pc.size
            merged_points[offset:offset + n] = pc.points
            offset += n

        # Merge colors (fill with white if missing)
        merged_colors = cls._merge_optional_array(
            point_clouds, 'colors', (3,), fill_value=1.0
        )

        # Merge normals (fill with zero if missing)
        merged_normals = cls._merge_optional_array(
            point_clouds, 'normals', (3,), fill_value=0.0
        )

        # Merge legacy attributes (intensity, distToGround)
        merged_intensity = cls._merge_optional_array(
            point_clouds, 'intensity', (), fill_value=0.0
        )
        merged_dist_to_ground = cls._merge_optional_array(
            point_clouds, 'distToGround', (), fill_value=0.0
        )

        # Create merged point cloud
        merged_pc = cls(
            points=merged_points,
            colors=merged_colors,
            normals=merged_normals,
            intensity=merged_intensity if merged_intensity is not None else np.array([]),
            distToGround=merged_dist_to_ground if merged_dist_to_ground is not None else np.array([]),
            params=name
        )

        # Merge custom attributes
        cls._merge_attributes_into(point_clouds, merged_pc, total_points)

        logger.debug(f"  Merge complete: {merged_pc.size:,} points")
        return merged_pc

    @classmethod
    def _merge_optional_array(
        cls,
        point_clouds: list,
        attr_name: str,
        shape_suffix: tuple,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Merge an optional array attribute from multiple point clouds.

        Memory-optimized: pre-allocates output array and fills directly.

        Args:
            point_clouds: List of point clouds
            attr_name: Name of the attribute (e.g., 'colors', 'normals')
            shape_suffix: Shape suffix for each point (e.g., (3,) for RGB)
            fill_value: Value to use when attribute is missing

        Returns:
            np.ndarray or None: Merged array, or None if none have the attribute
        """
        # Check if any point cloud has this attribute with data
        has_attr = any(
            hasattr(pc, attr_name) and getattr(pc, attr_name) is not None
            and len(getattr(pc, attr_name)) > 0
            for pc in point_clouds
        )

        if not has_attr:
            return None

        # Calculate total size and pre-allocate
        total_points = sum(pc.size for pc in point_clouds)
        output_shape = (total_points,) + shape_suffix
        merged = np.empty(output_shape, dtype=np.float32)

        # Fill directly (memory efficient - no intermediate arrays)
        offset = 0
        for pc in point_clouds:
            n = pc.size
            attr_val = getattr(pc, attr_name, None)

            if attr_val is not None and len(attr_val) > 0:
                merged[offset:offset + n] = attr_val
            else:
                merged[offset:offset + n] = fill_value
            offset += n

        return merged

    @classmethod
    def _merge_attributes_into(
        cls,
        point_clouds: list,
        merged_pc: 'PointCloud',
        total_points: int
    ) -> None:
        """
        Merge custom attributes from all point clouds into the merged result.

        Memory-optimized: pre-allocates arrays and fills directly.
        Attributes present in some but not all point clouds will have NaN/fill
        values for points from clouds where the attribute was missing.

        Args:
            point_clouds: Source point clouds
            merged_pc: Target merged point cloud
            total_points: Total number of points (for pre-allocation)
        """
        # Collect all unique attribute names
        all_attr_names = set()
        for pc in point_clouds:
            if hasattr(pc, 'attributes') and pc.attributes:
                all_attr_names.update(pc.attributes.keys())

        if not all_attr_names:
            return

        # Merge each attribute
        for attr_name in all_attr_names:
            # Determine attribute shape and dtype from first cloud that has it
            attr_shape = ()
            attr_dtype = np.float32
            is_array_attr = False
            for pc in point_clouds:
                if hasattr(pc, 'attributes') and attr_name in pc.attributes:
                    sample = pc.attributes[attr_name]
                    # Skip non-array attributes (like _cluster_names, _cluster_colors dicts)
                    if not isinstance(sample, np.ndarray):
                        break
                    is_array_attr = True
                    if sample.ndim == 1:
                        attr_shape = ()  # Scalar per point
                    else:
                        attr_shape = sample.shape[1:]  # Shape after first dim
                    attr_dtype = sample.dtype
                    break

            # Skip non-array attributes
            if not is_array_attr:
                continue

            # Pre-allocate merged array
            output_shape = (total_points,) + attr_shape
            merged_attr = np.empty(output_shape, dtype=attr_dtype)

            # Determine fill value based on dtype
            if np.issubdtype(attr_dtype, np.floating):
                fill_value = np.nan
            else:
                fill_value = -1

            # Fill directly (memory efficient - no intermediate arrays)
            offset = 0
            for pc in point_clouds:
                n = pc.size
                if hasattr(pc, 'attributes') and attr_name in pc.attributes:
                    merged_attr[offset:offset + n] = pc.attributes[attr_name]
                else:
                    merged_attr[offset:offset + n] = fill_value
                offset += n

            # Add to merged point cloud
            merged_pc.add_attribute(attr_name, merged_attr)

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

    def get_attribute(self, name):
        """Get a per-point attribute by name."""
        return self.attributes.get(name)

    def with_attribute(self, name: str, values: np.ndarray) -> 'PointCloud':
        """
        Return a new PointCloud with an additional attribute.

        Memory-efficient: creates a shallow copy that shares array references
        with the original, only adding the new attribute.

        Args:
            name: Name of the attribute to add
            values: Array of values (first dimension must match point count)

        Returns:
            PointCloud: New point cloud with the added attribute

        Example:
            pc_with_labels = point_cloud.with_attribute('labels', label_array)
        """
        # Create shallow copy - shares array references (memory efficient)
        result = PointCloud(
            points=self.points,
            colors=self.colors,
            normals=self.normals,
            intensity=getattr(self, 'intensity', np.array([])),
            distToGround=getattr(self, 'distToGround', np.array([])),
            params=self.name
        )
        # Copy attributes dict (shallow) and add new attribute
        result.attributes = self.attributes.copy()
        result.add_attribute(name, values)
        return result

    def translate(self, translation):
        """
        Translate the cluster by a given translation vector.

        Parameters:
        - translation (np.ndarray): A 1D numpy array representing the translation vector [dx, dy, dz].
        """
        self.points += translation
        self.translation += translation
        self._obb_calculated = False  # Invalidate OBB cache (center changes)

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

        self._obb_calculated = False  # Invalidate OBB cache (points changed)

    def dbscan(self, eps=0.05, min_points=10, return_clusters_object=False):
        """
        Apply DBSCAN clustering to the points in the cluster.

        Uses sklearn DBSCAN with ball_tree algorithm for O(n log n) performance.

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
        from config.config import global_variables

        if len(self.points) == 0:
            raise ValueError("The cluster has no points for DBSCAN clustering.")

        # Get DBSCAN backend from registry (automatically selects best available)
        backend_registry = global_variables.global_backend_registry

        if backend_registry is not None:
            # Use the backend registry system
            dbscan_backend = backend_registry.get_dbscan()
            labels = dbscan_backend.run(self.points, eps, min_points)
        else:
            # Fallback if registry not initialized (e.g., during testing)
            labels = self._dbscan_fallback(eps, min_points)

        if return_clusters_object:
            from core.entities.clusters import Clusters
            clusters = Clusters(labels)
            clusters.set_random_color()
            return clusters
        else:
            return labels

    def _dbscan_fallback(self, eps, min_points):
        """
        Fallback DBSCAN implementation when backend registry is not available.

        Uses sklearn with ball_tree algorithm for O(n log n) performance.
        This is used during testing or if the registry is not initialized.
        """
        import time
        start_time = time.time()

        try:
            from sklearn.cluster import DBSCAN
            import sklearn

            print(f"Using scikit-learn DBSCAN with ball_tree (version {sklearn.__version__})")
            print(f"Processing {len(self.points):,} points with eps={eps}, min_samples={min_points}")

            db = DBSCAN(eps=eps, min_samples=min_points, algorithm='ball_tree', n_jobs=-1)
            labels = db.fit_predict(self.points)

            self._print_dbscan_results(labels, start_time, "scikit-learn ball_tree")
            return labels

        except ImportError:
            # Fall back to Open3D
            print("scikit-learn not available. Using Open3D DBSCAN.")
            print(f"Processing {len(self.points):,} points with eps={eps}, min_points={min_points}")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
                labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

            self._print_dbscan_results(labels, start_time, "Open3D CPU")
            return labels

    def _print_dbscan_results(self, labels, start_time, backend_name):
        """Print DBSCAN results summary."""
        import time
        elapsed_time = time.time() - start_time
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        print(f"\n{'='*60}")
        print(f"DBSCAN COMPLETED ({backend_name})")
        print(f"{'='*60}")
        print(f"  Total points:     {len(self.points):,}")
        print(f"  Clusters found:   {n_clusters}")
        print(f"  Noise points:     {n_noise:,} ({100*n_noise/len(self.points):.1f}%)")
        print(f"  Processing time:  {elapsed_time:.2f} seconds")
        print(f"{'='*60}\n")

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

        # Use singleton to preserve KD-tree cache across calls
        analyser = _get_eigenvalue_utils()

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
        Uses the backend registry to automatically select GPU (CuPy) or CPU (NumPy) acceleration.

        This method is memory-optimized: instead of deepcopy + mask, it creates filtered arrays directly.

        Parameters:
        - mask (np.ndarray): A boolean array where True values indicate points to include in the subset.
        - inplace (bool): If True, modifies the current point cloud in-place. Default is False.

        Returns:
        - PointCloud: A new PointCloud instance containing the subset if inplace is False.
        """
        from config.config import global_variables

        # Check if mask selects enough points (at least 4)
        if not np.count_nonzero(mask) >= 4:
            print("Not enough points found for subset.")
            if inplace:
                self.points = np.array([])
                self.parent = 0
                self.prediction = ''
                self.probability = 0
                self.model_weight = ''
                self.length = 0
                self.width = 0
                self.height = 0
                self.feature = []
                self.metadata = {}
            else:
                return PointCloud(np.empty((1, 3), dtype=np.float32))
            return

        # Get masking backend from registry
        backend_registry = global_variables.global_backend_registry
        if backend_registry is not None:
            masking_backend = backend_registry.get_masking()
        else:
            masking_backend = None

        if inplace:
            self._apply_mask_inplace(mask, masking_backend)
        else:
            # Memory-optimized: create new PointCloud with filtered arrays directly
            # This avoids deepcopy which would duplicate all arrays before filtering
            return self._create_masked_subset(mask, masking_backend)

    def _create_masked_subset(self, mask, masking_backend=None):
        """
        Create a new PointCloud with masked data directly (no deepcopy).

        This is more memory-efficient than deepcopy + inplace mask, as it only
        allocates memory for the filtered arrays, not the full copy.

        Uses batch masking when GPU backend is available to minimize GPU transfers.

        Args:
            mask: Boolean mask array
            masking_backend: MaskingBackend instance or None for fallback

        Returns:
            PointCloud: New point cloud with only masked points
        """
        true_count = np.sum(mask)
        logger.debug(f"_create_masked_subset() called, mask has {true_count:,} True values")

        # Optimization: if all points pass the mask, return self (no copy needed)
        # This prevents expensive no-op copies for large point clouds
        if true_count == len(mask):
            logger.debug("  All points pass mask - returning self (no copy)")
            return self

        # Check if batch masking is available (GPU backend with batch method)
        use_batch = (masking_backend is not None and
                     hasattr(masking_backend, 'apply_mask_to_arrays_batch'))

        if use_batch:
            # Collect all arrays for batch processing
            arrays_to_mask = {
                'points': self.points,
                'colors': self.colors if self.colors is not None and len(self.colors) > 0 else None,
                'normals': self.normals if self.normals is not None and len(self.normals) > 0 else None,
                'intensity': self.intensity if hasattr(self, 'intensity') and hasattr(self.intensity, 'shape') and self.intensity.shape[0] > 0 else None,
                'distToGround': self.distToGround if hasattr(self, 'distToGround') and hasattr(self.distToGround, 'shape') and self.distToGround.shape[0] > 0 else None,
            }

            # Add custom attributes that need masking
            for attr_name, attr_value in self.attributes.items():
                if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == len(mask):
                    arrays_to_mask[f'attr_{attr_name}'] = attr_value

            # Single batch operation - mask transferred to GPU only ONCE
            masked_results = masking_backend.apply_mask_to_arrays_batch(arrays_to_mask, mask)

            # Extract results with dtype preservation
            new_points = masked_results['points']
            if new_points is not None:
                new_points = new_points.astype(self.points.dtype)

            new_colors = masked_results['colors']
            if new_colors is not None:
                new_colors = new_colors.astype(self.colors.dtype)

            new_normals = masked_results['normals']
            if new_normals is not None:
                new_normals = new_normals.astype(self.normals.dtype)

            new_intensity = masked_results['intensity'] if masked_results['intensity'] is not None else np.array([])
            new_dist_to_ground = masked_results['distToGround'] if masked_results['distToGround'] is not None else np.array([])

        else:
            # Fallback: apply mask to each array individually
            def apply_mask(array):
                if array is None:
                    return None
                if masking_backend is not None:
                    return masking_backend.apply_mask_to_array(array, mask)
                else:
                    return array[mask]

            new_points = apply_mask(self.points)
            if new_points is not None:
                new_points = new_points.astype(self.points.dtype)

            new_colors = None
            if self.colors is not None and len(self.colors) > 0:
                new_colors = apply_mask(self.colors)
                if new_colors is not None:
                    new_colors = new_colors.astype(self.colors.dtype)

            new_normals = None
            if self.normals is not None and len(self.normals) > 0:
                new_normals = apply_mask(self.normals)
                if new_normals is not None:
                    new_normals = new_normals.astype(self.normals.dtype)

            new_intensity = np.array([])
            if hasattr(self, 'intensity') and hasattr(self.intensity, 'shape') and self.intensity.shape[0] > 0:
                new_intensity = apply_mask(self.intensity)

            new_dist_to_ground = np.array([])
            if hasattr(self, 'distToGround') and hasattr(self.distToGround, 'shape') and self.distToGround.shape[0] > 0:
                new_dist_to_ground = apply_mask(self.distToGround)

        # Create new PointCloud with filtered data
        subset = PointCloud(
            points=new_points,
            colors=new_colors,
            normals=new_normals,
            intensity=new_intensity,
            distToGround=new_dist_to_ground,
            params=self.name,
            feature=self.feature.copy() if isinstance(self.feature, list) else self.feature,
            metadata=self.metadata.copy() if isinstance(self.metadata, dict) else self.metadata
        )

        # Copy non-array attributes (shallow copy is fine for these)
        subset.translation = self.translation.copy()
        subset.uuid = self.uuid
        subset.parent_uuid = self.parent_uuid
        subset.child_uuid = self.child_uuid
        subset.prediction = self.prediction
        subset.probability = self.probability
        subset.model_weight = self.model_weight

        # Handle custom attributes
        if use_batch:
            # Extract from batch results
            for attr_name, attr_value in self.attributes.items():
                if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == len(mask):
                    subset.attributes[attr_name] = masked_results[f'attr_{attr_name}']
                else:
                    subset.attributes[attr_name] = attr_value
        else:
            # Apply mask individually (fallback path)
            for attr_name, attr_value in self.attributes.items():
                if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == len(mask):
                    subset.attributes[attr_name] = apply_mask(attr_value)
                else:
                    subset.attributes[attr_name] = attr_value

        logger.debug(f"_create_masked_subset() completed: {subset.size:,} points")

        # Single cleanup after all masking operations (instead of per-array cleanup)
        if masking_backend is not None and masking_backend.is_gpu:
            from services.memory_manager import MemoryManager
            MemoryManager.cleanup()

        return subset

    def _apply_mask_inplace(self, mask, masking_backend=None):
        """
        Apply mask to all point cloud attributes in-place.

        Uses batch masking when GPU backend is available to minimize GPU transfers.

        Args:
            mask: Boolean mask array
            masking_backend: MaskingBackend instance or None for fallback
        """
        logger.debug(f"_apply_mask_inplace() called, mask has {np.sum(mask):,} True values")

        # Check if batch masking is available (GPU backend with batch method)
        use_batch = (masking_backend is not None and
                     hasattr(masking_backend, 'apply_mask_to_arrays_batch'))

        # Store original dtypes for preservation
        points_dtype = self.points.dtype if hasattr(self, 'points') and len(self.points) > 0 else None
        colors_dtype = self.colors.dtype if hasattr(self, 'colors') and self.colors is not None and self.colors.shape[0] > 0 else None
        normals_dtype = self.normals.dtype if hasattr(self, 'normals') and self.normals is not None and self.normals.shape[0] > 0 else None

        if use_batch:
            # Collect all arrays for batch processing
            arrays_to_mask = {
                'points': self.points if points_dtype else None,
                'colors': self.colors if colors_dtype else None,
                'normals': self.normals if normals_dtype else None,
                'intensity': self.intensity if hasattr(self, 'intensity') and hasattr(self.intensity, 'shape') and self.intensity.shape[0] > 0 else None,
                'distToGround': self.distToGround if hasattr(self, 'distToGround') and hasattr(self.distToGround, 'shape') and self.distToGround.shape[0] > 0 else None,
            }

            # Add custom attributes that need masking
            for attr_name, attr_value in self.attributes.items():
                if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == len(mask):
                    arrays_to_mask[f'attr_{attr_name}'] = attr_value

            # Single batch operation - mask transferred to GPU only ONCE
            masked_results = masking_backend.apply_mask_to_arrays_batch(arrays_to_mask, mask)

            # Apply results back to self
            if masked_results['points'] is not None:
                self.points = masked_results['points'].astype(points_dtype)
                logger.debug(f"  Points masked: {len(self.points):,} points remaining")

            if masked_results['colors'] is not None:
                self.colors = masked_results['colors'].astype(colors_dtype)

            if masked_results['normals'] is not None:
                self.normals = masked_results['normals'].astype(normals_dtype)

            if masked_results['intensity'] is not None:
                self.intensity = masked_results['intensity']

            if masked_results['distToGround'] is not None:
                self.distToGround = masked_results['distToGround']

            # Apply custom attributes from batch results
            for attr_name in list(self.attributes.keys()):
                attr_value = self.attributes[attr_name]
                if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == len(mask):
                    self.attributes[attr_name] = masked_results[f'attr_{attr_name}']

        else:
            # Fallback: apply mask to each array individually
            def apply_mask(array):
                if masking_backend is not None:
                    return masking_backend.apply_mask_to_array(array, mask)
                else:
                    return array[mask]

            # Apply mask to points
            if points_dtype:
                logger.debug(f"  Applying mask to points ({len(self.points):,} points)...")
                self.points = apply_mask(self.points).astype(points_dtype)
                logger.debug(f"  Points masked: {len(self.points):,} points remaining")

            # Apply mask to colors
            if colors_dtype:
                logger.debug(f"  Applying mask to colors...")
                self.colors = apply_mask(self.colors).astype(colors_dtype)
                logger.debug(f"  Colors masked")

            # Apply mask to normals
            if normals_dtype:
                logger.debug(f"  Applying mask to normals...")
                self.normals = apply_mask(self.normals).astype(normals_dtype)
                logger.debug(f"  Normals masked")

            # Apply mask to intensity
            if hasattr(self, 'intensity') and hasattr(self.intensity, 'shape') and self.intensity.shape[0] > 0:
                logger.debug(f"  Applying mask to intensity...")
                self.intensity = apply_mask(self.intensity)
                logger.debug(f"  Intensity masked")

            # Apply mask to distToGround
            if hasattr(self, 'distToGround') and hasattr(self.distToGround, 'shape') and self.distToGround.shape[0] > 0:
                logger.debug(f"  Applying mask to distToGround...")
                self.distToGround = apply_mask(self.distToGround)
                logger.debug(f"  distToGround masked")

            # Process custom attributes
            logger.debug(f"  Processing {len(self.attributes)} custom attributes...")
            for attr_name in list(self.attributes.keys()):
                attr_value = self.attributes[attr_name]
                if isinstance(attr_value, np.ndarray) and attr_value.shape[0] == len(mask):
                    logger.debug(f"    Masking attribute: {attr_name}")
                    self.attributes[attr_name] = apply_mask(attr_value)

        # Invalidate OBB cache - will be recalculated lazily when accessed
        self._obb_calculated = False
        logger.debug(f"  OBB cache invalidated (lazy recalculation on next access)")

        # Single cleanup after all masking operations (instead of per-array cleanup)
        if masking_backend is not None and masking_backend.is_gpu:
            from services.memory_manager import MemoryManager
            MemoryManager.cleanup()

        logger.debug(f"_apply_mask_inplace() completed")

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

        self._obb_calculated = False  # Invalidate OBB cache (points changed)

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

        self._obb_calculated = False  # Invalidate OBB cache (points transformed)

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

        try:
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
            self.center = list(obb.center)

        except RuntimeError:
            # OBB calculation failed (degenerate geometry) - use 2D bounding box fallback
            self._update_2d_bbox()

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

        logger.debug(
            f"Calculated 2D bounding box: L={self._length:.2f}, W={self._width:.2f}, H={self._height:.6f}")

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
