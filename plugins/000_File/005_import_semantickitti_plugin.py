"""
SemanticKITTI .bin File Import Plugin

This plugin imports SemanticKITTI .bin point cloud files with support for:
- Multi-file selection via file dialog
- Semantic labels (.label files) with class coloring
- Pose transformations (poses.txt) for world coordinates
- Separate or merged import modes
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import logging
import traceback
import gc
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.point_cloud import PointCloud
from core.data_node import DataNode
from core.feature_classes import FeatureClasses

# Get logger for this module
logger = logging.getLogger(__name__)


# SemanticKITTI semantic class colors (RGB 0-255)
SEMANTICKITTI_COLORS = {
    0: (0, 0, 0),          # unlabeled
    1: (0, 0, 255),        # outlier
    10: (245, 150, 100),   # car
    11: (245, 230, 100),   # bicycle
    13: (250, 80, 100),    # bus
    15: (150, 60, 30),     # motorcycle
    18: (255, 0, 0),       # truck
    20: (180, 30, 80),     # other-vehicle
    30: (255, 0, 255),     # person
    31: (255, 150, 255),   # bicyclist
    32: (75, 0, 75),       # motorcyclist
    40: (75, 0, 175),      # road
    44: (175, 0, 75),      # parking
    48: (255, 200, 0),     # sidewalk
    49: (255, 120, 50),    # other-ground
    50: (0, 175, 0),       # building
    51: (135, 60, 0),      # fence
    52: (150, 240, 80),    # other-structure
    60: (255, 240, 150),   # lane-marking
    70: (0, 175, 0),       # vegetation
    71: (135, 60, 0),      # trunk
    72: (80, 240, 150),    # terrain
    80: (150, 240, 255),   # pole
    81: (0, 0, 255),       # traffic-sign
    99: (255, 255, 50),    # other-object
    252: (245, 150, 100),  # moving-car
    253: (200, 40, 255),   # moving-bicyclist
    254: (30, 30, 255),    # moving-person
    255: (90, 30, 150),    # moving-motorcyclist
    256: (255, 0, 0),      # moving-truck
    257: (250, 80, 100),   # moving-bus
    258: (180, 30, 80),    # moving-other-vehicle
    259: (255, 0, 0),      # moving-on-rails
}

# SemanticKITTI semantic class names
SEMANTICKITTI_CLASS_NAMES = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-truck",
    257: "moving-bus",
    258: "moving-other-vehicle",
}


class ImportSemanticKITTIPlugin(ActionPlugin):
    """Plugin for importing SemanticKITTI .bin point cloud files."""

    def get_name(self) -> str:
        return "import_semantickitti"

    def get_parameters(self) -> Dict[str, Any]:
        """Return parameter schema for import options."""
        return {
            "merge_scans": {
                "type": "bool",
                "default": False,
                "label": "Merge Scans",
                "description": "Combine selected scans into single point cloud (requires poses)"
            },
            "load_labels": {
                "type": "bool",
                "default": True,
                "label": "Load Labels",
                "description": "Load .label files and color by semantic class"
            },
            "apply_poses": {
                "type": "bool",
                "default": False,
                "label": "Apply Poses",
                "description": "Transform points to world coordinates using poses.txt"
            },
            "max_distance": {
                "type": "float",
                "default": 30.0,
                "min": 0.0,
                "max": 120.0,
                "label": "Max Distance (m)",
                "description": "Keep points within this distance from scanner (0 = unlimited)"
            },
            "subsample_ratio": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 1.0,
                "label": "Subsample Ratio",
                "description": "Keep this fraction of points (0.25 = 25%). Use < 1.0 for large imports."
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the import operation.

        Args:
            main_window: The main application window
            params: Dictionary containing import options
        """
        logger.info("=" * 60)
        logger.info("SemanticKITTI Import Started")
        logger.info(f"Parameters: {params}")

        # Multi-file selection dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            main_window,
            "Select SemanticKITTI .bin Files",
            "",
            "SemanticKITTI Files (*.bin);;All Files (*)"
        )

        if not file_paths:
            logger.info("No files selected, aborting import")
            return

        # Sort files to maintain order (000000.bin, 000001.bin, etc.)
        file_paths = sorted(file_paths)
        logger.info(f"Selected {len(file_paths)} files for import")
        logger.info(f"First file: {file_paths[0]}")
        logger.info(f"Last file: {file_paths[-1]}")

        # Import based on merge_scans option
        if params.get("merge_scans", False):
            logger.info("Using MERGED import mode")
            self._import_merged(file_paths, params, main_window)
        else:
            logger.info("Using SEPARATE import mode")
            self._import_separate(file_paths, params, main_window)

    # Helper methods for loading labels and poses

    def _get_label_path(self, bin_path: str) -> Optional[str]:
        """
        Get corresponding .label path for a .bin file.

        Args:
            bin_path: Path to .bin file (e.g., .../sequences/00/velodyne/000000.bin)

        Returns:
            Path to .label file if it exists, None otherwise
        """
        dir_path = os.path.dirname(bin_path)
        parent_dir = os.path.dirname(dir_path)
        filename = os.path.basename(bin_path).replace('.bin', '.label')
        label_path = os.path.join(parent_dir, 'labels', filename)
        return label_path if os.path.exists(label_path) else None

    def _get_poses_path(self, bin_path: str) -> Optional[str]:
        """
        Get poses.txt path for a sequence folder.

        Args:
            bin_path: Path to any .bin file in the sequence

        Returns:
            Path to poses.txt if it exists, None otherwise
        """
        dir_path = os.path.dirname(bin_path)
        parent_dir = os.path.dirname(dir_path)
        poses_path = os.path.join(parent_dir, 'poses.txt')
        return poses_path if os.path.exists(poses_path) else None

    def _get_calib_path(self, bin_path: str) -> Optional[str]:
        """
        Get calib.txt path for a sequence folder.

        Args:
            bin_path: Path to any .bin file in the sequence

        Returns:
            Path to calib.txt if it exists, None otherwise
        """
        dir_path = os.path.dirname(bin_path)
        parent_dir = os.path.dirname(dir_path)
        calib_path = os.path.join(parent_dir, 'calib.txt')
        return calib_path if os.path.exists(calib_path) else None

    def _load_calib(self, calib_path: str) -> Optional[np.ndarray]:
        """
        Load calibration matrix (Tr: velodyne to camera) from calib.txt.

        Args:
            calib_path: Path to calib.txt

        Returns:
            4x4 transformation matrix from velodyne to camera, or None if not found
        """
        try:
            with open(calib_path, 'r') as f:
                for line in f:
                    if line.startswith('Tr:'):
                        values = [float(x) for x in line.split()[1:]]
                        if len(values) == 12:
                            Tr = np.eye(4)
                            Tr[:3, :4] = np.array(values).reshape(3, 4)
                            return Tr
        except:
            pass
        return None

    def _load_labels(self, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load .label file and extract semantic labels and instance IDs.

        Args:
            label_path: Path to .label file

        Returns:
            Tuple of (semantic_labels, instance_ids)
        """
        labels = np.fromfile(label_path, dtype=np.uint32)
        semantic = labels & 0xFFFF  # Lower 16 bits
        instance = labels >> 16     # Upper 16 bits
        return semantic, instance

    def _load_poses(self, poses_path: str) -> List[np.ndarray]:
        """
        Load poses.txt and return list of 4x4 transformation matrices.

        KITTI poses represent camera-to-world transformations.

        Args:
            poses_path: Path to poses.txt

        Returns:
            List of 4x4 transformation matrices (camera-to-world)
        """
        poses = []
        with open(poses_path, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                if len(values) == 12:  # Valid pose line
                    T = np.eye(4)
                    T[:3, :4] = np.array(values).reshape(3, 4)
                    poses.append(T)
        return poses

    def _apply_pose(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Transform points using 4x4 pose matrix.

        Args:
            points: (N, 3) array of XYZ coordinates
            pose: (4, 4) transformation matrix

        Returns:
            Transformed (N, 3) array
        """
        ones = np.ones((points.shape[0], 1))
        points_h = np.hstack([points, ones])  # Homogeneous coordinates
        transformed = (pose @ points_h.T).T
        return transformed[:, :3]

    def _labels_to_colors(self, semantic_labels: np.ndarray) -> np.ndarray:
        """
        Convert semantic labels to RGB colors.

        Args:
            semantic_labels: Array of semantic class IDs

        Returns:
            (N, 3) array of RGB colors in range [0, 1]
        """
        colors = np.zeros((len(semantic_labels), 3), dtype=np.float32)
        for label, color in SEMANTICKITTI_COLORS.items():
            mask = semantic_labels == label
            colors[mask] = np.array(color) / 255.0
        return colors

    def _extract_frame_number(self, filename: str) -> int:
        """Extract frame number from filename (e.g., '000123.bin' -> 123)."""
        return int(os.path.splitext(filename)[0])

    # Import methods

    def _import_separate(self, file_paths: List[str], params: Dict[str, Any], main_window) -> None:
        """
        Import multiple .bin files as separate point clouds.

        Args:
            file_paths: List of paths to .bin files
            params: Import parameters
            main_window: Main application window
        """
        logger.info("-" * 40)
        logger.info("_import_separate() started")

        load_labels = params.get("load_labels", True)
        apply_poses = params.get("apply_poses", False)
        max_distance = params.get("max_distance", 30.0)
        subsample_ratio = params.get("subsample_ratio", 1.0)

        logger.info(f"  load_labels: {load_labels}")
        logger.info(f"  apply_poses: {apply_poses}")
        logger.info(f"  max_distance: {max_distance}")
        logger.info(f"  subsample_ratio: {subsample_ratio}")

        # Load poses and calibration if needed
        poses = None
        Tr_velo_to_cam = None
        trajectory_min_bound = None
        if apply_poses:
            poses_path = self._get_poses_path(file_paths[0])
            if poses_path:
                logger.info(f"Loading poses from: {poses_path}")
                poses = self._load_poses(poses_path)
                logger.info(f"Loaded {len(poses)} poses")

                # Load calibration (velodyne to camera transform)
                calib_path = self._get_calib_path(file_paths[0])
                if calib_path:
                    Tr_velo_to_cam = self._load_calib(calib_path)
                    logger.info("Loaded calibration matrix")

                # Calculate consistent origin from trajectory (in KITTI coords)
                scanner_positions = np.array([pose[:3, 3] for pose in poses])
                trajectory_min_bound_kitti = scanner_positions.min(axis=0)

                # Convert to viewer coordinates: [X, Y, Z] -> [X, Z, -Y]
                trajectory_min_bound = np.array([
                    trajectory_min_bound_kitti[0],   # X unchanged
                    trajectory_min_bound_kitti[2],   # Y = Z_old
                    -trajectory_min_bound_kitti[1]   # Z = -Y_old
                ])
                logger.info(f"Trajectory min bound: {trajectory_min_bound}")
            else:
                logger.warning("poses.txt not found")
                QMessageBox.warning(
                    main_window,
                    "Poses Not Found",
                    "poses.txt not found. Importing without pose transformation."
                )
                apply_poses = False

        success_count = 0
        failed_files = []
        total_points_imported = 0

        for file_idx, file_path in enumerate(file_paths):
            logger.info(f"--- Processing file {file_idx + 1}/{len(file_paths)}: {os.path.basename(file_path)} ---")
            try:
                # Load .bin file
                logger.debug(f"  Reading binary file...")
                raw_data = np.fromfile(file_path, dtype=np.float32)
                logger.debug(f"  Raw data size: {len(raw_data)} floats ({raw_data.nbytes / 1024 / 1024:.2f} MB)")

                if len(raw_data) % 4 != 0:
                    logger.error(f"  Invalid data size: {len(raw_data)} not divisible by 4")
                    failed_files.append((os.path.basename(file_path), "Invalid data size"))
                    continue

                data = raw_data.reshape(-1, 4)
                points = data[:, :3].astype(np.float64)  # Convert to float64 for transformation precision
                intensity = data[:, 3]  # remission/intensity
                logger.debug(f"  Loaded {len(points)} points")

                # Load labels BEFORE filtering
                semantic_labels = None
                instance_ids = None
                if load_labels:
                    label_path = self._get_label_path(file_path)
                    if label_path:
                        logger.debug(f"  Loading labels from: {label_path}")
                        semantic_labels, instance_ids = self._load_labels(label_path)
                        logger.debug(f"  Loaded {len(semantic_labels)} labels")

                # Apply distance filter if requested
                original_count = len(points)
                if max_distance > 0:
                    distances = np.linalg.norm(points, axis=1)
                    mask = distances <= max_distance
                    points = points[mask]
                    intensity = intensity[mask]
                    if semantic_labels is not None:
                        semantic_labels = semantic_labels[mask]
                        instance_ids = instance_ids[mask]

                    filtered_count = len(points)
                    logger.debug(f"  Distance filter: {original_count} -> {filtered_count} points ({100*filtered_count/original_count:.1f}%)")

                # Apply subsample if ratio < 1.0
                if subsample_ratio < 1.0:
                    pre_subsample_count = len(points)
                    n_keep = max(1, int(len(points) * subsample_ratio))
                    indices = np.random.choice(len(points), n_keep, replace=False)
                    indices.sort()  # Maintain spatial order
                    points = points[indices]
                    intensity = intensity[indices]
                    if semantic_labels is not None:
                        semantic_labels = semantic_labels[indices]
                        instance_ids = instance_ids[indices]
                    logger.debug(f"  Subsample: {pre_subsample_count} -> {len(points)} points ({100*subsample_ratio:.1f}%)")

                # Apply pose if requested
                if apply_poses and poses:
                    frame_num = self._extract_frame_number(os.path.basename(file_path))
                    if frame_num < len(poses):
                        logger.debug(f"  Applying pose transformation for frame {frame_num}")
                        # Compose calibration with pose: world = pose @ Tr @ velodyne
                        if Tr_velo_to_cam is not None:
                            T_combined = poses[frame_num] @ Tr_velo_to_cam
                        else:
                            T_combined = poses[frame_num]
                        points = self._apply_pose(points, T_combined)

                # Convert KITTI coordinate system (X forward, Y left, Z up)
                # to viewer coordinate system (X forward, Y up, Z right)
                # Rotation: 90 degrees around X axis (counter-clockwise)
                # New: [X, Y, Z] = [X_old, Z_old, -Y_old]
                # Create new array to avoid in-place modification issues
                logger.debug(f"  Converting coordinate system...")
                points_converted = np.column_stack([
                    points[:, 0],   # X unchanged
                    points[:, 2],   # Y = Z_old
                    -points[:, 1]   # Z = -Y_old
                ])

                # Translate to origin for float32 precision
                # Use trajectory-based origin if poses were applied, otherwise use point cloud bounds
                if trajectory_min_bound is not None:
                    min_bound = trajectory_min_bound
                else:
                    min_bound = points_converted.min(axis=0)
                points_translated = (points_converted - min_bound).astype(np.float32)
                logger.debug(f"  Points array: {points_translated.shape}, {points_translated.nbytes / 1024 / 1024:.2f} MB")

                # Always use intensity for PointCloud colors (labels already loaded and filtered above)
                intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
                colors = np.column_stack([intensity_norm] * 3).astype(np.float32)
                logger.debug(f"  Colors array: {colors.shape}, {colors.nbytes / 1024 / 1024:.2f} MB")

                # Create PointCloud
                logger.debug(f"  Creating PointCloud object...")
                filename = os.path.basename(file_path)
                point_cloud = PointCloud(points_translated, colors=colors)
                point_cloud.name = filename
                point_cloud.translation = min_bound
                logger.debug(f"  PointCloud created with {point_cloud.size} points")

                # Add intensity attribute
                point_cloud.add_attribute('intensity', intensity)

                # Add PointCloud to data manager and get UID
                # Block signals if we have labels to add (prevents double render)
                has_labels = semantic_labels is not None
                logger.info(f"  Adding to DataManager (has_labels={has_labels})...")
                pc_uid = self._add_to_data_manager(point_cloud, block_signals=has_labels)
                logger.info(f"  Added with UID: {pc_uid}")

                # If labels loaded, create FeatureClasses as child
                # This triggers the single render for both branches
                if has_labels:
                    logger.debug(f"  Creating FeatureClasses...")
                    feature_classes = self._create_feature_classes(semantic_labels)
                    logger.debug(f"  Adding FeatureClasses to DataManager...")
                    self._add_feature_classes_to_data_manager(feature_classes, pc_uid, filename)

                success_count += 1
                total_points_imported += len(points_translated)
                logger.info(f"  SUCCESS: File {file_idx + 1} imported ({len(points_translated)} points)")

                # Force garbage collection every 10 files to prevent memory buildup
                if (file_idx + 1) % 10 == 0:
                    logger.debug(f"  Running garbage collection...")
                    gc.collect()

            except Exception as e:
                logger.error(f"  FAILED: {os.path.basename(file_path)}")
                logger.error(f"  Error: {str(e)}")
                logger.error(f"  Traceback:\n{traceback.format_exc()}")
                failed_files.append((os.path.basename(file_path), str(e)))

        # Show summary
        logger.info("=" * 40)
        logger.info(f"IMPORT COMPLETE: {success_count}/{len(file_paths)} files successful")
        logger.info(f"Total points imported: {total_points_imported:,}")
        if failed_files:
            logger.warning(f"Failed files: {len(failed_files)}")
            for filename, error in failed_files:
                logger.warning(f"  - {filename}: {error}")

        message = f"Successfully imported {success_count} of {len(file_paths)} files."
        if failed_files:
            message += "\n\nFailed files:\n"
            for filename, error in failed_files[:5]:
                message += f"- {filename}: {error}\n"
            if len(failed_files) > 5:
                message += f"... and {len(failed_files) - 5} more"
            QMessageBox.warning(main_window, "Import Complete", message)
        else:
            QMessageBox.information(main_window, "Import Complete", message)

        logger.info("_import_separate() completed")
        logger.info("=" * 60)

    def _import_merged(self, file_paths: List[str], params: Dict[str, Any], main_window) -> None:
        """
        Import multiple .bin files and merge into single point cloud.

        Args:
            file_paths: List of paths to .bin files
            params: Import parameters
            main_window: Main application window
        """
        load_labels = params.get("load_labels", True)
        max_distance = params.get("max_distance", 30.0)
        subsample_ratio = params.get("subsample_ratio", 1.0)

        # Load poses (mandatory for merge)
        poses_path = self._get_poses_path(file_paths[0])
        if not poses_path:
            QMessageBox.critical(
                main_window,
                "Poses Required",
                "poses.txt not found. Merged import requires poses for alignment."
            )
            return

        poses = self._load_poses(poses_path)

        # Load calibration (velodyne to camera transform)
        Tr_velo_to_cam = None
        calib_path = self._get_calib_path(file_paths[0])
        if calib_path:
            Tr_velo_to_cam = self._load_calib(calib_path)

        # Calculate consistent origin from trajectory (scanner positions)
        # This ensures all imports of same sequence align, regardless of filtering
        scanner_positions = np.array([pose[:3, 3] for pose in poses])
        trajectory_min_bound_kitti = scanner_positions.min(axis=0)

        # Convert trajectory_min_bound from KITTI to viewer coordinates
        # KITTI: X forward, Y left, Z up -> Viewer: X forward, Y up, Z right
        # [X, Y, Z] -> [X, Z, -Y]
        trajectory_min_bound = np.array([
            trajectory_min_bound_kitti[0],   # X unchanged
            trajectory_min_bound_kitti[2],   # Y = Z_old
            -trajectory_min_bound_kitti[1]   # Z = -Y_old
        ])

        # Lists to accumulate data
        all_points = []
        all_intensity = []
        all_semantic = []
        all_instance = []

        failed_files = []

        # Merging scans
        progress_interval = max(1, len(file_paths) // 20)  # Show progress every 5%

        for idx, file_path in enumerate(file_paths):
            try:
                # Load .bin file
                data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
                points = data[:, :3].astype(np.float64)  # Convert to float64 for transformation precision
                intensity = data[:, 3]

                # Load labels BEFORE filtering (so we can filter them too)
                semantic_labels = None
                instance_labels = None
                if load_labels:
                    label_path = self._get_label_path(file_path)
                    if label_path:
                        semantic_labels, instance_labels = self._load_labels(label_path)

                # Apply distance filter if requested
                original_count = len(points)
                if max_distance > 0:
                    distances = np.linalg.norm(points, axis=1)
                    mask = distances <= max_distance
                    points = points[mask]
                    intensity = intensity[mask]
                    if semantic_labels is not None:
                        semantic_labels = semantic_labels[mask]
                        instance_labels = instance_labels[mask]

                    filtered_count = len(points)
                    if idx % progress_interval == 0:
                        pass  # Filtered count logging

                # Apply subsample if ratio < 1.0
                if subsample_ratio < 1.0:
                    n_keep = max(1, int(len(points) * subsample_ratio))
                    indices = np.random.choice(len(points), n_keep, replace=False)
                    indices.sort()  # Maintain spatial order
                    points = points[indices]
                    intensity = intensity[indices]
                    if semantic_labels is not None:
                        semantic_labels = semantic_labels[indices]
                        instance_labels = instance_labels[indices]

                # Apply pose transformation (mandatory)
                frame_num = self._extract_frame_number(os.path.basename(file_path))
                if frame_num >= len(poses):
                    raise ValueError(f"No pose available for frame {frame_num}")

                # Compose calibration with pose: world = pose @ Tr @ velodyne
                if Tr_velo_to_cam is not None:
                    T_combined = poses[frame_num] @ Tr_velo_to_cam
                else:
                    T_combined = poses[frame_num]
                points_transformed = self._apply_pose(points, T_combined)

                # Convert KITTI coordinate system (X forward, Y left, Z up)
                # to viewer coordinate system (X forward, Y up, Z right)
                # Rotation: 90 degrees around X axis (counter-clockwise)
                # New: [X, Y, Z] = [X_old, Z_old, -Y_old]
                # Create new array to avoid in-place modification issues
                points_converted = np.column_stack([
                    points_transformed[:, 0],   # X unchanged
                    points_transformed[:, 2],   # Y = Z_old
                    -points_transformed[:, 1]   # Z = -Y_old
                ])

                # Accumulate
                all_points.append(points_converted)
                all_intensity.append(intensity)

                # Accumulate labels (already loaded and filtered above)
                if load_labels:
                    if semantic_labels is not None:
                        all_semantic.append(semantic_labels)
                        all_instance.append(instance_labels)
                    else:
                        # No labels for this file, use zeros (use filtered point count)
                        all_semantic.append(np.zeros(len(points), dtype=np.uint16))
                        all_instance.append(np.zeros(len(points), dtype=np.uint16))

            except Exception as e:
                failed_files.append((os.path.basename(file_path), str(e)))

        if not all_points:
            QMessageBox.critical(
                main_window,
                "Import Failed",
                "No files could be loaded successfully."
            )
            return

        # Memory-optimized merge: pre-allocate and fill directly
        # This avoids creating intermediate arrays from vstack/concatenate
        try:
            # Calculate total size for pre-allocation
            total_points = sum(len(pts) for pts in all_points)
            logger.info(f"Pre-allocating arrays for {total_points:,} points...")

            # Pre-allocate output arrays as float32 directly
            points_translated = np.empty((total_points, 3), dtype=np.float32)
            merged_intensity = np.empty(total_points, dtype=np.float32)

            # Fill directly with translation applied (avoids intermediate float64 array)
            offset = 0
            for pts, intensity in zip(all_points, all_intensity):
                n = len(pts)
                # Translate and convert to float32 in one step
                points_translated[offset:offset + n] = (pts - trajectory_min_bound).astype(np.float32)
                merged_intensity[offset:offset + n] = intensity
                offset += n

            # Clear source lists to free memory
            del all_points
            del all_intensity
            gc.collect()

        except MemoryError as e:
            QMessageBox.critical(
                main_window,
                "Memory Error",
                f"Not enough memory to merge {len(all_points)} scans.\n\n"
                f"Try importing as separate scans instead."
            )
            return
        except Exception as e:
            QMessageBox.critical(
                main_window,
                "Merge Failed",
                f"Failed to merge point clouds:\n{str(e)}"
            )
            return

        # Create intensity-based grayscale colors (memory efficient)
        intensity_min = merged_intensity.min()
        intensity_range = merged_intensity.max() - intensity_min + 1e-6
        # Pre-allocate colors array and fill directly (avoids column_stack copy)
        colors = np.empty((total_points, 3), dtype=np.float32)
        intensity_norm = (merged_intensity - intensity_min) / intensity_range
        colors[:, 0] = intensity_norm
        colors[:, 1] = intensity_norm
        colors[:, 2] = intensity_norm

        # Merge labels if loaded (pre-allocate and fill)
        merged_semantic = None
        if all_semantic:
            merged_semantic = np.empty(total_points, dtype=np.int32)
            offset = 0
            for sem in all_semantic:
                n = len(sem)
                merged_semantic[offset:offset + n] = sem
                offset += n
            del all_semantic
            gc.collect()

        # Check GPU memory before proceeding
        total_points = len(points_translated)
        data_size_mb = (points_translated.nbytes + colors.nbytes) / (1024 * 1024)
        logger.info(f"Merged data: {total_points:,} points, {data_size_mb:.1f} MB")

        try:
            from services.hardware_detector import HardwareDetector

            # Estimate GPU memory needed:
            # - VBO: points + colors combined (~1.5x data size for overhead)
            # - Masking operations during reconstruction: 3x data size
            # - Labels if present: additional memory
            vbo_required = data_size_mb * 1.5
            masking_required = data_size_mb * 3
            total_gpu_required = vbo_required + masking_required

            free_gpu_mb = HardwareDetector.get_free_gpu_memory_mb()

            if free_gpu_mb > 0:
                logger.info(f"GPU memory: {free_gpu_mb} MB free, need ~{total_gpu_required:.0f} MB")

                if total_gpu_required > free_gpu_mb:
                    # Warn user but don't block - operations will fall back to CPU
                    logger.warning(
                        f"Large dataset may exceed GPU memory. "
                        f"Operations will automatically fall back to CPU if needed."
                    )
                    QMessageBox.warning(
                        main_window,
                        "Large Dataset Warning",
                        f"This dataset ({total_points:,} points, {data_size_mb:.0f} MB) "
                        f"may exceed available GPU memory ({free_gpu_mb} MB free).\n\n"
                        f"Operations will automatically fall back to CPU if needed, "
                        f"but rendering may be slow or limited.\n\n"
                        f"Consider importing fewer scans or using a larger max_distance filter."
                    )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")

        # Create PointCloud with standard name
        logger.info(f"Creating PointCloud with {total_points:,} points...")
        point_cloud = PointCloud(points_translated, colors=colors)
        point_cloud.name = "SemanticKITTI"
        point_cloud.translation = trajectory_min_bound

        # Add intensity attribute
        point_cloud.add_attribute('intensity', merged_intensity)

        # Add PointCloud to data manager and get UID
        # Block signals if we have labels to add (prevents double render)
        has_labels = merged_semantic is not None
        pc_uid = self._add_to_data_manager(point_cloud, block_signals=has_labels)

        # If labels loaded, create FeatureClasses as child
        # This triggers the single render for both branches
        if has_labels:
            feature_classes = self._create_feature_classes(merged_semantic)
            self._add_feature_classes_to_data_manager(feature_classes, pc_uid, "SemanticKITTI")

        # Show summary
        first_frame = self._extract_frame_number(os.path.basename(file_paths[0]))
        last_frame = self._extract_frame_number(os.path.basename(file_paths[-1]))
        message = f"Merged {len(file_paths) - len(failed_files)} scans (frames {first_frame:06d}-{last_frame:06d}) into 'SemanticKITTI'"
        message += f"\nTotal points: {len(points_translated):,}"
        if failed_files:
            message += f"\n\n{len(failed_files)} files failed:\n"
            for filename, error in failed_files[:5]:
                message += f"- {filename}: {error}\n"
        QMessageBox.information(main_window, "Merge Complete", message)

    def _create_feature_classes(self, semantic_labels: np.ndarray) -> FeatureClasses:
        """
        Create FeatureClasses object from semantic labels.

        Args:
            semantic_labels: Array of semantic class IDs (SemanticKITTI format)

        Returns:
            FeatureClasses object with original SemanticKITTI label IDs
        """
        # Get unique labels present in data
        unique_labels = np.unique(semantic_labels)

        # Build class mapping using ORIGINAL label IDs (no remapping)
        class_mapping = {}
        class_colors = {}
        for label in unique_labels:
            original_id = int(label)
            class_name = SEMANTICKITTI_CLASS_NAMES.get(original_id, f"class_{original_id}")
            class_mapping[original_id] = class_name
            rgb = np.array(SEMANTICKITTI_COLORS.get(original_id, (128, 128, 128))) / 255.0
            class_colors[class_name] = rgb

        return FeatureClasses(
            labels=semantic_labels.astype(np.int32),
            class_mapping=class_mapping,
            class_colors=class_colors
        )

    def _add_feature_classes_to_data_manager(self, feature_classes: FeatureClasses,
                                              parent_uid: str, parent_name: str) -> None:
        """
        Add FeatureClasses as child node to data manager and tree widget.

        Args:
            feature_classes: The FeatureClasses object to add
            parent_uid: UID of parent PointCloud node
            parent_name: Name of parent PointCloud (for display)
        """
        logger.debug(f"    _add_feature_classes_to_data_manager() called for parent: {parent_name}")

        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Create DataNode for FeatureClasses (use standard name to match classification workflow)
        logger.debug(f"    Creating FeatureClasses DataNode...")
        fc_node = DataNode(
            params="feature_classes",
            data=feature_classes,
            data_type="feature_classes",
            parent_uid=parent_uid,
            depends_on=[parent_uid],
            tags=[]
        )

        # Calculate memory size (labels array + dicts)
        labels_size = feature_classes.labels.nbytes
        # Format as MB
        size_mb = labels_size / (1024 * 1024)
        fc_node.memory_size = f"{size_mb:.2f} MB"
        logger.debug(f"    FeatureClasses memory size: {fc_node.memory_size}")

        # Add to data nodes collection
        logger.debug(f"    Adding FeatureClasses to DataNodes collection...")
        fc_uid = data_nodes.add_node(fc_node)
        logger.debug(f"    FeatureClasses added with UID: {fc_uid}")

        # Hide parent BEFORE adding child (prevents rendering both when signal triggers)
        # This prevents rendering both parent and child which doubles memory usage
        from PyQt5.QtCore import Qt

        parent_uid_str = str(parent_uid)
        child_uid_str = str(fc_uid)

        # Update parent visibility BEFORE adding child branch
        if parent_uid_str in tree_widget.visibility_status:
            tree_widget.visibility_status[parent_uid_str] = False
        parent_item = tree_widget.branches_dict.get(parent_uid_str)
        if parent_item:
            parent_item.setCheckState(0, Qt.Unchecked)
            logger.debug(f"    Parent branch visibility set to False BEFORE adding child")

        # Now add the child branch (this triggers branch_added signal)
        logger.debug(f"    Adding FeatureClasses branch to tree widget...")
        try:
            tree_widget.add_branch(str(fc_uid), str(parent_uid), "feature_classes", is_root=False)
            tree_widget.update_cache_tooltip(str(fc_uid), fc_node.memory_size)
            logger.debug(f"    FeatureClasses branch added to tree")

        except Exception as e:
            logger.error(f"    Error adding FeatureClasses to tree: {e}")
            logger.error(f"    Traceback:\n{traceback.format_exc()}")
            raise

    def _add_to_data_manager(self, point_cloud: PointCloud, block_signals: bool = False) -> str:
        """
        Add loaded point cloud to data manager and tree widget.

        Args:
            point_cloud: The PointCloud object to add
            block_signals: If True, block signals to prevent auto-render
                          (use when adding child branches after this)

        Returns:
            UID of the created DataNode
        """
        logger.debug(f"    _add_to_data_manager() called for: {point_cloud.name}")

        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        if data_manager is None:
            logger.error("    global_data_manager is None!")
            raise RuntimeError("DataManager not initialized")
        if data_nodes is None:
            logger.error("    global_data_nodes is None!")
            raise RuntimeError("DataNodes not initialized")
        if tree_widget is None:
            logger.error("    global_tree_structure_widget is None!")
            raise RuntimeError("TreeStructureWidget not initialized")

        # Create DataNode
        logger.debug(f"    Creating DataNode...")
        data_node = DataNode(
            params=point_cloud.name,
            data=point_cloud,
            data_type="point_cloud",
            parent_uid=None,
            depends_on=None,
            tags=[]
        )

        # Calculate memory size
        logger.debug(f"    Calculating memory size...")
        data_node.memory_size = data_manager._calculate_point_cloud_memory(point_cloud)
        logger.debug(f"    Memory size: {data_node.memory_size}")

        # Add to data nodes collection
        logger.debug(f"    Adding to DataNodes collection...")
        uid = data_nodes.add_node(data_node)
        logger.debug(f"    DataNode added with UID: {uid}")

        # Update tree widget (root nodes are always cached)
        # Block signals if we're adding child branches after this to prevent double render
        logger.debug(f"    Updating tree widget (block_signals={block_signals})...")
        if block_signals:
            tree_widget.blockSignals(True)
        try:
            tree_widget.add_branch(str(uid), "", point_cloud.name, is_root=True)
            logger.debug(f"    Branch added to tree")
            tree_widget.update_cache_tooltip(str(uid), data_node.memory_size)
            logger.debug(f"    Cache tooltip updated")
        except Exception as e:
            logger.error(f"    Error adding branch to tree: {e}")
            logger.error(f"    Traceback:\n{traceback.format_exc()}")
            raise
        finally:
            if block_signals:
                tree_widget.blockSignals(False)

        logger.debug(f"    _add_to_data_manager() completed")
        return uid
