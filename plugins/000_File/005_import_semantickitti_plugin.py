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
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from plugins.interfaces import ActionPlugin
from config.config import global_variables
from core.point_cloud import PointCloud
from core.data_node import DataNode
from core.feature_classes import FeatureClasses


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
            }
        }

    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the import operation.

        Args:
            main_window: The main application window
            params: Dictionary containing import options
        """
        # Multi-file selection dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            main_window,
            "Select SemanticKITTI .bin Files",
            "",
            "SemanticKITTI Files (*.bin);;All Files (*)"
        )

        if not file_paths:
            return

        # Sort files to maintain order (000000.bin, 000001.bin, etc.)
        file_paths = sorted(file_paths)

        # Import based on merge_scans option
        if params.get("merge_scans", False):
            self._import_merged(file_paths, params, main_window)
        else:
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
        load_labels = params.get("load_labels", True)
        apply_poses = params.get("apply_poses", False)
        max_distance = params.get("max_distance", 30.0)

        # Load poses and calibration if needed
        poses = None
        Tr_velo_to_cam = None
        trajectory_min_bound = None
        if apply_poses:
            poses_path = self._get_poses_path(file_paths[0])
            if poses_path:
                poses = self._load_poses(poses_path)

                # Load calibration (velodyne to camera transform)
                calib_path = self._get_calib_path(file_paths[0])
                if calib_path:
                    Tr_velo_to_cam = self._load_calib(calib_path)

                # Calculate consistent origin from trajectory (in KITTI coords)
                scanner_positions = np.array([pose[:3, 3] for pose in poses])
                trajectory_min_bound_kitti = scanner_positions.min(axis=0)

                # Convert to viewer coordinates: [X, Y, Z] -> [X, Z, -Y]
                trajectory_min_bound = np.array([
                    trajectory_min_bound_kitti[0],   # X unchanged
                    trajectory_min_bound_kitti[2],   # Y = Z_old
                    -trajectory_min_bound_kitti[1]   # Z = -Y_old
                ])
            else:
                QMessageBox.warning(
                    main_window,
                    "Poses Not Found",
                    "poses.txt not found. Importing without pose transformation."
                )
                apply_poses = False

        success_count = 0
        failed_files = []

        for file_path in file_paths:
            try:
                # Load .bin file
                raw_data = np.fromfile(file_path, dtype=np.float32)

                if len(raw_data) % 4 != 0:
                    failed_files.append((os.path.basename(file_path), "Invalid data size"))
                    continue

                data = raw_data.reshape(-1, 4)
                points = data[:, :3].astype(np.float64)  # Convert to float64 for transformation precision
                intensity = data[:, 3]  # remission/intensity

                # Load labels BEFORE filtering
                semantic_labels = None
                instance_ids = None
                if load_labels:
                    label_path = self._get_label_path(file_path)
                    if label_path:
                        semantic_labels, instance_ids = self._load_labels(label_path)

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

                # Apply pose if requested
                if apply_poses and poses:
                    frame_num = self._extract_frame_number(os.path.basename(file_path))
                    if frame_num < len(poses):
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

                # Always use intensity for PointCloud colors (labels already loaded and filtered above)
                intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
                colors = np.column_stack([intensity_norm] * 3).astype(np.float32)

                # Create PointCloud
                filename = os.path.basename(file_path)
                point_cloud = PointCloud(points_translated, colors=colors)
                point_cloud.name = filename
                point_cloud.translation = min_bound

                # Add intensity attribute
                point_cloud.add_attribute('intensity', intensity)

                # Add PointCloud to data manager and get UID
                pc_uid = self._add_to_data_manager(point_cloud)

                # If labels loaded, create FeatureClasses as child
                if semantic_labels is not None:
                    feature_classes = self._create_feature_classes(semantic_labels)
                    self._add_feature_classes_to_data_manager(feature_classes, pc_uid, filename)

                success_count += 1

            except Exception as e:
                failed_files.append((os.path.basename(file_path), str(e)))

        # Show summary
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

        # Concatenate all data
        try:
            merged_points = np.vstack(all_points)
            merged_intensity = np.concatenate(all_intensity)
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

        # Translate to origin for float32 precision
        # Use trajectory-based origin for consistent alignment across different filters
        points_translated = (merged_points - trajectory_min_bound).astype(np.float32)

        # Always use intensity for PointCloud colors
        intensity_norm = (merged_intensity - merged_intensity.min()) / (merged_intensity.max() - merged_intensity.min() + 1e-6)
        colors = np.column_stack([intensity_norm] * 3).astype(np.float32)

        # Concatenate labels if loaded
        merged_semantic = None
        if all_semantic:
            merged_semantic = np.concatenate(all_semantic)

        # Create PointCloud with standard name
        point_cloud = PointCloud(points_translated, colors=colors)
        point_cloud.name = "SemanticKITTI"
        point_cloud.translation = trajectory_min_bound

        # Add intensity attribute
        point_cloud.add_attribute('intensity', merged_intensity)

        # Add PointCloud to data manager and get UID
        pc_uid = self._add_to_data_manager(point_cloud)

        # If labels loaded, create FeatureClasses as child
        if merged_semantic is not None:
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
        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Create DataNode for FeatureClasses (use standard name to match classification workflow)
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

        # Add to data nodes collection
        fc_uid = data_nodes.add_node(fc_node)

        # Update tree widget as child of PointCloud (use standard name)
        tree_widget.add_branch(str(fc_uid), str(parent_uid), "feature_classes", is_root=False)
        tree_widget.update_cache_tooltip(str(fc_uid), fc_node.memory_size)

    def _add_to_data_manager(self, point_cloud: PointCloud) -> str:
        """
        Add loaded point cloud to data manager and tree widget.

        Args:
            point_cloud: The PointCloud object to add

        Returns:
            UID of the created DataNode
        """
        data_manager = global_variables.global_data_manager
        data_nodes = global_variables.global_data_nodes
        tree_widget = global_variables.global_tree_structure_widget

        # Create DataNode
        data_node = DataNode(
            params=point_cloud.name,
            data=point_cloud,
            data_type="point_cloud",
            parent_uid=None,
            depends_on=None,
            tags=[]
        )

        # Calculate memory size
        data_node.memory_size = data_manager._calculate_point_cloud_memory(point_cloud)

        # Add to data nodes collection
        uid = data_nodes.add_node(data_node)

        # Update tree widget (root nodes are always cached)
        tree_widget.add_branch(str(uid), "", point_cloud.name, is_root=True)
        tree_widget.update_cache_tooltip(str(uid), data_node.memory_size)

        return uid
