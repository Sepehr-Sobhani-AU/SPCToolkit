# core/node_reconstruction_manager.py
"""
    This module implements a Dynamic Task Execution Framework that allows for the execution of different types of
    tasks based on the task type. The Task class is an abstract base class that defines the interface for all tasks.
    For example the SubsampleTask and ClusterTask classes are concrete implementations of the task classes that
    perform subsampling and clustering tasks, respectively. The task_registry dictionary is a registry that maps task
    types to task classes.

    The "execute" method in each task class is responsible for executing the task.
    The main function demonstrates how to use the different task classes and execute them.
"""

from core.point_cloud import PointCloud
from core.data_node import DataNode

from tasks.apply_values import ApplyValues
from tasks.apply_clusters import ApplyClusters
from tasks.apply_masks import ApplyMasks
from tasks.apply_eigenvalues import ApplyEigenvalues
from tasks.apply_colors import ApplyColors
from tasks.apply_dist_to_ground import ApplyDistToGround
from tasks.apply_feature_classes import ApplyFeatureClasses
from tasks.apply_class_reference import ApplyClassReference



class NodeReconstructionManager:

    def __init__(self):
        """
        Initializes the NodeReconstructionManager with a task registry.

        The task registry maps data node types to their corresponding
        reconstruction tasks.
        """

        super().__init__()
        self.tasks_registry = {
            "masks": ApplyMasks,
            "cluster_labels": ApplyClusters,
            "values": ApplyValues,
            "eigenvalues": ApplyEigenvalues,
            "colors": ApplyColors,
            "dist_to_ground": ApplyDistToGround,
            "feature_classes": ApplyFeatureClasses,
            "class_reference": ApplyClassReference,
        }

    def reconstruct_node(self, point_cloud: PointCloud, data_node: DataNode) -> PointCloud:
        """
        Reconstructs a DataNode using the appropriate task based on the data type.

        Args:
            point_cloud (PointCloud): The root PointCloud instance to reconstruct data node from.
            data_node (DataNode): The DataNode instance to reconstruct.

        Returns:
            PointCloud: The reconstructed point cloud with data node properties applied.

        Raises:
            ValueError: If the data node type is not found in the task registry.
        """

        analysis_type = data_node.data_type
        if analysis_type not in self.tasks_registry:
            raise ValueError(f"Analysis type '{analysis_type}' not found in the task registry.")

        task = self.tasks_registry[analysis_type]
        task_instance = task(point_cloud, data_node.data)
        derived_point_cloud = task_instance.execute()

        return derived_point_cloud