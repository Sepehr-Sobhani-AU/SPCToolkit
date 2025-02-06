"""
    This module implements a Dynamic Task Execution Framework that allows for the execution of different types of
    tasks based on the task type. The Task class is an abstract base class that defines the interface for all tasks.
    For example the SubsampleTask and ClusterTask classes are concrete implementations of the task classes that
    perform subsampling and clustering tasks, respectively. The task_registry dictionary is a registry that maps task
    types to task classes.

    The "execute" method in each task class is responsible for executing the task.
    The main function demonstrates how to use the different task classes and execute them.
"""
import uuid
from typing import Dict, Any

from PyQt5.QtCore import pyqtSignal, QObject

from tasks.subsampling import Subsampling
from tasks.clustering import Clustering
from tasks.filtering import Filtering

from core.data_node import DataNode


class AnalysisManager(QObject):
    """
    Manages and executes analyses on DataNodes and tracks the results.

    Attributes:
        analyses (dict): Dictionary mapping UUIDs to their corresponding AnalysisResult instances.
    """
    analysis_completed = pyqtSignal(object, str, list, object, str, dict)

    def __init__(self):
        """
        Initializes the AnalysisManager with an empty analyses dictionary.
        """

        super().__init__()
        self.tasks_registry = {"subsampling": Subsampling,
                               "clustering": Clustering,
                               "filtering": Filtering
                               }

    def apply_analysis(self, data: DataNode, analysis_type: str, params: Dict[str, Any]) -> None:
        """
        Applies an analysis task to the given DataNode.

        Args:
            data (DataNode): The DataNode to analyze.
            analysis_type (str): The type of analysis to perform.
            params (Dict[str, Any]): Parameters for the analysis task.

        Returns:
            AnalysisResult: The result of the analysis.
        """

        if analysis_type not in self.tasks_registry:
            raise ValueError(f"Analysis type '{analysis_type}' not found in the task registry.")

        task = self.tasks_registry[analysis_type]
        task_instance = task(params)
        result, result_type, dependencies = task_instance.execute(data)

        # Emit signal to add the result to the DataNode
        self.analysis_completed.emit(result, result_type, dependencies, data, analysis_type, params)

    def __repr__(self) -> str:
        """
        Provides a string representation of the AnalysisManager instance.

        Returns:
            str: A string describing the AnalysisManager and its managed analyses.
        """
        return f"AnalysisManager(analyses={len(self.analyses)} results)"
