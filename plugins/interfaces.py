
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from core.entities.data_node import DataNode


class Plugin(ABC):
    """
    Base interface for data processing plugins.

    Plugins implement specific functionality for processing point cloud data.
    These plugins operate on selected data branches and produce new data nodes.
    Each plugin must implement methods to identify itself, define its parameters,
    and execute its operation.

    The menu location for a plugin is automatically determined by its folder location:
    - plugins/Points/cluster.py        -> Menu: Points > Cluster
    - plugins/Analysis/Advanced/pca.py -> Menu: Analysis > Advanced > PCA
    - plugins/core_plugin.py           -> Not in menu (root-level system plugin)
    """

    def __init__(self):
        self.progress_callback = None  # Set by executor: fn(percent, message)

    def report_progress(self, percent: int, message: str = ""):
        """Report progress to the UI. Percent 0-100."""
        cb = getattr(self, 'progress_callback', None)
        if cb:
            cb(percent, message)

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name of the plugin.

        This name is used to identify the plugin in the system and should be unique
        across all plugins. It will also be used as the menu item display name.

        Returns:
            str: The plugin's unique identifier name
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters schema for the dialog box.

        This defines what parameters the plugin needs and how they should be
        presented in the user interface.

        Returns:
            Dict[str, Any]: A dictionary defining parameter names, types, default values,
                           and UI presentation hints. Return empty dict {} if no parameters needed.
        """
        pass

    @abstractmethod
    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the plugin operation on the provided data.

        Args:
            data_node (DataNode): The data node containing the data to process
            params (Dict[str, Any]): The parameters for the operation

        Returns:
            Tuple[Any, str, List]: A tuple containing:
                - The result of the operation (can be PointCloud, Masks, Clusters, etc.)
                - The type of the result (e.g., "point_cloud", "masks", "cluster_labels")
                - A list of dependencies (UIDs of data nodes this result depends on)
        """
        pass


class ActionPlugin(ABC):
    """
    Base interface for action plugins.

    Action plugins perform standalone operations that don't require selected data branches.
    Examples include: importing files, exporting data, opening preferences, etc.

    The menu location for an action plugin is automatically determined by its folder location:
    - plugins/File/import.py     -> Menu: File > Import
    - plugins/Edit/preferences.py -> Menu: Edit > Preferences
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name of the action plugin.

        This name is used to identify the plugin in the system and should be unique
        across all plugins. It will also be used as the menu item display name.

        Returns:
            str: The plugin's unique identifier name
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters schema for the dialog box.

        This defines what parameters the action needs and how they should be
        presented in the user interface.

        Returns:
            Dict[str, Any]: A dictionary defining parameter names, types, default values,
                           and UI presentation hints. Return empty dict {} if no parameters needed.
        """
        pass

    @abstractmethod
    def execute(self, main_window, params: Dict[str, Any]) -> None:
        """
        Execute the action plugin operation.

        Args:
            main_window: The main application window (provides access to all managers)
            params (Dict[str, Any]): The parameters for the operation

        Returns:
            None: Action plugins perform their operations directly (e.g., open dialogs, trigger events)
        """
        pass


# Legacy alias for backward compatibility during migration
AnalysisPlugin = Plugin
