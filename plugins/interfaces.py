
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from core.data_node import DataNode


class AnalysisPlugin(ABC):
    """
    Base interface for analysis plugins.

    Analysis plugins implement specific algorithms for processing point cloud data.
    Each plugin must implement methods to identify itself, define its parameters,
    and execute its analysis algorithm.
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name of the analysis plugin.

        This name is used to identify the plugin in the system and should be unique
        across all analysis plugins.

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
                           and UI presentation hints
        """
        pass

    @abstractmethod
    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        """
        Execute the analysis algorithm on the provided data.

        Args:
            data_node (DataNode): The data node containing the point cloud to analyze
            params (Dict[str, Any]): The parameters for the analysis algorithm

        Returns:
            Tuple[Any, str, List]: A tuple containing:
                - The result of the analysis
                - The type of the result (e.g., "masks", "cluster_labels")
                - A list of dependencies (UIDs of data nodes this result depends on)
        """
        pass


class MenuPlugin(ABC):
    """
    Base interface for menu plugins.

    Menu plugins define new menu items and actions that will appear in the
    application's menu bar. Each plugin can add multiple menu items to
    different locations in the menu structure.
    """

    @abstractmethod
    def get_menu_location(self) -> str:
        """
        Return the menu location where this plugin's items should appear.

        The location can be a top-level menu name (e.g., "File", "Action") or
        a submenu path separated by slashes (e.g., "Action/Clustering").

        Returns:
            str: The menu location path
        """
        pass

    @abstractmethod
    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Return a list of menu items with their properties.

        Each menu item is defined as a dictionary with properties that determine
        how it appears and behaves in the menu.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing at minimum:
                - "name": The display name of the menu item
                - "action": A unique identifier for the action
        """
        pass

    @abstractmethod
    def handle_action(self, action_name: str, main_window):
        """
        Handle the action when a menu item is triggered.

        This method is called when the user clicks on a menu item provided by this plugin.

        Args:
            action_name (str): The unique identifier of the triggered action
            main_window: A reference to the main application window
        """
        pass
