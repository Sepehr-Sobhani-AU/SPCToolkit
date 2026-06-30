
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
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

    def confirm_before_execute(self, data_node: DataNode, params: Dict[str, Any]) -> Optional[str]:
        """
        Optional pre-flight check, run on the main thread *before* the worker
        thread starts. Return a warning message to show the user as a Yes/No
        confirmation (e.g. "this will be slow, proceed?"); the operation only
        runs if they confirm. Return None to run without prompting (default).

        Must be cheap and side-effect free -- it runs synchronously on the UI
        thread. Do NOT do the heavy work here; just decide whether to warn.

        Args:
            data_node (DataNode): The target branch the plugin will run on.
            params (Dict[str, Any]): The parameters collected from the dialog.

        Returns:
            Optional[str]: Warning text to confirm, or None for no prompt.
        """
        return None

    def requires_selection(self) -> Optional[str]:
        """What selection this plugin consumes at execute time, if any:

        * ``None`` / ``False`` — needs no selection (default).
        * ``"points"`` — picked or polygon-selected points in the viewer.
        * ``"branches"`` — one or more selected tree branches.
        * ``"either"`` — at least one of the above.

        The legacy boolean contract still works: ``True`` is read as
        ``"points"``. Consumed in two places (via ``application/selection_gate``):
        a normal menu run prompts — non-modally — only when the needed selection
        is *absent*, then proceeds; pipeline replay *always* pauses here so the
        user can select on the freshly-produced intermediate (e.g. the clusters)
        rather than rely on a stale selection.
        """
        return None

    def build_param_dialog(self, parent, last_params=None) -> Optional[Any]:
        """Optional custom parameter dialog, replacing the auto-generated form.

        Return a ``QDialog`` whose ``exec_()`` collects input and whose
        ``get_parameters()`` returns the params dict passed to ``execute`` — the
        same contract ``DynamicDialog`` satisfies. Return None (default) to use
        the schema-driven dialog built from ``get_parameters()``.

        ``last_params`` is the params dict from this plugin's previous run (or
        None on first use), so a custom dialog can pre-populate its widgets the
        way the schema dialog restores last-used defaults.

        Use this only when the schema dialog can't express the input (e.g. an
        attribute query builder with a variable number of condition rows). The
        returned params MUST stay JSON-serialisable so the step replays from
        saved pipelines without reopening the dialog.
        """
        return None


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

    def requires_selection(self) -> Optional[str]:
        """What selection this action plugin consumes at execute time, if any —
        e.g. a seed point for region growing. Same contract as
        ``Plugin.requires_selection``: ``None``/``False`` (none, default),
        ``"points"``, ``"branches"``, or ``"either"`` (legacy ``True`` ⇒
        ``"points"``). A normal menu run prompts non-modally only when the needed
        selection is absent; pipeline replay always pauses here.
        """
        return None


# Legacy alias for backward compatibility during migration
AnalysisPlugin = Plugin
