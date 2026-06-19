# plugins/030_Selection/020_query_select_plugin.py
"""
Select points on the selected branch by an attribute query (ArcGIS-style
"Select By Attributes"), separating the matches into a new branch.

The query is a list of conditions (field / operator / value) combined with a
single AND/OR combiner, evaluated against every per-point field exposed by the
branch (geometry, normals, intensity, distance-to-ground, and every key in the
cloud's ``attributes`` dict). Field discovery and resolution are shared with
``color_by_value`` via ``services.point_fields``.

The condition list lives entirely in ``params`` (built by a custom dialog), so a
saved pipeline replays the same query on a new cloud with no dialog -- the plugin
returns a ``Masks`` result exactly like ``separate_selected_points``, so branch
creation, naming, and pipeline capture all happen on the standard analysis path.
"""

from typing import Dict, Any, List, Tuple

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.masks import Masks
from services.attribute_query import evaluate_query


class QuerySelectPlugin(Plugin):
    """Separate points matching an attribute query into a new branch."""

    def get_name(self) -> str:
        return "query_select"

    def get_parameters(self) -> Dict[str, Any]:
        # Input is collected by the custom builder dialog, not a flat schema.
        return {}

    def build_param_dialog(self, parent):
        from config.config import global_variables
        from gui.dialog_boxes.query_builder_dialog import QueryBuilderDialog
        from services.point_fields import representative_cloud
        import uuid

        controller = global_variables.global_application_controller
        data_nodes = global_variables.global_data_nodes
        node = None
        if controller is not None and data_nodes is not None:
            selected = getattr(controller, "selected_branches", None) or []
            if selected:
                try:
                    node = data_nodes.get_node(uuid.UUID(str(selected[0])))
                except Exception:
                    node = None

        pc = representative_cloud(node)
        return QueryBuilderDialog(node, pc, parent)

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud = data_node.data

        clauses = params.get("clauses", [])
        combiner = params.get("combiner", "AND")
        if not clauses:
            raise ValueError("Query has no conditions.")

        mask = evaluate_query(point_cloud, clauses, combiner)
        if not mask.any():
            raise ValueError("No points match the query.")

        return Masks(mask), "masks", [data_node.uid]
