# plugins/020_Points/040_Color/000_color_by_branch_plugin.py
from typing import Dict, Any, List, Tuple
import uuid

import numpy as np
from scipy.spatial import cKDTree

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.colors import Colors


class ColorByBranchPlugin(Plugin):
    """
    Colour a branch using the RGB colours of another (source) branch.

    The user picks a source branch from a dropdown. When the source is an
    ancestor of the selected branch (the common case -- e.g. restoring the
    original scan RGB onto a derived subset), the colours are mapped by index by
    composing the lineage masks: fast and exact, no coordinate search. Otherwise
    each point in the selected branch is assigned the colour of its nearest point
    in the source branch, so unrelated branches with different point counts/order
    still work. The result is emitted as a ``colors`` node on the selected
    branch; the source branch is left unchanged.
    """

    # Node types whose transformer preserves point identity, order and count
    # (they only attach attributes / colours or pass the cloud through). Mirrors
    # SubtractPlugin._IDENTITY_TYPES so lineage reasoning stays consistent.
    _IDENTITY_TYPES = frozenset({
        "values", "eigenvalues", "colors", "normals", "dist_to_ground",
        "cluster_labels", "container", "vector_feature", "cad_object",
    })

    def get_name(self) -> str:
        return "color_by_branch"

    def get_parameters(self) -> Dict[str, Any]:
        # Build a dropdown of all branches to pick the colour source from.
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes

        node_options: Dict[str, str] = {}
        if data_nodes is not None:
            for node_uid, node in data_nodes.data_nodes.items():
                node_options[str(node_uid)] = node.alias or node.params

        default_value = next(iter(node_options)) if node_options else ""

        return {
            "source_node": {
                "type": "dropdown",
                "options": node_options,
                "default": default_value,
                "label": "Colour Source Branch",
                "description": "Branch whose RGB colours are copied onto the selected branch"
            }
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        from config.config import global_variables
        data_nodes = global_variables.global_data_nodes
        controller = global_variables.global_application_controller

        target_pc: PointCloud = data_node.data
        if target_pc.points is None or target_pc.size == 0:
            raise ValueError("This branch has no points to colour.")

        # Resolve and reconstruct the source branch.
        try:
            source_uid = uuid.UUID(params["source_node"])
        except Exception as e:
            raise ValueError(f"Invalid colour source branch: {e}")

        source_node = data_nodes.get_node(source_uid)
        if source_node is None:
            raise ValueError(f"Colour source branch {source_uid} not found.")

        global_variables.global_progress = (None, "Reconstructing colour source branch...")
        source_pc = controller.reconstruct(source_uid)

        source_colors = source_pc.colors
        if source_colors is None or len(source_colors) == 0:
            raise ValueError(
                "The selected source branch has no RGB colours to copy."
            )
        if len(source_colors) != source_pc.size:
            raise ValueError(
                f"Source colour count ({len(source_colors)}) does not match its "
                f"point count ({source_pc.size})."
            )

        source_colors = np.asarray(source_colors, dtype=np.float32)

        # Fast path: if the source is an ancestor of the target, compose the
        # lineage masks into an index map (source point per target point) instead
        # of a coordinate search. The real (non-temporary) target node carries
        # the lineage, so fetch it from the collection by uid.
        target_node = data_nodes.get_node(data_node.uid)
        idx = None
        if target_node is not None:
            idx = self._lineage_index_map(
                target_node, source_node, data_nodes, source_pc.size
            )
            if idx is not None and idx.shape[0] != target_pc.size:
                idx = None  # Lineage assumption violated -- fall back.

        if idx is not None:
            mapped_colors = source_colors[idx]
        else:
            # General case: assign each target point the colour of its nearest
            # source point.
            global_variables.global_progress = (
                50, f"Matching {target_pc.size:,} points to source colours..."
            )
            tree = cKDTree(source_pc.points)
            _, nn_idx = tree.query(target_pc.points, k=1, workers=-1)
            mapped_colors = source_colors[nn_idx]

        result = Colors(mapped_colors)

        dependencies = [data_node.uid, source_uid]
        return result, "colors", dependencies

    @classmethod
    def _lineage_index_map(cls, target_node, source_node, data_nodes, source_size):
        """
        If ``source_node`` is an ancestor of ``target_node`` reachable through a
        pure subset/identity chain, return an int array ``idx`` of length
        ``target.size`` where target point i corresponds to source point
        ``idx[i]``. Returns None when the source isn't such an ancestor, in which
        case the caller falls back to coordinate matching.
        """
        # Ancestor chain of the target, from target up to root.
        chain = []
        node = target_node
        while node is not None:
            chain.append(node)
            node = data_nodes.get_node(node.parent_uid) if node.parent_uid else None

        # The source must lie on the target's ancestor chain.
        pos = None
        for i, n in enumerate(chain):
            if n.uid == source_node.uid:
                pos = i
                break
        if pos is None:
            return None

        # Compose the subset chain from source's child down to target. Process
        # top-down so masks apply in order over the source's point indices.
        idx = np.arange(source_size)
        for node in reversed(chain[:pos]):
            data_type = node.data_type
            if data_type in cls._IDENTITY_TYPES:
                continue  # Count and order preserved; nothing to select.
            if data_type == "masks":
                mask = node.data.mask
                if mask.shape[0] != idx.shape[0]:
                    return None
                idx = idx[mask]
            elif data_type == "class_reference":
                labels = cls._find_cluster_labels(node, data_nodes)
                if labels is None or labels.shape[0] != idx.shape[0]:
                    return None
                sel = np.isin(labels, node.data.cluster_ids)
                idx = idx[sel]
            else:
                return None  # Unknown / reordering transform -> fall back.

        return idx

    @staticmethod
    def _find_cluster_labels(node, data_nodes):
        """
        Walk up from ``node`` to the nearest ``cluster_labels`` ancestor and
        return its per-point integer labels, or None if there is none.
        """
        current = data_nodes.get_node(node.parent_uid) if node.parent_uid else None
        while current is not None:
            if current.data_type == "cluster_labels":
                return getattr(current.data, "labels", None)
            current = data_nodes.get_node(current.parent_uid) if current.parent_uid else None
        return None
