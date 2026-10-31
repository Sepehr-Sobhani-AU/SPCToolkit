"""
Plugin for classifying points as Linear or Non-Linear based on eigenvalue linearity.

Requires eigenvalues to be already computed on the selected node (run Compute Eigenvalues first).
Produces 2 clusters: Linear (high linearity) and Non-Linear (everything else).
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.eigenvalue_utils import EigenvalueUtils


CLASS_NAMES = {0: "Linear", 1: "Non-Linear"}
CLASS_COLORS = {
    "Linear":     np.array([0.1, 0.8, 0.2], dtype=np.float32),
    "Non-Linear": np.array([0.5, 0.5, 0.5], dtype=np.float32),
}


class LinearClassificationPlugin(Plugin):
    """Classify each point as Linear or Non-Linear based on eigenvalue linearity."""

    def get_name(self) -> str:
        return "linear_classification"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "linear_threshold": {
                "type": "float",
                "default": 0.4,
                "min": 0.0,
                "max": 1.0,
                "label": "Linear Threshold",
                "description": "Linearity cutoff — points above this are classified as Linear"
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud = data_node.data
        if not isinstance(point_cloud, PointCloud):
            raise ValueError(
                "Linear Classification requires a PointCloud with eigenvalues. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        eigenvalues = point_cloud.attributes.get('eigenvalues')
        if eigenvalues is None:
            raise ValueError(
                "No eigenvalues found in point cloud attributes. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        features = EigenvalueUtils().compute_geometric_features(eigenvalues)
        match_mask = features["linearity"] > params["linear_threshold"]

        n_points = len(eigenvalues)
        labels = np.ones(n_points, dtype=np.int32)  # default: Non-Linear (1)
        labels[match_mask] = 0  # Linear

        n_match = int(np.sum(match_mask))
        pct = 100.0 * n_match / n_points if n_points > 0 else 0.0
        print(f"  Linear:     {n_match:>8,d} points ({pct:5.1f}%)")
        print(f"  Non-Linear: {n_points - n_match:>8,d} points ({100.0 - pct:5.1f}%)")

        clusters = Clusters(
            labels=labels,
            cluster_names=dict(CLASS_NAMES),
            cluster_colors=dict(CLASS_COLORS),
        )

        return clusters, "cluster_labels", [data_node.uid]
