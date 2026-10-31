"""
Plugin for classifying points as Scatter or Non-Scatter based on eigenvalue sphericity.

Requires eigenvalues to be already computed on the selected node (run Compute Eigenvalues first).
Produces 2 clusters: Scatter (high sphericity) and Non-Scatter (everything else).
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from plugins.interfaces import Plugin
from core.entities.data_node import DataNode
from core.entities.point_cloud import PointCloud
from core.entities.clusters import Clusters
from core.services.eigenvalue_utils import EigenvalueUtils


CLASS_NAMES = {0: "Scatter", 1: "Non-Scatter"}
CLASS_COLORS = {
    "Scatter":     np.array([0.2, 0.3, 0.9], dtype=np.float32),
    "Non-Scatter": np.array([0.5, 0.5, 0.5], dtype=np.float32),
}


class ScatterClassificationPlugin(Plugin):
    """Classify each point as Scatter or Non-Scatter based on eigenvalue sphericity."""

    def get_name(self) -> str:
        return "scatter_classification"

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "sphericity_threshold": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "label": "Sphericity Threshold",
                "description": "Sphericity cutoff — points above this are classified as Scatter"
            },
        }

    def execute(self, data_node: DataNode, params: Dict[str, Any]) -> Tuple[Any, str, List]:
        point_cloud = data_node.data
        if not isinstance(point_cloud, PointCloud):
            raise ValueError(
                "Scatter Classification requires a PointCloud with eigenvalues. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        eigenvalues = point_cloud.attributes.get('eigenvalues')
        if eigenvalues is None:
            raise ValueError(
                "No eigenvalues found in point cloud attributes. "
                "Run Compute Eigenvalues first, then select the eigenvalues node."
            )

        features = EigenvalueUtils().compute_geometric_features(eigenvalues)
        match_mask = features["sphericity"] > params["sphericity_threshold"]

        n_points = len(eigenvalues)
        labels = np.ones(n_points, dtype=np.int32)  # default: Non-Scatter (1)
        labels[match_mask] = 0  # Scatter

        n_match = int(np.sum(match_mask))
        pct = 100.0 * n_match / n_points if n_points > 0 else 0.0
        print(f"  Scatter:     {n_match:>8,d} points ({pct:5.1f}%)")
        print(f"  Non-Scatter: {n_points - n_match:>8,d} points ({100.0 - pct:5.1f}%)")

        clusters = Clusters(
            labels=labels,
            cluster_names=dict(CLASS_NAMES),
            cluster_colors=dict(CLASS_COLORS),
        )

        return clusters, "cluster_labels", [data_node.uid]
